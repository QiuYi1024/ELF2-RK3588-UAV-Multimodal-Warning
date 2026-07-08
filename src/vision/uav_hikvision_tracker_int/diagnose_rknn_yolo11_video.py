#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RK3588 RKNN YOLO11n 单输出模型诊断脚本

用途：
  用 _assets/videos/drone_test.mp4 直接测试 best_uav_headless_i8.rknn，判断问题到底是：
    1. 模型量化后输出全 0
    2. RGB/BGR 预处理不匹配
    3. 输出布局 CHW/HWC 解析错
    4. 置信度通道不是第 4 通道
    5. 坐标尺度/letterbox 还原错
    6. 阈值太高

推荐放在 RK3588 工程目录：
  <repo_root>/src/vision/uav_hikvision_tracker_int

运行：
  conda activate yolo_env
  cd <repo_root>/src/vision/uav_hikvision_tracker_int

  python3 diagnose_rknn_yolo11_video.py \
    --model ./model/best_uav_headless_i8.rknn \
    --video ./_assets/videos/drone_test.mp4 \
    --outdir ./diag_rknn_drone \
    --max-frames 120 \
    --conf 0.05

如果测试视频不在 ./_assets/videos/ 目录，请改成实际路径。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# RKNNLite import
# ============================================================

def import_rknn_lite():
    try:
        from rknnlite.api import RKNNLite
        return RKNNLite
    except Exception as e1:
        try:
            from rknn_lite.api import RKNNLite
            return RKNNLite
        except Exception as e2:
            print("[FATAL] 无法导入 RKNNLite。")
            print("尝试过：from rknnlite.api import RKNNLite")
            print("尝试过：from rknn_lite.api import RKNNLite")
            print("错误1:", repr(e1))
            print("错误2:", repr(e2))
            print("\n请确认 RK3588 Python 环境安装了 rknn-toolkit-lite2。")
            sys.exit(1)


# ============================================================
# Letterbox
# ============================================================

def letterbox_bgr(img_bgr, new_shape=640, color=(114, 114, 114)):
    h, w = img_bgr.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dw = new_shape - new_w
    dh = new_shape - new_h
    left = int(round(dw / 2 - 0.1))
    right = int(round(dw / 2 + 0.1))
    top = int(round(dh / 2 - 0.1))
    bottom = int(round(dh / 2 + 0.1))

    out = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )
    return out, r, (left, top)


def xywh_to_xyxy_original(cx, cy, bw, bh, ratio, pad, orig_w, orig_h):
    left, top = pad

    x1 = (cx - bw / 2 - left) / ratio
    y1 = (cy - bh / 2 - top) / ratio
    x2 = (cx + bw / 2 - left) / ratio
    y2 = (cy + bh / 2 - top) / ratio

    x1 = max(0, min(orig_w - 1, x1))
    y1 = max(0, min(orig_h - 1, y1))
    x2 = max(0, min(orig_w - 1, x2))
    y2 = max(0, min(orig_h - 1, y2))

    return int(x1), int(y1), int(x2), int(y2)


# ============================================================
# Decode helpers
# ============================================================

def sigmoid(x):
    x = np.clip(x, -80, 80)
    return 1.0 / (1.0 + np.exp(-x))


def normalize_conf(arr, use_sigmoid=False):
    arr = arr.astype(np.float32)
    if use_sigmoid:
        return sigmoid(arr)
    # 如果已经 0~1，则原样；否则当作 logit sigmoid
    if np.nanmin(arr) >= 0.0 and np.nanmax(arr) <= 1.0:
        return arr
    return sigmoid(arr)


def as_float_array(out):
    a = np.asarray(out)
    return a.astype(np.float32).reshape(-1), str(a.dtype), list(a.shape)


def channel_stats(name, vals):
    vals = np.asarray(vals, dtype=np.float32)
    if vals.size == 0:
        return f"{name}: empty"
    return (
        f"{name}: min={float(np.nanmin(vals)):.6g}, "
        f"max={float(np.nanmax(vals)):.6g}, "
        f"mean={float(np.nanmean(vals)):.6g}, "
        f"std={float(np.nanstd(vals)):.6g}"
    )


def decode_layout(flat, layout, attrs, n, img_size, conf_channel, conf_thres, use_sigmoid):
    if layout == "CHW":
        # [attrs, n]
        get_ch = lambda c: flat[c * n:(c + 1) * n]
        cx = get_ch(0).astype(np.float32)
        cy = get_ch(1).astype(np.float32)
        bw = get_ch(2).astype(np.float32)
        bh = get_ch(3).astype(np.float32)
        cf_raw = get_ch(conf_channel).astype(np.float32)
    elif layout == "HWC":
        # [n, attrs]
        arr = flat.reshape(n, attrs).astype(np.float32)
        cx = arr[:, 0]
        cy = arr[:, 1]
        bw = arr[:, 2]
        bh = arr[:, 3]
        cf_raw = arr[:, conf_channel]
    else:
        raise ValueError(layout)

    conf = normalize_conf(cf_raw, use_sigmoid=use_sigmoid)

    # 自动处理归一化坐标
    coord_max = max(float(np.nanmax(np.abs(cx))), float(np.nanmax(np.abs(cy))),
                    float(np.nanmax(np.abs(bw))), float(np.nanmax(np.abs(bh))))
    if coord_max <= 2.0:
        cx = cx * img_size
        cy = cy * img_size
        bw = bw * img_size
        bh = bh * img_size

    # 合理框约束
    finite = np.isfinite(cx) & np.isfinite(cy) & np.isfinite(bw) & np.isfinite(bh) & np.isfinite(conf)
    reasonable = (
        finite &
        (bw > 1.0) & (bh > 1.0) &
        (bw < img_size * 1.2) & (bh < img_size * 1.2) &
        (cx > -img_size * 0.2) & (cx < img_size * 1.2) &
        (cy > -img_size * 0.2) & (cy < img_size * 1.2)
    )

    keep = reasonable & (conf >= conf_thres)

    boxes = []
    idxs = np.where(keep)[0]
    for i in idxs:
        boxes.append({
            "i": int(i),
            "cx": float(cx[i]),
            "cy": float(cy[i]),
            "w": float(bw[i]),
            "h": float(bh[i]),
            "conf": float(conf[i]),
            "raw_conf": float(cf_raw[i]),
        })

    # 按置信度排序
    boxes.sort(key=lambda x: x["conf"], reverse=True)

    return {
        "layout": layout,
        "attrs": attrs,
        "n": n,
        "conf_channel": conf_channel,
        "use_sigmoid": use_sigmoid,
        "coord_abs_max": coord_max,
        "max_conf": float(np.nanmax(conf)),
        "mean_conf": float(np.nanmean(conf)),
        "raw_conf_min": float(np.nanmin(cf_raw)),
        "raw_conf_max": float(np.nanmax(cf_raw)),
        "raw_conf_mean": float(np.nanmean(cf_raw)),
        "reasonable": int(np.sum(reasonable)),
        "kept": int(len(boxes)),
        "top": boxes[:10],
    }


def nms_boxes(decoded, ratio, pad, orig_w, orig_h, iou_thres=0.45, max_det=20):
    boxes_cv = []
    scores = []
    raw_boxes = []

    for b in decoded["top"][:2000]:
        x1, y1, x2, y2 = xywh_to_xyxy_original(
            b["cx"], b["cy"], b["w"], b["h"],
            ratio, pad, orig_w, orig_h
        )
        if x2 <= x1 or y2 <= y1:
            continue
        boxes_cv.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(float(b["conf"]))
        raw_boxes.append((x1, y1, x2, y2, float(b["conf"]), b))

    if not boxes_cv:
        return []

    idxs = cv2.dnn.NMSBoxes(boxes_cv, scores, score_threshold=0.0, nms_threshold=iou_thres)
    if len(idxs) == 0:
        return []

    idxs = np.array(idxs).reshape(-1).tolist()
    out = []
    for idx in idxs[:max_det]:
        out.append(raw_boxes[idx])
    return out


def draw_dets(frame, dets, label):
    out = frame.copy()
    for x1, y1, x2, y2, conf, _ in dets:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out, f"{label} {conf:.3f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2
        )
    return out


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./model/best_uav_headless_i8.rknn")
    ap.add_argument("--video", default="./_assets/videos/drone_test.mp4")
    ap.add_argument("--outdir", default="./diag_rknn_drone")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--rotate180", action="store_true")
    ap.add_argument("--save-video", action="store_true")
    args = ap.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    video_path = Path(args.video).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print("[FATAL] 模型不存在:", model_path)
        sys.exit(1)
    if not video_path.exists():
        print("[FATAL] 视频不存在:", video_path)
        sys.exit(1)

    print("========== RKNN YOLO11 VIDEO DIAG ==========")
    print("model:", model_path)
    print("video:", video_path)
    print("outdir:", outdir)
    print("conf:", args.conf)

    RKNNLite = import_rknn_lite()
    rknn = RKNNLite()

    print("[RKNN] load_rknn...")
    ret = rknn.load_rknn(str(model_path))
    if ret != 0:
        print("[FATAL] load_rknn failed:", ret)
        sys.exit(ret)

    print("[RKNN] init_runtime...")
    # 先用 Core 0，避免多核干扰诊断
    try:
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    except Exception:
        ret = rknn.init_runtime()
    if ret != 0:
        print("[FATAL] init_runtime failed:", ret)
        sys.exit(ret)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("[FATAL] OpenCV 无法打开视频:", video_path)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[VIDEO] {orig_w}x{orig_h}, fps={fps:.2f}, frames={total}")

    writer = None
    if args.save_video:
        out_video = str(outdir / "diag_best_layout.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video, fourcc, max(1.0, fps / max(1, args.stride)), (orig_w, orig_h))
        print("[SAVE VIDEO]", out_video)

    preprocess_modes = ["RGB", "BGR"]
    layout_candidates = []

    report = {
        "model": str(model_path),
        "video": str(video_path),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "frames": [],
        "summary": {},
    }

    global_scores = {}

    frame_id = 0
    used_frames = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1

        if frame_id % args.stride != 0:
            continue
        if args.rotate180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        used_frames += 1
        if used_frames > args.max_frames:
            break

        lb_bgr, ratio, pad = letterbox_bgr(frame, args.imgsz)

        frame_info = {
            "frame_id": frame_id,
            "used_idx": used_frames,
            "modes": {},
        }

        best_key = None
        best_dets = []

        for mode in preprocess_modes:
            if mode == "RGB":
                inp = cv2.cvtColor(lb_bgr, cv2.COLOR_BGR2RGB)
            else:
                inp = lb_bgr.copy()

            inp = np.expand_dims(inp, axis=0).astype(np.uint8)

            # RKNNLite 默认 NHWC；部分版本支持 data_format 参数，做兼容。
            try:
                outs = rknn.inference(inputs=[inp], data_format="nhwc")
            except TypeError:
                outs = rknn.inference(inputs=[inp])

            if outs is None or len(outs) == 0:
                frame_info["modes"][mode] = {"error": "no outputs"}
                continue

            flat, dtype, shape = as_float_array(outs[0])
            elem = flat.size

            # 统计 raw output
            raw_stats = {
                "dtype": dtype,
                "shape": shape,
                "elem": int(elem),
                "min": float(np.nanmin(flat)) if elem else None,
                "max": float(np.nanmax(flat)) if elem else None,
                "mean": float(np.nanmean(flat)) if elem else None,
                "std": float(np.nanstd(flat)) if elem else None,
                "nonzero": int(np.count_nonzero(flat)),
            }

            candidates = []
            if elem % 5 == 0:
                n = elem // 5
                for layout in ["CHW", "HWC"]:
                    for conf_channel in range(4, 5):
                        for use_sigmoid in [False, True]:
                            d = decode_layout(
                                flat=flat,
                                layout=layout,
                                attrs=5,
                                n=n,
                                img_size=args.imgsz,
                                conf_channel=conf_channel,
                                conf_thres=args.conf,
                                use_sigmoid=use_sigmoid,
                            )
                            candidates.append(d)

            # 如果输出不是 5 attrs，也尝试 6 attrs
            if elem % 6 == 0:
                n = elem // 6
                for layout in ["CHW", "HWC"]:
                    for conf_channel in [4, 5]:
                        for use_sigmoid in [False, True]:
                            d = decode_layout(
                                flat=flat,
                                layout=layout,
                                attrs=6,
                                n=n,
                                img_size=args.imgsz,
                                conf_channel=conf_channel,
                                conf_thres=args.conf,
                                use_sigmoid=use_sigmoid,
                            )
                            candidates.append(d)

            # 给候选打分：优先有 reasonable/kept，但不要 thousands 个假框
            for d in candidates:
                key = f"{mode}_{d['layout']}_a{d['attrs']}_c{d['conf_channel']}_sig{int(d['use_sigmoid'])}"
                score = 0.0
                if 0 < d["kept"] <= 200:
                    score += 1000 + d["kept"]
                elif d["kept"] > 200:
                    score += 100
                score += min(1.0, d["max_conf"]) * 10
                global_scores.setdefault(key, {"score": 0.0, "frames_with": 0, "max_conf": 0.0, "kept_sum": 0})
                global_scores[key]["score"] += score
                global_scores[key]["max_conf"] = max(global_scores[key]["max_conf"], d["max_conf"])
                global_scores[key]["kept_sum"] += d["kept"]
                if d["kept"] > 0:
                    global_scores[key]["frames_with"] += 1

            # 排序输出前几种
            candidates.sort(
                key=lambda d: (
                    1 if (0 < d["kept"] <= 200) else 0,
                    min(200, d["kept"]),
                    d["max_conf"]
                ),
                reverse=True
            )

            frame_info["modes"][mode] = {
                "raw": raw_stats,
                "top_candidates": candidates[:6],
            }

            # 本帧选最优候选画图
            if candidates:
                cand = candidates[0]
                label = f"{mode}-{cand['layout']}-a{cand['attrs']}-c{cand['conf_channel']}-sig{int(cand['use_sigmoid'])}"
                dets = nms_boxes(cand, ratio, pad, frame.shape[1], frame.shape[0], args.iou)
                if len(dets) > len(best_dets):
                    best_dets = dets
                    best_key = label

        if used_frames <= 10 or used_frames % 20 == 0:
            print(f"\n[FRAME {used_frames}] source_frame={frame_id}")
            for mode, info in frame_info["modes"].items():
                raw = info.get("raw", {})
                print(f"  [{mode}] raw shape={raw.get('shape')} dtype={raw.get('dtype')} "
                      f"min={raw.get('min')} max={raw.get('max')} mean={raw.get('mean')} "
                      f"nonzero={raw.get('nonzero')}")
                for d in info.get("top_candidates", [])[:3]:
                    print(f"    cand layout={d['layout']} attrs={d['attrs']} c={d['conf_channel']} "
                          f"sig={d['use_sigmoid']} max_conf={d['max_conf']:.6f} "
                          f"reasonable={d['reasonable']} kept={d['kept']} "
                          f"top={d['top'][:2]}")

        if best_dets:
            vis = draw_dets(frame, best_dets, best_key or "best")
        else:
            vis = frame.copy()
            cv2.putText(vis, "NO DET", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        if used_frames <= 20 or best_dets:
            img_path = outdir / f"frame_{used_frames:04d}.jpg"
            cv2.imwrite(str(img_path), vis)

        if writer:
            writer.write(vis)

        frame_info["best_key"] = best_key
        frame_info["best_det_count"] = len(best_dets)
        report["frames"].append(frame_info)

    cap.release()
    if writer:
        writer.release()
    rknn.release()

    dt = time.time() - t0
    print("\n========== GLOBAL CANDIDATE SCORE ==========")
    ranked = sorted(global_scores.items(), key=lambda kv: kv[1]["score"], reverse=True)
    for k, v in ranked[:20]:
        print(f"{k}: score={v['score']:.1f}, frames_with={v['frames_with']}, "
              f"max_conf={v['max_conf']:.6f}, kept_sum={v['kept_sum']}")

    report["summary"] = {
        "used_frames": used_frames,
        "elapsed_sec": dt,
        "approx_fps": used_frames / dt if dt > 0 else 0.0,
        "global_scores": ranked[:50],
    }

    report_path = outdir / "diag_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n[OK] report saved:", report_path)
    print("[OK] images saved to:", outdir)
    print("\n下一步：把终端输出里的前 10 帧 raw/cand 信息，以及 GLOBAL CANDIDATE SCORE 发给我。")


if __name__ == "__main__":
    main()
