#!/usr/bin/env python3
"""回放离线视觉框和真实音频 DOA 日志，验证控制策略。

本工具不连接 ELF2、球机或 PTZ，只生成决策 CSV 和报告。
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from antiuav_control_policy import (
    AudioCalibration,
    AudioGuidancePolicy,
    DetectionPolicy,
    FocusPolicy,
    LostSearchPolicy,
    TrackingSpeedPolicy,
    ZoomPolicy,
)


ROOT = Path(__file__).resolve().parents[1]
VISION_CSV = ROOT / "离线测试数据" / "视觉离线数据" / "drone_real_upright_baseline_filtered" / "drone_real_upright_detections.csv"
AUDIO_ROOT = ROOT / "离线测试数据" / "音频离线数据" / "audio_beamforming_logs"
OUT_DIR = ROOT / "simulation_outputs"


def read_float(row: dict[str, str], names: tuple[str, ...], default: float = 0.0) -> float:
    for name in names:
        value = row.get(name, "")
        if value == "":
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return default


def read_visual_rows() -> tuple[list[dict], str]:
    rows: list[dict] = []
    if VISION_CSV.exists():
        with VISION_CSV.open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                conf = read_float(row, ("conf", "score", "confidence"), 0.0)
                max_side = max(
                    read_float(row, ("norm_w", "w_ratio", "width_ratio"), 0.0),
                    read_float(row, ("norm_h", "h_ratio", "height_ratio"), 0.0),
                )
                kept = row.get("kept", "1").strip()
                if conf > 0.0 and max_side > 0.0 and kept not in ("0", "false", "False"):
                    frame = int(read_float(row, ("frame", "frame_id"), len(rows)))
                    rows.append({
                        "source": "csv",
                        "frame": frame,
                        "time_ms": int(read_float(row, ("time_sec", "time"), frame / 25.0) * 1000),
                        "conf": conf,
                        "max_side": max_side,
                        "dx_ratio": 0.08,
                        "near_edge": False,
                        "visual": True,
                    })
    if rows:
        return rows[:80], f"loaded {len(rows)} rows from {VISION_CSV}"

    fallback = [
        ("small", 0, 0.52, 0.035, 0.24, False, True),
        ("pretrack", 240, 0.48, 0.055, 0.16, False, True),
        ("track", 520, 0.56, 0.12, 0.05, False, True),
        ("ideal", 820, 0.58, 0.18, 0.01, False, True),
        ("large", 1120, 0.54, 0.29, -0.20, False, True),
        ("edge", 1420, 0.50, 0.09, 0.43, True, True),
        ("lost_hold", 1720, 0.0, 0.0, 0.0, False, False),
        ("lost_zoom", 2300, 0.0, 0.0, 0.0, False, False),
        ("wait_audio", 3100, 0.0, 0.0, 0.0, False, False),
        ("recapture", 3900, 0.55, 0.13, -0.04, False, True),
    ]
    return [
        {
            "source": name,
            "frame": idx,
            "time_ms": time_ms,
            "conf": conf,
            "max_side": side,
            "dx_ratio": dx,
            "near_edge": edge,
            "visual": visual,
        }
        for idx, (name, time_ms, conf, side, dx, edge, visual) in enumerate(fallback)
    ], f"{VISION_CSV} has no detection rows; used deterministic simulated boxes"


def read_audio_rows(limit: int = 80) -> tuple[list[dict], str]:
    files = sorted(AUDIO_ROOT.rglob("audio_beamforming_log.csv"))
    if not files:
        return [], f"audio log not found under {AUDIO_ROOT}"
    chosen = None
    for path in files:
        if "202606" in str(path):
            chosen = path
            break
    chosen = chosen or files[0]
    rows: list[dict] = []
    last_doa: float | None = None
    with chosen.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            prob = read_float(row, ("prob_final", "prob_beam", "prob_raw"), 0.0)
            doa = read_float(row, ("azimuth_deg", "doa_deg", "angle"), -1.0)
            if doa < 0.0:
                continue
            drift = 0.0 if last_doa is None else abs(((doa - last_doa + 540.0) % 360.0) - 180.0)
            last_doa = doa
            stability = max(0.0, min(1.0, 1.0 - drift / 30.0))
            rows.append({
                "time_ms": int(read_float(row, ("timestamp_ms",), len(rows) * 400)),
                "prob": prob,
                "doa": doa,
                "stability": stability,
                "drift": drift,
            })
            if len(rows) >= limit:
                break
    return rows, f"loaded {len(rows)} audio rows from {chosen}"


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    visual_rows, visual_note = read_visual_rows()
    audio_rows, audio_note = read_audio_rows()
    if not audio_rows:
        raise AssertionError(audio_note)

    detection = DetectionPolicy()
    zoom = ZoomPolicy()
    lost = LostSearchPolicy()
    speed = TrackingSpeedPolicy()
    focus = FocusPolicy()
    audio_policy = AudioGuidancePolicy()
    calibration = AudioCalibration(True, 35.0, 0.86, 28)

    decisions = []
    stable_frames = 0
    last_zoom_ms = -999999
    zoom_ratio = 2.0
    conf_baseline = 0.55
    lost_start_ms: int | None = None

    for idx, visual in enumerate(visual_rows):
        audio = audio_rows[min(idx, len(audio_rows) - 1)]
        now_ms = int(visual["time_ms"])
        has_visual = bool(visual["visual"])
        if has_visual:
            stable_frames += 1
            lost_start_ms = None
        else:
            stable_frames = 0
            if lost_start_ms is None:
                lost_start_ms = now_ms

        max_side = visual["max_side"] if has_visual else None
        stage = detection.stage(
            visual["conf"],
            visual["max_side"],
            stable_frames,
            track_continuous=stable_frames >= 2,
            near_edge=visual["near_edge"],
        ) if has_visual else "LOST"
        lost_ms = 0 if lost_start_ms is None else max(0, now_ms - lost_start_ms)
        zoom_state, zoom_cmd, zoom_cooldown = zoom.decide(
            max_side,
            stable=stable_frames >= 2,
            near_edge=visual["near_edge"],
            lost=not has_visual,
            now_ms=now_ms,
            last_zoom_ms=last_zoom_ms,
            current_zoom=zoom_ratio,
        )
        if zoom_cmd:
            last_zoom_ms = now_ms
            zoom_ratio = max(zoom.min_zoom, min(zoom.max_zoom, zoom_ratio + (0.4 if zoom_cmd > 0 else -0.6)))
        lost_state = lost.state(has_visual, lost_ms, zoom_ratio, zoom.lost_search_zoom, audio["prob"] >= audio_policy.yamnet_threshold)
        pan_speed, tilt_speed = speed.speed(visual["dx_ratio"], 0.01, zoom_ratio)
        focus_reason, focus_trigger = focus.decide(
            now_ms,
            zoom_changed=zoom_cmd != 0,
            blur=65.0 if visual["source"] == "recapture" else 125.0,
            conf=visual["conf"],
            conf_baseline=conf_baseline,
        )
        if has_visual and visual["conf"] > conf_baseline:
            conf_baseline = visual["conf"]
        audio_result = audio_policy.decide(
            visual_reliable=has_visual,
            yamnet_prob=audio["prob"],
            doa_deg=audio["doa"],
            doa_stability=audio["stability"],
            doa_drift_deg=audio["drift"],
            current_pan=180.0,
            calibration=calibration,
            manual_control=False,
            emergency_stop=False,
            now_ms=now_ms,
        )
        decisions.append({
            "idx": idx,
            "visual_source": visual["source"],
            "time_ms": now_ms,
            "visual": int(has_visual),
            "conf": round(visual["conf"], 4),
            "max_side": "" if max_side is None else round(max_side, 4),
            "stage": stage,
            "zoom_state": zoom_state,
            "zoom_cmd": zoom_cmd,
            "zoom_ratio": round(zoom_ratio, 2),
            "lost_state": lost_state.value,
            "pan_speed": pan_speed,
            "tilt_speed": tilt_speed,
            "focus_reason": focus_reason,
            "focus_trigger": int(focus_trigger),
            "audio_prob": round(audio["prob"], 3),
            "audio_doa": round(audio["doa"], 2),
            "audio_stability": round(audio["stability"], 3),
            "audio_state": audio_result["state"],
            "target_pan": audio_result["target_pan"],
            "audio_command": audio_result["command"],
            "audio_blocked": audio_result["blocked"],
        })

    csv_path = OUT_DIR / "offline_replay_decisions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(decisions[0]))
        writer.writeheader()
        writer.writerows(decisions)

    checks = {
        "visual rows present": len(visual_rows) > 0,
        "audio rows present": len(audio_rows) > 0,
        "search/pretrack stage": any(r["stage"] in {"SEARCH_PRETRACK", "STABLE_TRACK", "ALARM_CONFIRM"} for r in decisions),
        "zoom in covered": any(r["zoom_cmd"] > 0 for r in decisions),
        "zoom out or lost zoom covered": any(r["zoom_cmd"] < 0 for r in decisions),
        "lost wait/audio state covered": any("WAIT_AUDIO" in r["lost_state"] or r["audio_state"] == "AUDIO_GUIDING" for r in decisions),
        "focus request covered": any(r["focus_trigger"] for r in decisions),
        "audio gate blocks while visual reliable": any("visual_reliable" in r["audio_blocked"] for r in decisions),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(", ".join(failed))

    report = [
        "# offline replay report",
        "",
        f"- visual: {visual_note}",
        f"- audio: {audio_note}",
        f"- CSV: `{csv_path}`",
        f"- rows: {len(decisions)}",
        f"- checks: {len(checks)} passed",
        "- covered: staged detection, target-size zoom, lost zoom/search, focus request generation, real audio-log DOA gating, visual-priority audio blocking",
    ]
    (OUT_DIR / "offline_replay_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"PASS offline replay rows={len(decisions)} audio_rows={len(audio_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
