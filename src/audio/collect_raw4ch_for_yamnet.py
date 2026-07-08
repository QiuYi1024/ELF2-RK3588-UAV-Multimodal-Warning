import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pyaudio
import soundfile as sf


# ==========================================================
# USB 麦克风阵列 6 通道输入布局
# ch0: 内部处理后的音频，不使用
# ch1~ch4: 四个原始麦克风通道，需要保存
# ch5: playback/loopback，不使用
# ==========================================================

RATE = 16000
DEVICE_CHANNELS = 6
CHUNK = 2000

# 采集保存原始四通道：设备 ch1~ch4
RAW_MIC_CHANNELS = [1, 2, 3, 4]

# 和 drone_tracker.py 一致，后续 YAMNet 用 ch1
YAMNET_SOURCE_CHANNEL = 1


def sanitize_name(name: str) -> str:
    name = str(name).strip()
    if not name:
        return "unknown"

    out = []
    for c in name:
        if c.isalnum() or c in ["_", "-"]:
            out.append(c)
        else:
            out.append("_")
    return "".join(out)


def list_audio_devices():
    p = pyaudio.PyAudio()
    print("\n[INFO] PyAudio 输入设备列表：")

    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if int(dev.get("maxInputChannels", 0)) > 0:
            print(
                f"  index={i:2d} | "
                f"name={dev.get('name')} | "
                f"input_channels={dev.get('maxInputChannels')} | "
                f"default_rate={dev.get('defaultSampleRate')}"
            )

    p.terminate()
    print()


def find_audio_array_device():
    p = pyaudio.PyAudio()
    target_index = None
    fallback_index = None

    print("\n[INFO] 正在搜索 6 通道 USB 麦克风阵列输入设备...")

    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        name = dev.get("name", "")
        lower_name = name.lower()
        max_input = int(dev.get("maxInputChannels", 0))

        if max_input > 0:
            print(f"  index={i:2d} | name={name} | input_channels={max_input}")

        if max_input >= DEVICE_CHANNELS and any(token in lower_name for token in ("usb", "array", "mic", "microphone")):
            target_index = i
            print(f"\n[OK] 找到 6 通道 USB 麦克风阵列: index={target_index}")
            break
        if max_input >= DEVICE_CHANNELS and fallback_index is None:
            fallback_index = i

    p.terminate()
    return target_index if target_index is not None else fallback_index


def calc_dbfs_int16(x: np.ndarray):
    x = x.astype(np.float32)
    peak = np.max(np.abs(x)) + 1e-12
    rms = np.sqrt(np.mean(x ** 2) + 1e-12)

    peak_dbfs = 20.0 * np.log10(peak / 32768.0 + 1e-12)
    rms_dbfs = 20.0 * np.log10(rms / 32768.0 + 1e-12)

    return float(rms_dbfs), float(peak_dbfs)


def create_session_dirs(base_dir, label, scene, distance_m):
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    label_clean = sanitize_name(label)
    scene_clean = sanitize_name(scene)

    if distance_m is None or distance_m < 0:
        distance_str = "unknown_distance"
    else:
        distance_str = f"{distance_m:g}m"

    session_name = f"{time_str}_{scene_clean}_{distance_str}"

    session_dir = Path(base_dir).expanduser().resolve() / label_clean / session_name

    raw4ch_dir = session_dir / "0_raw_ch1_ch4"
    yamnet_ch1_dir = session_dir / "1_yamnet_ch1_raw"
    metadata_dir = session_dir / "metadata"

    raw4ch_dir.mkdir(parents=True, exist_ok=False)
    yamnet_ch1_dir.mkdir(parents=True, exist_ok=False)
    metadata_dir.mkdir(parents=True, exist_ok=False)

    return session_dir, raw4ch_dir, yamnet_ch1_dir, metadata_dir


def save_json(path: Path, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Collect raw ch1~ch4 audio from a USB microphone array, and extract raw ch1 for YAMNet dataset."
    )

    parser.add_argument("--list-devices", action="store_true", help="只列出音频设备，不采集")
    parser.add_argument("--device", type=int, default=None, help="手动指定 USB 麦克风阵列设备 index")

    parser.add_argument(
        "--base-dir",
        type=str,
        default="./data/dataset_raw4ch",
        help="数据集根目录"
    )

    parser.add_argument("--label", type=str, default="drone", help="标签，例如 drone")
    parser.add_argument("--scene", type=str, default="unknown_scene", help="场景，例如 outdoor_open")
    parser.add_argument("--distance-m", type=float, default=-1, help="距离，单位 m，不知道填 -1")
    parser.add_argument("--note", type=str, default="", help="备注")

    parser.add_argument("--segment-sec", type=float, default=10.0, help="每段音频长度，默认 10 秒")
    parser.add_argument("--num-segments", type=int, default=0, help="采集段数，0 表示一直采到 Ctrl+C")

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    if args.device is None:
        device_index = find_audio_array_device()
    else:
        device_index = args.device

    if device_index is None:
        print("\n[ERROR] 没找到 6 通道 USB 麦克风阵列设备。")
        print("先运行：")
        print("  python collect_raw4ch_for_yamnet.py --list-devices")
        print("然后手动指定：")
        print("  python collect_raw4ch_for_yamnet.py --device 设备编号 --label drone --scene outdoor_open")
        sys.exit(1)

    p = pyaudio.PyAudio()
    dev_info = p.get_device_info_by_index(device_index)

    if int(dev_info.get("maxInputChannels", 0)) < DEVICE_CHANNELS:
        print("[ERROR] 当前设备输入通道数不足 6。")
        print(dev_info)
        p.terminate()
        sys.exit(1)

    session_dir, raw4ch_dir, yamnet_ch1_dir, metadata_dir = create_session_dirs(
        base_dir=args.base_dir,
        label=args.label,
        scene=args.scene,
        distance_m=args.distance_m,
    )

    index_csv = session_dir / "index.csv"
    session_info_json = session_dir / "session_info.json"

    session_info = {
        "created_at": datetime.now().isoformat(),
        "purpose": "collect raw microphone array ch1~ch4, then extract raw ch1 for YAMNet dataset",
        "label": args.label,
        "scene": args.scene,
        "distance_m": args.distance_m,
        "note": args.note,
        "sample_rate": RATE,
        "device_channels": DEVICE_CHANNELS,
        "chunk": CHUNK,
        "saved_raw_channels": {
            "raw4ch_wav_channel_0": "device ch1 raw mic, same source as drone_tracker.py audio_matrix[:, 1]",
            "raw4ch_wav_channel_1": "device ch2 raw mic",
            "raw4ch_wav_channel_2": "device ch3 raw mic",
            "raw4ch_wav_channel_3": "device ch4 raw mic",
        },
        "yamnet_channel": {
            "source": "device ch1",
            "code_equivalent": "audio_matrix[:, 1]",
            "processing": "raw int16 wav only, no gain, no tanh, no averaging",
        },
        "excluded_channels": {
            "device_ch0": "internal processed audio, not saved",
            "device_ch5": "playback/loopback, not saved",
        },
        "device_index": device_index,
        "device_name": dev_info.get("name"),
        "session_dir": str(session_dir),
    }

    save_json(session_info_json, session_info)

    print("\n" + "=" * 90)
    print("[DATASET SESSION CREATED]")
    print(f"  session_dir     : {session_dir}")
    print(f"  raw4ch_dir      : {raw4ch_dir}")
    print(f"  yamnet_ch1_dir  : {yamnet_ch1_dir}")
    print(f"  metadata_dir    : {metadata_dir}")
    print(f"  index_csv       : {index_csv}")
    print("=" * 90)

    print("\n[IMPORTANT]")
    print("  采集阶段保存原始四通道 ch1~ch4。")
    print("  不保存 ch0。")
    print("  不做 ch1~ch4 平均。")
    print("  不做 /32768、增益、tanh。")
    print("  额外导出的 YAMNet ch1 wav 也是原始 int16 数据。")
    print()

    stream = None

    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=DEVICE_CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK,
        )

        samples_per_segment = int(RATE * args.segment_sec)
        segment_id = 0

        with open(index_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "segment_id",
                    "label",
                    "scene",
                    "distance_m",
                    "start_time",
                    "end_time",
                    "duration_sec",
                    "sample_rate",
                    "raw4ch_wav",
                    "yamnet_ch1_raw_wav",
                    "metadata_json",
                    "rms_dbfs_ch1",
                    "peak_dbfs_ch1",
                    "note",
                ],
            )
            writer.writeheader()

            print("[START] 开始采集，按 Ctrl+C 停止。\n")

            while True:
                if args.num_segments > 0 and segment_id >= args.num_segments:
                    break

                start_time = datetime.now()
                print(f"[REC] segment {segment_id:06d}, duration={args.segment_sec:g}s ...")

                chunks_raw4ch = []
                chunks_ch1 = []
                collected = 0

                while collected < samples_per_segment:
                    need = samples_per_segment - collected
                    frames_to_read = min(CHUNK, need)

                    data = stream.read(frames_to_read, exception_on_overflow=False)

                    # 原始 6 通道输入
                    audio_matrix = np.frombuffer(data, dtype=np.int16).reshape(-1, DEVICE_CHANNELS)

                    # 保存原始 ch1~ch4，不包含 ch0
                    raw4ch = audio_matrix[:, RAW_MIC_CHANNELS].copy()

                    # 单独导出 ch1，和 drone_tracker.py 的 audio_matrix[:, 1] 对齐
                    yamnet_ch1 = audio_matrix[:, YAMNET_SOURCE_CHANNEL].copy()

                    chunks_raw4ch.append(raw4ch)
                    chunks_ch1.append(yamnet_ch1)

                    collected += raw4ch.shape[0]

                raw4ch_int16 = np.vstack(chunks_raw4ch).astype(np.int16)
                yamnet_ch1_int16 = np.concatenate(chunks_ch1).astype(np.int16)

                end_time = datetime.now()

                raw4ch_path = raw4ch_dir / f"seg_{segment_id:06d}_raw_ch1_ch4.wav"
                yamnet_ch1_path = yamnet_ch1_dir / f"seg_{segment_id:06d}_ch1_for_yamnet_raw.wav"
                metadata_path = metadata_dir / f"seg_{segment_id:06d}.json"

                # 1. 保存原始 4 通道 ch1~ch4
                sf.write(
                    file=str(raw4ch_path),
                    data=raw4ch_int16,
                    samplerate=RATE,
                    subtype="PCM_16",
                )

                # 2. 保存原始 ch1 单通道，给 YAMNet 数据集用
                sf.write(
                    file=str(yamnet_ch1_path),
                    data=yamnet_ch1_int16,
                    samplerate=RATE,
                    subtype="PCM_16",
                )

                rms_dbfs_ch1, peak_dbfs_ch1 = calc_dbfs_int16(yamnet_ch1_int16)

                metadata = {
                    "segment_id": segment_id,
                    "label": args.label,
                    "scene": args.scene,
                    "distance_m": args.distance_m,
                    "note": args.note,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_sec": args.segment_sec,
                    "sample_rate": RATE,
                    "raw4ch_wav": str(raw4ch_path),
                    "yamnet_ch1_raw_wav": str(yamnet_ch1_path),
                    "raw4ch_channel_mapping": {
                        "wav_channel_0": "device ch1 raw mic",
                        "wav_channel_1": "device ch2 raw mic",
                        "wav_channel_2": "device ch3 raw mic",
                        "wav_channel_3": "device ch4 raw mic",
                    },
                    "yamnet_source": {
                        "source_channel": "device ch1",
                        "code_equivalent": "audio_matrix[:, 1]",
                        "processing": "raw int16 wav, no average, no gain, no tanh",
                    },
                    "excluded_channels": {
                        "ch0": "processed channel, not used",
                        "ch5": "loopback/playback channel, not used",
                    },
                    "rms_dbfs_ch1": rms_dbfs_ch1,
                    "peak_dbfs_ch1": peak_dbfs_ch1,
                }

                save_json(metadata_path, metadata)

                writer.writerow(
                    {
                        "segment_id": segment_id,
                        "label": args.label,
                        "scene": args.scene,
                        "distance_m": args.distance_m,
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "duration_sec": args.segment_sec,
                        "sample_rate": RATE,
                        "raw4ch_wav": str(raw4ch_path),
                        "yamnet_ch1_raw_wav": str(yamnet_ch1_path),
                        "metadata_json": str(metadata_path),
                        "rms_dbfs_ch1": f"{rms_dbfs_ch1:.2f}",
                        "peak_dbfs_ch1": f"{peak_dbfs_ch1:.2f}",
                        "note": args.note,
                    }
                )
                f.flush()

                print(f"[SAVE] raw 4ch        : {raw4ch_path}")
                print(f"[SAVE] yamnet ch1 raw : {yamnet_ch1_path}")
                print(f"[SAVE] metadata       : {metadata_path}")
                print(f"[STAT] ch1 RMS={rms_dbfs_ch1:.1f} dBFS, PEAK={peak_dbfs_ch1:.1f} dBFS")

                if peak_dbfs_ch1 > -1.0:
                    print("[WARN] ch1 峰值接近 0 dBFS，可能削波/爆音。")
                elif peak_dbfs_ch1 < -45.0:
                    print("[WARN] ch1 声音偏小，可能距离较远或环境声太弱。")

                print()

                segment_id += 1

    except KeyboardInterrupt:
        print("\n[INFO] 用户停止采集。")

    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        p.terminate()

    print("\n[DONE] 本次采集结束。")
    print(f"[DONE] session 目录：{session_dir}")
    print(f"[DONE] 索引文件：{index_csv}")


if __name__ == "__main__":
    main()
