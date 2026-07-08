import argparse
import json
import math
import os
import socket
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pyaudio
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import usb.core
from rknnlite.api import RKNNLite

try:
    from tuning import Tuning
except ImportError:
    print("找不到 tuning.py，请确保脚本在 usb_4_mic_array 目录下。")
    sys.exit(1)


GLOBAL_ANGLE = -1.0
GLOBAL_IS_VOICE = 0
GLOBAL_DOA_TS_MS = 0
GLOBAL_DOA_LOCK = threading.Lock()

DEFAULT_QT_HOST = os.environ.get("ANTI_UAV_QT_HOST", "DEVICE_IP")
DEFAULT_QT_PORT = int(os.environ.get("ANTI_UAV_AUDIO_PORT", "5006"))
DEFAULT_FUSION_HOST = os.environ.get("ANTI_UAV_FUSION_HOST", "127.0.0.1")
DEFAULT_FUSION_PORT = int(os.environ.get("ANTI_UAV_FUSION_PORT", "5007"))
SCRIPT_DIR = Path(__file__).resolve().parent


def default_rknn_model_path():
    env_model = os.environ.get("ANTI_UAV_YAMNET_MODEL")
    if env_model:
        return env_model

    candidates = [
        SCRIPT_DIR.parent / "drone_yamnet.rknn",
        Path(os.environ.get("ANTI_UAV_YAMNET_RKNN", "models/drone_yamnet.rknn")),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[-1])


DEFAULT_RKNN_MODEL = default_rknn_model_path()


def parse_args():
    parser = argparse.ArgumentParser(description="RK3588 YAMNET audio tracker")
    parser.add_argument("--qt-host", default=DEFAULT_QT_HOST, help="本机 Qt UDP 地址")
    parser.add_argument("--qt-port", type=int, default=DEFAULT_QT_PORT, help="本机 Qt 音频 UDP 端口")
    parser.add_argument("--fusion-host", default=DEFAULT_FUSION_HOST, help="RK 本机 YOLO 融合 UDP 地址")
    parser.add_argument("--fusion-port", type=int, default=DEFAULT_FUSION_PORT, help="RK 本机 YOLO 融合 UDP 端口")
    parser.add_argument("--model", default=DEFAULT_RKNN_MODEL, help="YAMNET RKNN 模型路径")
    parser.add_argument("--on-threshold", type=float, default=float(os.environ.get("ANTI_UAV_AUDIO_ON_THRESHOLD", "10.0")))
    parser.add_argument("--off-threshold", type=float, default=float(os.environ.get("ANTI_UAV_AUDIO_OFF_THRESHOLD", "5.0")))
    parser.add_argument("--confirm-hits", type=int, default=int(os.environ.get("ANTI_UAV_AUDIO_CONFIRM_HITS", "2")))
    parser.add_argument("--confirm-window", type=int, default=int(os.environ.get("ANTI_UAV_AUDIO_CONFIRM_WINDOW", "6")))
    parser.add_argument("--hold-frames", type=int, default=int(os.environ.get("ANTI_UAV_AUDIO_HOLD_FRAMES", "12")))
    parser.add_argument("--doa-window", type=int, default=int(os.environ.get("ANTI_UAV_AUDIO_DOA_WINDOW", "12")))
    parser.add_argument("--min-stability", type=float, default=float(os.environ.get("ANTI_UAV_AUDIO_MIN_STABILITY", "0.55")))
    parser.add_argument("--doa-stale-ms", type=int, default=int(os.environ.get("ANTI_UAV_AUDIO_DOA_STALE_MS", "700")))
    parser.add_argument("--jump-reject-deg", type=float, default=float(os.environ.get("ANTI_UAV_AUDIO_JUMP_REJECT_DEG", "75.0")))
    return parser.parse_args()


def mono_ms():
    return int(time.monotonic() * 1000)


def wall_ms():
    return int(time.time() * 1000)


def wrap_deg(angle):
    if angle is None:
        return -1.0
    return float(angle) % 360.0


def circular_delta_deg(a, b):
    return ((float(a) - float(b) + 540.0) % 360.0) - 180.0


def circular_mean_deg(values):
    if not values:
        return -1.0, 0.0
    radians = np.deg2rad(np.asarray(values, dtype=np.float32))
    sin_mean = float(np.mean(np.sin(radians)))
    cos_mean = float(np.mean(np.cos(radians)))
    stability = math.sqrt(sin_mean * sin_mean + cos_mean * cos_mean)
    angle = math.degrees(math.atan2(sin_mean, cos_mean))
    return wrap_deg(angle), stability


def dbfs(samples):
    rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
    if rms <= 1e-9:
        return -120.0
    return 20.0 * math.log10(min(1.0, rms))


def read_global_doa():
    with GLOBAL_DOA_LOCK:
        return float(GLOBAL_ANGLE), int(GLOBAL_IS_VOICE), int(GLOBAL_DOA_TS_MS)


def hardware_doa_thread():
    global GLOBAL_ANGLE, GLOBAL_IS_VOICE, GLOBAL_DOA_TS_MS

    while True:
        dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
        if not dev:
            time.sleep(1)
            continue

        try:
            mic_tuning = Tuning(dev)
            while True:
                angle = mic_tuning.direction
                is_voice = mic_tuning.is_voice()
                with GLOBAL_DOA_LOCK:
                    GLOBAL_ANGLE = wrap_deg(angle)
                    GLOBAL_IS_VOICE = int(is_voice)
                    GLOBAL_DOA_TS_MS = mono_ms()
                time.sleep(0.02)
        except Exception as exc:
            print(f"[WARN] DOA thread reconnecting: {exc}")
            time.sleep(0.5)


class YAMNetProcessor:
    def __init__(self, model_path):
        self.rknn = RKNNLite()
        if self.rknn.load_rknn(model_path) != 0:
            print(f"YAMNET 模型加载失败: {model_path}")
            sys.exit(1)
        if self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_2) != 0:
            print("YAMNET NPU 运行时初始化失败。")
            sys.exit(1)

        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            win_length=400,
            hop_length=160,
            f_min=125.0,
            f_max=7500.0,
            n_mels=64,
            power=2.0,
            center=True,
            mel_scale="slaney",
            norm="slaney",
        )

    def waveform_to_patch(self, waveform_tensor):
        mel_spec = self.mel_transform(waveform_tensor)
        log_mel = torch.log(mel_spec + 0.001).transpose(1, 2)
        if log_mel.size(1) > 96:
            log_mel = log_mel[:, :96, :]
        elif log_mel.size(1) < 96:
            log_mel = F.pad(log_mel, (0, 0, 0, 96 - log_mel.size(1)))
        return log_mel.unsqueeze(-1).numpy().astype(np.float32)

    def softmax(self, x):
        x = x.astype(np.float32)
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def infer(self, audio_buffer_1sec):
        waveform_tensor = torch.from_numpy(audio_buffer_1sec).unsqueeze(0)
        input_data = self.waveform_to_patch(waveform_tensor)
        t0 = time.perf_counter()
        outputs = self.rknn.inference(inputs=[input_data], data_format=["nhwc"])
        infer_ms = (time.perf_counter() - t0) * 1000.0
        probs = self.softmax(outputs[0][0]) * 100.0
        return float(probs[1]), float(infer_ms)

    def release(self):
        self.rknn.release()


class AudioStateMachine:
    def __init__(self, args):
        self.on_threshold = args.on_threshold
        self.off_threshold = args.off_threshold
        self.confirm_hits = max(1, args.confirm_hits)
        self.hits = deque(maxlen=max(1, args.confirm_window))
        self.hold_frames = max(1, args.hold_frames)
        self.score_ema = -1.0
        self.state = "IDLE"
        self.hold_left = 0

    def update(self, score):
        if self.score_ema < 0:
            self.score_ema = float(score)
        else:
            self.score_ema = 0.75 * self.score_ema + 0.25 * float(score)

        strong = self.score_ema >= self.on_threshold
        weak = self.score_ema >= self.off_threshold
        self.hits.append(1 if strong else 0)

        confirmed_by_window = strong and sum(self.hits) >= self.confirm_hits

        if confirmed_by_window:
            self.state = "CONFIRMED"
            self.hold_left = self.hold_frames
        elif self.state == "CONFIRMED" and weak:
            self.hold_left = self.hold_frames
        elif self.hold_left > 0 and weak:
            self.state = "HOLD"
            self.hold_left -= 1
        elif weak and any(self.hits):
            self.state = "CANDIDATE"
            self.hold_left = max(0, self.hold_left - 1)
        else:
            self.state = "IDLE"
            self.hold_left = 0

        detected = weak and self.state in ("CONFIRMED", "HOLD")
        return detected, self.state, self.score_ema


def select_audio_input_device(pyaudio_obj):
    keyword = os.environ.get("ANTI_UAV_AUDIO_DEVICE_KEYWORD", "").strip().lower()
    fallback_index = None
    for i in range(pyaudio_obj.get_device_count()):
        dev_info = pyaudio_obj.get_device_info_by_index(i)
        max_inputs = int(dev_info.get("maxInputChannels", 0))
        if max_inputs < 6:
            continue

        name = dev_info.get("name", "").lower()
        if keyword and keyword in name:
            return i
        if any(token in name for token in ("usb", "array", "mic", "microphone")):
            return i
        if fallback_index is None:
            fallback_index = i
    return fallback_index


def main():
    args = parse_args()
    print("=" * 65)
    print("AntiUAV YAMNET audio tracker with fusion output")
    print("=" * 65)

    qt_address = (args.qt_host, args.qt_port)
    fusion_address = (args.fusion_host, args.fusion_port)
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[INFO] Qt audio UDP: {qt_address[0]}:{qt_address[1]}")
    print(f"[INFO] YOLO fusion UDP: {fusion_address[0]}:{fusion_address[1]}")

    threading.Thread(target=hardware_doa_thread, daemon=True).start()

    print(f"[INFO] YAMNET RKNN model: {args.model}")
    rknn_engine = YAMNetProcessor(args.model)
    state_machine = AudioStateMachine(args)
    doa_history = deque(maxlen=max(3, args.doa_window))
    last_smooth = -1.0
    last_doa_accept_ms = 0

    chunk = 2000
    rate = 16000
    channels = 6
    audio_buffer = np.zeros(rate, dtype=np.float32)

    p = pyaudio.PyAudio()
    target_device_index = select_audio_input_device(p)
    if target_device_index is None:
        print("未在 PyAudio 列表中找到支持 6 通道的 USB 麦克风阵列设备。")
        sys.exit(1)

    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=target_device_index,
            frames_per_buffer=chunk,
        )
    except OSError as exc:
        print(f"声卡打开失败: {exc}")
        sys.exit(1)

    try:
        while True:
            data = stream.read(chunk, exception_on_overflow=False)
            audio_matrix = np.frombuffer(data, dtype=np.int16).reshape(-1, channels)

            # 与数据集采集说明保持一致：ch1 raw mic 对应 audio_matrix[:, 1]。
            raw_mic_1 = audio_matrix[:, 1].astype(np.float32) / 32768.0
            rms_dbfs = dbfs(raw_mic_1)

            # 不再默认 tanh 大增益压缩，避免改变训练/采集分布。
            final_audio = np.clip(raw_mic_1, -1.0, 1.0)
            audio_buffer[: rate - chunk] = audio_buffer[chunk:]
            audio_buffer[rate - chunk :] = final_audio

            score, infer_ms = rknn_engine.infer(audio_buffer)
            detected_raw, audio_state, score_ema = state_machine.update(score)

            raw_doa, is_voice, doa_sample_ts_ms = read_global_doa()
            raw_doa_valid = 0.0 <= raw_doa < 360.0
            if detected_raw and raw_doa_valid:
                if last_smooth < 0 or abs(circular_delta_deg(raw_doa, last_smooth)) <= args.jump_reject_deg:
                    doa_history.append(raw_doa)
                    last_doa_accept_ms = mono_ms()
                elif len(doa_history) < 3:
                    doa_history.append(raw_doa)
                    last_doa_accept_ms = mono_ms()
            elif not detected_raw:
                doa_history.clear()
                last_smooth = -1.0
                last_doa_accept_ms = 0

            smooth_doa, doa_stability = circular_mean_deg(list(doa_history))
            if smooth_doa >= 0:
                last_smooth = smooth_doa

            doa_recent = last_doa_accept_ms > 0 and (mono_ms() - last_doa_accept_ms) <= args.doa_stale_ms
            angle_valid = detected_raw and doa_recent and smooth_doa >= 0 and doa_stability >= args.min_stability
            audio_detected = bool(detected_raw and angle_valid)
            out_angle = smooth_doa if audio_detected else -1.0

            payload = {
                "type": "audio",
                "audio_detected": audio_detected,
                "detected": audio_detected,
                "audio_state": audio_state,
                "angle": out_angle,
                "doa_deg": out_angle,
                "raw_doa_deg": raw_doa if raw_doa_valid else -1.0,
                "doa_stability": float(doa_stability),
                "confidence": float(score),
                "yamnet_score": float(score),
                "score_ema": float(score_ema),
                "rms_dbfs": float(rms_dbfs),
                "is_voice": int(is_voice),
                "audio_infer_ms": float(infer_ms),
                "ts_ms": wall_ms(),
                "ts": time.time(),
                "ts_mono_ms": mono_ms(),
                "doa_sample_ts_mono_ms": int(doa_sample_ts_ms),
            }

            encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            udp_sock.sendto(encoded, qt_address)
            udp_sock.sendto(encoded, fusion_address)

            if audio_detected:
                bar = "#" * int(min(20, max(0, score_ema / 5.0)))
                print(
                    f"[AUDIO] {audio_state:9s} doa={out_angle:6.1f} raw={raw_doa:6.1f} "
                    f"stab={doa_stability:.2f} score={score:5.1f}% ema={score_ema:5.1f}% [{bar:<20}]"
                )
            else:
                print(
                    f"[AUDIO] {audio_state:9s} doa=--- raw={raw_doa:6.1f} "
                    f"stab={doa_stability:.2f} score={score:5.1f}% ema={score_ema:5.1f}%   ",
                    end="\r",
                )

    except KeyboardInterrupt:
        print("\n[INFO] audio tracker stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        rknn_engine.release()
        udp_sock.close()


if __name__ == "__main__":
    main()
