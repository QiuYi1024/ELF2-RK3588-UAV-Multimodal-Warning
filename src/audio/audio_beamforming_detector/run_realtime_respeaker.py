from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import socket
import sys
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent
PARENT_DIR = PACKAGE_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from audio_beamforming_detector.array_processing import (  # noqa: E402
    FrequencyDelaySumBeamformer,
    DOAStabilizer,
    SrpPhatEstimator,
    build_mic_positions,
    create_doa_provider,
    estimate_snr_db,
    post_filter_waveform,
    preprocess_array,
)
from audio_beamforming_detector.backends import MockArrayBackend, ReSpeaker4Backend  # noqa: E402
from audio_beamforming_detector.detector import WindowAlarmLogic, create_detector, fuse_max  # noqa: E402
from audio_beamforming_detector.utils import load_config  # noqa: E402


def wall_ms() -> int:
    return int(time.time() * 1000)


def mono_ms() -> int:
    return int(time.monotonic() * 1000)


def _env_or_config(env_name: str, value, default):
    env_value = os.environ.get(env_name)
    if env_value is not None and env_value != "":
        return env_value
    if value is not None and value != "":
        return value
    return default


def _parse_target_present(value) -> int:
    if value is None or value == "":
        return -1
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "present", "drone"):
        return 1
    if text in ("0", "false", "no", "n", "absent", "none", "noise"):
        return 0
    return -1


def _json_compact(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _channel_stats_int16(matrix: np.ndarray) -> dict:
    data = np.asarray(matrix, dtype=np.int16)
    if data.ndim != 2 or data.shape[1] == 0:
        return {
            "rms": [],
            "peak": [],
            "clip_ratio": [],
            "zero_ratio": [],
            "mean": [],
            "std": [],
        }
    x = data.astype(np.float32)
    return {
        "rms": [float(np.sqrt(np.mean(ch * ch) + 1e-9)) for ch in x],
        "peak": [int(np.max(np.abs(ch.astype(np.int32)))) for ch in data],
        "clip_ratio": [float(np.mean(np.abs(ch.astype(np.int32)) >= 32760)) for ch in data],
        "zero_ratio": [float(np.mean(ch == 0)) for ch in data],
        "mean": [float(np.mean(ch)) for ch in x],
        "std": [float(np.std(ch)) for ch in x],
    }


def _waveform_stats_float(waveform: np.ndarray) -> dict:
    x = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return {
            "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0,
            "rms": 0.0, "peak": 0.0, "clip_ratio": 0.0, "zero_ratio": 1.0,
        }
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "rms": float(np.sqrt(np.mean(x * x) + 1e-12)),
        "peak": float(np.max(np.abs(x))),
        "clip_ratio": float(np.mean(np.abs(x) >= 0.999)),
        "zero_ratio": float(np.mean(x == 0.0)),
    }


def _dbfs_from_rms(rms: float) -> float:
    return float(20.0 * np.log10(max(float(rms), 1e-9)))


def _experiment_context(cfg: dict, args) -> dict:
    exp_cfg = cfg.get("experiment", {})
    return {
        "experiment_id": _env_or_config("ANTI_UAV_EXPERIMENT_ID", args.experiment_id or exp_cfg.get("experiment_id"), ""),
        "scene_type": _env_or_config("ANTI_UAV_SCENE_TYPE", args.scene or exp_cfg.get("scene_type"), ""),
        "location": _env_or_config("ANTI_UAV_EXPERIMENT_LOCATION", exp_cfg.get("location"), ""),
        "distance_m": float(_env_or_config("ANTI_UAV_DISTANCE_M", args.distance_m if args.distance_m is not None else exp_cfg.get("distance_m"), -1)),
        "noise_type": _env_or_config("ANTI_UAV_NOISE_TYPE", args.background_noise or exp_cfg.get("noise_type"), ""),
        "background_noise": _env_or_config("ANTI_UAV_BACKGROUND_NOISE", args.background_noise or exp_cfg.get("background_noise"), ""),
        "drone_model": _env_or_config("ANTI_UAV_DRONE_MODEL", exp_cfg.get("drone_model"), ""),
        "flight_mode": _env_or_config("ANTI_UAV_FLIGHT_MODE", exp_cfg.get("flight_mode"), ""),
        "method_label": _env_or_config("ANTI_UAV_EXPERIMENT_METHOD", args.method_label or exp_cfg.get("method_label"), "only_acoustic_yamnet_beam"),
        "module_label": _env_or_config("ANTI_UAV_EXPERIMENT_MODULE", exp_cfg.get("module_label"), "only_acoustic"),
        "trial_id": _env_or_config("ANTI_UAV_TRIAL_ID", args.trial_id or exp_cfg.get("trial_id"), ""),
        "target_present": _parse_target_present(_env_or_config("ANTI_UAV_TARGET_PRESENT", args.target_present if args.target_present is not None else exp_cfg.get("target_present"), "")),
        "power_w": float(_env_or_config("ANTI_UAV_POWER_W", exp_cfg.get("power_w"), -1)),
    }


def _valid_angle(angle: float) -> bool:
    return 0.0 <= float(angle) < 360.0


def _angle_delta_deg(a: float, b: float) -> float:
    return (float(a) - float(b) + 180.0) % 360.0 - 180.0


def _circular_mean_deg(angles) -> float:
    values = [float(a) for a in angles if _valid_angle(float(a))]
    if not values:
        return -1.0
    radians = np.deg2rad(values)
    sin_mean = float(np.mean(np.sin(radians)))
    cos_mean = float(np.mean(np.cos(radians)))
    return float((np.rad2deg(np.arctan2(sin_mean, cos_mean)) + 360.0) % 360.0)


def _circular_std_deg(angles) -> float:
    values = [float(a) for a in angles if _valid_angle(float(a))]
    if len(values) <= 1:
        return 0.0 if values else 999.0
    radians = np.deg2rad(values)
    sin_mean = float(np.mean(np.sin(radians)))
    cos_mean = float(np.mean(np.cos(radians)))
    r = max(1e-6, min(1.0, float(np.hypot(sin_mean, cos_mean))))
    return float(np.rad2deg(np.sqrt(max(0.0, -2.0 * np.log(r)))))


def resolve_model_path(config_path: Path, model_path: str) -> str:
    if not model_path:
        return ""
    p = Path(model_path)
    if p.is_absolute():
        return str(p)
    candidates = [
        config_path.parent / p,
        PACKAGE_DIR.parent / p,
        PACKAGE_DIR.parent.parent / p,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return str(config_path.parent / p)


class RawMultichannelWavRecorder:
    """按需保存阵列原始多通道音频，供后续离线 DOA 和波束分析。"""

    def __init__(
        self,
        root_dir: Path,
        sample_rate: int,
        num_mics: int,
        array_name: str,
        selected_channel_index: int = 0,
    ):
        self.root_dir = root_dir
        self.sample_rate = int(sample_rate)
        self.num_mics = int(num_mics)
        self.array_name = str(array_name)
        self.selected_channel_index = max(0, int(selected_channel_index))
        self._wav = None
        self._selected_wav = None
        self.path = ""
        self.selected_path = ""
        self.session_id = ""
        self.samples = 0
        self.started_ms = 0

    @property
    def active(self) -> bool:
        return self._wav is not None

    def start(self, record_id: str = "", session_id: str = "") -> str:
        if self.active:
            self.stop()
        safe_session = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in str(session_id or "")
        ).strip("_")
        record_root_dir = self.root_dir
        if safe_session and self.root_dir.name == "raw_6ch" and self.root_dir.parent.name == "audio":
            sessions_dir = self.root_dir.parent.parent.parent
            if sessions_dir.name == "sessions":
                record_root_dir = sessions_dir / safe_session / "audio" / "raw_6ch"
        record_root_dir.mkdir(parents=True, exist_ok=True)
        selected_dir = record_root_dir.parent / "selected_channel"
        selected_dir.mkdir(parents=True, exist_ok=True)
        stamp = record_id or time.strftime("%Y%m%d_%H%M%S")
        safe_stamp = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(stamp))
        path = record_root_dir / f"mic_raw_{self.array_name}_{safe_stamp}.wav"
        selected_path = selected_dir / f"mic_selected_ch{self.selected_channel_index}_{safe_stamp}.wav"
        self._wav = wave.open(str(path), "wb")
        self._wav.setnchannels(self.num_mics)
        self._wav.setsampwidth(2)
        self._wav.setframerate(self.sample_rate)
        self._selected_wav = wave.open(str(selected_path), "wb")
        self._selected_wav.setnchannels(1)
        self._selected_wav.setsampwidth(2)
        self._selected_wav.setframerate(self.sample_rate)
        self.path = str(path)
        self.selected_path = str(selected_path)
        self.session_id = str(safe_session or safe_stamp)
        self.samples = 0
        self.started_ms = wall_ms()
        print(f"[REC] raw mic recording started: {self.path}")
        print(f"[REC] selected channel recording started: {self.selected_path}")
        return self.path

    def write(self, mic_data: np.ndarray) -> None:
        if not self.active:
            return
        data = np.asarray(mic_data)
        if data.ndim != 2 or data.shape[0] != self.num_mics:
            return
        if np.issubdtype(data.dtype, np.integer):
            pcm = np.clip(data, -32768, 32767).astype(np.int16, copy=False)
        else:
            pcm = np.clip(data, -1.0, 1.0)
            pcm = (pcm * 32767.0).astype(np.int16)
        interleaved = np.ascontiguousarray(pcm.T)
        self._wav.writeframes(interleaved.tobytes())
        if self._selected_wav is not None:
            selected_index = min(self.selected_channel_index, pcm.shape[0] - 1)
            selected_pcm = np.ascontiguousarray(pcm[selected_index])
            self._selected_wav.writeframes(selected_pcm.tobytes())
        self.samples += int(data.shape[1])
        try:
            # 原始音频用于后续重训，写入后立即刷盘，降低异常退出时的数据损失。
            self._wav._file.flush()
            if self._selected_wav is not None:
                self._selected_wav._file.flush()
        except Exception:
            pass

    def stop(self) -> None:
        if not self.active:
            return
        path = self.path
        selected_path = self.selected_path
        wav = self._wav
        selected_wav = self._selected_wav
        self._wav = None
        self._selected_wav = None
        try:
            wav.close()
        except Exception as exc:
            print(f"[WARN] failed to close raw mic wav cleanly: {exc}")
        if selected_wav is not None:
            try:
                selected_wav.close()
            except Exception as exc:
                print(f"[WARN] failed to close selected channel wav cleanly: {exc}")
        meta = {
            "path": path,
            "selected_channel_path": selected_path,
            "selected_channel_index": self.selected_channel_index,
            "array_name": self.array_name,
            "session_id": self.session_id,
            "num_mics": self.num_mics,
            "sample_rate": self.sample_rate,
            "samples_per_channel": self.samples,
            "started_ms": self.started_ms,
            "ended_ms": wall_ms(),
        }
        try:
            Path(path).with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            if selected_path:
                selected_meta = dict(meta)
                selected_meta["path"] = selected_path
                selected_meta["source_multichannel_path"] = path
                selected_meta["num_mics"] = 1
                Path(selected_path).with_suffix(".json").write_text(
                    json.dumps(selected_meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        except OSError as exc:
            print(f"[WARN] failed to write raw mic metadata: {exc}")
        print(f"[REC] raw mic recording stopped: {path} samples={self.samples}")
        if selected_path:
            print(f"[REC] selected channel recording stopped: {selected_path} samples={self.samples}")
        self.session_id = ""
        self.selected_path = ""


def open_control_socket(host: str, port: int):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, int(port)))
    sock.setblocking(False)
    print(f"[CTRL] listening on UDP {host}:{port}")
    return sock


def poll_control_messages(sock):
    if sock is None:
        return []
    messages = []
    while True:
        try:
            data, _addr = sock.recvfrom(4096)
        except BlockingIOError:
            break
        except OSError as exc:
            print(f"[WARN] control socket recv failed: {exc}")
            break
        try:
            obj = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            print(f"[WARN] invalid control json: {exc}")
            continue
        if isinstance(obj, dict):
            messages.append(obj)
    return messages


def control_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on", "start", "enable", "enabled"):
        return True
    if text in ("0", "false", "no", "off", "stop", "disable", "disabled"):
        return False
    return default


def parse_args():
    ap = argparse.ArgumentParser(description="Realtime ReSpeaker4 beamforming + YAMNet-RKNN detector")
    ap.add_argument("--config", default=str(PACKAGE_DIR / "configs" / "respeaker4.yaml"))
    ap.add_argument("--model-path", default="")
    ap.add_argument("--qt-host", default="DEVICE_IP")
    ap.add_argument("--qt-port", type=int, default=5006)
    ap.add_argument("--fusion-host", default="127.0.0.1")
    ap.add_argument("--fusion-port", type=int, default=5007)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--alsa-device", default="", help="稳定 ALSA PCM 名称，例如 hw:CARD=ArrayUAC10,DEV=0。")
    ap.add_argument("--disable-beamforming", action="store_true")
    ap.add_argument("--disable-post-filter", action="store_true")
    ap.add_argument("--doa-source", choices=["config", "respeaker_hardware", "srp_phat", "none"], default="config")
    ap.add_argument("--disable-software-doa", action="store_true", help="硬件 DOA 不可用时不回退 SRP-PHAT。")
    ap.add_argument("--experiment-id", default="")
    ap.add_argument("--scene", default="")
    ap.add_argument("--distance-m", type=float, default=None)
    ap.add_argument("--background-noise", default="")
    ap.add_argument("--target-present", default=None, help="1/0：当前窗口所在实验是否真实存在无人机，用于后处理统计 Accuracy/Recall/F1。")
    ap.add_argument("--method-label", default="")
    ap.add_argument("--trial-id", default="")
    ap.add_argument("--mock-audio", action="store_true")
    ap.add_argument("--model-path-optional", action="store_true", help="Allow DummyDetector if the model path is unavailable.")
    ap.add_argument("--log-dir", default="")
    ap.add_argument("--control-host", default="0.0.0.0")
    ap.add_argument("--control-port", type=int, default=5008)
    ap.add_argument("--disable-control", action="store_true")
    ap.add_argument("--raw-record-dir", default="")
    ap.add_argument("--max-windows", type=int, default=0, help="测试用：处理指定窗口数后自动退出，0 表示持续运行。")
    return ap.parse_args()


def main():
    args = parse_args()
    stop_requested = {"value": False}
    backend_holder = {"value": None}

    def request_stop(signum, _frame):
        if not stop_requested["value"]:
            print(f"\n[INFO] received signal {signum}, realtime audio detector stopping...")
        stop_requested["value"] = True
        backend = backend_holder["value"]
        if backend is not None and hasattr(backend, "request_stop"):
            backend.request_stop()

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            signal.signal(sig, request_stop)

    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)
    array_cfg = cfg["array"]
    beam_cfg = cfg.get("beamforming", {})
    prep_cfg = cfg.get("preprocess", {})
    det_cfg = cfg.get("detector", {})

    if args.model_path:
        det_cfg["model_path"] = args.model_path
    det_cfg["model_path"] = resolve_model_path(config_path, det_cfg.get("model_path", ""))

    sample_rate = int(array_cfg["sample_rate"])
    window_samples = int(float(det_cfg.get("window_sec", 1.0)) * sample_rate)
    hop_samples = int(float(det_cfg.get("hop_sec", 0.5)) * sample_rate)
    ref_channel = int(array_cfg.get("ref_channel", 0))
    mic_positions = build_mic_positions(array_cfg)
    experiment = _experiment_context(cfg, args)

    doa_source = str(beam_cfg.get("doa_source", beam_cfg.get("doa_method", "srp_phat"))).lower()
    if args.doa_source != "config":
        doa_source = args.doa_source
    doa_fallback = str(beam_cfg.get("doa_fallback", "srp_phat")).lower()
    trust_hardware_doa = bool(beam_cfg.get("trust_hardware_doa", True))

    if args.mock_audio:
        backend = MockArrayBackend(sample_rate, int(array_cfg["num_mics"]), mic_positions=mic_positions)
    else:
        backend = ReSpeaker4Backend(
            sample_rate=sample_rate,
            raw_channels=array_cfg.get("raw_channels", [1, 2, 3, 4]),
            device_channels=int(array_cfg.get("device_channels", 6)),
            device_index=args.device,
            frames_per_buffer=hop_samples,
            alsa_device=args.alsa_device,
            reopen_attempts=int(os.environ.get("ANTI_UAV_AUDIO_REOPEN_ATTEMPTS", "30")),
        )
    backend_holder["value"] = backend

    srp = SrpPhatEstimator(
        mic_positions=mic_positions,
        sample_rate=sample_rate,
        freq_min=float(beam_cfg.get("doa_freq_min", 500)),
        freq_max=float(beam_cfg.get("doa_freq_max", 5500)),
        angle_step_deg=float(beam_cfg.get("angle_step_deg", 5)),
    )
    doa_provider = create_doa_provider(
        doa_source,
        backend,
        srp,
        doa_fallback=doa_fallback,
        disable_software_doa=args.disable_software_doa,
        array_name=str(array_cfg.get("name", "")),
    )
    doa_stabilizer = DOAStabilizer(
        median_window=int(beam_cfg.get("doa_median_window", 5)),
        ema_alpha=float(beam_cfg.get("doa_ema_alpha", 0.35)),
        jump_reject_deg=float(beam_cfg.get("doa_jump_reject_deg", 40.0)),
        stable_window=int(beam_cfg.get("doa_stable_window", 3)),
        stable_max_std_deg=float(beam_cfg.get("doa_stable_max_std_deg", 18.0)),
        hold_sec=float(beam_cfg.get("doa_hold_sec", 0.8)),
        rebase_window=int(beam_cfg.get("doa_rebase_window", 3)),
        rebase_min_count=int(beam_cfg.get("doa_rebase_min_count", 2)),
        rebase_max_std_deg=float(beam_cfg.get("doa_rebase_max_std_deg", 12.0)),
    )
    beamformer = FrequencyDelaySumBeamformer(mic_positions, sample_rate)
    detector = create_detector(det_cfg.get("model_path"), allow_dummy=args.model_path_optional)
    alarm_logic = WindowAlarmLogic(threshold_percent=float(det_cfg.get("threshold_percent", 30.0)))
    guidance_threshold = float(det_cfg.get("guidance_threshold_percent", det_cfg.get("threshold_percent", 30.0)))
    guidance_hold_sec = float(det_cfg.get("guidance_hold_sec", 1.8))
    guidance_window_count = max(1, int(det_cfg.get("guidance_window_count", 3)))
    guidance_min_windows = max(1, int(det_cfg.get("guidance_min_windows", 2)))
    guidance_min_snr_db = float(det_cfg.get("guidance_min_snr_db", 1.5))
    guidance_weak_min_snr_db = float(det_cfg.get("guidance_weak_min_snr_db", max(0.6, guidance_min_snr_db - 0.4)))
    guidance_min_doa_confidence = float(det_cfg.get("guidance_min_doa_confidence", 0.65))
    guidance_require_stable_doa = bool(det_cfg.get("guidance_require_stable_doa", True))
    guidance_max_angle_std_deg = float(det_cfg.get("guidance_max_angle_std_deg", 18.0))
    guidance_max_jump_deg = float(det_cfg.get("guidance_max_jump_deg", 42.0))

    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    qt_addr = (args.qt_host, args.qt_port)
    fusion_addr = (args.fusion_host, args.fusion_port)

    log_dir = Path(args.log_dir or (Path.cwd() / "audio_beamforming_logs" / time.strftime("run_%Y%m%d_%H%M%S")))
    log_dir.mkdir(parents=True, exist_ok=True)
    raw_record_dir = Path(args.raw_record_dir).expanduser() if args.raw_record_dir else (log_dir / "raw_mic_recordings")
    capture_channels = int(getattr(backend, "capture_channels", array_cfg["num_mics"]))
    processing_channels = int(getattr(backend, "processing_channels", array_cfg["num_mics"]))
    selected_device_channels = [int(ch) for ch in array_cfg.get("raw_channels", range(processing_channels))]
    main_device_channel = selected_device_channels[ref_channel]
    raw_recorder = RawMultichannelWavRecorder(
        raw_record_dir,
        sample_rate=sample_rate,
        num_mics=capture_channels,
        array_name=f"{array_cfg.get('name', 'array')}_{capture_channels}ch",
        selected_channel_index=main_device_channel,
    )
    control_sock = None if args.disable_control else open_control_socket(args.control_host, args.control_port)
    csv_path = log_dir / "audio_beamforming_log.csv"
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "timestamp_ms", "window_id", "prob_raw", "prob_beam", "prob_final", "azimuth_deg",
            "raw_doa_deg", "smooth_doa_deg", "doa_valid", "stable_doa", "doa_confidence",
            "srp_score", "doa_source", "hardware_doa_deg", "hardware_voice", "snr_db",
            "raw_infer_ms", "beam_infer_ms", "doa_compute_ms", "beamform_ms",
            "beamforming_enabled", "post_filter_enabled", "alarm", "experiment_id",
            "scene_type", "location", "distance_m", "background_noise", "noise_type",
            "drone_model", "flight_mode", "target_present", "method_label",
            "module_label", "trial_id", "power_w", "audio_state", "guidance_threshold_percent",
            "guidance_hold_sec", "guidance_hold_remaining_ms",
            "guidance_positive_windows", "guidance_angle_std_deg", "guidance_min_snr_db",
            "guidance_weak_min_snr_db", "guidance_min_doa_confidence", "guidance_require_stable_doa",
            "channel_rms_json", "channel_peak_json", "channel_clip_ratio_json", "channel_zero_ratio_json",
            "main_device_channel", "main_channel_mean", "main_channel_std", "main_channel_rms",
            "main_channel_peak", "main_channel_clip_ratio", "main_channel_zero_ratio",
            "log_mel_min", "log_mel_max", "log_mel_mean", "log_mel_std",
            "rknn_input_shape", "rknn_input_dtype", "raw_logits_json", "softmax_percent_json",
            "drone_class_index",
            "raw_recording_active", "raw_recording_path", "raw_recording_session_id", "raw_recording_samples",
        ],
    )
    writer.writeheader()

    rolling = np.zeros((int(array_cfg["num_mics"]), window_samples), dtype=np.float32)
    filled = 0
    window_id = 0
    beamforming_enabled = bool(beam_cfg.get("enable", True)) and not args.disable_beamforming
    post_filter_enabled = bool(beam_cfg.get("enable_post_filter", False)) and not args.disable_post_filter
    last_guidance_until = 0.0
    last_guidance_angle = -1.0
    last_guidance_score = 0.0
    guidance_history = deque(maxlen=guidance_window_count)
    log_flush_interval = max(1, int(os.environ.get("ANTI_UAV_AUDIO_LOG_FLUSH_INTERVAL", "5")))

    print("[INFO] realtime audio beamforming detector started")
    print(f"[INFO] config={config_path}")
    print(f"[INFO] model={det_cfg.get('model_path')} dummy={getattr(detector, 'is_dummy', False)}")
    print(f"[INFO] doa_source={doa_source} fallback={doa_fallback}")
    print(
        f"[INFO] capture_channels={capture_channels} processing_channels={processing_channels} "
        f"selected_device_channels={selected_device_channels} main_device_channel={main_device_channel}"
    )
    print(f"[INFO] experiment={experiment}")
    if getattr(detector, "is_dummy", False):
        print(f"[WARN] detector fallback: {getattr(detector, 'fallback_reason', '')}")
    print(f"[INFO] log={csv_path}")

    try:
        while not stop_requested["value"]:
            chunk = backend.read_frame(hop_samples)
            if chunk.shape[0] != rolling.shape[0]:
                raise RuntimeError(f"backend returned {chunk.shape[0]} channels, expected {rolling.shape[0]}")
            device_chunk = backend.last_device_frame() if hasattr(backend, "last_device_frame") else chunk
            if device_chunk.shape[0] != capture_channels:
                raise RuntimeError(
                    f"backend returned {device_chunk.shape[0]} capture channels, expected {capture_channels}"
                )
            channel_stats = _channel_stats_int16(device_chunk)
            main_device_wave = device_chunk[main_device_channel] if 0 <= main_device_channel < device_chunk.shape[0] else np.array([], dtype=np.int16)
            main_device_stats = _waveform_stats_float(main_device_wave.astype(np.float32) / 32768.0)

            for msg in poll_control_messages(control_sock):
                command = str(msg.get("command", ""))
                if command == "set_raw_recording" or "raw_recording" in msg:
                    enabled = control_bool(msg.get("raw_recording", msg.get("enabled")), False)
                    if enabled and not raw_recorder.active:
                        raw_recorder.start(str(msg.get("record_id", "")), str(msg.get("session_id", "")))
                    elif not enabled and raw_recorder.active:
                        raw_recorder.stop()

            raw_recorder.write(device_chunk)
            rolling[:, :-hop_samples] = rolling[:, hop_samples:]
            rolling[:, -hop_samples:] = chunk[:, :hop_samples]
            filled = min(window_samples, filled + hop_samples)
            if filled < window_samples:
                continue

            t_window = time.perf_counter()
            raw_float = preprocess_array(
                rolling,
                sample_rate,
                remove_dc=bool(prep_cfg.get("remove_dc", True)),
                rms_normalize=False,
                highpass_hz=float(prep_cfg.get("highpass_hz", 100) or 0),
            )
            array_float = preprocess_array(
                rolling,
                sample_rate,
                remove_dc=bool(prep_cfg.get("remove_dc", True)),
                rms_normalize=bool(prep_cfg.get("rms_normalize", True)),
                highpass_hz=float(prep_cfg.get("highpass_hz", 100) or 0),
            )
            raw_ref = raw_float[ref_channel]
            raw_ref_stats = _waveform_stats_float(raw_ref)
            rms_dbfs = _dbfs_from_rms(raw_ref_stats["rms"])

            doa_ms = 0.0
            beam_ms = 0.0
            azimuth_deg = -1.0
            srp_score = 0.0
            actual_doa_source = "none"
            hardware_doa_deg = -1.0
            hardware_voice = 0
            hardware_doa_error = ""
            raw_doa_deg_for_log = -1.0
            doa_valid = False
            stable_doa = False
            doa_confidence = 0.0
            if beamforming_enabled:
                t0 = time.perf_counter()
                raw_doa_result = doa_provider.estimate(array_float)
                if (
                    trust_hardware_doa
                    and raw_doa_result.doa_source == "respeaker_hardware"
                    and raw_doa_result.doa_valid
                    and _valid_angle(raw_doa_result.raw_doa_deg)
                ):
                    # ReSpeaker 硬件 DOA 已经做过阵列定位；这里不再套 EMA/跳变拒绝，
                    # 避免真实换方向时被旧角度保持逻辑拖住。
                    raw_doa_result.raw_doa_deg = float(raw_doa_result.raw_doa_deg) % 360.0
                    raw_doa_result.smooth_doa_deg = raw_doa_result.raw_doa_deg
                    raw_doa_result.stable_doa = True
                    raw_doa_result.doa_confidence = min(max(float(raw_doa_result.doa_confidence), 0.0), 1.0)
                    raw_doa_result.raw_confidence = min(max(float(raw_doa_result.raw_confidence), 0.0), 1.0)
                    doa_result = raw_doa_result
                else:
                    doa_result = doa_stabilizer.update(raw_doa_result)
                doa_ms = (time.perf_counter() - t0) * 1000.0
                azimuth_deg = doa_result.smooth_doa_deg if doa_result.doa_valid else -1.0
                srp_score = doa_result.doa_confidence
                actual_doa_source = doa_result.doa_source
                hardware_doa_deg = doa_result.hardware_doa_deg
                hardware_voice = doa_result.hardware_voice
                hardware_doa_error = doa_result.error
                raw_doa_deg_for_log = doa_result.raw_doa_deg
                doa_valid = bool(doa_result.doa_valid)
                stable_doa = bool(doa_result.stable_doa)
                doa_confidence = float(doa_result.doa_confidence)

                t0 = time.perf_counter()
                beam_mono = beamformer.process(array_float, azimuth_deg) if _valid_angle(azimuth_deg) else raw_ref
                beam_mono = post_filter_waveform(
                    beam_mono,
                    sample_rate,
                    enable=post_filter_enabled,
                    strength=float(beam_cfg.get("post_filter_strength", 0.2)),
                )
                beam_ms = (time.perf_counter() - t0) * 1000.0
            else:
                beam_mono = raw_ref

            prob_raw, raw_infer_ms, _raw_extra = detector.infer(raw_ref)
            if beamforming_enabled:
                prob_beam, beam_infer_ms, _beam_extra = detector.infer(beam_mono)
            else:
                prob_beam, beam_infer_ms, _beam_extra = prob_raw, 0.0, _raw_extra
            log_mel_stats = dict(_raw_extra.get("log_mel_stats", {}) or {})
            fused = fuse_max(prob_raw, prob_beam, azimuth_deg, srp_score)
            snr_db = estimate_snr_db(raw_ref)
            alarm, alarm_record = alarm_logic.update({
                **fused,
                "timestamp_ms": wall_ms(),
                "window_end_ms": wall_ms(),
            })

            window_positive = (
                beamforming_enabled
                and _valid_angle(fused["azimuth_deg"])
                and doa_valid
                and (stable_doa or not guidance_require_stable_doa)
                and doa_confidence >= guidance_min_doa_confidence
                and fused["prob_final"] > guidance_threshold
                and snr_db >= guidance_min_snr_db
            )
            guidance_now = time.monotonic()
            guidance_history.append({
                "positive": bool(window_positive),
                "angle": float(fused["azimuth_deg"]),
                "score": float(fused["srp_score"]),
                "prob": float(fused["prob_final"]),
                "snr_db": float(snr_db),
            })
            positive_angles = [
                item["angle"]
                for item in guidance_history
                if item["positive"] and _valid_angle(item["angle"])
            ]
            guidance_positive_windows = len(positive_angles)
            guidance_angle_std_deg = _circular_std_deg(positive_angles)
            stable_guidance = (
                beamforming_enabled
                and guidance_positive_windows >= guidance_min_windows
                and guidance_angle_std_deg <= guidance_max_angle_std_deg
            )
            candidate_angle = _circular_mean_deg(positive_angles[-guidance_min_windows:])
            jump_allowed = (
                not _valid_angle(last_guidance_angle)
                or guidance_now > last_guidance_until
                or (_valid_angle(candidate_angle) and abs(_angle_delta_deg(candidate_angle, last_guidance_angle)) <= guidance_max_jump_deg)
            )
            if stable_guidance and jump_allowed and _valid_angle(candidate_angle):
                last_guidance_until = guidance_now + guidance_hold_sec
                last_guidance_angle = candidate_angle
                last_guidance_score = float(fused["srp_score"])
            cue_active = (
                beamforming_enabled
                and _valid_angle(last_guidance_angle)
                and guidance_now <= last_guidance_until
            )
            weak_candidate = (
                beamforming_enabled
                and not cue_active
                and _valid_angle(fused["azimuth_deg"])
                and doa_valid
                and stable_doa
                and doa_confidence >= guidance_min_doa_confidence
                and fused["prob_final"] > guidance_threshold
                and snr_db >= guidance_weak_min_snr_db
            )
            send_guidance = cue_active or weak_candidate
            cue_angle = last_guidance_angle if cue_active else (fused["azimuth_deg"] if weak_candidate else -1.0)
            current_doa_score = float(fused["srp_score"])
            guidance_hold_score = float(last_guidance_score) if cue_active else 0.0
            guidance_age_ms = max(0.0, (last_guidance_until - guidance_now) * 1000.0) if cue_active else 0.0
            state = "CONFIRMED" if alarm and cue_active else (
                "CANDIDATE" if send_guidance else ("DETECTED_UNSTABLE" if alarm or window_positive else "IDLE")
            )
            payload = {
                "type": "audio",
                "audio_detected": bool(send_guidance),
                "detected": bool(send_guidance),
                "audio_state": state,
                "angle": cue_angle,
                "doa_deg": cue_angle,
                "raw_doa_deg": raw_doa_deg_for_log,
                "smooth_doa_deg": fused["azimuth_deg"],
                "doa_valid": int(doa_valid),
                "stable_doa": int(stable_doa),
                "doa_confidence": doa_confidence,
                "doa_source": actual_doa_source,
                "hardware_doa_deg": hardware_doa_deg,
                "hardware_voice": int(hardware_voice),
                "hardware_doa_error": hardware_doa_error,
                "doa_stability": current_doa_score,
                "doa_stable": int(stable_doa),
                "guidance_hold_score": guidance_hold_score,
                "confidence": fused["prob_final"],
                "yamnet_score": fused["prob_final"],
                "prob_raw": fused["prob_raw"],
                "prob_beam": fused["prob_beam"],
                "prob_final": fused["prob_final"],
                "score_ema": fused["prob_final"],
                "srp_score": fused["srp_score"],
                "snr_db": snr_db,
                "beamforming_enabled": int(beamforming_enabled),
                "post_filter_enabled": int(post_filter_enabled),
                "beam_angle_deg": fused["azimuth_deg"],
                "beam_gain_db": fused["prob_beam"] - fused["prob_raw"],
                "raw_infer_ms": raw_infer_ms,
                "beam_infer_ms": beam_infer_ms,
                "audio_infer_ms": raw_infer_ms + beam_infer_ms,
                "doa_compute_ms": doa_ms,
                "beamform_ms": beam_ms,
                "alarm": bool(alarm),
                "trigger_sent": int(send_guidance),
                "ptz_triggered": int(send_guidance),
                "threshold_percent": float(det_cfg.get("threshold_percent", 30.0)),
                "guidance_threshold_percent": guidance_threshold,
                "guidance_hold_sec": guidance_hold_sec,
                "guidance_hold_remaining_ms": guidance_age_ms,
                "guidance_positive_windows": guidance_positive_windows,
                "guidance_angle_std_deg": guidance_angle_std_deg,
                "guidance_min_snr_db": guidance_min_snr_db,
                "guidance_weak_min_snr_db": guidance_weak_min_snr_db,
                "guidance_min_doa_confidence": guidance_min_doa_confidence,
                "guidance_require_stable_doa": int(guidance_require_stable_doa),
                "window_id": window_id,
                "mic_array_type": f"{capture_channels}ch",
                "sample_rate": sample_rate,
                "channels": capture_channels,
                "processing_channels": processing_channels,
                "selected_device_channels": selected_device_channels,
                "main_device_channel": main_device_channel,
                "channel_rms": channel_stats["rms"],
                "channel_peak": channel_stats["peak"],
                "channel_clip_ratio": channel_stats["clip_ratio"],
                "channel_zero_ratio": channel_stats["zero_ratio"],
                "main_channel_mean": main_device_stats["mean"],
                "main_channel_std": main_device_stats["std"],
                "main_channel_rms": main_device_stats["rms"],
                "main_channel_peak": main_device_stats["peak"],
                "main_channel_clip_ratio": main_device_stats["clip_ratio"],
                "main_channel_zero_ratio": main_device_stats["zero_ratio"],
                "raw_ref_mean": raw_ref_stats["mean"],
                "raw_ref_std": raw_ref_stats["std"],
                "raw_ref_rms": raw_ref_stats["rms"],
                "raw_ref_peak": raw_ref_stats["peak"],
                "rms_dbfs": rms_dbfs,
                "log_mel_stats": log_mel_stats,
                "rknn_input_shape": _raw_extra.get("rknn_input_shape", []),
                "rknn_input_dtype": str(_raw_extra.get("rknn_input_dtype", "")),
                "raw_logits": _raw_extra.get("raw_logits", []),
                "softmax_percent": _raw_extra.get("softmax_percent", []),
                "drone_class_index": int(_raw_extra.get("drone_class_index", 1)),
                "audio_window_ms": int(float(det_cfg.get("window_sec", 1.0)) * 1000),
                "raw_recording_active": int(raw_recorder.active),
                "raw_recording_path": raw_recorder.path,
                "raw_recording_session_id": raw_recorder.session_id,
                "raw_recording_samples": raw_recorder.samples,
                **experiment,
                "ts_ms": wall_ms(),
                "ts": time.time(),
                "ts_mono_ms": mono_ms(),
            }
            encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            udp_sock.sendto(encoded, qt_addr)
            udp_sock.sendto(encoded, fusion_addr)

            row = {
                "timestamp_ms": payload["ts_ms"],
                "window_id": window_id,
                "prob_raw": fused["prob_raw"],
                "prob_beam": fused["prob_beam"],
                "prob_final": fused["prob_final"],
                "azimuth_deg": fused["azimuth_deg"],
                "raw_doa_deg": raw_doa_deg_for_log,
                "smooth_doa_deg": fused["azimuth_deg"],
                "doa_valid": int(doa_valid),
                "stable_doa": int(stable_doa),
                "doa_confidence": doa_confidence,
                "srp_score": fused["srp_score"],
                "doa_source": actual_doa_source,
                "hardware_doa_deg": hardware_doa_deg,
                "hardware_voice": int(hardware_voice),
                "snr_db": payload["snr_db"],
                "raw_infer_ms": raw_infer_ms,
                "beam_infer_ms": beam_infer_ms,
                "doa_compute_ms": doa_ms,
                "beamform_ms": beam_ms,
                "beamforming_enabled": int(beamforming_enabled),
                "post_filter_enabled": int(post_filter_enabled),
                "alarm": int(alarm),
                "audio_state": state,
                "guidance_threshold_percent": guidance_threshold,
                "guidance_hold_sec": guidance_hold_sec,
                "guidance_hold_remaining_ms": guidance_age_ms,
                "guidance_positive_windows": guidance_positive_windows,
                "guidance_angle_std_deg": guidance_angle_std_deg,
                "guidance_min_snr_db": guidance_min_snr_db,
                "guidance_weak_min_snr_db": guidance_weak_min_snr_db,
                "guidance_min_doa_confidence": guidance_min_doa_confidence,
                "guidance_require_stable_doa": int(guidance_require_stable_doa),
                "channel_rms_json": _json_compact(channel_stats["rms"]),
                "channel_peak_json": _json_compact(channel_stats["peak"]),
                "channel_clip_ratio_json": _json_compact(channel_stats["clip_ratio"]),
                "channel_zero_ratio_json": _json_compact(channel_stats["zero_ratio"]),
                "main_device_channel": main_device_channel,
                "main_channel_mean": main_device_stats["mean"],
                "main_channel_std": main_device_stats["std"],
                "main_channel_rms": main_device_stats["rms"],
                "main_channel_peak": main_device_stats["peak"],
                "main_channel_clip_ratio": main_device_stats["clip_ratio"],
                "main_channel_zero_ratio": main_device_stats["zero_ratio"],
                "log_mel_min": float(log_mel_stats.get("min", 0.0)),
                "log_mel_max": float(log_mel_stats.get("max", 0.0)),
                "log_mel_mean": float(log_mel_stats.get("mean", 0.0)),
                "log_mel_std": float(log_mel_stats.get("std", 0.0)),
                "rknn_input_shape": _json_compact(_raw_extra.get("rknn_input_shape", [])),
                "rknn_input_dtype": str(_raw_extra.get("rknn_input_dtype", "")),
                "raw_logits_json": _json_compact(_raw_extra.get("raw_logits", [])),
                "softmax_percent_json": _json_compact(_raw_extra.get("softmax_percent", [])),
                "drone_class_index": int(_raw_extra.get("drone_class_index", 1)),
                "raw_recording_active": int(raw_recorder.active),
                "raw_recording_path": raw_recorder.path,
                "raw_recording_session_id": raw_recorder.session_id,
                "raw_recording_samples": raw_recorder.samples,
                **experiment,
            }
            writer.writerow(row)
            if window_id % log_flush_interval == 0:
                csv_file.flush()
            elapsed_ms = (time.perf_counter() - t_window) * 1000.0
            print(
                f"[AUDIO-BF] raw={fused['prob_raw']:5.1f}% beam={fused['prob_beam']:5.1f}% "
                f"final={fused['prob_final']:5.1f}% doa={fused['azimuth_deg']:6.1f} src={actual_doa_source} "
                f"srp={fused['srp_score']:.3f} state={state} alarm={int(alarm)} elapsed={elapsed_ms:.1f}ms"
            )
            window_id += 1
            if args.max_windows > 0 and window_id >= args.max_windows:
                print(f"[INFO] max_windows={args.max_windows}, realtime audio detector stopping...")
                break
    except InterruptedError:
        if not stop_requested["value"]:
            raise
        print("[INFO] audio capture stopped")
    except KeyboardInterrupt:
        print("\n[INFO] realtime audio detector stopping...")
    finally:
        raw_recorder.stop()
        csv_file.close()
        backend.close()
        detector.release()
        udp_sock.close()
        if control_sock is not None:
            control_sock.close()


if __name__ == "__main__":
    main()
