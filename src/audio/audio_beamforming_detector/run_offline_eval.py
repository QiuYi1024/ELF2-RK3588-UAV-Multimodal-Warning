from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent
PARENT_DIR = PACKAGE_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from audio_beamforming_detector.array_processing import (  # noqa: E402
    FrequencyDelaySumBeamformer,
    SrpPhatEstimator,
    build_mic_positions,
    estimate_snr_db,
    post_filter_waveform,
    preprocess_array,
    to_float32_audio,
)
from audio_beamforming_detector.detector import WindowAlarmLogic, create_detector, fuse_max  # noqa: E402
from audio_beamforming_detector.utils import dump_simple_yaml, load_config  # noqa: E402


def env_or_value(env_name: str, value, default):
    env_value = os.environ.get(env_name)
    if env_value is not None and env_value != "":
        return env_value
    if value is not None and value != "":
        return value
    return default


def parse_target_present(value) -> int:
    if value is None or value == "":
        return -1
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "present", "drone"):
        return 1
    if text in ("0", "false", "no", "absent", "noise"):
        return 0
    return -1


def classification_metrics(rows: list[dict], threshold: float, target_present: int) -> dict:
    if target_present not in (0, 1) or not rows:
        return {"accuracy": -1.0, "recall": -1.0, "f1": -1.0, "false_positive_count": -1, "false_positive_rate": -1.0}

    tp = fp = tn = fn = 0
    for row in rows:
        pred = int(row["alarm"]) if "alarm" in row else (1 if float(row.get("prob_final", 0.0)) > threshold else 0)
        if target_present == 1 and pred == 1:
            tp += 1
        elif target_present == 1 and pred == 0:
            fn += 1
        elif target_present == 0 and pred == 1:
            fp += 1
        else:
            tn += 1
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "accuracy": (tp + tn) / total if total else -1.0,
        "recall": recall,
        "f1": f1,
        "false_positive_count": fp,
        "false_positive_rate": fpr,
    }


def read_pcm_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
    data = np.frombuffer(frames, dtype=np.int16).reshape(-1, channels)
    return data.T.copy(), sample_rate


def write_pcm_wav(path: Path, data_ch_samples: np.ndarray, sample_rate: int) -> None:
    data = np.asarray(data_ch_samples, dtype=np.float32)
    pcm = np.clip(data.T, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(pcm.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def generate_mock_wav(path: Path, mic_positions: np.ndarray, sample_rate: int, seconds: float = 5.0) -> None:
    samples = int(sample_rate * seconds)
    t = np.arange(samples, dtype=np.float32) / float(sample_rate)
    angle = math.radians(65.0)
    direction = np.array([math.cos(angle), math.sin(angle), 0.0], dtype=np.float64)
    delays = mic_positions.dot(direction) / 343.0
    delays -= float(np.mean(delays))
    rng = np.random.default_rng(20260608)
    channels = []
    for tau in delays:
        tt = t - float(tau)
        sig = 0.11 * np.sin(2 * math.pi * 185 * tt) + 0.07 * np.sin(2 * math.pi * 245 * tt)
        sig *= 0.5 + 0.5 * np.sin(2 * math.pi * 7.5 * tt) ** 2
        sig += 0.02 * rng.standard_normal(samples)
        channels.append(sig.astype(np.float32))
    write_pcm_wav(path, np.vstack(channels), sample_rate)


def select_config_channels(wav_data: np.ndarray, array_cfg: dict) -> np.ndarray:
    raw_channels = [int(x) for x in array_cfg.get("raw_channels", list(range(int(array_cfg["num_mics"]))))]
    if wav_data.shape[0] == int(array_cfg["num_mics"]):
        return wav_data
    if max(raw_channels) < wav_data.shape[0]:
        return wav_data[raw_channels]
    raise RuntimeError(f"input wav has {wav_data.shape[0]} channels, cannot select {raw_channels}")


def process_window(
    mic_window: np.ndarray,
    cfg: dict,
    mic_positions: np.ndarray,
    srp: SrpPhatEstimator,
    beamformer: FrequencyDelaySumBeamformer,
    detector,
    disable_beamforming: bool,
    disable_post_filter: bool,
):
    sample_rate = int(cfg["array"]["sample_rate"])
    ref_channel = int(cfg["array"].get("ref_channel", 0))
    prep_cfg = cfg.get("preprocess", {})
    beam_cfg = cfg.get("beamforming", {})

    raw_float = preprocess_array(
        mic_window,
        sample_rate,
        remove_dc=bool(prep_cfg.get("remove_dc", True)),
        rms_normalize=False,
        highpass_hz=float(prep_cfg.get("highpass_hz", 100) or 0),
    )
    array_float = preprocess_array(
        mic_window,
        sample_rate,
        remove_dc=bool(prep_cfg.get("remove_dc", True)),
        rms_normalize=bool(prep_cfg.get("rms_normalize", True)),
        highpass_hz=float(prep_cfg.get("highpass_hz", 100) or 0),
    )
    raw_ref = raw_float[ref_channel]

    doa_ms = 0.0
    beam_ms = 0.0
    azimuth_deg = -1.0
    srp_score = 0.0
    if disable_beamforming or not bool(beam_cfg.get("enable", True)):
        beam_mono = raw_ref
    else:
        t0 = time.perf_counter()
        azimuth_deg, srp_score = srp.estimate(array_float)
        doa_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        beam_mono = beamformer.process(array_float, azimuth_deg)
        if not disable_post_filter:
            beam_mono = post_filter_waveform(
                beam_mono,
                sample_rate,
                enable=bool(beam_cfg.get("enable_post_filter", False)),
                strength=float(beam_cfg.get("post_filter_strength", 0.2)),
            )
        beam_ms = (time.perf_counter() - t0) * 1000.0

    prob_raw, raw_infer_ms, raw_extra = detector.infer(raw_ref)
    prob_beam, beam_infer_ms, beam_extra = detector.infer(beam_mono)
    fused = fuse_max(prob_raw, prob_beam, azimuth_deg, srp_score)
    fused.update({
        "snr_db": estimate_snr_db(raw_ref),
        "doa_compute_ms": doa_ms,
        "beamform_ms": beam_ms,
        "raw_infer_ms": raw_infer_ms,
        "beam_infer_ms": beam_infer_ms,
        "raw_detector": raw_extra.get("detector", "unknown"),
        "beam_detector": beam_extra.get("detector", "unknown"),
    })
    return fused


def main():
    ap = argparse.ArgumentParser(description="Offline evaluation for array beamforming YAMNet-RKNN detector")
    ap.add_argument("--config", default=str(PACKAGE_DIR / "configs" / "respeaker4.yaml"))
    ap.add_argument("--input", default="", help="Multi-channel wav. If absent, a mock wav is generated.")
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--model-path", default="")
    ap.add_argument("--disable-beamforming", action="store_true")
    ap.add_argument("--disable-post-filter", action="store_true")
    ap.add_argument("--experiment-id", default="")
    ap.add_argument("--scene", default="")
    ap.add_argument("--distance-m", type=float, default=None)
    ap.add_argument("--background-noise", default="")
    ap.add_argument("--target-present", default=None)
    ap.add_argument("--method-label", default="")
    ap.add_argument("--trial-id", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model_path:
        cfg.setdefault("detector", {})["model_path"] = args.model_path

    out_dir = Path(args.output_dir or f"beamforming_eval_{time.strftime('%Y%m%d_%H%M%S')}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "used_config.yaml").write_text(dump_simple_yaml(cfg), encoding="utf-8")

    mic_positions = build_mic_positions(cfg["array"])
    sample_rate = int(cfg["array"]["sample_rate"])
    wav_path = Path(args.input).expanduser() if args.input else out_dir / "mock_multichannel.wav"
    if not args.input or not wav_path.exists():
        generate_mock_wav(wav_path, mic_positions, sample_rate)

    wav_data, wav_sr = read_pcm_wav(wav_path)
    if wav_sr != sample_rate:
        raise RuntimeError(f"Expected {sample_rate} Hz wav, got {wav_sr}")
    mic_data = select_config_channels(wav_data, cfg["array"])

    beam_cfg = cfg.get("beamforming", {})
    det_cfg = cfg.get("detector", {})
    exp_cfg = cfg.get("experiment", {})
    threshold = float(det_cfg.get("threshold_percent", 30.0))
    experiment = {
        "experiment_id": env_or_value("ANTI_UAV_EXPERIMENT_ID", args.experiment_id or exp_cfg.get("experiment_id"), ""),
        "scene_type": env_or_value("ANTI_UAV_SCENE_TYPE", args.scene or exp_cfg.get("scene_type"), ""),
        "distance_m": float(env_or_value("ANTI_UAV_DISTANCE_M", args.distance_m if args.distance_m is not None else exp_cfg.get("distance_m"), -1)),
        "background_noise": env_or_value("ANTI_UAV_BACKGROUND_NOISE", args.background_noise or exp_cfg.get("background_noise"), ""),
        "noise_type": env_or_value("ANTI_UAV_NOISE_TYPE", args.background_noise or exp_cfg.get("noise_type"), ""),
        "target_present": parse_target_present(env_or_value("ANTI_UAV_TARGET_PRESENT", args.target_present if args.target_present is not None else exp_cfg.get("target_present"), "")),
        "method_label": env_or_value("ANTI_UAV_EXPERIMENT_METHOD", args.method_label or exp_cfg.get("method_label"), "only_acoustic_yamnet_beam"),
        "trial_id": env_or_value("ANTI_UAV_TRIAL_ID", args.trial_id or exp_cfg.get("trial_id"), ""),
    }
    srp = SrpPhatEstimator(
        mic_positions=mic_positions,
        sample_rate=sample_rate,
        freq_min=float(beam_cfg.get("doa_freq_min", 500)),
        freq_max=float(beam_cfg.get("doa_freq_max", 5500)),
        angle_step_deg=float(beam_cfg.get("angle_step_deg", 5)),
    )
    beamformer = FrequencyDelaySumBeamformer(mic_positions, sample_rate)
    detector = create_detector(det_cfg.get("model_path"), allow_dummy=True)
    alarm = WindowAlarmLogic(threshold_percent=float(det_cfg.get("threshold_percent", 30.0)))

    window_samples = int(float(det_cfg.get("window_sec", 1.0)) * sample_rate)
    hop_samples = int(float(det_cfg.get("hop_sec", 0.5)) * sample_rate)
    csv_path = out_dir / "window_results.csv"
    rows = []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "window_index", "start_sec", "end_sec", "prob_raw", "prob_beam", "prob_final",
                "azimuth_deg", "srp_score", "snr_db", "raw_infer_ms", "beam_infer_ms",
                "doa_compute_ms", "beamform_ms", "elapsed_ms", "alarm", "experiment_id",
                "scene_type", "distance_m", "background_noise", "noise_type",
                "target_present", "method_label", "trial_id",
            ],
        )
        writer.writeheader()
        idx = 0
        for start in range(0, max(1, mic_data.shape[1] - window_samples + 1), hop_samples):
            win = mic_data[:, start:start + window_samples]
            if win.shape[1] < window_samples:
                win = np.pad(win, ((0, 0), (0, window_samples - win.shape[1])))
            t0 = time.perf_counter()
            result = process_window(
                win, cfg, mic_positions, srp, beamformer, detector,
                args.disable_beamforming, args.disable_post_filter,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            record = {
                "window_index": idx,
                "start_sec": start / sample_rate,
                "end_sec": (start + window_samples) / sample_rate,
                **result,
                **experiment,
                "elapsed_ms": elapsed_ms,
            }
            alarm_state, enriched = alarm.update(record)
            record["alarm"] = alarm_state
            writer.writerow({k: record.get(k, "") for k in writer.fieldnames})
            rows.append(record)
            idx += 1

    report = out_dir / "detection_report.md"
    detected = sum(1 for r in rows if r["prob_final"] > threshold)
    metrics = classification_metrics(rows, threshold, int(experiment["target_present"]))
    mean_raw = float(np.mean([r["prob_raw"] for r in rows])) if rows else 0.0
    mean_beam = float(np.mean([r["prob_beam"] for r in rows])) if rows else 0.0
    mean_final = float(np.mean([r["prob_final"] for r in rows])) if rows else 0.0
    report.write_text(
        "\n".join([
            "# Detection Report",
            "",
            f"- input_wav: `{wav_path}`",
            f"- config: `{Path(args.config)}`",
            f"- windows: {len(rows)}",
            f"- threshold_percent: {det_cfg.get('threshold_percent', 30.0)}",
            f"- experiment_id: {experiment['experiment_id']}",
            f"- scene_type: {experiment['scene_type']}",
            f"- distance_m: {experiment['distance_m']}",
            f"- background_noise: {experiment['background_noise']}",
            f"- target_present: {experiment['target_present']}",
            f"- windows_above_threshold: {detected}",
            f"- accuracy: {metrics['accuracy']:.4f}",
            f"- recall: {metrics['recall']:.4f}",
            f"- f1: {metrics['f1']:.4f}",
            f"- false_positive_count: {metrics['false_positive_count']}",
            f"- false_positive_rate: {metrics['false_positive_rate']:.4f}",
            f"- mean_prob_raw: {mean_raw:.3f}",
            f"- mean_prob_beam: {mean_beam:.3f}",
            f"- mean_prob_final: {mean_final:.3f}",
            f"- detector_dummy: {getattr(detector, 'is_dummy', False)}",
            f"- detector_fallback_reason: {getattr(detector, 'fallback_reason', '')}",
            f"- window_csv: `{csv_path}`",
        ]) + "\n",
        encoding="utf-8",
    )
    detector.release()
    print(f"[OK] offline eval complete: {out_dir}")


if __name__ == "__main__":
    main()
