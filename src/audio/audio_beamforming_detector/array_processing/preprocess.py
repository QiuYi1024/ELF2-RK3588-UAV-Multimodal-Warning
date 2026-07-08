from __future__ import annotations

import math

import numpy as np


def to_float32_audio(mic_data: np.ndarray) -> np.ndarray:
    data = np.asarray(mic_data)
    if data.dtype == np.int16:
        return (data.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    return data.astype(np.float32, copy=False)


def remove_dc_offset(mic_data: np.ndarray) -> np.ndarray:
    return mic_data - np.mean(mic_data, axis=1, keepdims=True)


def rms_normalize_channels(mic_data: np.ndarray, target_rms: float = 0.08, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt(np.mean(np.square(mic_data), axis=1, keepdims=True) + eps)
    gain = np.minimum(float(target_rms) / rms, 8.0)
    return mic_data * gain


def highpass_one_pole(mic_data: np.ndarray, sample_rate: int, cutoff_hz: float = 100.0) -> np.ndarray:
    if cutoff_hz <= 0:
        return mic_data
    rc = 1.0 / (2.0 * math.pi * float(cutoff_hz))
    dt = 1.0 / float(sample_rate)
    alpha = rc / (rc + dt)

    out = np.zeros_like(mic_data, dtype=np.float32)
    out[:, 0] = mic_data[:, 0]
    for n in range(1, mic_data.shape[1]):
        out[:, n] = alpha * (out[:, n - 1] + mic_data[:, n] - mic_data[:, n - 1])
    return out


def preprocess_array(
    mic_data: np.ndarray,
    sample_rate: int,
    remove_dc: bool = True,
    rms_normalize: bool = True,
    highpass_hz: float | None = 100.0,
) -> np.ndarray:
    out = to_float32_audio(mic_data).copy()
    if remove_dc:
        out = remove_dc_offset(out)
    if highpass_hz and highpass_hz > 0:
        out = highpass_one_pole(out, sample_rate=sample_rate, cutoff_hz=float(highpass_hz))
    if rms_normalize:
        out = rms_normalize_channels(out)
    return out.astype(np.float32, copy=False)


def estimate_snr_db(waveform: np.ndarray, eps: float = 1e-9) -> float:
    x = np.asarray(waveform, dtype=np.float32)
    frame = max(160, min(1600, x.size // 8 if x.size >= 1600 else x.size))
    if frame <= 0:
        return -99.0
    powers = []
    for start in range(0, max(1, x.size - frame + 1), frame):
        seg = x[start:start + frame]
        if seg.size:
            powers.append(float(np.mean(seg * seg)))
    if not powers:
        return -99.0
    noise = max(float(np.percentile(powers, 20)), eps)
    signal = max(float(np.percentile(powers, 90)), eps)
    return float(10.0 * np.log10(signal / noise))
