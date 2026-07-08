from __future__ import annotations

import numpy as np


def post_filter_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    enable: bool = False,
    strength: float = 0.2,
    n_fft: int = 512,
) -> np.ndarray:
    if not enable:
        return np.asarray(waveform, dtype=np.float32)

    strength = float(np.clip(strength, 0.0, 0.6))
    x = np.asarray(waveform, dtype=np.float32)
    if x.size < n_fft:
        return x.copy()

    hop = n_fft // 4
    window = np.hanning(n_fft).astype(np.float32)
    frames = []
    starts = []
    for start in range(0, x.size - n_fft + 1, hop):
        starts.append(start)
        frames.append(np.fft.rfft(x[start:start + n_fft] * window))
    if not frames:
        return x.copy()

    spec = np.stack(frames, axis=0)
    power = np.abs(spec) ** 2
    noise = np.percentile(power, 20, axis=0, keepdims=True)
    gain = power / np.maximum(power + strength * noise, 1e-9)
    gain = np.clip(gain, 1.0 - strength, 1.0)
    spec_f = spec * gain

    out = np.zeros_like(x, dtype=np.float64)
    norm = np.zeros_like(x, dtype=np.float64)
    for frame_spec, start in zip(spec_f, starts):
        frame = np.fft.irfft(frame_spec, n=n_fft)
        out[start:start + n_fft] += frame * window
        norm[start:start + n_fft] += window * window
    valid = norm > 1e-8
    out[valid] /= norm[valid]
    return out.astype(np.float32)
