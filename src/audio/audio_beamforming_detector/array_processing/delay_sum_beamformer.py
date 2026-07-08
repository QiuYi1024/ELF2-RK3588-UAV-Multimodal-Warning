from __future__ import annotations

import numpy as np

from .mic_geometry import far_field_delays


def _stft_multi(x: np.ndarray, n_fft: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    num_mics, samples = x.shape
    window = np.hanning(n_fft).astype(np.float32)
    if samples < n_fft:
        pad = n_fft - samples
        x = np.pad(x, ((0, 0), (0, pad)))
        samples = x.shape[1]
    frames = 1 + int(np.ceil((samples - n_fft) / float(hop)))
    total = n_fft + (frames - 1) * hop
    if total > samples:
        x = np.pad(x, ((0, 0), (0, total - samples)))
    out = []
    for frame_idx in range(frames):
        start = frame_idx * hop
        chunk = x[:, start:start + n_fft] * window[None, :]
        out.append(np.fft.rfft(chunk, n=n_fft, axis=1))
    return np.stack(out, axis=1), window


def _istft_mono(spec: np.ndarray, window: np.ndarray, hop: int, output_len: int) -> np.ndarray:
    frames, _bins = spec.shape
    n_fft = (spec.shape[1] - 1) * 2
    total = n_fft + (frames - 1) * hop
    out = np.zeros(total, dtype=np.float64)
    norm = np.zeros(total, dtype=np.float64)
    for frame_idx in range(frames):
        start = frame_idx * hop
        frame = np.fft.irfft(spec[frame_idx], n=n_fft)
        out[start:start + n_fft] += frame * window
        norm[start:start + n_fft] += window * window
    valid = norm > 1e-8
    out[valid] /= norm[valid]
    if out.size < output_len:
        out = np.pad(out, (0, output_len - out.size))
    return out[:output_len].astype(np.float32)


class FrequencyDelaySumBeamformer:
    def __init__(
        self,
        mic_positions: np.ndarray,
        sample_rate: int,
        n_fft: int = 512,
        hop: int | None = None,
        sound_speed: float = 343.0,
    ):
        self.mic_positions = np.asarray(mic_positions, dtype=np.float64)
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop = int(hop or n_fft // 4)
        self.sound_speed = float(sound_speed)

    def process(self, mic_data: np.ndarray, azimuth_deg: float) -> np.ndarray:
        x = np.asarray(mic_data, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"mic_data must be [num_mics, samples], got {x.shape}")
        output_len = x.shape[1]
        spec, window = _stft_multi(x, self.n_fft, self.hop)
        freqs = np.fft.rfftfreq(self.n_fft, 1.0 / self.sample_rate)
        delays = far_field_delays(self.mic_positions, azimuth_deg, self.sound_speed)

        aligned = []
        for mic_idx, tau in enumerate(delays):
            phase = np.exp(-1j * 2.0 * np.pi * freqs * tau)
            aligned.append(spec[mic_idx] * phase[None, :])
        mono_spec = np.mean(np.stack(aligned, axis=0), axis=0)
        return _istft_mono(mono_spec, window, self.hop, output_len)
