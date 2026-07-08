from __future__ import annotations

import itertools

import numpy as np

from .mic_geometry import far_field_delays


class SrpPhatEstimator:
    def __init__(
        self,
        mic_positions: np.ndarray,
        sample_rate: int,
        freq_min: float = 500.0,
        freq_max: float = 5500.0,
        angle_step_deg: float = 5.0,
        sound_speed: float = 343.0,
        local_search_deg: float = 20.0,
        stable_score_threshold: float = 0.08,
    ):
        self.mic_positions = np.asarray(mic_positions, dtype=np.float64)
        self.sample_rate = int(sample_rate)
        self.freq_min = float(freq_min)
        self.freq_max = float(freq_max)
        self.angle_step_deg = float(angle_step_deg)
        self.sound_speed = float(sound_speed)
        self.local_search_deg = float(local_search_deg)
        self.stable_score_threshold = float(stable_score_threshold)
        self.prev_azimuth_deg: float | None = None
        self.prev_score = 0.0

    def _candidate_angles(self) -> np.ndarray:
        if self.prev_azimuth_deg is not None and self.prev_score >= self.stable_score_threshold:
            offsets = np.arange(-self.local_search_deg, self.local_search_deg + 0.1, self.angle_step_deg)
            return (self.prev_azimuth_deg + offsets) % 360.0
        return np.arange(0.0, 360.0, self.angle_step_deg, dtype=np.float64)

    def estimate(self, mic_data: np.ndarray) -> tuple[float, float]:
        x = np.asarray(mic_data, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"mic_data must be [num_mics, samples], got {x.shape}")
        num_mics, samples = x.shape
        if num_mics < 2 or samples < 32:
            return -1.0, 0.0

        n_fft = int(2 ** np.ceil(np.log2(samples)))
        window = np.hanning(samples).astype(np.float32)
        spectrum = np.fft.rfft(x * window[None, :], n=n_fft, axis=1)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sample_rate)
        mask = (freqs >= self.freq_min) & (freqs <= self.freq_max)
        if not np.any(mask):
            return -1.0, 0.0
        freqs_sel = freqs[mask]

        pair_data = []
        for i, j in itertools.combinations(range(num_mics), 2):
            cross = spectrum[i, mask] * np.conj(spectrum[j, mask])
            cross /= np.maximum(np.abs(cross), 1e-9)
            pair_data.append((i, j, cross))

        angles = self._candidate_angles()
        scores = []
        for angle in angles:
            delays = far_field_delays(self.mic_positions, angle, self.sound_speed)
            score = 0.0
            for i, j, cross in pair_data:
                delta_tau = delays[i] - delays[j]
                phase = np.exp(1j * 2.0 * np.pi * freqs_sel * delta_tau)
                score += float(np.mean(np.real(cross * phase)))
            scores.append(score / max(1, len(pair_data)))

        scores_np = np.asarray(scores, dtype=np.float64)
        best_idx = int(np.argmax(scores_np))
        best_angle = float(angles[best_idx] % 360.0)
        peak = float(scores_np[best_idx])
        median = float(np.median(scores_np))
        score_norm = float(max(0.0, min(1.0, (peak - median) / (abs(peak) + 1e-6))))

        self.prev_azimuth_deg = best_angle
        self.prev_score = score_norm
        return best_angle, score_norm
