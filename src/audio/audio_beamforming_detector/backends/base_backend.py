from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np


class AudioArrayBackend(ABC):
    def __init__(self, sample_rate: int, num_mics: int):
        self.sample_rate = int(sample_rate)
        self.num_mics = int(num_mics)

    @abstractmethod
    def read_frame(self, num_samples: int) -> np.ndarray:
        """
        Returns:
            mic_data: np.ndarray, shape [num_mics, num_samples], dtype float32 or int16
        """
        raise NotImplementedError

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class MockArrayBackend(AudioArrayBackend):
    """Deterministic multi-channel mock source used when hardware is absent."""

    def __init__(
        self,
        sample_rate: int,
        num_mics: int,
        mic_positions: np.ndarray | None = None,
        azimuth_deg: float = 65.0,
        noise_scale: float = 0.015,
    ):
        super().__init__(sample_rate=sample_rate, num_mics=num_mics)
        self.mic_positions = mic_positions
        self.azimuth_deg = float(azimuth_deg)
        self.noise_scale = float(noise_scale)
        self.cursor = 0
        self.rng = np.random.default_rng(20260608)

    def read_frame(self, num_samples: int) -> np.ndarray:
        num_samples = int(num_samples)
        t = (np.arange(num_samples, dtype=np.float32) + self.cursor) / float(self.sample_rate)
        self.cursor += num_samples

        carrier = (
            0.11 * np.sin(2.0 * math.pi * 185.0 * t)
            + 0.07 * np.sin(2.0 * math.pi * 245.0 * t)
            + 0.025 * np.sin(2.0 * math.pi * 370.0 * t)
        )
        carrier *= 0.5 + 0.5 * np.sin(2.0 * math.pi * 7.5 * t) ** 2

        out = []
        if self.mic_positions is None:
            delays = np.linspace(-1.5, 1.5, self.num_mics) / self.sample_rate
        else:
            angle = math.radians(self.azimuth_deg)
            direction = np.array([math.cos(angle), math.sin(angle), 0.0], dtype=np.float64)
            delays = np.asarray(self.mic_positions, dtype=np.float64).dot(direction) / 343.0
            delays -= float(np.mean(delays))

        for delay in delays:
            shifted_t = t - float(delay)
            ch = (
                0.11 * np.sin(2.0 * math.pi * 185.0 * shifted_t)
                + 0.07 * np.sin(2.0 * math.pi * 245.0 * shifted_t)
                + 0.025 * np.sin(2.0 * math.pi * 370.0 * shifted_t)
            )
            ch *= 0.5 + 0.5 * np.sin(2.0 * math.pi * 7.5 * shifted_t) ** 2
            ch += self.noise_scale * self.rng.standard_normal(num_samples)
            out.append(ch.astype(np.float32))

        return np.vstack(out)
