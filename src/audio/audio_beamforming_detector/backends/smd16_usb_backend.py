from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .base_backend import AudioArrayBackend, MockArrayBackend


class SMD16UsbBackend(AudioArrayBackend):
    """
    Reserved backend for an SMD USB 16-element array.

    The future device payload is treated as 18 channels: channels 0..15 are raw
    microphones and channels 16..17 are loopback channels. This first version can
    replay a PCM wav file or return deterministic mock data so the downstream
    array processing stack remains testable before the real device SDK lands.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        input_wav: str | None = None,
        expected_channels: int = 18,
        raw_channels: list[int] | None = None,
        mic_positions: np.ndarray | None = None,
    ):
        raw_channels = raw_channels or list(range(16))
        super().__init__(sample_rate=sample_rate, num_mics=len(raw_channels))
        self.expected_channels = int(expected_channels)
        self.raw_channels = [int(x) for x in raw_channels]
        self.input_wav = Path(input_wav).expanduser() if input_wav else None
        self._wav = None
        self._mock = MockArrayBackend(sample_rate, len(raw_channels), mic_positions=mic_positions)

    def _open_wav(self):
        if self.input_wav is None or self._wav is not None:
            return
        self._wav = wave.open(str(self.input_wav), "rb")
        if self._wav.getframerate() != self.sample_rate:
            raise RuntimeError(f"SMD16 wav sample rate must be {self.sample_rate}, got {self._wav.getframerate()}")
        if self._wav.getnchannels() < self.expected_channels:
            raise RuntimeError(
                f"SMD16 wav must contain at least {self.expected_channels} channels, got {self._wav.getnchannels()}"
            )

    def read_frame(self, num_samples: int) -> np.ndarray:
        if self.input_wav is None:
            return self._mock.read_frame(num_samples)

        self._open_wav()
        raw = self._wav.readframes(int(num_samples))
        if not raw:
            self._wav.rewind()
            raw = self._wav.readframes(int(num_samples))
        channels = self._wav.getnchannels()
        data = np.frombuffer(raw, dtype=np.int16).reshape(-1, channels)
        if data.shape[0] < num_samples:
            pad = np.zeros((num_samples - data.shape[0], channels), dtype=np.int16)
            data = np.vstack([data, pad])
        return data[:, self.raw_channels].T.copy()

    def close(self) -> None:
        if self._wav is not None:
            self._wav.close()
            self._wav = None
