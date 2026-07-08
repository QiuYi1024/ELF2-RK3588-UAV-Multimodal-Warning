from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .base_backend import AudioArrayBackend


class ReSpeaker4Backend(AudioArrayBackend):
    """PyAudio backend for ReSpeaker Mic Array v2 raw microphone channels."""

    def __init__(
        self,
        sample_rate: int = 16000,
        raw_channels: list[int] | None = None,
        device_channels: int = 6,
        device_index: int | None = None,
        frames_per_buffer: int = 2000,
        device_keyword: str | None = None,
        alsa_device: str | None = None,
        reopen_attempts: int = 30,
    ):
        raw_channels = raw_channels or [1, 2, 3, 4]
        super().__init__(sample_rate=sample_rate, num_mics=len(raw_channels))
        self.raw_channels = [int(x) for x in raw_channels]
        self.device_channels = int(device_channels)
        self.device_index = device_index
        self.frames_per_buffer = int(frames_per_buffer)
        self.device_keyword = (device_keyword or os.environ.get("ANTI_UAV_AUDIO_DEVICE_KEYWORD", "")).lower()
        self.alsa_device = str(alsa_device or "").strip()
        self.reopen_attempts = max(1, int(reopen_attempts))
        self._pyaudio = None
        self._stream = None
        self._arecord = None
        self._arecord_reader = None
        self._arecord_stop = threading.Event()
        self._arecord_condition = threading.Condition()
        self._arecord_buffer = bytearray()
        self._arecord_error = ""
        self._stopping = False
        self._last_device_frame = np.empty((self.device_channels, 0), dtype=np.int16)
        self._tuning = None
        self._hardware_doa_error = ""
        invalid_channels = [ch for ch in self.raw_channels if ch < 0 or ch >= self.device_channels]
        if invalid_channels:
            raise ValueError(
                f"raw_channels contains invalid device channels: {invalid_channels}; "
                f"device_channels={self.device_channels}"
            )

    @property
    def capture_channels(self) -> int:
        return self.device_channels

    @property
    def processing_channels(self) -> int:
        return len(self.raw_channels)

    def last_device_frame(self) -> np.ndarray:
        return self._last_device_frame.copy()

    def _select_device(self):
        import pyaudio

        if self._pyaudio is None:
            self._pyaudio = pyaudio.PyAudio()
        if self.device_index is not None:
            return int(self.device_index)

        fallback = None
        for i in range(self._pyaudio.get_device_count()):
            info = self._pyaudio.get_device_info_by_index(i)
            max_inputs = int(info.get("maxInputChannels", 0))
            if max_inputs < self.device_channels:
                continue
            name = str(info.get("name", "")).lower()
            if self.device_keyword and self.device_keyword in name:
                return i
            if any(token in name for token in ("usb", "array", "mic", "microphone")):
                return i
            if fallback is None:
                fallback = i
        return fallback

    def _ensure_stream(self):
        import pyaudio

        if self.alsa_device:
            self._ensure_arecord()
            return
        if self._stream is not None:
            return
        if self._pyaudio is None:
            self._pyaudio = pyaudio.PyAudio()
        device = self._select_device()
        if device is None:
            raise RuntimeError("No ReSpeaker-compatible 6-channel input device was found.")

        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=self.device_channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=device,
            frames_per_buffer=self.frames_per_buffer,
        )

    def _ensure_arecord(self) -> None:
        if self._stopping:
            raise InterruptedError("audio capture is stopping")
        if self._arecord is not None and self._arecord.poll() is None:
            return
        self._close_arecord()
        command = [
            "arecord",
            "-q",
            "-D",
            self.alsa_device,
            "-t",
            "raw",
            "-f",
            "S16_LE",
            "-r",
            str(self.sample_rate),
            "-c",
            str(self.device_channels),
        ]
        self._arecord = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
        )
        if self._arecord.stdout is None:
            self._close_arecord()
            raise RuntimeError("arecord stdout pipe was not created")
        self._arecord_stop.clear()
        with self._arecord_condition:
            self._arecord_buffer.clear()
            self._arecord_error = ""
        self._arecord_reader = threading.Thread(
            target=self._capture_arecord,
            args=(self._arecord,),
            name="respeaker-alsa-reader",
            daemon=True,
        )
        self._arecord_reader.start()
        print(
            f"[AUDIO_CAPTURE] source=alsa device={self.alsa_device} "
            f"format=S16_LE rate={self.sample_rate} channels={self.device_channels}"
        )

    def _capture_arecord(self, process: subprocess.Popen) -> None:
        bytes_per_second = self.sample_rate * self.device_channels * np.dtype(np.int16).itemsize
        max_buffer_bytes = bytes_per_second * 2
        try:
            while not self._arecord_stop.is_set():
                if process.stdout is None:
                    raise RuntimeError("arecord stdout pipe is unavailable")
                data = process.stdout.read(4096)
                if not data:
                    return_code = process.poll()
                    raise RuntimeError(f"arecord stopped while capturing PCM, return_code={return_code}")
                with self._arecord_condition:
                    self._arecord_buffer.extend(data)
                    excess = len(self._arecord_buffer) - max_buffer_bytes
                    if excess > 0:
                        del self._arecord_buffer[:excess]
                    self._arecord_condition.notify_all()
        except Exception as exc:
            if not self._arecord_stop.is_set():
                with self._arecord_condition:
                    self._arecord_error = str(exc)
                    self._arecord_condition.notify_all()

    def _read_arecord_exact(self, byte_count: int) -> bytes:
        self._ensure_arecord()
        deadline = time.monotonic() + max(3.0, byte_count / (
            self.sample_rate * self.device_channels * np.dtype(np.int16).itemsize
        ) + 2.0)
        with self._arecord_condition:
            while len(self._arecord_buffer) < byte_count:
                if self._arecord_error:
                    raise RuntimeError(self._arecord_error)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"timed out waiting for {byte_count} PCM bytes; "
                        f"available={len(self._arecord_buffer)}"
                    )
                self._arecord_condition.wait(timeout=remaining)

            # 只消费最新一段音频，推理繁忙时丢弃旧样本，避免延迟持续累积。
            excess = len(self._arecord_buffer) - byte_count
            if excess > 0:
                del self._arecord_buffer[:excess]
            data = bytes(self._arecord_buffer[:byte_count])
            del self._arecord_buffer[:byte_count]
            return data

    def _close_arecord(self) -> None:
        process = self._arecord
        self._arecord = None
        self._arecord_stop.set()
        with self._arecord_condition:
            self._arecord_condition.notify_all()
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        if process.stdout is not None:
            try:
                process.stdout.close()
            except OSError:
                pass
        reader = self._arecord_reader
        self._arecord_reader = None
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=2.0)
        with self._arecord_condition:
            self._arecord_buffer.clear()
            self._arecord_error = ""

    def read_frame(self, num_samples: int) -> np.ndarray:
        if self._stopping:
            raise InterruptedError("audio capture is stopping")
        if self.alsa_device:
            byte_count = int(num_samples) * self.device_channels * np.dtype(np.int16).itemsize
            last_error = None
            for attempt in range(1, self.reopen_attempts + 1):
                try:
                    data = self._read_arecord_exact(byte_count)
                    break
                except (OSError, RuntimeError) as exc:
                    if self._stopping:
                        raise InterruptedError("audio capture is stopping") from exc
                    last_error = exc
                    self._close_arecord()
                    print(
                        f"[AUDIO_CAPTURE] read failed attempt={attempt}/{self.reopen_attempts}: {exc}"
                    )
                    if attempt < self.reopen_attempts:
                        time.sleep(1.0)
            else:
                raise RuntimeError(
                    f"ReSpeaker did not recover after {self.reopen_attempts} attempts: {last_error}"
                )
        else:
            self._ensure_stream()
            data = self._stream.read(int(num_samples), exception_on_overflow=False)
        if self._stopping:
            raise InterruptedError("audio capture is stopping")
        matrix = np.frombuffer(data, dtype=np.int16).reshape(-1, self.device_channels)
        self._last_device_frame = matrix.T.copy()
        selected = matrix[:, self.raw_channels]
        return selected.T.copy()

    def _ensure_tuning(self):
        if self._tuning is not None:
            return self._tuning

        try:
            root_dir = Path(__file__).resolve().parents[2]
            if str(root_dir) not in sys.path:
                sys.path.insert(0, str(root_dir))

            import usb.core
            from tuning import Tuning

            dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
            if dev is None:
                self._hardware_doa_error = "respeaker_usb_not_found"
                return None
            self._tuning = Tuning(dev)
            self._hardware_doa_error = ""
            return self._tuning
        except Exception as exc:
            self._hardware_doa_error = str(exc)
            self._tuning = None
            return None

    def read_hardware_doa(self) -> tuple[float, int, str]:
        """读取 ReSpeaker 固件 DOA 角度；音频波束仍只使用 ch1~ch4 原始通道。"""
        tuning = self._ensure_tuning()
        if tuning is None:
            return -1.0, 0, self._hardware_doa_error

        try:
            angle = float(tuning.direction) % 360.0
            is_voice = int(tuning.is_voice())
            self._hardware_doa_error = ""
            return angle, is_voice, ""
        except Exception as exc:
            self._hardware_doa_error = str(exc)
            self._tuning = None
            return -1.0, 0, self._hardware_doa_error

    def close(self) -> None:
        self._stopping = True
        self._close_arecord()
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._tuning is not None:
            try:
                self._tuning.close()
            except Exception:
                pass
            self._tuning = None
        if self._pyaudio is not None:
            self._pyaudio.terminate()
            self._pyaudio = None

    def request_stop(self) -> None:
        self._stopping = True
        self._close_arecord()
