from __future__ import annotations

import math
import time

import numpy as np


def _hz_to_mel_slaney(freq: float) -> float:
    f_sp = 200.0 / 3.0
    mel = float(freq) / f_sp
    if freq >= 1000.0:
        mel = 15.0 + math.log(float(freq) / 1000.0) / (math.log(6.4) / 27.0)
    return mel


def _create_slaney_mel_filterbank(
    sample_rate: int = 16000,
    n_fft: int = 400,
    n_mels: int = 64,
    f_min: float = 125.0,
    f_max: float = 7500.0,
) -> np.ndarray:
    all_freqs = np.linspace(0.0, sample_rate // 2, n_fft // 2 + 1, dtype=np.float32)
    mel_points = np.linspace(
        _hz_to_mel_slaney(f_min),
        _hz_to_mel_slaney(f_max),
        n_mels + 2,
        dtype=np.float32,
    )

    f_sp = np.float32(200.0 / 3.0)
    freq_points = mel_points * f_sp
    min_log_hz = np.float32(1000.0)
    min_log_mel = min_log_hz / f_sp
    logstep = np.float32(math.log(6.4) / 27.0)
    log_region = mel_points >= min_log_mel
    freq_points[log_region] = min_log_hz * np.exp(
        logstep * (mel_points[log_region] - min_log_mel)
    )

    freq_diff = freq_points[1:] - freq_points[:-1]
    slopes = freq_points[None, :] - all_freqs[:, None]
    down_slopes = -slopes[:, :-2] / freq_diff[:-1]
    up_slopes = slopes[:, 2:] / freq_diff[1:]
    filterbank = np.maximum(0.0, np.minimum(down_slopes, up_slopes))
    filterbank *= (2.0 / (freq_points[2:] - freq_points[:-2]))[None, :]
    return filterbank.astype(np.float32)


def _waveform_to_log_mel_nchw(
    waveform_1sec: np.ndarray,
    mel_filterbank: np.ndarray,
    n_fft: int = 400,
    hop_length: int = 160,
    target_frames: int = 96,
) -> np.ndarray:
    waveform = np.asarray(waveform_1sec, dtype=np.float32).reshape(-1)
    if waveform.size < 2:
        waveform = np.pad(waveform, (0, 2 - waveform.size))

    padded = np.pad(waveform, (n_fft // 2, n_fft // 2), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop_length]
    window = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    spectrum = np.fft.rfft(frames * window[None, :], n=n_fft, axis=1)
    power = np.abs(spectrum).astype(np.float32) ** 2
    log_mel = np.log(power @ mel_filterbank + 0.001).astype(np.float32)

    if log_mel.shape[0] > target_frames:
        log_mel = log_mel[:target_frames, :]
    elif log_mel.shape[0] < target_frames:
        log_mel = np.pad(log_mel, ((0, target_frames - log_mel.shape[0]), (0, 0)))
    return log_mel[None, None, :, :].astype(np.float32)


def _array_stats(values: np.ndarray) -> dict:
    x = np.asarray(values, dtype=np.float32)
    if x.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }


class DummyDetector:
    """Fallback detector used when RKNN runtime or model files are unavailable."""

    is_dummy = True
    fallback_reason = "dummy detector selected"

    def infer(self, waveform_1sec: np.ndarray) -> tuple[float, float, dict]:
        t0 = time.perf_counter()
        x = np.asarray(waveform_1sec, dtype=np.float32)
        if x.size == 0:
            return 0.0, 0.0, {"detector": "dummy", "reason": "empty waveform"}
        rms = float(np.sqrt(np.mean(x * x) + 1e-9))
        spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
        freqs = np.fft.rfftfreq(x.size, 1.0 / 16000.0)
        rotor_band = (freqs >= 120.0) & (freqs <= 650.0)
        full_band = (freqs >= 80.0) & (freqs <= 7500.0)
        band_ratio = float(np.sum(spec[rotor_band]) / max(np.sum(spec[full_band]), 1e-9))
        prob = float(np.clip(100.0 * (0.35 * band_ratio + 4.2 * rms), 0.0, 99.0))
        infer_ms = (time.perf_counter() - t0) * 1000.0
        return prob, infer_ms, {"detector": "dummy", "rms": rms, "rotor_band_ratio": band_ratio}

    def release(self) -> None:
        return None


class YAMNetRKNNDetector:
    @property
    def is_dummy(self) -> bool:
        return self._dummy is not None

    @property
    def fallback_reason(self) -> str:
        return getattr(self, "_dummy_reason", "")

    def __init__(self, model_path: str, allow_dummy: bool = True, drone_class_index: int = 1):
        self.model_path = str(model_path)
        self.allow_dummy = bool(allow_dummy)
        self.drone_class_index = int(drone_class_index)
        self._dummy = None
        self._init_runtime()

    def _init_runtime(self) -> None:
        try:
            from rknnlite.api import RKNNLite
        except Exception as exc:
            if self.allow_dummy:
                self._dummy = DummyDetector()
                self._dummy_reason = f"import failed: {exc}"
                return
            raise

        self.RKNNLite = RKNNLite
        self.rknn = RKNNLite()
        if self.rknn.load_rknn(self.model_path) != 0:
            if self.allow_dummy:
                self._dummy = DummyDetector()
                self._dummy_reason = f"model load failed: {self.model_path}"
                return
            raise RuntimeError(f"YAMNet RKNN model load failed: {self.model_path}")
        if self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_2) != 0:
            if self.allow_dummy:
                self._dummy = DummyDetector()
                self._dummy_reason = "RKNN runtime init failed"
                return
            raise RuntimeError("YAMNet RKNN runtime init failed")

        self.mel_filterbank = _create_slaney_mel_filterbank()

    def waveform_to_feature_nchw(self, waveform_1sec: np.ndarray) -> np.ndarray:
        return _waveform_to_log_mel_nchw(waveform_1sec, self.mel_filterbank)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        e_x = np.exp(x - np.max(x))
        return e_x / np.maximum(e_x.sum(axis=-1, keepdims=True), 1e-9)

    def infer(self, waveform_1sec: np.ndarray) -> tuple[float, float, dict]:
        if self._dummy is not None:
            prob, infer_ms, extra = self._dummy.infer(waveform_1sec)
            extra["fallback_reason"] = self._dummy_reason
            return prob, infer_ms, extra

        feature_nchw = self.waveform_to_feature_nchw(waveform_1sec)
        feature_nhwc = np.transpose(feature_nchw, (0, 2, 3, 1))
        t0 = time.perf_counter()
        outputs = self.rknn.inference(inputs=[feature_nhwc], data_format=["nhwc"])
        infer_ms = (time.perf_counter() - t0) * 1000.0
        raw_output = np.asarray(outputs[0], dtype=np.float32)
        logits = raw_output.reshape(-1)
        if logits.size == 1:
            prob = float(100.0 / (1.0 + np.exp(-float(logits[0]))))
            score_mode = "sigmoid_single_output"
            softmax_percent = []
            drone_class_index = 0
        else:
            probs = self.softmax(logits) * 100.0
            drone_class_index = min(max(self.drone_class_index, 0), int(probs.size) - 1)
            prob = float(probs[drone_class_index])
            score_mode = "softmax_multi_output"
            softmax_percent = [float(v) for v in probs[:8]]
        return prob, float(infer_ms), {
            "detector": "yamnet_rknn",
            "feature_shape_nchw": list(feature_nchw.shape),
            "rknn_input_shape": list(feature_nhwc.shape),
            "rknn_input_dtype": str(feature_nhwc.dtype),
            "log_mel_stats": _array_stats(feature_nchw),
            "raw_output_shape": list(raw_output.shape),
            "raw_logits": [float(v) for v in logits[:8]],
            "softmax_percent": softmax_percent,
            "drone_class_index": drone_class_index,
            "score_mode": score_mode,
        }

    def release(self) -> None:
        if self._dummy is not None:
            return
        self.rknn.release()


def create_detector(model_path: str | None, allow_dummy: bool = True, drone_class_index: int = 1):
    if not model_path:
        return DummyDetector()
    return YAMNetRKNNDetector(
        model_path=model_path,
        allow_dummy=allow_dummy,
        drone_class_index=drone_class_index,
    )
