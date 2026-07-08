from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .srp_phat import SrpPhatEstimator


def _valid_angle(angle: float) -> bool:
    return 0.0 <= float(angle) < 360.0


def _angle_delta_deg(a: float, b: float) -> float:
    return (float(a) - float(b) + 180.0) % 360.0 - 180.0


def _circular_mean_deg(angles: list[float]) -> float:
    values = [float(a) for a in angles if _valid_angle(float(a))]
    if not values:
        return -1.0
    radians = np.deg2rad(values)
    s = float(np.mean(np.sin(radians)))
    c = float(np.mean(np.cos(radians)))
    return float((np.rad2deg(np.arctan2(s, c)) + 360.0) % 360.0)


def _circular_median_deg(angles: list[float]) -> float:
    values = [float(a) for a in angles if _valid_angle(float(a))]
    if not values:
        return -1.0
    best = min(values, key=lambda candidate: sum(abs(_angle_delta_deg(v, candidate)) for v in values))
    return float(best)


def _circular_std_deg(angles: list[float]) -> float:
    values = [float(a) for a in angles if _valid_angle(float(a))]
    if len(values) <= 1:
        return 0.0 if values else 999.0
    radians = np.deg2rad(values)
    s = float(np.mean(np.sin(radians)))
    c = float(np.mean(np.cos(radians)))
    r = max(1e-6, min(1.0, float(math.hypot(s, c))))
    return float(np.rad2deg(np.sqrt(max(0.0, -2.0 * np.log(r)))))


@dataclass
class DOAResult:
    raw_doa_deg: float = -1.0
    smooth_doa_deg: float = -1.0
    doa_valid: bool = False
    stable_doa: bool = False
    doa_source: str = "none"
    doa_confidence: float = 0.0
    raw_confidence: float = 0.0
    hardware_doa_deg: float = -1.0
    hardware_voice: int = 0
    error: str = ""


class DOAProvider:
    """阵列无关 DOA Provider 基类，输出原始角度，不负责报警判定。"""

    source_name = "none"

    def estimate(self, mic_data: np.ndarray) -> DOAResult:
        return DOAResult(doa_source=self.source_name)


class SRPPHATDOAProvider(DOAProvider):
    source_name = "srp_phat"

    def __init__(self, estimator: SrpPhatEstimator, source_name: str = "srp_phat"):
        self.estimator = estimator
        self.source_name = source_name

    def estimate(self, mic_data: np.ndarray) -> DOAResult:
        angle, score = self.estimator.estimate(mic_data)
        return DOAResult(
            raw_doa_deg=float(angle),
            smooth_doa_deg=float(angle),
            doa_valid=_valid_angle(angle),
            stable_doa=_valid_angle(angle),
            doa_source=self.source_name,
            doa_confidence=float(score),
            raw_confidence=float(score),
        )


class SMD16DOAProvider(SRPPHATDOAProvider):
    def __init__(self, estimator: SrpPhatEstimator):
        super().__init__(estimator, source_name="smd16_srp_phat")


class ReSpeakerHardwareDOAProvider(DOAProvider):
    source_name = "respeaker_hardware"

    def __init__(self, backend, fallback: Optional[DOAProvider] = None):
        self.backend = backend
        self.fallback = fallback

    def estimate(self, mic_data: np.ndarray) -> DOAResult:
        if hasattr(self.backend, "read_hardware_doa"):
            angle, hardware_voice, error = self.backend.read_hardware_doa()
            if _valid_angle(angle):
                confidence = 0.75 if int(hardware_voice) else 0.45
                return DOAResult(
                    raw_doa_deg=float(angle),
                    smooth_doa_deg=float(angle),
                    doa_valid=True,
                    stable_doa=True,
                    doa_source=self.source_name,
                    doa_confidence=confidence,
                    raw_confidence=confidence,
                    hardware_doa_deg=float(angle),
                    hardware_voice=int(hardware_voice),
                    error=str(error or ""),
                )
            if self.fallback is None:
                return DOAResult(
                    raw_doa_deg=-1.0,
                    smooth_doa_deg=-1.0,
                    doa_valid=False,
                    stable_doa=False,
                    doa_source=self.source_name,
                    hardware_doa_deg=float(angle),
                    hardware_voice=int(hardware_voice),
                    error=str(error or "invalid_hardware_doa"),
                )

        if self.fallback is not None:
            result = self.fallback.estimate(mic_data)
            if result.doa_source == "srp_phat":
                result.doa_source = "srp_phat_fallback"
            return result
        return DOAResult(doa_source=self.source_name)


class DisabledDOAProvider(DOAProvider):
    source_name = "none"


class DOAStabilizer:
    """环形中值 + EMA + 跳变门控，输出给波束和云台引导使用的平滑 DOA。

    大跳变默认先拦截；如果连续多个原始角度稳定落在新方向，则认为目标真实换向，
    立即重置平滑角，避免声学引导长时间粘在旧方向。
    """

    def __init__(
        self,
        median_window: int = 5,
        ema_alpha: float = 0.35,
        jump_reject_deg: float = 40.0,
        stable_window: int = 3,
        stable_max_std_deg: float = 18.0,
        hold_sec: float = 0.8,
        rebase_window: int = 3,
        rebase_min_count: int = 2,
        rebase_max_std_deg: float = 12.0,
    ):
        self.median_window = max(1, int(median_window))
        self.ema_alpha = float(max(0.01, min(1.0, ema_alpha)))
        self.jump_reject_deg = float(jump_reject_deg)
        self.stable_window = max(1, int(stable_window))
        self.stable_max_std_deg = float(stable_max_std_deg)
        self.hold_sec = float(max(0.0, hold_sec))
        self.rebase_window = max(1, int(rebase_window))
        self.rebase_min_count = max(1, min(int(rebase_min_count), self.rebase_window))
        self.rebase_max_std_deg = float(max(1.0, rebase_max_std_deg))
        self.raw_history: deque[float] = deque(maxlen=self.median_window)
        self.stable_history: deque[float] = deque(maxlen=self.stable_window)
        self.rebase_history: deque[float] = deque(maxlen=self.rebase_window)
        self.smooth_angle = -1.0
        self.last_valid_time = 0.0

    def _reset_to_angle(self, angle: float) -> None:
        self.smooth_angle = float(angle) % 360.0
        self.raw_history.clear()
        self.stable_history.clear()
        self.rebase_history.clear()
        self.raw_history.append(self.smooth_angle)
        self.stable_history.append(self.smooth_angle)

    def update(self, raw: DOAResult, now: float | None = None) -> DOAResult:
        now = time.monotonic() if now is None else float(now)
        if not raw.doa_valid or not _valid_angle(raw.raw_doa_deg):
            held = _valid_angle(self.smooth_angle) and (now - self.last_valid_time) <= self.hold_sec
            return DOAResult(
                raw_doa_deg=raw.raw_doa_deg,
                smooth_doa_deg=self.smooth_angle if held else -1.0,
                doa_valid=held,
                stable_doa=False,
                doa_source=raw.doa_source,
                doa_confidence=0.0,
                raw_confidence=raw.raw_confidence,
                hardware_doa_deg=raw.hardware_doa_deg,
                hardware_voice=raw.hardware_voice,
                error=raw.error,
            )

        raw_angle = float(raw.raw_doa_deg) % 360.0
        self.raw_history.append(raw_angle)
        median_angle = _circular_median_deg(list(self.raw_history))

        if _valid_angle(self.smooth_angle):
            jump = abs(_angle_delta_deg(median_angle, self.smooth_angle))
            if jump > self.jump_reject_deg:
                self.rebase_history.append(raw_angle)
                rebase_angles = list(self.rebase_history)
                rebase_mean = _circular_mean_deg(rebase_angles)
                rebase_std = _circular_std_deg(rebase_angles)
                rebase_jump = abs(_angle_delta_deg(rebase_mean, self.smooth_angle)) if _valid_angle(rebase_mean) else 0.0
                if (
                    len(rebase_angles) >= self.rebase_min_count
                    and _valid_angle(rebase_mean)
                    and rebase_std <= self.rebase_max_std_deg
                    and rebase_jump > self.jump_reject_deg
                ):
                    self._reset_to_angle(rebase_mean)
                    self.last_valid_time = now
                    confidence = max(0.0, min(1.0, max(raw.raw_confidence, 0.85)))
                    return DOAResult(
                        raw_doa_deg=raw_angle,
                        smooth_doa_deg=self.smooth_angle,
                        doa_valid=True,
                        stable_doa=True,
                        doa_source=raw.doa_source,
                        doa_confidence=confidence,
                        raw_confidence=raw.raw_confidence,
                        hardware_doa_deg=raw.hardware_doa_deg,
                        hardware_voice=raw.hardware_voice,
                        error="jump_rebased",
                    )
                held = (now - self.last_valid_time) <= self.hold_sec
                confidence = max(0.0, min(1.0, raw.raw_confidence * 0.35))
                return DOAResult(
                    raw_doa_deg=raw_angle,
                    smooth_doa_deg=self.smooth_angle if held else -1.0,
                    doa_valid=held,
                    stable_doa=False,
                    doa_source=raw.doa_source,
                    doa_confidence=confidence,
                    raw_confidence=raw.raw_confidence,
                    hardware_doa_deg=raw.hardware_doa_deg,
                    hardware_voice=raw.hardware_voice,
                    error="jump_rejected",
                )
            self.rebase_history.clear()
            self.smooth_angle = (self.smooth_angle + self.ema_alpha * _angle_delta_deg(median_angle, self.smooth_angle)) % 360.0
        else:
            self._reset_to_angle(median_angle)

        self.last_valid_time = now
        self.stable_history.append(self.smooth_angle)
        angle_std = _circular_std_deg(list(self.stable_history))
        enough = len(self.stable_history) >= self.stable_window
        stable = enough and angle_std <= self.stable_max_std_deg
        stability_score = 1.0 / (1.0 + angle_std / max(1.0, self.stable_max_std_deg))
        confidence = max(0.0, min(1.0, max(raw.raw_confidence, stability_score if stable else stability_score * 0.75)))
        return DOAResult(
            raw_doa_deg=raw_angle,
            smooth_doa_deg=self.smooth_angle,
            doa_valid=True,
            stable_doa=stable,
            doa_source=raw.doa_source,
            doa_confidence=confidence,
            raw_confidence=raw.raw_confidence,
            hardware_doa_deg=raw.hardware_doa_deg,
            hardware_voice=raw.hardware_voice,
            error=raw.error,
        )


def create_doa_provider(doa_source: str, backend, srp: SrpPhatEstimator, doa_fallback: str = "srp_phat", disable_software_doa: bool = False, array_name: str = "") -> DOAProvider:
    source = str(doa_source or "srp_phat").lower()
    fallback: Optional[DOAProvider] = None
    if not disable_software_doa and str(doa_fallback).lower() in ("srp_phat", "software_srp_phat"):
        fallback = SRPPHATDOAProvider(srp)

    if source in ("none", "disabled", "off"):
        return DisabledDOAProvider()
    if source in ("respeaker_hardware", "hardware"):
        return ReSpeakerHardwareDOAProvider(backend, fallback=fallback)
    if source in ("smd16", "smd16_usb"):
        return SMD16DOAProvider(srp)
    if source in ("srp_phat", "software_srp_phat"):
        if "smd16" in str(array_name).lower():
            return SMD16DOAProvider(srp)
        return SRPPHATDOAProvider(srp)
    return fallback or DisabledDOAProvider()
