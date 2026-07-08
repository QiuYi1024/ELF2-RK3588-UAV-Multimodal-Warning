#!/usr/bin/env python3
"""AntiUAV 离线控制策略。

本模块只生成状态和命令，不连接球机、不发送真实 PTZ/对焦请求。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import copysign
from typing import Optional


class DetectionMode(Enum):
    HIGH_RECALL_SEARCH = 0
    BALANCED_SEARCH = 1
    LOW_FALSE_CONFIRM = 2
    CONSERVATIVE_ALARM = 3
    CUSTOM = 4


class LostState(Enum):
    VISUAL_TRACKING = "VISUAL_TRACKING"
    LOST_HOLD = "LOST_HOLD"
    LOST_ZOOM_OUT = "LOST_ZOOM_OUT"
    WAIT_AUDIO_GUIDE = "WAIT_AUDIO_GUIDE"
    AUDIO_GUIDING = "AUDIO_GUIDING"
    WAIT_VISUAL_RECAPTURE = "WAIT_VISUAL_RECAPTURE"


@dataclass
class DetectionPolicy:
    mode: DetectionMode = DetectionMode.BALANCED_SEARCH
    search_conf: float = 0.35
    search_min_box: float = 0.04
    track_conf: float = 0.45
    track_min_box: float = 0.06
    alarm_conf: float = 0.45
    alarm_min_box: float = 0.08
    search_confirm_frames: int = 3
    track_confirm_frames: int = 3
    alarm_confirm_frames: int = 2

    def stage(self, conf: float, max_side: float, stable_frames: int, track_continuous: bool, near_edge: bool) -> str:
        if conf < self.search_conf or max_side < self.search_min_box:
            return "REJECT"
        if (
            conf >= self.alarm_conf
            and max_side >= self.alarm_min_box
            and stable_frames >= self.alarm_confirm_frames
            and not near_edge
        ):
            return "ALARM_CONFIRM"
        if (
            conf >= self.track_conf
            and max_side >= self.track_min_box
            and stable_frames >= self.track_confirm_frames
            and track_continuous
            and not near_edge
        ):
            return "STABLE_TRACK"
        if stable_frames < self.search_confirm_frames:
            return "REJECT"
        return "SEARCH_PRETRACK"


@dataclass
class ZoomPolicy:
    target_min: float = 0.10
    ideal_low: float = 0.12
    ideal_high: float = 0.22
    target_max: float = 0.25
    step_in: int = 35
    step_out: int = 80
    cooldown_ms: int = 120
    lost_zoom_out_step: int = 160
    min_zoom: float = 1.0
    max_zoom: float = 25.0
    lost_search_zoom: float = 1.2

    def decide(
        self,
        max_side: Optional[float],
        stable: bool,
        near_edge: bool,
        lost: bool,
        now_ms: int,
        last_zoom_ms: int,
        current_zoom: Optional[float] = None,
    ) -> tuple[str, int, bool]:
        in_cooldown = now_ms - last_zoom_ms < self.cooldown_ms
        if current_zoom is not None:
            if current_zoom <= self.min_zoom and lost:
                return ("HOLD_MIN_ZOOM", 0, in_cooldown)
            if current_zoom <= self.lost_search_zoom and lost:
                return ("WAIT_AUDIO_GUIDE", 0, in_cooldown)
        if lost:
            return ("LOST_ZOOM_OUT", -self.lost_zoom_out_step, in_cooldown)
        if max_side is None or not stable:
            return ("HOLD_UNSTABLE", 0, in_cooldown)
        if near_edge:
            if max_side > self.ideal_high and not in_cooldown:
                return ("EDGE_ZOOM_OUT", -self.step_out, False)
            return ("EDGE_HOLD", 0, in_cooldown)
        if self.ideal_low <= max_side <= self.ideal_high:
            return ("HOLD_IDEAL", 0, in_cooldown)
        if current_zoom is not None and current_zoom >= self.max_zoom and max_side < self.target_min:
            return ("HOLD_MAX_ZOOM", 0, in_cooldown)
        if current_zoom is not None and current_zoom <= self.min_zoom and max_side > self.target_max:
            return ("HOLD_MIN_ZOOM", 0, in_cooldown)
        if max_side < self.target_min and not in_cooldown:
            return ("ZOOM_IN", self.step_in, False)
        if max_side > self.target_max and not in_cooldown:
            return ("ZOOM_OUT", -self.step_out, False)
        return ("HOLD_HYSTERESIS", 0, in_cooldown)


@dataclass
class LostSearchPolicy:
    hold_ms: int = 350
    zoom_out_start_ms: int = 700
    max_search_time_ms: int = 5000

    def state(self, visual: bool, lost_ms: int, zoom_ratio: float, search_zoom: float, audio_ready: bool) -> LostState:
        if visual:
            return LostState.VISUAL_TRACKING
        if lost_ms < self.hold_ms:
            return LostState.LOST_HOLD
        if lost_ms < self.zoom_out_start_ms or zoom_ratio > search_zoom:
            return LostState.LOST_ZOOM_OUT
        if audio_ready:
            return LostState.AUDIO_GUIDING
        if lost_ms < self.max_search_time_ms:
            return LostState.WAIT_AUDIO_GUIDE
        return LostState.WAIT_VISUAL_RECAPTURE


@dataclass
class TrackingSpeedPolicy:
    pan_gain: float = 1.0
    tilt_gain: float = 1.0
    max_speed: int = 66
    high_zoom_max_speed: int = 30
    dead_zone: float = 0.055
    feedforward_gain: float = 0.10

    def speed(self, dx_ratio: float, dy_ratio: float, zoom_ratio: float, vx_ratio: float = 0.0, vy_ratio: float = 0.0) -> tuple[int, int]:
        cap = self.high_zoom_max_speed if zoom_ratio >= 10.0 else self.max_speed

        def axis(error: float, velocity: float, gain: float) -> int:
            if abs(error) <= self.dead_zone:
                return 0
            e = min(1.0, (abs(error) - self.dead_zone) / max(0.001, 0.5 - self.dead_zone))
            raw = 10 + e ** 1.45 * (cap - 10)
            raw += abs(velocity) * self.feedforward_gain * cap
            return int(copysign(min(cap, max(10, raw * gain)), error))

        return axis(dx_ratio, vx_ratio, self.pan_gain), axis(dy_ratio, vy_ratio, self.tilt_gain)


@dataclass
class AudioCalibration:
    calibrated: bool = False
    offset_deg: float = 0.0
    confidence: float = 0.0
    samples: int = 0

    def target_pan(self, doa_deg: float) -> float:
        return (doa_deg + self.offset_deg) % 360.0


@dataclass
class AudioGuidancePolicy:
    yamnet_threshold: float = 20.0
    min_stability: float = 0.85
    max_doa_drift_deg: float = 12.0
    guide_cooldown_ms: int = 1200
    max_turn_deg: float = 42.0
    last_guide_ms: int = -999999

    def decide(
        self,
        visual_reliable: bool,
        yamnet_prob: float,
        doa_deg: float,
        doa_stability: float,
        doa_drift_deg: float,
        current_pan: float,
        calibration: AudioCalibration,
        manual_control: bool,
        emergency_stop: bool,
        now_ms: int,
    ) -> dict:
        blocked = []
        if visual_reliable:
            blocked.append("visual_reliable")
        if yamnet_prob < self.yamnet_threshold:
            blocked.append("yamnet_low")
        if doa_stability < self.min_stability or doa_drift_deg > self.max_doa_drift_deg:
            blocked.append("doa_unstable")
        if not calibration.calibrated or calibration.confidence < 0.75:
            blocked.append("not_calibrated")
        if manual_control:
            blocked.append("manual_control")
        if emergency_stop:
            blocked.append("emergency_stop")
        if now_ms - self.last_guide_ms < self.guide_cooldown_ms:
            blocked.append("cooldown")
        if blocked:
            return {"state": "WAIT_AUDIO_GUIDE", "blocked": "|".join(blocked), "target_pan": "", "pan_delta": 0.0, "command": "hold"}

        target_pan = calibration.target_pan(doa_deg)
        delta = circular_error(target_pan, current_pan)
        bounded = max(-self.max_turn_deg, min(self.max_turn_deg, delta))
        self.last_guide_ms = now_ms
        return {
            "state": "AUDIO_GUIDING",
            "blocked": "",
            "target_pan": round(target_pan, 2),
            "pan_delta": round(bounded, 2),
            "command": "ptz_pan_pulse",
        }


@dataclass
class FocusPolicy:
    enabled: bool = True
    cooldown_ms: int = 5000
    blur_threshold: float = 70.0
    confidence_drop: float = 0.18
    last_focus_ms: int = -999999

    def decide(self, now_ms: int, zoom_changed: bool, blur: float, conf: float, conf_baseline: float, user_request: bool = False) -> tuple[str, bool]:
        in_cooldown = now_ms - self.last_focus_ms < self.cooldown_ms
        reason = ""
        if user_request:
            reason = "USER_REQUEST"
        elif self.enabled and zoom_changed:
            reason = "AFTER_ZOOM"
        elif self.enabled and blur < self.blur_threshold:
            reason = "LOW_BLUR"
        elif self.enabled and conf_baseline - conf >= self.confidence_drop:
            reason = "CONFIDENCE_DROP"
        if reason and not in_cooldown:
            self.last_focus_ms = now_ms
            return reason, True
        return ("COOLDOWN" if reason else "HOLD", False)


def circular_error(target: float, current: float) -> float:
    return (target - current + 540.0) % 360.0 - 180.0


@dataclass
class SimulationSummary:
    checks: list[str] = field(default_factory=list)

    def require(self, name: str, ok: bool) -> None:
        if not ok:
            raise AssertionError(name)
        self.checks.append(name)
