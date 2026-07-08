from __future__ import annotations

from collections import deque


class WindowAlarmLogic:
    def __init__(self, threshold_percent: float = 30.0, window_count: int = 5, min_hits: int = 2):
        self.threshold_percent = float(threshold_percent)
        self.history = deque(maxlen=int(window_count))
        self.min_hits = int(min_hits)
        self.first_alarm_time_ms = None
        self.records = []

    def update(self, record: dict) -> tuple[bool, dict]:
        hit = float(record.get("prob_final", 0.0)) > self.threshold_percent
        self.history.append(1 if hit else 0)
        alarm = sum(self.history) >= self.min_hits
        enriched = dict(record)
        enriched["threshold_percent"] = self.threshold_percent
        enriched["window_hit"] = bool(hit)
        enriched["alarm"] = bool(alarm)
        if alarm and self.first_alarm_time_ms is None:
            self.first_alarm_time_ms = record.get("window_end_ms") or record.get("timestamp_ms")
        enriched["first_alarm_time_ms"] = self.first_alarm_time_ms
        self.records.append(enriched)
        return bool(alarm), enriched
