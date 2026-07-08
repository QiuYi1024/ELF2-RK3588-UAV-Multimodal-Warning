#!/usr/bin/env python3
"""模拟目标大小、丢失搜索、PTZ 速度和自动对焦。"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from antiuav_control_policy import FocusPolicy, LostSearchPolicy, TrackingSpeedPolicy, ZoomPolicy


def main() -> int:
    out_dir = Path("simulation_outputs")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "tracking_zoom_decisions.csv"
    zoom = ZoomPolicy()
    lost = LostSearchPolicy()
    speed = TrackingSpeedPolicy()
    focus = FocusPolicy()
    last_zoom_ms = -9999
    rows = []

    scenarios = [
        ("small_target", 0, 0.03, True, False, False, 2.0, 0.35, 0.02),
        ("enter_range", 200, 0.12, True, False, False, 2.4, 0.08, 0.00),
        ("ideal_range", 400, 0.18, True, False, False, 2.4, 0.01, 0.00),
        ("too_large", 700, 0.30, True, False, False, 12.0, -0.30, 0.03),
        ("near_edge", 1000, 0.09, True, True, False, 6.0, 0.42, 0.00),
        ("lost_hold", 1300, None, False, False, True, 8.0, 0.00, 0.00),
        ("lost_zoom_out", 1900, None, False, False, True, 8.0, 0.00, 0.00),
        ("search_zoom_wait_audio", 2600, None, False, False, True, 1.1, 0.00, 0.00),
        ("recapture", 3400, 0.13, True, False, False, 1.2, -0.04, 0.00),
        ("max_zoom_stop", 3800, 0.03, True, False, False, 25.5, 0.22, 0.00),
        ("min_zoom_stop", 4200, 0.30, True, False, False, 1.0, -0.22, 0.00),
    ]

    for name, now_ms, box, stable, edge, is_lost, zoom_ratio, dx, vx in scenarios:
        zoom_state, zoom_cmd, cooldown = zoom.decide(box, stable, edge, is_lost, now_ms, last_zoom_ms, zoom_ratio)
        if zoom_cmd:
            last_zoom_ms = now_ms
        lost_state = lost.state(not is_lost and box is not None, now_ms - 1200 if is_lost else 0, zoom_ratio, zoom.lost_search_zoom, False)
        pan, tilt = speed.speed(dx, 0.01, zoom_ratio, vx, 0.0)
        focus_reason, focus_trigger = focus.decide(now_ms, zoom_cmd != 0, blur=65 if name == "recapture" else 120, conf=0.40, conf_baseline=0.60)
        rows.append({
            "scenario": name,
            "time_ms": now_ms,
            "max_side": "" if box is None else box,
            "zoom_state": zoom_state,
            "zoom_cmd": zoom_cmd,
            "cooldown": int(cooldown),
            "lost_state": lost_state.value,
            "pan_speed": pan,
            "tilt_speed": tilt,
            "focus_reason": focus_reason,
            "focus_trigger": int(focus_trigger),
        })

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    checks = {
        "small target zooms in": rows[0]["zoom_cmd"] > 0,
        "ideal range holds": rows[2]["zoom_cmd"] == 0,
        "large target zooms out": rows[3]["zoom_cmd"] < 0,
        "lost triggers zoom out": rows[6]["zoom_cmd"] < 0,
        "high zoom caps speed": abs(rows[3]["pan_speed"]) <= speed.high_zoom_max_speed,
        "dead-zone stops": rows[2]["pan_speed"] == 0,
        "focus generated": any(r["focus_trigger"] for r in rows),
        "lost search zoom waits audio": rows[7]["zoom_cmd"] == 0 and rows[7]["zoom_state"] == "WAIT_AUDIO_GUIDE",
        "max zoom blocks zoom in": rows[9]["zoom_cmd"] == 0 and rows[9]["zoom_state"] == "HOLD_MAX_ZOOM",
        "min zoom blocks zoom out": rows[10]["zoom_cmd"] == 0 and rows[10]["zoom_state"] == "HOLD_MIN_ZOOM",
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(", ".join(failed))

    report = [
        "# tracking / zoom simulation report",
        "",
        f"- CSV: `{csv_path}`",
        f"- checks: {len(checks)} passed",
        "- covered: small target, ideal target, oversized target, target lost, zoom-out search, search zoom wait, recapture, speed cap, autofocus request, min/max zoom limit",
    ]
    (out_dir / "simulation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"PASS tracking zoom simulation rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
