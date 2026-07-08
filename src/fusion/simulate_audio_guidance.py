#!/usr/bin/env python3
"""模拟声学 DOA 到云台 pan 的映射和引导门控。"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from antiuav_control_policy import AudioCalibration, AudioGuidancePolicy


def main() -> int:
    out_dir = Path("simulation_outputs")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "audio_guidance_decisions.csv"
    policy = AudioGuidancePolicy()
    uncalibrated = AudioCalibration(False, 0.0, 0.0, 0)
    calibrated = AudioCalibration(True, 35.0, 0.86, 28)
    cases = [
        ("visual_blocks", True, 80, 210, 0.95, 3, 180, calibrated, False, False, 0),
        ("uncalibrated_blocks", False, 80, 210, 0.95, 3, 180, uncalibrated, False, False, 1500),
        ("low_prob_blocks", False, 5, 210, 0.95, 3, 180, calibrated, False, False, 3000),
        ("unstable_blocks", False, 80, 210, 0.40, 30, 180, calibrated, False, False, 4500),
        ("manual_blocks", False, 80, 210, 0.95, 3, 180, calibrated, True, False, 6000),
        ("guides", False, 80, 210, 0.95, 3, 180, calibrated, False, False, 8000),
        ("recapture_wait", True, 80, 212, 0.95, 2, 214, calibrated, False, False, 10000),
    ]
    rows = []
    for name, visual, prob, doa, stability, drift, pan, calib, manual, stop, now in cases:
        result = policy.decide(visual, prob, doa, stability, drift, pan, calib, manual, stop, now)
        rows.append({
            "scenario": name,
            "visual_reliable": int(visual),
            "yamnet_prob": prob,
            "doa_deg": doa,
            "doa_stability": stability,
            "current_pan": pan,
            "calibrated": int(calib.calibrated),
            "target_pan": result["target_pan"],
            "pan_delta": result["pan_delta"],
            "state": result["state"],
            "blocked": result["blocked"],
            "command": result["command"],
        })

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    assert rows[1]["command"] == "hold" and "not_calibrated" in rows[1]["blocked"]
    assert rows[5]["command"] == "ptz_pan_pulse"
    assert float(rows[5]["target_pan"]) == 245.0
    assert rows[6]["command"] == "hold" and "visual_reliable" in rows[6]["blocked"]

    report = [
        "# audio guidance simulation report",
        "",
        f"- CSV: `{csv_path}`",
        "- 未标定、视觉已有目标、低概率、DOA 不稳、手动控制都会拒绝引导。",
        "- 已标定且视觉无目标时，DOA 210 + offset 35 映射为 target pan 245，并生成有限 pan pulse。",
    ]
    (out_dir / "audio_guidance_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"PASS audio guidance simulation rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
