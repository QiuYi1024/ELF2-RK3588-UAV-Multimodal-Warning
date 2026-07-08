from __future__ import annotations


def fuse_max(prob_raw: float, prob_beam: float, azimuth_deg: float, srp_score: float) -> dict:
    final = max(float(prob_raw), float(prob_beam))
    return {
        "prob_raw": float(prob_raw),
        "prob_beam": float(prob_beam),
        "prob_final": final,
        "azimuth_deg": float(azimuth_deg),
        "srp_score": float(srp_score),
    }
