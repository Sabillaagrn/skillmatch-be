"""Match calculation: compare a user's skills against a job's required skills.

Headline metric is the SIMPLE coverage ratio — identical to the training
notebook's end-to-end demo ("Match Score"):

    match = |user ∩ required| / |required|

We also expose a confidence-WEIGHTED variant for transparency:

    weighted = Σ(score_i · has_i) / Σ(score_i)   over required skills i

so a strongly-required skill the user has counts more than a borderline one.
"""
from __future__ import annotations

from ..core import skills as sk


def compute_match(user_codes: list[str], required: list[dict]) -> dict:
    """
    user_codes: skill codes the user claims.
    required:   [{'code','score'}, ...] from the model (already thresholded).
    """
    user_set = set(user_codes)
    req_codes = [r["code"] for r in required]
    req_set = set(req_codes)

    matched = [c for c in req_codes if c in user_set]
    missing = [c for c in req_codes if c not in user_set]

    # Simple coverage ratio — the headline metric (matches the demo).
    simple = (len(matched) / len(req_codes)) if req_codes else 0.0

    # Confidence-weighted variant, kept for transparency.
    total_w = sum(r["score"] for r in required) or 1e-9
    matched_w = sum(r["score"] for r in required if r["code"] in user_set)
    weighted = matched_w / total_w

    return {
        "match_pct": round(simple * 100, 1),
        "simple_match_pct": round(simple * 100, 1),
        "weighted_match_pct": round(weighted * 100, 1),
        "matching": sk.decorate(matched),
        "missing": sk.decorate(missing),
    }
