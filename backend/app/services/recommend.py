"""Recommendation logic.

Missing-skill recommendations:
  - Start from the job's missing required skills (ranked by model score = how
    important the gap is).
  - When no JD context is given (/recommend), fall back to co-occurrence: skills
    that most frequently appear alongside the user's existing skills in the
    training data, excluding what they already have.

Related-job recommendations:
  - Cosine similarity between the user's 35-dim skill vector and every training
    job's skill vector (both precomputed in job_index.npz). Fast, no model call.
"""
from __future__ import annotations

import numpy as np

from ..core import skills as sk
from ..core.config import settings


def recommend_skills_from_gap(missing: list[dict]) -> list[dict]:
    """missing: [{'code','score'}] -> recommended skills, most important first."""
    ranked = sorted(missing, key=lambda d: d.get("score", 0), reverse=True)
    out = []
    for m in ranked[: settings.MAX_RECOMMENDED_SKILLS]:
        out.append({
            "code": m["code"],
            "name": sk.name_of(m["code"]),
            "reason": f"Required by this job (importance {m.get('score', 0):.0%}) and not in your skills",
        })
    return out


def recommend_skills_by_cooccurrence(user_codes: list[str]) -> list[dict]:
    """Skills that co-occur with the user's skills in real jobs."""
    ji = sk.get_job_index()
    have = {ji.index[c] for c in user_codes if c in ji.index}
    if not have:
        return []
    # Sum co-occurrence rows for the user's skills, zero out owned skills.
    scores = ji.cooc[list(have)].sum(axis=0)
    for i in have:
        scores[i] = 0.0
    order = np.argsort(scores)[::-1]
    out = []
    for i in order[: settings.MAX_RECOMMENDED_SKILLS]:
        if scores[i] <= 0:
            break
        code = ji.classes[i]
        out.append({
            "code": code,
            "name": sk.name_of(code),
            "reason": "Frequently appears alongside your current skills",
        })
    return out


def related_jobs(user_codes: list[str]) -> list[dict]:
    """Top jobs by cosine similarity to the user's skill vector."""
    ji = sk.get_job_index()
    uv = ji.vectorize(user_codes)
    norm = np.linalg.norm(uv)
    if norm == 0:
        return []
    sims = (ji.vectors @ uv) / (ji._norms * norm)  # (J,)

    # Deduplicate by title, keep the best-scoring instance.
    best: dict[str, tuple[float, int]] = {}
    for idx in np.argsort(sims)[::-1]:
        title = str(ji.titles[idx])
        s = float(sims[idx])
        if s <= 0:
            break
        if title not in best:
            best[title] = (s, idx)
        if len(best) >= settings.MAX_RELATED_JOBS:
            break

    out = []
    for title, (s, idx) in sorted(best.items(), key=lambda kv: kv[1][0], reverse=True):
        out.append({
            "title": title,
            "job_id": str(ji.job_ids[idx]),
            "match_pct": round(s * 100, 1),
        })
    return out
