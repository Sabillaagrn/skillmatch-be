"""REST endpoints: /extract-skill, /match, /recommend, /health.

Inference is CPU-bound and blocking, so we run it in a threadpool
(`run_in_threadpool`) to keep the async event loop responsive under concurrency.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from ..core.config import settings
from ..core import skills as sk
from ..schemas import models as m
from ..services import matching, recommend

# Imported via REPO_ROOT on sys.path (set in core.config).
from ml.src import inference  # noqa: E402

router = APIRouter()


async def _extract(job_description: str, threshold: float | None, title: str = "") -> list[dict]:
    """Run the model (in a thread) and return [{'code','score'}] above threshold."""
    try:
        pairs = await run_in_threadpool(
            inference.extract_skills_cached, job_description, threshold, title
        )
    except Exception as exc:  # inference/model failure
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    return [{"code": c, "score": s} for c, s in pairs]


def _validate_codes(codes: list[str]) -> list[str]:
    """Keep only known skill codes; reject if the user sent none valid."""
    valid = [c for c in codes if c in sk.SKILL_NAMES]
    if codes and not valid:
        raise HTTPException(
            status_code=422,
            detail=f"No valid skill codes. Allowed codes: {', '.join(sk.ALL_CODES)}",
        )
    return valid


@router.get("/health", response_model=m.HealthResponse)
async def health() -> m.HealthResponse:
    return m.HealthResponse(status="ok", model_loaded=inference.is_ready())


@router.post("/extract-skill", response_model=m.ExtractResponse)
async def extract_skill(req: m.ExtractRequest) -> m.ExtractResponse:
    required = await _extract(req.job_description, req.threshold, req.title)
    return m.ExtractResponse(
        required_skills=[
            m.ScoredSkill(code=r["code"], name=sk.name_of(r["code"]), score=r["score"])
            for r in required
        ]
    )


@router.post("/match", response_model=m.MatchResponse)
async def match(req: m.MatchRequest) -> m.MatchResponse:
    user_codes = _validate_codes(req.user_skills)
    required = await _extract(req.job_description, req.threshold, req.title)

    result = matching.compute_match(user_codes, required)
    rec_skills = recommend.recommend_skills_from_gap(
        [r for r in required if r["code"] not in set(user_codes)]
    )
    jobs = await run_in_threadpool(recommend.related_jobs, user_codes)

    return m.MatchResponse(
        match_pct=result["match_pct"],
        simple_match_pct=result["simple_match_pct"],
        weighted_match_pct=result["weighted_match_pct"],
        matching=[m.Skill(**s) for s in result["matching"]],
        missing=[m.Skill(**s) for s in result["missing"]],
        required=[
            m.ScoredSkill(code=r["code"], name=sk.name_of(r["code"]), score=r["score"])
            for r in required
        ],
        recommended_skills=[m.RecommendedSkill(**s) for s in rec_skills],
        related_jobs=[m.RelatedJob(**j) for j in jobs],
    )


@router.post("/recommend", response_model=m.RecommendResponse)
async def recommend_endpoint(req: m.RecommendRequest) -> m.RecommendResponse:
    user_codes = _validate_codes(req.user_skills)
    rec_skills = recommend.recommend_skills_by_cooccurrence(user_codes)
    jobs = await run_in_threadpool(recommend.related_jobs, user_codes)
    return m.RecommendResponse(
        recommended_skills=[m.RecommendedSkill(**s) for s in rec_skills],
        related_jobs=[m.RelatedJob(**j) for j in jobs],
    )
