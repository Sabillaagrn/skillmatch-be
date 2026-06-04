"""Pydantic request/response schemas — the public API contract."""
from __future__ import annotations

from pydantic import BaseModel, Field


# --- shared pieces -----------------------------------------------------------
class Skill(BaseModel):
    code: str = Field(..., examples=["ANLS"])
    name: str = Field(..., examples=["Analysis & Analytics"])


class ScoredSkill(Skill):
    score: float = Field(..., ge=0, le=1, description="Model confidence (0-1)")


class RecommendedSkill(Skill):
    reason: str = Field(..., description="Why this skill is recommended")


class RelatedJob(BaseModel):
    title: str
    job_id: str
    match_pct: float = Field(..., description="This job's match against the user's skills")


# --- /extract-skill ----------------------------------------------------------
class ExtractRequest(BaseModel):
    job_description: str = Field(..., min_length=1, max_length=20000)
    title: str = Field("", max_length=300, description="Optional job title (joined as 'title [SEP] description')")
    threshold: float | None = Field(None, ge=0, le=1)


class ExtractResponse(BaseModel):
    required_skills: list[ScoredSkill]


# --- /match ------------------------------------------------------------------
class MatchRequest(BaseModel):
    user_skills: list[str] = Field(..., description="Skill codes the user has", examples=[["ANLS", "IT"]])
    job_description: str = Field(..., min_length=1, max_length=20000)
    title: str = Field("", max_length=300, description="Optional job title (joined as 'title [SEP] description')")
    threshold: float | None = Field(None, ge=0, le=1)


class MatchResponse(BaseModel):
    match_pct: float = Field(..., description="Headline match % = simple coverage ratio (matched/required)")
    simple_match_pct: float = Field(..., description="Same simple coverage ratio (kept for compatibility)")
    weighted_match_pct: float = Field(0.0, description="Confidence-weighted match %, for transparency")
    matching: list[Skill]
    missing: list[Skill]
    required: list[ScoredSkill]
    recommended_skills: list[RecommendedSkill]
    related_jobs: list[RelatedJob]


# --- /recommend --------------------------------------------------------------
class RecommendRequest(BaseModel):
    user_skills: list[str] = Field(..., min_length=1)


class RecommendResponse(BaseModel):
    recommended_skills: list[RecommendedSkill]
    related_jobs: list[RelatedJob]


# --- /health -----------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
