from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from supabase import create_client, Client

from ..core.config import settings
from ..core import skills as sk
from ..schemas import models as m
from ..services import matching, recommend
from typing import Optional

# Inisialisasi Supabase
# Pastikan settings.SUPABASE_URL dan KEY sudah terisi (cek config.py)
supabase: Optional[Client] = None

if settings.SUPABASE_URL and settings.SUPABASE_KEY:
    supabase = create_client(
        settings.SUPABASE_URL,
        settings.SUPABASE_KEY
    )

# Imported via REPO_ROOT on sys.path
from ml.src import inference

router = APIRouter()

# ==========================================
# SCHEMAS
# ==========================================

class UserRegister(BaseModel):
    name: str
    email: str
    password: str


class UserLogin(BaseModel):
    email: str
    password: str


class ProfileHistory(BaseModel):
    date: str
    role: str
    pct: int
    tone: str


class UserProfile(BaseModel):
    email: str
    name: str
    location: str
    photo: Optional[str] = None
    skills: List[str] = []
    history: List[ProfileHistory] = []


# ==========================================
# HELPER FUNCTIONS
# ==========================================

async def _extract(
    job_description: str,
    threshold: float | None,
    title: str = ""
) -> list[dict]:

    try:

        pairs = await run_in_threadpool(
            inference.extract_skills_cached,
            job_description,
            threshold,
            title
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}"
        ) from exc

    return [
        {
            "code": code,
            "score": score
        }
        for code, score in pairs
    ]


def _validate_codes(
    codes: list[str]
) -> list[str]:

    valid = [
        c
        for c in codes
        if c in sk.SKILL_NAMES
    ]

    if codes and not valid:

        raise HTTPException(
            status_code=422,
            detail=f"No valid skill codes. Allowed codes: {', '.join(sk.ALL_CODES)}"
        )

    return valid


# ==========================================
# AUTHENTICATION ENDPOINTS
# ==========================================

@router.post("/register")
async def register(user: UserRegister):

    response = (
        supabase
        .table("users")
        .select("id")
        .eq("email", user.email)
        .execute()
    )

    if response.data:

        raise HTTPException(
            status_code=400,
            detail="Email sudah digunakan."
        )

    (
        supabase
        .table("users")
        .insert({
            "name": user.name,
            "email": user.email,
            "password": user.password,
            "location": "Bandung, Jawa Barat",
            "skills": [],
            "history": []
        })
        .execute()
    )

    return {
        "message": "Registrasi berhasil",
        "user": {
            "name": user.name,
            "email": user.email
        }
    }


@router.post("/login")
async def login(user: UserLogin):

    response = (
        supabase
        .table("users")
        .select("*")
        .eq("email", user.email)
        .execute()
    )

    db_user = response.data[0] if response.data else None

    if not db_user or db_user["password"] != user.password:

        raise HTTPException(
            status_code=401,
            detail="Email atau password salah."
        )

    return {
        "message": "Login berhasil",
        "user": {
            "name": db_user["name"],
            "email": db_user["email"]
        }
    }


# ==========================================
# PROFILE ENDPOINTS
# ==========================================

@router.get("/profile")
async def get_profile(email: str):

    response = (
        supabase
        .table("users")
        .select("*")
        .eq("email", email)
        .execute()
    )

    db_user = response.data[0] if response.data else None

    if not db_user:

        raise HTTPException(
            status_code=404,
            detail="User tidak ditemukan"
        )

    return {
        "name": db_user.get("name"),
        "location": db_user.get(
            "location",
            "Bandung, Jawa Barat"
        ),
        "photo": db_user.get("photo"),
        "skills": db_user.get("skills", []),
        "history": db_user.get("history", [])
    }


@router.post("/profile")
async def update_profile(profile: UserProfile):

    response = (
        supabase
        .table("users")
        .update({
            "name": profile.name,
            "location": profile.location,
            "photo": profile.photo,
            "skills": profile.skills,
            "history": [h.dict() for h in profile.history]
        })
        .eq("email", profile.email)
        .execute()
    )

    if not response.data:

        raise HTTPException(
            status_code=404,
            detail="User tidak ditemukan"
        )

    return {
        "message": "Profil berhasil diperbarui"
    }


# ==========================================
# ML & CORE ENDPOINTS
# ==========================================

@router.get("/health", response_model=m.HealthResponse)
async def health():

    return m.HealthResponse(
        status="ok",
        model_loaded=inference.is_ready()
    )


@router.post("/extract-skill", response_model=m.ExtractResponse)
async def extract_skill(req: m.ExtractRequest):

    required = await _extract(
        req.job_description,
        req.threshold,
        req.title
    )

    return m.ExtractResponse(
        required_skills=[
            m.ScoredSkill(
                code=r["code"],
                name=sk.name_of(r["code"]),
                score=r["score"]
            )
            for r in required
        ]
    )


@router.post("/match", response_model=m.MatchResponse)
async def match(req: m.MatchRequest):

    user_codes = _validate_codes(
        req.user_skills
    )

    required = await _extract(
        req.job_description,
        req.threshold,
        req.title
    )

    result = matching.compute_match(
        user_codes,
        required
    )

    rec_skills = recommend.recommend_skills_from_gap(
        [
            r
            for r in required
            if r["code"] not in set(user_codes)
        ]
    )

    jobs = await run_in_threadpool(
        recommend.related_jobs,
        user_codes
    )

    return m.MatchResponse(
        match_pct=result["match_pct"],
        simple_match_pct=result["simple_match_pct"],
        weighted_match_pct=result["weighted_match_pct"],
        matching=[
            m.Skill(**s)
            for s in result["matching"]
        ],
        missing=[
            m.Skill(**s)
            for s in result["missing"]
        ],
        required=[
            m.ScoredSkill(
                code=r["code"],
                name=sk.name_of(r["code"]),
                score=r["score"]
            )
            for r in required
        ],
        recommended_skills=[
            m.RecommendedSkill(**s)
            for s in rec_skills
        ],
        related_jobs=[
            m.RelatedJob(**j)
            for j in jobs
        ]
    )


@router.post("/recommend", response_model=m.RecommendResponse)
async def recommend_endpoint(
    req: m.RecommendRequest
):

    user_codes = _validate_codes(
        req.user_skills
    )

    rec_skills = recommend.recommend_skills_by_cooccurrence(
        user_codes
    )

    jobs = await run_in_threadpool(
        recommend.related_jobs,
        user_codes
    )

    return m.RecommendResponse(
        recommended_skills=[
            m.RecommendedSkill(**s)
            for s in rec_skills
        ],
        related_jobs=[
            m.RelatedJob(**j)
            for j in jobs
        ]
    )