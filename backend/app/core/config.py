"""Backend settings. Reads from environment, with sensible local defaults."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Repo root = .../skill-match  (backend/ and ml/ are siblings).
REPO_ROOT = Path(__file__).resolve().parents[3]

# Make the `ml` package importable so the backend reuses the inference pipeline
# instead of duplicating it.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class Settings:
    # CORS: comma-separated list of allowed origins (the Vercel URL in prod).
    ALLOWED_ORIGINS: list[str] = [
        o.strip()
        for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
        if o.strip()
    ]

    # Default classification threshold; can be overridden per request.
    THRESHOLD: float = float(os.getenv("SKILLMATCH_THRESHOLD", "0.4"))

    # How many recommendations to return.
    MAX_RECOMMENDED_SKILLS: int = int(os.getenv("MAX_RECOMMENDED_SKILLS", "5"))
    MAX_RELATED_JOBS: int = int(os.getenv("MAX_RELATED_JOBS", "5"))

    # Load the heavy TF model eagerly at startup (True) or lazily on first
    # request (False). Eager avoids a slow first request; lazy speeds boot.
    EAGER_MODEL_LOAD: bool = os.getenv("EAGER_MODEL_LOAD", "true").lower() == "true"


settings = Settings()
