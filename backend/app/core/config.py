"""Backend settings. Reads from environment, with sensible local defaults."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))

# Repo root = .../skill-match
REPO_ROOT = Path(__file__).resolve().parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

class Settings:
    # CORS
    ALLOWED_ORIGINS: list[str] = [
        o.strip()
        for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
        if o.strip()
    ]

    # [BARU] Konfigurasi Supabase
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")

    # Default classification threshold
    THRESHOLD: float = float(os.getenv("SKILLMATCH_THRESHOLD", "0.4"))

    # How many recommendations to return
    MAX_RECOMMENDED_SKILLS: int = int(os.getenv("MAX_RECOMMENDED_SKILLS", "5"))
    MAX_RELATED_JOBS: int = int(os.getenv("MAX_RELATED_JOBS", "5"))

    # Model load setting
    EAGER_MODEL_LOAD: bool = os.getenv("EAGER_MODEL_LOAD", "true").lower() == "true"

settings = Settings()