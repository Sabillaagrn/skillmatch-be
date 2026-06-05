"""Backend settings. Reads from environment, with sensible local defaults."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from dotenv import load_dotenv


# Locate repo root dynamically (supports app/... or backend/app/...)
def _find_repo_root() -> Path:
    here = Path(__file__).resolve()

    for parent in here.parents:
        if (parent / "ml").is_dir():
            return parent

    # fallback to previous assumption
    return here.parents[3]


REPO_ROOT = _find_repo_root()

# backend/.env
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

load_dotenv(ENV_PATH)

# Load .env from repo root
load_dotenv(REPO_ROOT / ".env")

# Add repo root to python path
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class Settings:
    # CORS

    print("CORS ALLOWED_ORIGINS:", ALLOWED_ORIGINS)

    ALLOWED_ORIGINS: list[str] = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,https://skill-match-fe-two.vercel.app"
    ).split(",")
    if o.strip()
]

print("CORS ALLOWED_ORIGINS:", ALLOWED_ORIGINS)

    # Supabase
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")

    # Default classification threshold
    THRESHOLD: float = float(
        os.getenv("SKILLMATCH_THRESHOLD", "0.4")
    )

    # Recommendation limits
    MAX_RECOMMENDED_SKILLS: int = int(
        os.getenv("MAX_RECOMMENDED_SKILLS", "5")
    )

    MAX_RELATED_JOBS: int = int(
        os.getenv("MAX_RELATED_JOBS", "5")
    )

    # Model load setting
    EAGER_MODEL_LOAD: bool = (
        os.getenv("EAGER_MODEL_LOAD", "true")
        .lower() == "true"
    )


settings = Settings()