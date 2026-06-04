"""Canonical skill vocabulary + the precomputed job index.

Single source of truth for:
  - code <-> human-readable name  (master_skill_list.csv)
  - the 35-dim skill ordering used by the model and the job index
  - the job vectors / co-occurrence matrix for recommendations

Loaded once at import time (small files; the big TF model lives in ml.inference).
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import REPO_ROOT

# Artifacts live in ml/model/ (same layout as the training notebook).
ARTIFACTS = REPO_ROOT / "ml" / "model"
SKILLS_CSV = ARTIFACTS / "master_skill_list.csv"
JOB_INDEX = ARTIFACTS / "job_index.npz"


# --- code <-> name -----------------------------------------------------------
def _load_skill_names() -> dict[str, str]:
    names: dict[str, str] = {}
    with open(SKILLS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            names[row["skill_code"].strip()] = row["skill_name"].strip()
    return names


SKILL_NAMES: dict[str, str] = _load_skill_names()
ALL_CODES: list[str] = sorted(SKILL_NAMES.keys())


def name_of(code: str) -> str:
    return SKILL_NAMES.get(code, code)


def decorate(codes) -> list[dict]:
    """[code, ...] -> [{'code','name'}, ...] preserving order."""
    return [{"code": c, "name": name_of(c)} for c in codes]


# --- job index (lazy: only needed for recommendations) -----------------------
class JobIndex:
    def __init__(self, path: Path):
        data = np.load(path, allow_pickle=True)
        self.vectors: np.ndarray = data["vectors"].astype(np.float32)  # (J, 35)
        self.titles = data["titles"]
        self.job_ids = data["job_ids"]
        self.classes = [str(c) for c in data["classes"]]               # model order
        self.cooc: np.ndarray = data["cooc"].astype(np.float64)        # (35, 35)
        self.index = {c: i for i, c in enumerate(self.classes)}
        # Precompute L2 norms for cosine similarity.
        self._norms = np.linalg.norm(self.vectors, axis=1)
        self._norms[self._norms == 0] = 1.0

    def vectorize(self, codes) -> np.ndarray:
        v = np.zeros(len(self.classes), dtype=np.float32)
        for c in codes:
            i = self.index.get(c)
            if i is not None:
                v[i] = 1.0
        return v


@lru_cache(maxsize=1)
def get_job_index() -> JobIndex:
    return JobIndex(JOB_INDEX)
