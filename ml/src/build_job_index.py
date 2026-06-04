"""Offline: build the job index used for 'related jobs' recommendations and a
skill co-occurrence matrix used for 'recommended skills'.

Why offline: we never need the model to know a *training* job's skills — the
ground-truth labels are already in `target_skills_list`. So we precompute, once:

  1. For every job: a 35-dim binary skill vector (over mlb.classes_ order).
  2. A 35x35 co-occurrence matrix (how often skills appear together).

The backend loads the resulting `job_index.npz` at startup and ranks related
jobs by cosine similarity at request time — fast, no model calls.

Run:  python -m ml.src.build_job_index
Output: ml/artifacts/job_index.npz  (vectors, titles, job_ids, classes, cooc)
"""
from __future__ import annotations

import ast
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACTS = Path(__file__).resolve().parents[1] / "model"
# Training data lives in the user's Downloads project; copied into ml/data in setup.
DATA_CANDIDATES = [
    ARTIFACTS.parent / "data" / "skillmatch_train_data.csv",
    Path.home() / "Downloads" / "skillmatch 3" / "data" / "skillmatch_train_data.csv",
    Path.home() / "Downloads" / "skillmatch" / "data" / "skillmatch_train_data.csv",
]


def _find_data() -> Path:
    for p in DATA_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "skillmatch_train_data.csv not found. Place it in ml/data/ "
        f"(looked in: {[str(p) for p in DATA_CANDIDATES]})"
    )


def _parse_labels(cell) -> list[str]:
    """target_skills_list is stored as a stringified python list."""
    if isinstance(cell, list):
        return [str(x) for x in cell]
    if not isinstance(cell, str) or not cell.strip():
        return []
    try:
        val = ast.literal_eval(cell)
        return [str(x) for x in val] if isinstance(val, (list, tuple)) else [str(val)]
    except (ValueError, SyntaxError):
        # fall back to comma split
        return [s.strip() for s in cell.strip("[]").replace("'", "").split(",") if s.strip()]


def main() -> None:
    with open(ARTIFACTS / "mlb.pkl", "rb") as f:
        mlb = pickle.load(f)
    classes = [str(c) for c in mlb.classes_]
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)

    df = pd.read_csv(_find_data())
    vectors, titles, job_ids = [], [], []
    cooc = np.zeros((n, n), dtype=np.float64)

    for _, row in df.iterrows():
        codes = [c for c in _parse_labels(row.get("target_skills_list")) if c in idx]
        if not codes:
            continue
        vec = np.zeros(n, dtype=np.float32)
        ids = [idx[c] for c in codes]
        vec[ids] = 1.0
        vectors.append(vec)
        titles.append(str(row.get("title", "")))
        job_ids.append(str(row.get("job_id", "")))
        for i in ids:
            for j in ids:
                cooc[i, j] += 1

    vectors = np.vstack(vectors) if vectors else np.zeros((0, n), dtype=np.float32)
    np.savez_compressed(
        ARTIFACTS / "job_index.npz",
        vectors=vectors,
        titles=np.array(titles, dtype=object),
        job_ids=np.array(job_ids, dtype=object),
        classes=np.array(classes, dtype=object),
        cooc=cooc,
    )
    print(f"Wrote job_index.npz: {len(vectors)} jobs, {n} skill classes.")


if __name__ == "__main__":
    main()
