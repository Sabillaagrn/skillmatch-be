"""SkillMatch inference pipeline.

Loads the trained BiLSTM + Attention multi-label classifier and its preprocessing
assets ONCE, then turns raw job text into a ranked list of required skill codes.
The model classifies into the 35 canonical skill categories in master_skill_list.csv.

This mirrors the training notebook exactly (SkillMatch_AI_Fix):

    text = f"{title} [SEP] {description}".strip()
         -> TextVectorization(saved vocab, len 512)   # standardizes internally
         -> BiLSTM + Attention -> 35 sigmoid probabilities
         -> threshold -> mlb.classes_ -> skill codes + scores

Two details that must match the notebook or predictions go wrong:
  * Input is "title [SEP] description" — the same string fed at training time.
  * The vectorizer is rebuilt with the FULL saved vocabulary (including the
    reserved '' / '[UNK]' tokens) via set_vocabulary(vocab) — no token dropping.

Everything is loaded lazily and cached at module level, so a server process pays
the load cost only once.
"""
from __future__ import annotations

import json
import os
import pickle
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

# Mock mode: skip TensorFlow entirely and return a deterministic keyword-based
# heuristic. Lets the API boot and be tested (frontend dev, CI) on machines
# without the heavy TF dependency. Set SKILLMATCH_MOCK=true to enable.
MOCK = os.getenv("SKILLMATCH_MOCK", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Paths & config  — artifacts live in ml/model/ (same layout as the notebook).
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "model"
MODEL_PATH = ARTIFACTS_DIR / "skillmatch_model.keras"
MLB_PATH = ARTIFACTS_DIR / "mlb.pkl"
VOCAB_PATH = ARTIFACTS_DIR / "vectorizer_vocab.pkl"
CONFIG_PATH = ARTIFACTS_DIR / "config.json"
SKILLS_CSV = ARTIFACTS_DIR / "master_skill_list.csv"

# Separator token joining title and description — identical to training.
SEP = " [SEP] "


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


CONFIG = _load_config()
SEQ_LEN = CONFIG["sequence_length"]      # 512
MAX_TOKENS = CONFIG["max_tokens"]        # 20000
# Serving threshold for extracting a job's required skills.
# config.json stores 0.4 (the F1-tuned threshold for the multi-label metric),
# but that is too strict for matching: many job descriptions then expose only a
# single skill above the bar, so the match score can only land on 0% or 100%.
# The training notebook's end-to-end demo uses 0.20, which surfaces the 3-4
# genuinely-required skills per role and produces graded scores (e.g. 33%).
# Override with SKILLMATCH_THRESHOLD if you need a different operating point.
THRESHOLD = float(os.getenv("SKILLMATCH_THRESHOLD", "0.20"))


def build_input_text(description: str, title: str = "") -> str:
    """Compose the exact string the model was trained on: 'title [SEP] description'.

    TextVectorization does its own lowercasing + punctuation stripping, so we pass
    the raw text through unchanged (no manual cleaning) to match the notebook.
    """
    return (f"{(title or '').strip()}{SEP}{(description or '').strip()}").strip()


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------
_model = None
_mlb = None
_vectorizer = None


def _build_vectorizer():
    """Rebuild the TextVectorization layer from the saved vocabulary.

    Replicates the notebook's reload path exactly:
        vec = TextVectorization(max_tokens, output_sequence_length)
        vec.set_vocabulary(vocab)          # vocab is the FULL get_vocabulary()
    The saved vocab already includes the reserved '' (pad) and '[UNK]' (oov)
    tokens at indices 0 and 1, which is what set_vocabulary round-trips on, so
    we pass it through untouched — dropping them would shift every token id.
    """
    import tensorflow as tf

    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    vocab = [str(t) for t in vocab]  # unwrap numpy str_ -> plain str

    vec = tf.keras.layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_sequence_length=SEQ_LEN,
        standardize="lower_and_strip_punctuation",
        split="whitespace",
    )
    vec.set_vocabulary(vocab)
    return vec


def _load_mlb():
    """Unpickle the MultiLabelBinarizer.

    Normally scikit-learn is installed and this is a plain pickle.load. As a
    fallback (e.g. minimal/mock environments without sklearn), register stub
    classes so the object still deserializes — we only ever read `.classes_`.
    """
    try:
        with open(MLB_PATH, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError:
        import sys
        import types

        for mod in ("sklearn", "sklearn.preprocessing", "sklearn.preprocessing._label"):
            m = types.ModuleType(mod)
            m.MultiLabelBinarizer = type("MultiLabelBinarizer", (), {})
            sys.modules.setdefault(mod, m)
        with open(MLB_PATH, "rb") as f:
            return pickle.load(f)


def load():
    """Load model + mlb + vectorizer once. Safe to call repeatedly.

    In MOCK mode we only need the label classes (from mlb), not TF.
    """
    global _model, _mlb, _vectorizer
    if _mlb is None:
        _mlb = _load_mlb()
    if MOCK:
        return None, _mlb, None
    if _model is None:
        import tensorflow as tf
        # Registers AttentionLayer so load_model can deserialize the custom layer.
        from .attention import AttentionLayer
        _model = tf.keras.models.load_model(
            MODEL_PATH, custom_objects={"AttentionLayer": AttentionLayer}
        )
    if _vectorizer is None:
        _vectorizer = _build_vectorizer()
    return _model, _mlb, _vectorizer


def is_ready() -> bool:
    if MOCK:
        return _mlb is not None
    return _model is not None and _mlb is not None and _vectorizer is not None


# ---------------------------------------------------------------------------
# MOCK mode — keyword heuristic (only used when SKILLMATCH_MOCK=true)
# ---------------------------------------------------------------------------
_WS = re.compile(r"\s+")
_NONWORD = re.compile(r"[^a-z0-9\s]")


def _clean_for_keywords(text: str) -> str:
    text = _NONWORD.sub(" ", (text or "").lower())
    return _WS.sub(" ", text).strip()


_MOCK_HINTS = {
    "ACCT": ["account", "bookkeep", "ledger", "payroll"],
    "ADM": ["administrative", "clerical", "scheduling", "office"],
    "ADVR": ["advertis", "campaign", "media buy"],
    "ANLS": ["analyst", "analytic", "sql", "dashboard", "report", "data", "bi"],
    "ART": ["art", "creative", "illustration"],
    "BD": ["business development", "partnership", "growth", "lead generation"],
    "CNSL": ["consult", "advisory"],
    "CUST": ["customer", "support", "service", "client success"],
    "DIST": ["logistics", "distribution", "warehouse", "shipping"],
    "DSGN": ["design", "ux", "ui", "figma", "graphic"],
    "EDU": ["teach", "education", "curriculum", "instructor"],
    "ENG": ["engineer", "software", "developer", "python", "java", "react", "backend"],
    "FIN": ["finance", "financial", "investment", "budget"],
    "GENB": ["general business", "operations"],
    "HCPR": ["nurse", "clinical", "patient", "healthcare", "medical"],
    "HR": ["human resources", "recruit", "talent", "hr"],
    "IT": ["it ", "information technology", "infrastructure", "network", "system admin", "cloud"],
    "LGL": ["legal", "law", "compliance", "attorney"],
    "MGMT": ["manager", "management", "lead team", "director"],
    "MNFC": ["manufactur", "assembly", "production line"],
    "MRKT": ["marketing", "seo", "brand", "content"],
    "OTHR": [],
    "PR": ["public relations", "communications", "press"],
    "PRCH": ["procure", "purchasing", "sourcing"],
    "PRDM": ["product manage", "roadmap", "product owner"],
    "PRJM": ["project manage", "scrum", "agile", "stakeholder", "milestone"],
    "PROD": ["production"],
    "QA": ["quality assurance", "qa", "testing", "test cases"],
    "RSCH": ["research", "experiment", "study"],
    "SALE": ["sales", "quota", "account executive", "revenue", "selling"],
    "SCI": ["scientist", "scientific", "biology", "chemistry"],
    "STRA": ["strategy", "strategic planning"],
    "SUPL": ["supply chain", "inventory"],
    "TRNG": ["training", "onboarding", "coaching"],
    "WRT": ["writing", "writer", "copywrit", "editorial"],
}


def _mock_scores(text: str, classes: list[str]) -> np.ndarray:
    """Deterministic pseudo-probabilities from keyword hits (MOCK mode)."""
    t = " " + _clean_for_keywords(text) + " "
    scores = np.full(len(classes), 0.05, dtype=np.float32)
    for i, c in enumerate(classes):
        hits = sum(1 for kw in _MOCK_HINTS.get(c, []) if kw.strip() in t)
        if hits:
            scores[i] = min(0.95, 0.45 + 0.18 * hits)
    return scores


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def _predict_scores(texts: Iterable[str]) -> np.ndarray:
    """Return (n_texts, n_labels) sigmoid probability matrix.

    `texts` are already-composed 'title [SEP] description' strings.
    """
    _, mlb, _ = load()
    classes = [str(c) for c in mlb.classes_]
    if MOCK:
        return np.vstack([_mock_scores(t, classes) for t in texts])
    import tensorflow as tf

    model, _, vec = load()
    seqs = vec(tf.constant(list(texts)))       # (n, 512) int — standardized internally
    probs = model.predict(seqs, verbose=0)     # (n, 35) float
    return np.asarray(probs)


def extract_skills(
    job_description: str,
    threshold: float | None = None,
    title: str = "",
) -> list[dict]:
    """Extract required skill codes from a job posting.

    Returns [{code, score}] sorted by score desc, keeping labels above the
    threshold. If nothing clears the bar, returns the single highest so the UI
    is never empty.
    """
    thr = THRESHOLD if threshold is None else threshold
    _, mlb, _ = load()
    text = build_input_text(job_description, title)
    scores = _predict_scores([text])[0]              # (35,)
    classes = list(mlb.classes_)
    ranked = sorted(
        ({"code": str(c), "score": float(s)} for c, s in zip(classes, scores)),
        key=lambda d: d["score"], reverse=True,
    )
    above = [r for r in ranked if r["score"] >= thr]
    return above or ranked[:1]


@lru_cache(maxsize=512)
def extract_skills_cached(
    job_description: str,
    threshold: float | None = None,
    title: str = "",
) -> tuple:
    """Hashable-friendly cached wrapper. Identical inputs skip inference entirely.

    Returns a tuple of (code, score) pairs so it can be cached; callers convert
    to dicts. Used by /extract-skill and /match.
    """
    return tuple(
        (d["code"], d["score"])
        for d in extract_skills(job_description, threshold, title)
    )


if __name__ == "__main__":
    load()
    _, mlb, _ = load()
    print("Loaded. n_labels =", len(mlb.classes_), "classes =", list(mlb.classes_))
    print("Extracted:", extract_skills(
        "We need a Data Analyst skilled in Python, SQL, and statistics.",
        title="Data Analyst",
    ))
