# SkillMatch — Architecture (Step 1)

AI-powered job skill matching & recommendation system. This document defines the
monorepo architecture, data flow, API contracts, model-serving strategy, match
formula, and folder layout. **No feature code is written yet** — this is the
blueprint to approve before Steps 2–5.

---

## 1. What the model actually does (important)

Your trained artifacts in `Downloads/skillmatch/`:

| File | Purpose |
|------|---------|
| `model/skillmatch_model.keras` | BiLSTM multi-label classifier (~35 MB) |
| `model/mlb.pkl` | `MultiLabelBinarizer` — maps model outputs → skill codes |
| `model/vectorizer_vocab.pkl` | `TextVectorization` vocabulary (max_tokens 20000) |
| `model/config.json` | seq_length 512, embedding 128, lstm_units 128, **threshold 0.4** |
| `data/skillmatch_train_data.csv` | training rows (`description_cleaned` → `target_skills_list`) |
| `src/master_skill_list.csv` | 35 canonical skill codes ↔ names |

**Key fact that shapes the whole system:** the model classifies a job
description into ~35 **skill categories** (e.g. `ANLS` = Analysis & Analytics,
`BD` = Business Development, `CUST` = Customer Service), not granular tech
skills like "Python" or "React". So:

- "Extract required skills from a job description" = **run the BiLSTM**, take
  classes with probability ≥ threshold (0.4), decode via `mlb` to skill codes,
  then join to `master_skill_list.csv` for human-readable names.
- "The user's skills" must be expressed in / mapped to the **same 35 codes** so
  the two sides are comparable. The UI will let users pick from the canonical
  list (multi-select), which keeps matching clean and avoids free-text drift.

This is the single most important design decision: **both sides of the match
live in the same controlled vocabulary** (the 35 codes). Everything else follows.

---

## 2. Monorepo layout

```
skill-match/
├── ARCHITECTURE.md            ← this file
├── README.md                  (added in a later step)
├── frontend/                  React + Vite + Tailwind  → Vercel
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── .env.example           VITE_API_BASE_URL=...
│   └── src/
│       ├── main.jsx
│       ├── App.jsx            routes
│       ├── pages/
│       │   ├── Landing.jsx        hero + CTA
│       │   └── Analyze.jsx        the core flow (inputs → results)
│       ├── components/
│       │   ├── SkillSelect.jsx    multi-select over the 35 codes
│       │   ├── JobDescriptionInput.jsx
│       │   ├── ResultDashboard.jsx
│       │   ├── MatchGauge.jsx     match % ring
│       │   ├── SkillChips.jsx     matched / missing chips
│       │   ├── RecommendationList.jsx (skills + related jobs)
│       │   ├── Loader.jsx
│       │   ├── ErrorBanner.jsx
│       │   └── ui/                Button, Card, Badge (reusable primitives)
│       ├── hooks/
│       │   └── useAnalyze.js      calls backend, manages loading/error
│       ├── lib/
│       │   └── api.js             fetch wrapper, typed responses
│       └── assets/
│
├── backend/                   FastAPI  → Render (free tier)
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── .env.example
│   └── app/
│       ├── main.py            app factory, CORS, router mount, model warmup
│       ├── api/
│       │   └── routes.py      POST /extract-skill /match /recommend + /health
│       ├── schemas/
│       │   └── models.py      Pydantic request/response models
│       ├── services/
│       │   ├── inference.py   load model+mlb+vectorizer, predict (cached)
│       │   ├── matching.py    match % + gap calculation
│       │   └── recommend.py   missing-skill & related-job recommendations
│       └── core/
│           ├── config.py      settings (paths, threshold, CORS origins)
│           └── skills.py      loads master_skill_list.csv, code↔name maps
│
├── ml/                        model assets + offline scripts
│   ├── artifacts/             ← copy of .keras, mlb.pkl, vocab, config.json
│   ├── src/
│   │   ├── export_savedmodel.py   .keras → TF SavedModel (faster load/serve)
│   │   └── build_job_index.py     precompute job→skills index for recommendations
│   └── notebooks/             (your existing training notebooks, optional)
│
└── .github/workflows/         CI (lint/test) — optional, added in Step 5
```

The backend reads model artifacts from `ml/artifacts/` (single source of truth);
Step 2 copies your `Downloads/skillmatch/model/*` there so the repo is
self-contained.

---

## 3. Data flow

```
                ┌──────────────── Frontend (Vercel) ─────────────────┐
 user picks     │  SkillSelect (35 codes)   JobDescriptionInput      │
 skills + JD →  │            └──────────────┬──────────────┘          │
                └───────────────────────────│──────────────────────────┘
                                             │  POST /match  {user_skills, job_description}
                                             ▼
                ┌──────────────── Backend (FastAPI, Render) ─────────┐
                │  inference.py: vectorize JD → BiLSTM → probs        │
                │      → threshold 0.4 → mlb.inverse → required codes │
                │  matching.py: compare user_skills vs required codes │
                │      → match %, matched[], missing[]                │
                │  recommend.py: rank missing skills + related jobs   │
                └───────────────────────────│──────────────────────────┘
                                             ▼
                         { match_pct, matching, missing,
                           recommended_skills, related_jobs }
```

`/extract-skill` (JD → required skills only) and `/recommend` (skills/jobs from a
skill set) are the decomposed pieces; `/match` is the end-to-end convenience
endpoint the UI uses by default.

---

## 4. API contracts

**`POST /extract-skill`** — JD only.
```json
// req
{ "job_description": "We are hiring a data analyst..." }
// res
{ "required_skills": [ {"code":"ANLS","name":"Analysis & Analytics","score":0.91} ] }
```

**`POST /match`** — full analysis (primary endpoint).
```json
// req
{ "user_skills": ["ANLS","IT"], "job_description": "..." }
// res
{
  "match_pct": 66.7,
  "matching":  [ {"code":"ANLS","name":"Analysis & Analytics"} ],
  "missing":   [ {"code":"BD","name":"Business Development"} ],
  "required":  [ {"code":"ANLS","score":0.91}, {"code":"BD","score":0.55} ],
  "recommended_skills": [ {"code":"BD","name":"...","reason":"co-occurs with your skills"} ],
  "related_jobs": [ {"title":"Data Analyst","match_pct":80.0,"job_id":"..."} ]
}
```

**`POST /recommend`** — from a skill set, no JD.
```json
{ "user_skills": ["ANLS","IT"] }  →  { "recommended_skills":[...], "related_jobs":[...] }
```

**`GET /health`** — liveness + model-loaded flag (used by Render & frontend warmup).

All errors return `{ "detail": "..." }` with proper HTTP codes (422 validation,
503 model-not-ready, 500 inference error). Frontend `ErrorBanner` renders `detail`.

---

## 5. Match formula

Primary (your requested weighted scoring), with required skills weighted by the
model's confidence so a strongly-required skill counts more than a borderline one:

```
weighted_match = Σ(score_i · 1[user has skill_i])  /  Σ(score_i)
                 over all required skills i (score_i = model probability)
```

When you want the simple version (all required skills equal):

```
simple_match = |user_skills ∩ required_skills| / |required_skills|
```

The default response uses **weighted_match** (more faithful to how essential each
skill is) and also returns `simple_match` for transparency. Alternatives I'll
document in code comments:

- **Jaccard** `|∩| / |∪|` — penalizes having *extra* skills too; good for
  "how similar are these two skill sets" but unfair for over-qualified users.
- **F1 of precision/recall** — balances "covered the JD" vs "didn't pad".
- **Cosine similarity** over score vectors — needed anyway for related-job
  recommendations (§6), so we compute it regardless.

Recommendation: weighted match as the headline number; it directly answers
"do my skills cover what this job needs, weighted by importance?"

---

## 6. Recommendation system

**Missing skills** = `required − user_skills`, ranked by model score (most
important gaps first). Optional boost: skills that frequently **co-occur** with
the user's existing skills in the training data (a precomputed co-occurrence
matrix from `skillmatch_train_data.csv`), so suggestions feel coherent.

**Related jobs** = nearest jobs by **cosine similarity** between the user's skill
vector (35-dim binary) and each job's skill vector. We precompute every training
job's skill vector once (`ml/src/build_job_index.py` → a small JSON/NPZ the
backend loads at startup), then rank at request time — fast, no extra model
calls. Returns title + that job's match_pct against the user.

---

## 7. Model serving & low-latency strategy

Decision (from your choice): **in-process SavedModel, cached in memory.**

1. **Convert once**: `.keras` → **TF SavedModel** (`ml/src/export_savedmodel.py`).
   SavedModel loads faster and is the stable serving format; we ship it in the image.
2. **Load once at startup** (FastAPI lifespan/`startup` event), keep the model,
   `mlb`, and vectorizer as module-level singletons. No per-request reload.
3. **Async endpoints**: routes are `async def`; the blocking TF `predict` call
   runs in a threadpool (`run_in_executor` / `anyio.to_thread`) so the event loop
   stays responsive under concurrent requests.
4. **Caching**: LRU cache keyed by a hash of the JD text → identical JDs skip
   inference entirely. (Cheap win; JDs are often re-submitted.)
5. **Batching-friendly**: vectorize + predict accept lists, so future batch
   endpoints are trivial.
6. **TF Lite**: deliberately *not* used for v1 — the model is small enough that
   in-process SavedModel hits real-time latency on a free CPU instance, and TFLite
   adds conversion/ops-compatibility risk for BiLSTM. Documented as a future
   optimization if we move to edge/serverless.

Cold-start note: free hosts sleep idle instances. The frontend pings `/health`
on app load to warm the model before the user submits.

---

## 8. Deployment (free tier)

- **Frontend → Vercel**: `frontend/` as project root, build `npm run build`,
  output `dist/`, env `VITE_API_BASE_URL`. SPA rewrite to `/index.html`.
- **Backend → Render (free Web Service)**: Docker deploy from `backend/`,
  `/health` as health check, `ALLOWED_ORIGINS` set to the Vercel URL. (Railway
  works the same way — both free; I'll include notes for either in Step 5.)

---

## 9. Build order (incremental, each awaits your OK)

1. **Architecture** ← *you are here*
2. **ML layer** — copy artifacts to `ml/artifacts/`, write `inference.py`,
   `export_savedmodel.py`, `build_job_index.py`; verify a real prediction.
3. **Backend** — FastAPI endpoints, schemas, matching + recommend services, caching/async.
4. **Frontend** — React/Vite/Tailwind UI from your Figma design (needs Figma access resolved).
5. **Deployment & optimization** — Dockerfile, Vercel + Render guides, CI, tuning.

---

## Open items before Step 4

- **Figma access**: the linked SkillMatch file couldn't be read by the Figma
  connector. To build the UI to your design, please open the file in the Figma
  desktop app (with the MCP/Dev Mode server enabled) or confirm this Google
  account has access. Until then Steps 2–3 are unaffected.
```
