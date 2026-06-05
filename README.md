  # SkillMatch — AI-Powered Job Skill Matching & Recommendation

  Monorepo: `frontend/` (React + Vite + Tailwind → Vercel), `backend/` (FastAPI →
  Render free tier), `ml/` (BiLSTM model artifacts + inference). See
  [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full design.

  ## Status
  - [x] Architecture & folder structure
  - [x] ML inference layer (custom AttentionLayer, vectorizer, job index)
  - [x] FastAPI backend (`/extract-skill`, `/match`, `/recommend`, `/health`) — tested
  - [ ] React frontend (pending Figma access)
  - [ ] Deployment & optimization

  ---

  ## Run the backend locally

  ### Option A — full model (real inference)
  Requires TensorFlow (the trained `.keras` model is loaded in-process).

  ```bash
  cd backend
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  uvicorn app.main:app --reload --port 8000
  ```

  The model artifacts are read from `ml/artifacts/`. First boot loads the model
  (a few seconds) because `EAGER_MODEL_LOAD=true`.

  ### Option B — mock mode (no TensorFlow)
  For frontend development and CI. Returns deterministic keyword-based predictions
  so the API runs without the heavy ML stack.

  ```bash
  cd backend
  pip install fastapi "uvicorn[standard]" pydantic numpy
  SKILLMATCH_MOCK=true uvicorn app.main:app --reload --port 8000
  ```

  ### Try it
  Interactive docs at http://localhost:8000/docs , or:

  ```bash
  curl localhost:8000/health

  curl -X POST localhost:8000/match -H 'Content-Type: application/json' -d '{
    "user_skills": ["ANLS","IT","SALE"],
    "job_description": "Data analyst: SQL, dashboards, reporting, business development."
  }'
  ```

  Skill codes come from `ml/artifacts/master_skill_list.csv` (35 categories, e.g.
  `ANLS` Analysis & Analytics, `ENG` Engineering, `SALE` Sales).

  ---

  ## Environment variables (backend)
  See `backend/.env.example`:

  | Var | Default | Purpose |
  |-----|---------|---------|
  | `ALLOWED_ORIGINS` | `http://localhost:5173` | CORS allow-list (the frontend URL) |
  | `SKILLMATCH_THRESHOLD` | `0.4` | Classification cutoff |
  | `SKILLMATCH_MOCK` | `false` | Skip TF, use heuristic predictions |
  | `EAGER_MODEL_LOAD` | `true` | Load model at startup vs first request |
  | `MAX_RECOMMENDED_SKILLS` / `MAX_RELATED_JOBS` | `5` | Recommendation counts |

  ---

  ## Rebuild ML artifacts (optional)
  ```bash
  cd <repo root>
  python -m ml.src.build_job_index      # regenerates job_index.npz
  python -m ml.src.export_savedmodel    # .keras -> SavedModel (serving)
  python -m ml.src.inference            # smoke test a prediction
  ```
