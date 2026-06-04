# SkillMatch — Deployment Guide (Vercel + Railway)

This project is a monorepo with two independently deployed parts:

| Part | Stack | Host | Why |
|------|-------|------|-----|
| `frontend/` | React + Vite + Tailwind | **Vercel** | Best-in-class static/SPA hosting, instant CDN, free tier |
| `backend/` + `ml/` | FastAPI + TensorFlow (BiLSTM + Attention) | **Railway** | Runs the Python TF model natively; Vercel can't host a 35 MB Keras model + TensorFlow within its serverless size limits |

The React app talks to the Railway API over HTTPS via `VITE_API_BASE_URL`. The model stays in Python — no conversion needed.

---

## 0. Prerequisites

1. Push this repository to GitHub (the model files in `ml/model/` must be committed — see note below).
2. Free accounts on [Railway](https://railway.app) and [Vercel](https://vercel.com).

> **Important — commit the model.** `ml/model/` (~35 MB: `skillmatch_model.keras`, `mlb.pkl`, `vectorizer_vocab.pkl`, `config.json`, `master_skill_list.csv`, `job_index.npz`) is **not** git-ignored on purpose. Railway serves these files directly. Verify they're tracked:
>
> ```bash
> git add ml/model && git status   # should list the 6 artifact files
> ```

---

## 1. Backend → Railway

1. On Railway: **New Project → Deploy from GitHub repo** and pick this repo.
2. Leave **Root Directory** as the repository root (`/`). The backend imports the `ml` package from the root, so it must build from there. `nixpacks.toml` and `railway.json` are already configured to:
   - install `backend/requirements.txt`
   - start `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
   - health-check `GET /health`
3. Under **Variables**, add:

   | Variable | Value | Notes |
   |----------|-------|-------|
   | `ALLOWED_ORIGINS` | `http://localhost:5173` | Update to your Vercel URL in step 3 |
   | `EAGER_MODEL_LOAD` | `true` | Load the model at boot so the first request isn't slow |
   | `SKILLMATCH_THRESHOLD` | `0.4` | Optional — matches `config.json` |

4. Deploy. When it's live, open the generated domain and check:
   - `https://<your-app>.up.railway.app/health` → `{"status":"ok","model_loaded":true}`
   - `https://<your-app>.up.railway.app/docs` → interactive API docs

> **Memory.** TensorFlow + the model need roughly **1–2 GB RAM**. If the deploy crashes on boot with an OOM/killed signal, raise the service memory in Railway settings (or set `EAGER_MODEL_LOAD=false` so memory is allocated lazily on first request).

> **First boot is slow** (installing TensorFlow + loading the model). The 300 s health-check timeout in `railway.json` covers this.

---

## 2. Frontend → Vercel

1. On Vercel: **Add New → Project**, import this repo.
2. Set **Root Directory** to `frontend`. Vercel auto-detects Vite (`vercel.json` is already included with SPA rewrites so client-side routes like `/analysis` don't 404).
3. Add an **Environment Variable**:

   | Variable | Value |
   |----------|-------|
   | `VITE_API_BASE_URL` | `https://<your-app>.up.railway.app` |

   (No trailing slash.)
4. Deploy. You'll get a URL like `https://skill-match.vercel.app`.

---

## 3. Connect them (CORS)

Back on Railway, set `ALLOWED_ORIGINS` to your Vercel URL and redeploy:

```
ALLOWED_ORIGINS=https://skill-match.vercel.app
```

You can list several origins comma-separated (e.g. include a preview domain):

```
ALLOWED_ORIGINS=https://skill-match.vercel.app,http://localhost:5173
```

Open the Vercel URL, go to **Analisis Skill**, pick skills + a role, and run an analysis. A request to `/match` should return a real match score.

---

## Local development

**Frontend**
```bash
cd frontend
npm install
echo "VITE_API_BASE_URL=http://localhost:8000" > .env.local
npm run dev            # http://localhost:5173
```

> **Use Python 3.11 or 3.12** (TensorFlow has no wheels for 3.13+ yet). Check with `python3 --version`.

**Backend (real model)**
```bash
python3.11 -m venv backend/venv
source backend/venv/bin/activate
pip install -r backend/requirements.txt    # picks `tensorflow` on macOS, `tensorflow-cpu` on Linux
./run-backend.sh                            # always runs uvicorn from the repo root
```
The helper script runs `uvicorn backend.app.main:app` from the repo root — important, because the backend imports the `ml` package from there. Running it from inside `backend/` will fail with `ModuleNotFoundError: No module named 'ml'`.

**Backend (mock mode — no TensorFlow needed)**
For UI work or CI where you don't want the heavy TF install:
```bash
pip install -r backend/requirements-mock.txt
./run-backend.sh mock
```

> **macOS note:** install `tensorflow` (not `tensorflow-cpu` — that package has **no macOS wheels**). The `requirements.txt` markers already handle this. The model also needs `keras>=3.13` (it was exported with Keras 3.13.2); this is pinned in `requirements.txt`.

---

## Retraining / updating the model

The notebook (`SkillMatch_AI_Fix`) saves six artifacts into its `model/` folder. To ship a new model:

1. Copy the new `skillmatch_model.keras`, `mlb.pkl`, `vectorizer_vocab.pkl`, `config.json` into `ml/model/`, and the updated `master_skill_list.csv`.
2. Rebuild the job index from the training CSV:
   ```bash
   python -m ml.src.build_job_index    # writes ml/model/job_index.npz
   ```
3. Commit `ml/model/` and push — Railway redeploys automatically.

The inference code (`ml/src/inference.py`) mirrors the notebook exactly: it composes input as `"{title} [SEP] {description}"` and rebuilds `TextVectorization` from the full saved vocabulary. If you change preprocessing in the notebook, mirror it there.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Railway build OK, crashes on boot | Out of memory — raise service memory to ≥2 GB, or set `EAGER_MODEL_LOAD=false`. |
| `Could not load model` / Keras deserialization error | The training TensorFlow version differs from `tensorflow-cpu==2.17.0`. Pin `backend/requirements.txt` to the version printed by the notebook (`tf.__version__`). |
| Browser console: CORS error | `ALLOWED_ORIGINS` on Railway doesn't exactly match the Vercel origin (scheme + host, no trailing slash). |
| Frontend routes 404 on refresh | Ensure `frontend/vercel.json` (SPA rewrites) is deployed and Root Directory = `frontend`. |
| `/match` returns 422 | Frontend sent skill codes not in `master_skill_list.csv`. The UI uses the canonical 35 codes, so this only happens with manual API calls. |
