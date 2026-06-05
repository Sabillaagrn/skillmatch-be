# SkillMatch backend image (FastAPI + TensorFlow).
# Builds from the repo ROOT so the `ml/` package ships together with the backend
# (the API imports `ml` and loads the model from ml/model/).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    EAGER_MODEL_LOAD=true \
    SKILLMATCH_THRESHOLD=0.20

# libgomp1 is required by TensorFlow's CPU runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better layer caching.
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Application code + ML package (model artifacts live in ml/model/).
COPY backend ./backend
COPY ml ./ml

EXPOSE 8000
# Railway injects $PORT at runtime.
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
