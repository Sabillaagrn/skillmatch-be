"""FastAPI application entrypoint.

- Loads the TF model once at startup (configurable) so the first real request
  isn't slow.
- CORS restricted to the configured frontend origin(s).
- Mounts the REST router.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.EAGER_MODEL_LOAD:
        # Import here so module import stays cheap; load weights into memory once.
        from ml.src import inference
        inference.load()
    yield


app = FastAPI(
    title="SkillMatch API",
    version="1.0.0",
    description="AI-powered job skill matching & recommendation.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {"service": "SkillMatch API", "docs": "/docs", "health": "/health"}
