"""Shared application configuration constants."""

from __future__ import annotations

import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, "").strip())
    except ValueError:
        return default
    return value if value > 0 else default


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, "").strip())
    except ValueError:
        return default
    return value if value > 0 else default


def _env_value(name: str, default: str = "") -> str:
    return (os.getenv(name) or os.getenv(f"\ufeff{name}") or default).strip()


RESULTS_DIR = Path("data/results")
SAMPLE_DIR = Path("data/sample_papers")
PIPELINE_TIMEOUT_SECONDS = _env_int("SORTPAPER_PIPELINE_TIMEOUT_SECONDS", 900)
BATCH_ESTIMATED_MINUTES_PER_PAPER = _env_int("SORTPAPER_BATCH_ESTIMATED_MINUTES_PER_PAPER", 5)
BATCH_MAX_PAPER_WORKERS = _env_int("SORTPAPER_BATCH_MAX_PAPER_WORKERS", 2)

LLM_MAX_CONCURRENT_REQUESTS = _env_int("SORTPAPER_LLM_MAX_CONCURRENT_REQUESTS", 4)
LLM_RATE_LIMIT_RETRY_MAX = _env_int("SORTPAPER_LLM_RATE_LIMIT_RETRY_MAX", 3)
LLM_RATE_LIMIT_BACKOFF_SECONDS = _env_float("SORTPAPER_LLM_RATE_LIMIT_BACKOFF_SECONDS", 2.0)
EMBEDDING_PROVIDER = _env_value("SORTPAPER_EMBEDDING_PROVIDER", "dashscope").lower()
OPENAI_EMBEDDING_MODEL = _env_value("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_EMBEDDING_BASE_URL = _env_value(
    "OPENAI_EMBEDDING_BASE_URL",
    "https://api.openai.com/v1",
)
DASHSCOPE_EMBEDDING_MODEL = _env_value("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3")

if EMBEDDING_PROVIDER == "dashscope":
    EMBEDDING_API_KEY_ENV = _env_value("SORTPAPER_EMBEDDING_API_KEY_ENV", "DASHSCOPE_API_KEY")
    EMBEDDING_MODEL = DASHSCOPE_EMBEDDING_MODEL
    EMBEDDING_DIM = _env_int("SORTPAPER_EMBEDDING_DIM", 1024)
else:
    EMBEDDING_PROVIDER = "openai"
    EMBEDDING_API_KEY_ENV = _env_value("SORTPAPER_EMBEDDING_API_KEY_ENV", "OPENAI_API_KEY")
    EMBEDDING_MODEL = OPENAI_EMBEDDING_MODEL
    EMBEDDING_DIM = _env_int("SORTPAPER_EMBEDDING_DIM", 3072)
