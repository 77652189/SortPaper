"""Shared runtime settings for application and infrastructure modules."""

from __future__ import annotations

import os
from pathlib import Path

from src.runtime_env import load_project_env


load_project_env()


def env_value(name: str, default: str = "") -> str:
    return (os.getenv(name) or os.getenv(f"\ufeff{name}") or default).strip()


def env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, "").strip())
    except ValueError:
        return default
    return value if value > 0 else default


def env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, "").strip())
    except ValueError:
        return default
    return value if value > 0 else default


def env_bool(name: str, default: bool) -> bool:
    value = env_value(name).lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


RESULTS_DIR = Path("data/results")
SAMPLE_DIR = Path("data/sample_papers")
MINERU_CACHE_DIR = Path("data/mineru_cache")
EXTERNAL_IMPORTS_DIR = Path("data/external_imports")
PIPELINE_TIMEOUT_SECONDS = env_int("SORTPAPER_PIPELINE_TIMEOUT_SECONDS", 900)
BATCH_ESTIMATED_MINUTES_PER_PAPER = env_int("SORTPAPER_BATCH_ESTIMATED_MINUTES_PER_PAPER", 5)
BATCH_MAX_PAPER_WORKERS = env_int("SORTPAPER_BATCH_MAX_PAPER_WORKERS", 2)
MINERU_FIGURE_VISION_MAX_WORKERS = env_int("MINERU_FIGURE_VISION_MAX_WORKERS", 2)
EXTERNAL_CONTACT_EMAIL = env_value("SORTPAPER_EXTERNAL_CONTACT_EMAIL", "")
EXTERNAL_HTTP_TIMEOUT_SECONDS = env_float("SORTPAPER_EXTERNAL_HTTP_TIMEOUT_SECONDS", 20.0)
EXTERNAL_MAX_PDF_MB = env_int("SORTPAPER_EXTERNAL_MAX_PDF_MB", 50)
EXTERNAL_MONITOR_AUTO_RUN = env_bool("SORTPAPER_EXTERNAL_MONITOR_AUTO_RUN", True)
DAILY_BRIEF_SMALL_MODEL = env_value("SORTPAPER_DAILY_BRIEF_SMALL_MODEL", "deepseek-chat")
DAILY_BRIEF_JUDGE_MODEL = env_value("SORTPAPER_DAILY_BRIEF_JUDGE_MODEL", "deepseek-v4-pro")
DAILY_BRIEF_MAX_WORKERS = env_int("SORTPAPER_DAILY_BRIEF_MAX_WORKERS", 4)

LLM_MAX_CONCURRENT_REQUESTS = env_int("SORTPAPER_LLM_MAX_CONCURRENT_REQUESTS", 4)
LLM_RATE_LIMIT_RETRY_MAX = env_int("SORTPAPER_LLM_RATE_LIMIT_RETRY_MAX", 3)
LLM_RATE_LIMIT_BACKOFF_SECONDS = env_float("SORTPAPER_LLM_RATE_LIMIT_BACKOFF_SECONDS", 2.0)

EMBEDDING_PROVIDER = env_value("SORTPAPER_EMBEDDING_PROVIDER", "dashscope").lower()
OPENAI_EMBEDDING_MODEL = env_value("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_EMBEDDING_BASE_URL = env_value(
    "OPENAI_EMBEDDING_BASE_URL",
    "https://api.openai.com/v1",
)
DASHSCOPE_EMBEDDING_MODEL = env_value("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3")
QWEN3_LOCAL_EMBEDDING_MODEL_PATH = env_value(
    "QWEN3_LOCAL_EMBEDDING_MODEL_PATH",
    "data/models/Qwen3-Embedding-0.6B",
)
QWEN3_LOCAL_EMBEDDING_MAX_LENGTH = env_int("QWEN3_LOCAL_EMBEDDING_MAX_LENGTH", 8192)
QWEN3_LOCAL_EMBEDDING_QUERY_INSTRUCT = env_value(
    "QWEN3_LOCAL_EMBEDDING_QUERY_INSTRUCT",
    "Given a scientific paper search query, retrieve relevant passages that answer the query.",
)

if EMBEDDING_PROVIDER == "dashscope":
    EMBEDDING_API_KEY_ENV = env_value("SORTPAPER_EMBEDDING_API_KEY_ENV", "DASHSCOPE_API_KEY")
    EMBEDDING_MODEL = DASHSCOPE_EMBEDDING_MODEL
    EMBEDDING_DIM = env_int("SORTPAPER_EMBEDDING_DIM", 1024)
elif EMBEDDING_PROVIDER in {"qwen3_local", "local_qwen3"}:
    EMBEDDING_PROVIDER = "qwen3_local"
    EMBEDDING_API_KEY_ENV = ""
    EMBEDDING_MODEL = QWEN3_LOCAL_EMBEDDING_MODEL_PATH
    EMBEDDING_DIM = env_int("SORTPAPER_EMBEDDING_DIM", 1024)
else:
    EMBEDDING_PROVIDER = "openai"
    EMBEDDING_API_KEY_ENV = env_value("SORTPAPER_EMBEDDING_API_KEY_ENV", "OPENAI_API_KEY")
    EMBEDDING_MODEL = OPENAI_EMBEDDING_MODEL
    EMBEDDING_DIM = env_int("SORTPAPER_EMBEDDING_DIM", 3072)

DEFAULT_QDRANT_COLLECTION = env_value(
    "SORTPAPER_QDRANT_COLLECTION",
    "papers_qwen3_local" if EMBEDDING_PROVIDER == "qwen3_local" else "papers",
)
RERANK_DEFAULT_ENABLED = env_bool(
    "SORTPAPER_RERANK_DEFAULT_ENABLED",
    EMBEDDING_PROVIDER != "qwen3_local",
)
SELECTIVE_RERANK_DEFAULT_ENABLED = env_bool(
    "SORTPAPER_SELECTIVE_RERANK_DEFAULT_ENABLED",
    EMBEDDING_PROVIDER == "qwen3_local",
)
