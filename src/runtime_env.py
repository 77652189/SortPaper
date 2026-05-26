"""Runtime environment loading helpers."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


def load_project_env() -> bool:
    """Load SortPaper's .env regardless of the process working directory."""
    return load_dotenv(ENV_PATH, override=False)
