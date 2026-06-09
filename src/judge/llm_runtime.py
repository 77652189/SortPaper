from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable
from typing import TypeVar

from src.application.settings import (
    LLM_MAX_CONCURRENT_REQUESTS,
    LLM_RATE_LIMIT_BACKOFF_SECONDS,
    LLM_RATE_LIMIT_RETRY_MAX,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

_LLM_SEMAPHORE = threading.BoundedSemaphore(LLM_MAX_CONCURRENT_REQUESTS)


def run_llm_call(operation: Callable[[], T], *, label: str = "llm") -> T:
    """Run an LLM/API call under global concurrency control with rate-limit backoff."""
    last_exc: Exception | None = None
    for attempt in range(LLM_RATE_LIMIT_RETRY_MAX + 1):
        with _LLM_SEMAPHORE:
            try:
                return operation()
            except Exception as exc:
                last_exc = exc
                if not _is_rate_limit_error(exc) or attempt >= LLM_RATE_LIMIT_RETRY_MAX:
                    raise

        delay = LLM_RATE_LIMIT_BACKOFF_SECONDS * (2 ** attempt)
        delay += random.uniform(0.0, min(1.0, LLM_RATE_LIMIT_BACKOFF_SECONDS))
        logger.warning(
            "%s rate-limited; retrying in %.1fs (%d/%d)",
            label,
            delay,
            attempt + 1,
            LLM_RATE_LIMIT_RETRY_MAX,
        )
        time.sleep(delay)

    assert last_exc is not None
    raise last_exc


def _is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    text = str(exc).lower()
    return "rate limit" in text or "rate_limit" in text or "429" in text
