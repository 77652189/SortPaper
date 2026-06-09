from __future__ import annotations

from typing import Any


def create_literature_agent(**kwargs: Any):
    """Create the legacy literature agent behind an application port boundary."""
    from src.agent.literature_agent import LiteratureAgent

    return LiteratureAgent(**kwargs)
