from __future__ import annotations

from typing import Protocol


class LiteratureAgentRunner(Protocol):
    def query(self, question: str, *, max_rounds: int) -> dict:
        ...
