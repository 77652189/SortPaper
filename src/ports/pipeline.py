from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol


class ParserRunner(Protocol):
    def __call__(self, pdf_bytes: bytes, paper_id: str, filename: str = "") -> dict[str, Any]:
        ...


class StoreRunner(Protocol):
    def __call__(self, result: dict[str, Any]) -> dict[str, Any]:
        ...


PreviewRunner = Callable[..., dict[str, Any]]
