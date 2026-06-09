from __future__ import annotations

from typing import Any, Protocol


class QualityEvaluator(Protocol):
    def evaluate(self, chunks: list[Any], title: str = "") -> dict[str, Any]:
        ...


class QualityEnrichmentStore(Protocol):
    def scroll_by_paper_id(self, paper_id: str) -> list[Any]:
        ...

    def update_paper_payload(
        self,
        paper_id: str,
        payload: dict[str, Any],
        *,
        point_ids: list[int | str] | None = None,
    ) -> int:
        ...

    def set_chunk_payload(self, point_id: int | str, payload: dict[str, Any]) -> None:
        ...

    def invalidate_lexical_cache(self) -> None:
        ...
