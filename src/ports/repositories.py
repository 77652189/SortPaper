from __future__ import annotations

from typing import Any, Protocol


class ChunkRepository(Protocol):
    def add(self, **kwargs: Any) -> None:
        ...

    def scroll_by_paper_id(self, paper_id: str, page_size: int = 500) -> list[Any]:
        ...

    def update_paper_payload(
        self,
        paper_id: str,
        payload: dict[str, Any],
        point_ids: list[int | str] | None = None,
    ) -> int:
        ...


class ParsedChunkStore(Protocol):
    def store(
        self,
        result: dict[str, Any],
        *,
        embedding_api_key_env: str,
        embed_content_for_chunk,
    ) -> dict[str, Any]:
        ...


class SearchRepository(Protocol):
    def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        ...

    def search_evidence(self, **kwargs: Any) -> list[dict[str, Any]]:
        ...


class VectorLibraryRepository(Protocol):
    @property
    def count(self) -> int:
        ...

    def paper_count(self, paper_id: str) -> int:
        ...

    def list_papers(self) -> list[dict[str, Any]]:
        ...

    def delete_by_paper_id(self, paper_id: str) -> int:
        ...

    def clear(self) -> None:
        ...

    def embed(self, text: str) -> list[float]:
        ...
