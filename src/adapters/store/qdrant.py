from __future__ import annotations

from typing import Any


class QdrantChunkRepository:
    """Chunk storage repository backed by the existing QdrantStore facade."""

    def __init__(self, store: Any | None = None) -> None:
        if store is None:
            from src.store.qdrant_store import QdrantStore

            store = QdrantStore()
        self.store = store

    def add(self, **kwargs: Any) -> None:
        self.store.add(**kwargs)

    def scroll_by_paper_id(self, paper_id: str, page_size: int = 500) -> list[Any]:
        return self.store.scroll_by_paper_id(paper_id, page_size=page_size)

    def update_paper_payload(
        self,
        paper_id: str,
        payload: dict[str, Any],
        point_ids: list[int | str] | None = None,
    ) -> int:
        return self.store.update_paper_payload(paper_id, payload, point_ids=point_ids)


class QdrantSearchRepository:
    """Search repository backed by the existing QdrantStore facade."""

    def __init__(self, store: Any | None = None) -> None:
        if store is None:
            from src.store.qdrant_store import QdrantStore

            store = QdrantStore()
        self.store = store

    def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.store.search(**kwargs)

    def search_evidence(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.store.search_evidence(**kwargs)

    def expand_neighbor_context(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        if not hasattr(self.store, "expand_neighbor_context"):
            return []
        return self.store.expand_neighbor_context(*args, **kwargs)

    def expand_paper_local_context(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        if not hasattr(self.store, "expand_paper_local_context"):
            return []
        return self.store.expand_paper_local_context(*args, **kwargs)


class QdrantQualityEnrichmentStore:
    """Quality-enrichment storage adapter backed by the existing QdrantStore facade."""

    def __init__(self, store: Any | None = None) -> None:
        if store is None:
            from src.store.qdrant_store import QdrantStore

            store = QdrantStore()
        self.store = store

    def scroll_by_paper_id(self, paper_id: str) -> list[Any]:
        return self.store.scroll_by_paper_id(paper_id)

    def update_paper_payload(
        self,
        paper_id: str,
        payload: dict[str, Any],
        *,
        point_ids: list[int | str] | None = None,
    ) -> int:
        return self.store.update_paper_payload(paper_id, payload, point_ids=point_ids)

    def set_chunk_payload(self, point_id: int | str, payload: dict[str, Any]) -> None:
        self.store.client.set_payload(
            collection_name=self.store.collection,
            payload=payload,
            points=[point_id],
        )

    def invalidate_lexical_cache(self) -> None:
        if hasattr(self.store, "_invalidate_lexical_cache"):
            self.store._invalidate_lexical_cache()


class QdrantVectorLibraryRepository:
    """Vector-library management adapter backed by the existing QdrantStore facade."""

    def __init__(self, store: Any | None = None) -> None:
        if store is None:
            from src.store.qdrant_store import QdrantStore

            store = QdrantStore()
        self.store = store

    @property
    def count(self) -> int:
        return self.store.count

    def paper_count(self, paper_id: str) -> int:
        from src.adapters.store import qdrant_library

        return qdrant_library.paper_count(
            self.store.client,
            collection=self.store.collection,
            paper_id=paper_id,
        )

    def list_papers(self) -> list[dict[str, Any]]:
        return self.store.list_papers()

    def delete_by_paper_id(self, paper_id: str) -> int:
        return self.store.delete_by_paper_id(paper_id)

    def clear(self) -> None:
        self.store.clear()

    def embed(self, text: str) -> list[float]:
        return self.store.embed(text)
