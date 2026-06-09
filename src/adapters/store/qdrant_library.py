from __future__ import annotations

from typing import Any, Callable

from qdrant_client.http import models

from src.adapters.store import qdrant_payload
from src.domain import paper_catalog


InvalidateCache = Callable[[], None]


def count_points(client: Any, *, collection: str) -> int:
    try:
        return client.count(collection_name=collection).count
    except Exception:
        return 0


def paper_count(client: Any, *, collection: str, paper_id: str) -> int:
    return client.count(
        collection_name=collection,
        count_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="paper_id",
                    match=models.MatchValue(value=paper_id),
                )
            ],
        ),
    ).count


def clear_collection(
    client: Any,
    *,
    collection: str,
    ensure_collection: Callable[[], None],
    invalidate_cache: InvalidateCache | None = None,
) -> None:
    client.delete_collection(collection_name=collection)
    ensure_collection()
    if invalidate_cache is not None:
        invalidate_cache()


def list_papers(client: Any, *, collection: str, page_size: int = 500) -> list[dict[str, Any]]:
    return paper_catalog.list_papers_from_points(
        qdrant_payload.scroll_all_points(
            client=client,
            collection=collection,
            page_size=page_size,
        )
    )
