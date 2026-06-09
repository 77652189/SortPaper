from __future__ import annotations

import logging
from typing import Any

from qdrant_client import models
from qdrant_client.models import FilterSelector


logger = logging.getLogger(__name__)


def paper_id_filter(paper_id: str) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="paper_id",
                match=models.MatchValue(value=paper_id),
            )
        ],
    )


def scroll_by_paper_id(
    *,
    client: Any,
    collection: str,
    paper_id: str,
    page_size: int = 500,
) -> list[Any]:
    points: list[Any] = []
    offset = None
    q_filter = paper_id_filter(paper_id)
    while True:
        batch, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=q_filter,
            limit=page_size,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        points.extend(batch)
        if next_offset is None:
            break
        offset = next_offset
    return points


def scroll_all_points(
    *,
    client: Any,
    collection: str,
    page_size: int = 500,
) -> list[Any]:
    points: list[Any] = []
    offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=collection,
            limit=page_size,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        points.extend(batch)
        if next_offset is None:
            break
        offset = next_offset
    return points


def update_paper_payload(
    *,
    client: Any,
    collection: str,
    paper_id: str,
    payload: dict[str, Any],
    point_ids: list[int | str] | None = None,
) -> int:
    normalized_payload = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in payload.items()
    }
    if point_ids:
        client.set_payload(
            collection_name=collection,
            payload=normalized_payload,
            points=point_ids,
        )
        return len(point_ids)

    q_filter = paper_id_filter(paper_id)
    client.set_payload(
        collection_name=collection,
        payload=normalized_payload,
        points=q_filter,
    )
    return client.count(
        collection_name=collection,
        count_filter=q_filter,
    ).count


def delete_by_paper_id(
    *,
    client: Any,
    collection: str,
    paper_id: str,
) -> int:
    q_filter = paper_id_filter(paper_id)
    count_result = client.count(
        collection_name=collection,
        count_filter=q_filter,
    )
    before = count_result.count
    if before == 0:
        return 0

    client.delete(
        collection_name=collection,
        points_selector=FilterSelector(filter=q_filter),
    )
    logger.info("Deleted %d points for paper_id='%s'", before, paper_id)
    return before
