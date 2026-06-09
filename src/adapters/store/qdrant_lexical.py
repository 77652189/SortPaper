from __future__ import annotations

from typing import Any, Callable

from src.domain import evidence_policy


TextBuilder = Callable[[dict[str, Any]], str]
TextNormalizer = Callable[[str], str]


def scroll_document_entries(
    client: Any,
    *,
    collection: str,
    qdrant_filter: Any | None,
    document_text_builder: TextBuilder,
    text_normalizer: TextNormalizer,
    page_size: int = 500,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=qdrant_filter,
            limit=page_size,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for point in points:
            payload = point.payload or {}
            entries.append({
                "id": point.id,
                "payload": payload,
                "normalized_text": text_normalizer(
                    payload.get("search_text") or document_text_builder(payload),
                ),
            })
        if offset is None:
            break
    return entries


def candidate_entries(
    *,
    query_terms: dict[str, float],
    qdrant_filter: Any | None,
    filtered_entries: list[dict[str, Any]],
    lexical_index: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    if qdrant_filter is not None:
        return filtered_entries
    return evidence_policy.lexical_candidate_entries(
        query_terms=query_terms,
        lexical_index=lexical_index,
        limit=limit,
    )


def lexical_candidates(
    *,
    entries: list[dict[str, Any]],
    query_terms: dict[str, float],
    requires_strict_anchor: bool,
    score_fn: Any = evidence_policy.lexical_match_score_from_terms,
    limit: int,
) -> list[dict[str, Any]]:
    return evidence_policy.lexical_candidates_from_entries(
        entries,
        query_terms=query_terms,
        requires_strict_anchor=requires_strict_anchor,
        score_fn=score_fn,
        limit=limit,
    )


def document_index(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return evidence_policy.lexical_document_index(entries)
