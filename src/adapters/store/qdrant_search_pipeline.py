from __future__ import annotations

from typing import Any, Callable


BackfillAnchorCount = Callable[[int], int]
SelectiveRerankReason = Callable[[str, list[dict[str, Any]]], str]
MergeLexicalCandidates = Callable[..., list[dict[str, Any]]]
MergeNeighborCandidates = Callable[..., list[dict[str, Any]]]
RerankCandidates = Callable[[str, list[dict[str, Any]], int], list[dict[str, Any]]]
ApplyQueryBoost = Callable[[str, list[dict[str, Any]]], list[dict[str, Any]]]
RestoreAnchorCandidates = Callable[[list[dict[str, Any]], list[dict[str, Any]], int], list[dict[str, Any]]]


def postprocess_results(
    *,
    query: str,
    results: list[dict[str, Any]],
    qdrant_filter: Any | None,
    limit: int,
    rerank: bool,
    selective_rerank: bool,
    rerank_top_n: int | None,
    lexical_backfill: bool,
    neighbor_backfill: bool,
    backfill_anchor_count: BackfillAnchorCount,
    selective_rerank_reason: SelectiveRerankReason,
    merge_lexical_candidates: MergeLexicalCandidates,
    merge_neighbor_candidates: MergeNeighborCandidates,
    rerank_candidates: RerankCandidates,
    apply_query_relevance_boost: ApplyQueryBoost,
    restore_anchor_candidates: RestoreAnchorCandidates,
) -> list[dict[str, Any]]:
    anchor_results = results[:backfill_anchor_count(limit)] if lexical_backfill else []

    selective_reason = ""
    should_rerank = rerank
    if selective_rerank and not should_rerank and results:
        selective_reason = selective_rerank_reason(query, results)
        should_rerank = bool(selective_reason)

    if should_rerank and results:
        results = merge_lexical_candidates(
            query=query,
            results=results,
            qdrant_filter=qdrant_filter,
            limit=limit * 4,
        )
        if neighbor_backfill:
            results = merge_neighbor_candidates(
                results=results,
                qdrant_filter=qdrant_filter,
                limit=max(limit * 4, 40),
            )
        results = rerank_candidates(query, results, rerank_top_n or limit)
        if selective_reason:
            for result in results:
                result["selective_rerank_reason"] = selective_reason
        results = apply_query_relevance_boost(query, results)
        if lexical_backfill:
            results = restore_anchor_candidates(results, anchor_results, limit)
    elif lexical_backfill or neighbor_backfill:
        if lexical_backfill:
            results = merge_lexical_candidates(
                query=query,
                results=results,
                qdrant_filter=qdrant_filter,
                limit=max(limit * 4, 40),
            )
        if neighbor_backfill:
            results = merge_neighbor_candidates(
                results=results,
                qdrant_filter=qdrant_filter,
                limit=max(limit * 4, 40),
            )
        results = apply_query_relevance_boost(query, results)
        if lexical_backfill:
            results = restore_anchor_candidates(results, anchor_results, limit)

    return results[:limit]
