from __future__ import annotations

import logging
import time
from typing import Callable

from src.application.retrieval_candidates import CandidateSearchRequest, retrieve_candidates
from src.application.retrieval_profiles import get_retrieval_profile
from src.ports.repositories import SearchRepository

logger = logging.getLogger(__name__)


def default_search_repository() -> SearchRepository:
    from src.adapters.store.defaults import default_search_repository as factory

    return factory()


def _format_search_results(results: list[dict], search_meta: dict) -> list[dict]:
    return [
        {
            "id": result["id"],
            "score": result["score"],
            "reranked": result.get("reranked", False),
            "selective_rerank_reason": result.get("selective_rerank_reason"),
            "matched_routes": result.get("matched_routes", []),
            "matched_queries": result.get("matched_queries", []),
            "neighbor_score": result.get("neighbor_score"),
            "neighbor_distance": result.get("neighbor_distance"),
            "neighbor_source_chunk_id": result.get("neighbor_source_chunk_id"),
            "search_meta": search_meta,
            "payload": result["payload"],
        }
        for result in results
    ]


def search_chunks(
    query: str,
    *,
    paper_id: str | None = None,
    top_k: int = 10,
    retrieval_profile: str | None = None,
    rerank: bool = False,
    selective_rerank: bool = False,
    rerank_top_n: int | None = None,
    filter_kwargs: dict | None = None,
    lexical_backfill: bool = True,
    neighbor_backfill: bool = False,
    query_rewrite: bool = False,
    multi_query: bool = False,
    repository: SearchRepository | None = None,
    multi_query_runner: Callable | None = None,
) -> list[dict]:
    repo = repository or default_search_repository()
    filters = dict(filter_kwargs or {})
    if paper_id:
        filters["paper_id"] = paper_id

    profile_meta: dict = {}
    if retrieval_profile:
        profile = get_retrieval_profile(retrieval_profile)
        rerank = profile.rerank
        selective_rerank = profile.selective_rerank
        rerank_top_n = profile.rerank_top_n
        lexical_backfill = profile.lexical_backfill
        neighbor_backfill = profile.neighbor_backfill
        query_rewrite = profile.query_rewrite
        multi_query = profile.multi_query
        profile_meta = {"profile": profile.to_dict()}

    started = time.monotonic()
    candidate_result = retrieve_candidates(
        repo,
        CandidateSearchRequest(
            query=query,
            limit=top_k,
            filter_kwargs=filters or None,
            rerank=rerank,
            selective_rerank=selective_rerank,
            rerank_top_n=rerank_top_n,
            lexical_backfill=lexical_backfill,
            neighbor_backfill=neighbor_backfill,
            query_rewrite=query_rewrite,
            multi_query=multi_query,
        ),
        multi_query_runner=multi_query_runner,
    )
    results = candidate_result.results
    search_meta: dict = {**profile_meta, **candidate_result.search_meta}

    elapsed_ms = (time.monotonic() - started) * 1000
    logger.info(
        "Qdrant search | top_k=%s rerank=%s selective_rerank=%s rerank_top_n=%s lexical_backfill=%s neighbor_backfill=%s query_rewrite=%s multi_query=%s filters=%s hits=%s elapsed_ms=%.1f",
        top_k,
        rerank,
        selective_rerank,
        rerank_top_n,
        lexical_backfill,
        neighbor_backfill,
        query_rewrite,
        multi_query,
        sorted(filters.keys()),
        len(results),
        elapsed_ms,
    )
    return _format_search_results(results, search_meta)
