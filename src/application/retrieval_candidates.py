from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from src.retrieval.query_rewrite import build_context_query


@dataclass(frozen=True)
class CandidateSearchRequest:
    query: str
    limit: int
    filter_kwargs: dict[str, Any] | None = None
    rerank: bool = False
    selective_rerank: bool = False
    rerank_top_n: int | None = None
    lexical_backfill: bool = True
    neighbor_backfill: bool = False
    query_rewrite: bool = False
    multi_query: bool = False


@dataclass(frozen=True)
class CandidateSearchResult:
    results: list[dict[str, Any]]
    search_meta: dict[str, Any] = field(default_factory=dict)
    context_query: str = ""
    content_type_preference: str | None = None


def retrieve_candidates(
    repository: Any,
    request: CandidateSearchRequest,
    *,
    multi_query_runner: Callable | None = None,
) -> CandidateSearchResult:
    if request.query_rewrite or request.multi_query:
        runner = multi_query_runner
        if runner is None:
            from src.retrieval.multi_query import multi_query_search

            runner = multi_query_search
        multi_result = runner(
            getattr(repository, "store", repository),
            request.query,
            limit=request.limit,
            filter_kwargs=request.filter_kwargs,
            rerank=request.rerank,
            selective_rerank=request.selective_rerank,
            rerank_top_n=request.rerank_top_n,
            lexical_backfill=request.lexical_backfill,
            neighbor_backfill=request.neighbor_backfill,
            use_query_rewrite=request.query_rewrite or request.multi_query,
            route_limit=4 if request.multi_query else 2,
            adaptive_route_limit=request.multi_query,
        )
        rewrite = getattr(multi_result, "rewrite", None)
        metadata = multi_result.metadata() if hasattr(multi_result, "metadata") else {}
        return CandidateSearchResult(
            results=list(multi_result.results),
            search_meta=metadata,
            context_query=build_context_query(rewrite, fallback_query=request.query),
            content_type_preference=_content_type_preference(rewrite),
        )

    results = repository.search(
        query=request.query,
        limit=request.limit,
        filter_kwargs=request.filter_kwargs,
        rerank=request.rerank,
        selective_rerank=request.selective_rerank,
        rerank_top_n=request.rerank_top_n,
        lexical_backfill=request.lexical_backfill,
        neighbor_backfill=request.neighbor_backfill,
    )
    return CandidateSearchResult(
        results=list(results),
        search_meta={},
        context_query=request.query,
        content_type_preference=None,
    )


def _content_type_preference(rewrite: Any | None) -> str | None:
    preference = str(getattr(rewrite, "evidence_preference", "") or "").strip().lower()
    return preference if preference in {"text", "table"} else None
