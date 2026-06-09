"""Multi-query retrieval orchestration."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from src.retrieval.query_rewrite import QueryRewrite, rewrite_query

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchRoute:
    query: str
    source: str
    weight: float = 1.0


@dataclass(frozen=True)
class MultiQuerySearchResult:
    results: list[dict[str, Any]]
    rewrite: QueryRewrite | None = None
    routes: list[SearchRoute] = field(default_factory=list)
    elapsed_ms: float = 0.0
    route_policy: dict[str, Any] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        return {
            "rewrite": self.rewrite.to_dict() if self.rewrite else None,
            "routes": [asdict(route) for route in self.routes],
            "route_policy": self.route_policy,
            "elapsed_ms": self.elapsed_ms,
        }


def build_search_routes(
    query: str,
    *,
    rewritten_query: QueryRewrite | None = None,
    use_query_rewrite: bool = False,
    query_rewrite_model: str | None = None,
    max_routes: int = 4,
) -> tuple[list[SearchRoute], QueryRewrite | None]:
    """Build stable retrieval routes from the original query and optional rewrite."""
    raw_query = str(query or "").strip()
    rewrite = rewritten_query
    if use_query_rewrite and rewrite is None and raw_query:
        rewrite = rewrite_query(raw_query, model=query_rewrite_model, use_llm=True)

    candidates: list[SearchRoute] = [SearchRoute(raw_query, "raw", 1.0)]
    if rewrite:
        if rewrite.normalized_query:
            candidates.append(SearchRoute(rewrite.normalized_query, "normalized", 0.95))
        candidates.extend(_structured_rewrite_routes(rewrite))
        for variant in rewrite.variants:
            candidates.append(SearchRoute(variant, "variant", 0.45))

    routes: list[SearchRoute] = []
    seen: set[str] = set()
    for route in candidates:
        normalized = " ".join(route.query.split()).lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        routes.append(route)
        if len(routes) >= max_routes:
            break
    return routes, rewrite


def _structured_rewrite_routes(rewrite: QueryRewrite) -> list[SearchRoute]:
    """Build RAGFlow-style keyword routes from structured rewrite fields."""
    routes: list[SearchRoute] = []
    entity_query = _join_route_terms(
        rewrite.products[:3]
        + rewrite.organisms[:2]
        + rewrite.genes[:4]
        + rewrite.enzymes[:3]
    )
    if entity_query:
        routes.append(SearchRoute(entity_query, "entity", 0.7))

    table_query = _table_evidence_query(rewrite)
    if table_query:
        routes.append(SearchRoute(table_query, "table_evidence", 0.65))

    alias_query = _join_route_terms(rewrite.aliases[:6])
    if alias_query:
        routes.append(SearchRoute(alias_query, "alias", 0.55))
    return routes


def _table_evidence_query(rewrite: QueryRewrite) -> str:
    if not rewrite.metrics and rewrite.evidence_preference != "table":
        return ""
    terms = (
        rewrite.metrics[:4]
        + rewrite.products[:3]
        + rewrite.organisms[:1]
        + rewrite.genes[:2]
    )
    query = _join_route_terms(terms)
    if not query:
        return ""
    if rewrite.evidence_preference == "table":
        query = f"{query} table"
    return query


def _join_route_terms(values: list[str], *, max_chars: int = 240) -> str:
    selected: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value or "").split()).strip()
        key = text.lower()
        if not text or key in seen:
            continue
        candidate = " ".join(selected + [text]).strip()
        if len(candidate) > max_chars:
            break
        seen.add(key)
        selected.append(text)
    return " ".join(selected)


def multi_query_search(
    store: Any,
    query: str,
    *,
    limit: int,
    filter_kwargs: dict[str, Any] | None = None,
    rerank: bool = False,
    selective_rerank: bool = False,
    rerank_top_n: int | None = None,
    lexical_backfill: bool = False,
    neighbor_backfill: bool = False,
    use_query_rewrite: bool = False,
    query_rewrite_model: str | None = None,
    rewritten_query: QueryRewrite | None = None,
    route_limit: int = 4,
    adaptive_route_limit: bool = True,
    route_fetch_multiplier: int = 3,
) -> MultiQuerySearchResult:
    """Run several query routes and merge results with weighted reciprocal rank."""
    started = time.perf_counter()
    rewrite = rewritten_query
    raw_query = str(query or "").strip()
    if use_query_rewrite and rewrite is None and raw_query:
        rewrite = rewrite_query(raw_query, model=query_rewrite_model, use_llm=True)
    requested_route_limit = max(1, route_limit)
    effective_route_limit, route_reason = _effective_route_limit(
        query,
        rewrite,
        requested_limit=requested_route_limit,
        adaptive=adaptive_route_limit,
    )
    routes, rewrite = build_search_routes(
        query,
        rewritten_query=rewrite,
        use_query_rewrite=False,
        query_rewrite_model=query_rewrite_model,
        max_routes=effective_route_limit,
    )
    if not routes:
        return MultiQuerySearchResult(
            results=[],
            rewrite=rewrite,
            routes=[],
            elapsed_ms=0.0,
            route_policy={
                "adaptive": adaptive_route_limit,
                "requested_limit": requested_route_limit,
                "effective_limit": effective_route_limit,
                "reason": route_reason,
            },
        )

    merged: dict[str, dict[str, Any]] = {}
    raw_order: list[str] = []
    fetch_limit = max(limit, limit * max(1, route_fetch_multiplier))
    for route_index, route in enumerate(routes):
        current_limit = limit if route_index == 0 else fetch_limit
        route_results = store.search(
            route.query,
            limit=current_limit,
            filter_kwargs=filter_kwargs,
            rerank=rerank,
            selective_rerank=selective_rerank,
            rerank_top_n=rerank_top_n,
            lexical_backfill=lexical_backfill,
            neighbor_backfill=neighbor_backfill,
        )
        for rank, item in enumerate(route_results, start=1):
            payload = item.get("payload") or {}
            key = _result_key(item, payload)
            if route_index == 0:
                raw_order.append(key)
            fused = route.weight / (60.0 + rank)
            if route_index == 0:
                fused *= 1.05
            existing = merged.get(key)
            if existing is None:
                copied = dict(item)
                copied["score"] = fused
                copied["raw_score"] = item.get("score")
                if route_index == 0:
                    copied["raw_rank"] = rank
                copied["matched_routes"] = [route.source]
                copied["matched_queries"] = [route.query]
                merged[key] = copied
            else:
                existing["score"] = float(existing.get("score") or 0.0) + fused
                existing["raw_score"] = max(
                    float(existing.get("raw_score") or 0.0),
                    float(item.get("score") or 0.0),
                )
                existing.setdefault("matched_routes", []).append(route.source)
                existing.setdefault("matched_queries", []).append(route.query)

    protected_count = _anchor_protected_count(limit)
    protected_keys = set(raw_order[:protected_count])
    protected_results = [merged[key] for key in raw_order[:protected_count] if key in merged]
    anchor_paper_ids = _paper_ids(protected_results)
    raw_candidates_are_sparse = len(raw_order) < protected_count
    remaining_results = [
        item for key, item in merged.items()
        if key not in protected_keys
        and _admit_tail_candidate(
            item,
            anchor_paper_ids=anchor_paper_ids,
            raw_candidates_are_sparse=raw_candidates_are_sparse,
        )
    ]
    ranked_remaining = sorted(
        remaining_results,
        key=lambda item: (
            _tail_priority(item),
            float(item.get("score") or 0.0),
            float(item.get("raw_score") or 0.0),
        ),
        reverse=True,
    )
    results = (protected_results + ranked_remaining)[:limit]
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "Multi-query search | routes=%s route_policy=%s hits=%s elapsed_ms=%.1f",
        [route.source for route in routes],
        route_reason,
        len(results),
        elapsed_ms,
    )
    return MultiQuerySearchResult(
        results=results,
        rewrite=rewrite,
        routes=routes,
        elapsed_ms=elapsed_ms,
        route_policy={
            "adaptive": adaptive_route_limit,
            "requested_limit": requested_route_limit,
            "effective_limit": effective_route_limit,
            "reason": route_reason,
        },
    )


def _effective_route_limit(
    query: str,
    rewrite: QueryRewrite | None,
    *,
    requested_limit: int,
    adaptive: bool,
) -> tuple[int, str]:
    if not adaptive:
        return requested_limit, "fixed"
    if requested_limit <= 2:
        return requested_limit, "already_compact"
    if _should_expand_routes(query, rewrite):
        return requested_limit, "evidence_or_structured_query"
    return min(requested_limit, 2), "compact_generic_query"


def _should_expand_routes(query: str, rewrite: QueryRewrite | None) -> bool:
    lowered = str(query or "").lower()
    evidence_terms = (
        "which",
        "evidence",
        "passage",
        "report",
        "reports",
        "table",
        "figure",
        "g/l",
        "mg/l",
        "yield",
        "titer",
    )
    if any(term in lowered for term in evidence_terms):
        return True
    if rewrite is None:
        return False
    if rewrite.evidence_preference == "table" or rewrite.metrics:
        return True
    if rewrite.aliases:
        return True
    entity_count = (
        len(rewrite.products)
        + len(rewrite.organisms)
        + len(rewrite.genes)
        + len(rewrite.enzymes)
    )
    return entity_count >= 3


def _result_key(item: dict[str, Any], payload: dict[str, Any]) -> str:
    paper_id = str(payload.get("paper_id") or "")
    chunk_id = str(payload.get("chunk_id") or "")
    if paper_id or chunk_id:
        return f"{paper_id}:{chunk_id}"
    return str(item.get("id"))


def _anchor_protected_count(limit: int) -> int:
    if limit <= 0:
        return 0
    if limit <= 5:
        return min(limit, 3)
    return min(limit, max(5, limit // 2))


def _admit_tail_candidate(
    item: dict[str, Any],
    *,
    anchor_paper_ids: set[str],
    raw_candidates_are_sparse: bool,
) -> bool:
    if item.get("raw_rank") is not None:
        return True
    if raw_candidates_are_sparse:
        return True
    if _route_consensus_count(item) >= 2:
        return True
    payload = item.get("payload") or {}
    paper_id = str(payload.get("paper_id") or "")
    return bool(paper_id and paper_id in anchor_paper_ids)


def _route_consensus_count(item: dict[str, Any]) -> int:
    queries = item.get("matched_queries") or []
    return len({" ".join(str(query).split()).lower() for query in queries if str(query).strip()})


def _tail_priority(item: dict[str, Any]) -> int:
    if item.get("raw_rank") is not None:
        return 3
    if _route_consensus_count(item) >= 2:
        return 2
    return 1


def _paper_ids(items: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for item in items:
        payload = item.get("payload") or {}
        paper_id = str(payload.get("paper_id") or "")
        if paper_id:
            ids.add(paper_id)
    return ids
