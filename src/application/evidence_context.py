from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvidenceContextOptions:
    expand_neighbor_context: bool = True
    expand_paper_local_context: bool = True
    paper_limit: int = 5
    per_paper_limit: int = 3
    total_limit: int = 5
    neighbor_per_result_limit: int = 2


def expand_evidence_context(
    repository: Any,
    query: str,
    results: list[dict[str, Any]],
    *,
    filter_kwargs: dict[str, Any] | None = None,
    content_type_preference: str | None = None,
    options: EvidenceContextOptions | None = None,
) -> list[dict[str, Any]]:
    opts = options or EvidenceContextOptions()
    context_results: list[dict[str, Any]] = []

    if (
        opts.expand_paper_local_context
        and results
        and hasattr(repository, "expand_paper_local_context")
    ):
        context_results.extend(
            repository.expand_paper_local_context(
                query,
                results,
                filter_kwargs=filter_kwargs,
                paper_limit=opts.paper_limit,
                per_paper_limit=opts.per_paper_limit,
                total_limit=opts.total_limit,
                content_type_preference=content_type_preference,
            )
        )

    if (
        opts.expand_neighbor_context
        and results
        and hasattr(repository, "expand_neighbor_context")
    ):
        existing_context_keys = {_result_key(item) for item in context_results}
        neighbor_limit = max(0, opts.total_limit - len(context_results))
        if neighbor_limit > 0:
            neighbor_results = repository.expand_neighbor_context(
                results + context_results,
                filter_kwargs=filter_kwargs,
                per_result_limit=opts.neighbor_per_result_limit,
                total_limit=neighbor_limit,
            )
            for item in neighbor_results:
                key = _result_key(item)
                if key not in existing_context_keys:
                    context_results.append(item)
                    existing_context_keys.add(key)

    return context_results


def _result_key(item: dict[str, Any]) -> tuple[str, str]:
    payload = item.get("payload") or {}
    return (
        str(payload.get("paper_id") or ""),
        str(payload.get("chunk_id") or item.get("id") or ""),
    )
