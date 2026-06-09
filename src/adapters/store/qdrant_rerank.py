from __future__ import annotations

from typing import Any, Callable

from src.adapters.rerank.dashscope import DashScopeReranker


ApiKeyGetter = Callable[[], str]
DocumentTextBuilder = Callable[[dict[str, Any]], str]


def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    top_n: int,
    api_key_getter: ApiKeyGetter,
    document_text_builder: DocumentTextBuilder,
) -> list[dict[str, Any]]:
    reranker = DashScopeReranker(
        api_key_getter=api_key_getter,
        document_text_builder=document_text_builder,
    )
    return reranker.rerank(query, candidates, top_n=top_n)
