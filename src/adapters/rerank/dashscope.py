from __future__ import annotations

import logging
from collections.abc import Callable
from http import HTTPStatus
from typing import Any

from src.application.settings import env_value

logger = logging.getLogger(__name__)


def dashscope_rerank_api_key() -> str:
    return env_value("DASHSCOPE_API_KEY")


class DashScopeReranker:
    """DashScope qwen3-rerank adapter."""

    def __init__(
        self,
        *,
        api_key_getter: Callable[[], str] = dashscope_rerank_api_key,
        document_text_builder: Callable[[dict[str, Any]], str],
        model: str = "qwen3-rerank",
        instruct: str = "Given a scientific paper search query, retrieve relevant passages that answer the query.",
        max_documents: int = 500,
    ) -> None:
        self.api_key_getter = api_key_getter
        self.document_text_builder = document_text_builder
        self.model = model
        self.instruct = instruct
        self.max_documents = max_documents

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        import dashscope

        rerank_candidates = candidates[: self.max_documents]
        documents = [self.document_text_builder(c["payload"]) for c in rerank_candidates]
        if not documents:
            return candidates

        api_key = self.api_key_getter()
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY is not configured; qwen3-rerank cannot run")

        resp = dashscope.TextReRank.call(
            model=self.model,
            query=query,
            documents=documents,
            top_n=min(top_n, len(documents)),
            return_documents=False,
            instruct=self.instruct,
            api_key=api_key,
        )

        if resp.status_code != HTTPStatus.OK:
            logger.warning(
                "Rerank failed; falling back to original ranking | code=%s msg=%s",
                getattr(resp, "code", None),
                getattr(resp, "message", None),
            )
            return candidates

        rank_map = {
            item.get("index", -1): item.get("relevance_score", 0.0)
            for item in (
                resp.output.get("results")
                if isinstance(resp.output, dict)
                else getattr(resp.output, "results", [])
            )
        }
        reranked_indexes = set(rank_map)
        ranked_pairs = sorted(
            (
                (index, item)
                for index, item in enumerate(rerank_candidates)
                if index in reranked_indexes
            ),
            key=lambda item: rank_map.get(item[0], 0.0),
            reverse=True,
        )
        reranked: list[dict[str, Any]] = []
        for original_index, item in ranked_pairs:
            item["score"] = rank_map.get(original_index, item["score"])
            item["reranked"] = True
            reranked.append(item)
        reranked.extend(
            item
            for original_index, item in enumerate(rerank_candidates)
            if original_index not in reranked_indexes
        )
        reranked.extend(candidates[self.max_documents :])

        usage = getattr(resp, "usage", None)
        logger.info(
            "Rerank %d -> %d | tokens=%s",
            len(candidates),
            len(reranked),
            usage.get("total_tokens", "?") if usage else "?",
        )
        return reranked
