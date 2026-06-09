from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


RetrievalProfileName = Literal["quick", "evidence", "agent"]


@dataclass(frozen=True)
class RetrievalProfile:
    name: RetrievalProfileName
    rerank: bool
    selective_rerank: bool
    rerank_top_n: int | None
    lexical_backfill: bool
    neighbor_backfill: bool
    query_rewrite: bool
    multi_query: bool
    expand_neighbor_context: bool
    expand_paper_local_context: bool
    default_top_k: int = 10
    context_total_limit: int = 5
    paper_local_paper_limit: int = 5
    paper_local_per_paper_limit: int = 3

    def search_options(self) -> dict:
        return {
            "rerank": self.rerank,
            "selective_rerank": self.selective_rerank,
            "rerank_top_n": self.rerank_top_n,
            "lexical_backfill": self.lexical_backfill,
            "neighbor_backfill": self.neighbor_backfill,
            "query_rewrite": self.query_rewrite,
            "multi_query": self.multi_query,
        }

    def to_dict(self) -> dict:
        return asdict(self)


PROFILE_DEFINITIONS: dict[RetrievalProfileName, RetrievalProfile] = {
    "quick": RetrievalProfile(
        name="quick",
        rerank=False,
        selective_rerank=False,
        rerank_top_n=None,
        lexical_backfill=True,
        neighbor_backfill=False,
        query_rewrite=False,
        multi_query=False,
        expand_neighbor_context=False,
        expand_paper_local_context=False,
        default_top_k=10,
    ),
    "evidence": RetrievalProfile(
        name="evidence",
        rerank=False,
        selective_rerank=True,
        rerank_top_n=10,
        lexical_backfill=True,
        neighbor_backfill=False,
        query_rewrite=False,
        multi_query=False,
        expand_neighbor_context=False,
        expand_paper_local_context=False,
        default_top_k=10,
    ),
    "agent": RetrievalProfile(
        name="agent",
        rerank=False,
        selective_rerank=True,
        rerank_top_n=10,
        lexical_backfill=True,
        neighbor_backfill=False,
        query_rewrite=False,
        multi_query=False,
        expand_neighbor_context=True,
        expand_paper_local_context=True,
        default_top_k=5,
        context_total_limit=5,
        paper_local_paper_limit=5,
        paper_local_per_paper_limit=3,
    ),
}


def coerce_profile_name(value: str | None, *, default: RetrievalProfileName = "quick") -> RetrievalProfileName:
    normalized = str(value or "").strip().lower()
    if normalized in PROFILE_DEFINITIONS:
        return normalized  # type: ignore[return-value]
    return default


def get_retrieval_profile(value: str | None, *, default: RetrievalProfileName = "quick") -> RetrievalProfile:
    return PROFILE_DEFINITIONS[coerce_profile_name(value, default=default)]
