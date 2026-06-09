from __future__ import annotations

from typing import Callable

from src.ports.agent import LiteratureAgentRunner


def run_literature_agent_query(
    question: str,
    *,
    max_rounds: int,
    retrieval_profile: str | None = None,
    lexical_backfill: bool = True,
    neighbor_backfill: bool = False,
    rerank: bool = False,
    selective_rerank: bool = False,
    rerank_top_n: int | None = None,
    use_query_rewrite: bool = False,
    multi_query_recall: bool = False,
    agent_factory: Callable[..., LiteratureAgentRunner] | None = None,
) -> dict:
    factory = agent_factory or _default_literature_agent_factory
    if retrieval_profile:
        from src.application.retrieval_profiles import get_retrieval_profile

        profile = get_retrieval_profile(retrieval_profile, default="agent")
        retrieval_profile = profile.name
        lexical_backfill = profile.lexical_backfill
        neighbor_backfill = profile.neighbor_backfill
        rerank = profile.rerank
        selective_rerank = profile.selective_rerank
        rerank_top_n = profile.rerank_top_n
        use_query_rewrite = profile.query_rewrite
        multi_query_recall = profile.multi_query
    elif (rerank or selective_rerank) and rerank_top_n is None:
        rerank_top_n = 5

    agent_kwargs = {
        "lexical_backfill": lexical_backfill,
        "neighbor_backfill": neighbor_backfill,
        "rerank": rerank,
        "selective_rerank": selective_rerank,
        "rerank_top_n": rerank_top_n,
        "use_query_rewrite": use_query_rewrite,
        "multi_query_recall": multi_query_recall,
    }
    if retrieval_profile:
        agent_kwargs["retrieval_profile"] = retrieval_profile
    agent = factory(**agent_kwargs)
    return agent.query(question, max_rounds=max_rounds)


def _default_literature_agent_factory(**kwargs) -> LiteratureAgentRunner:
    from src.adapters.agent.literature import create_literature_agent

    return create_literature_agent(**kwargs)
