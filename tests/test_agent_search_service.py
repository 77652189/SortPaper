from __future__ import annotations

from src.application.agent_search import run_literature_agent_query


def test_run_literature_agent_query_uses_injected_factory() -> None:
    captured: dict[str, object] = {}

    class Agent:
        def query(self, question: str, *, max_rounds: int) -> dict:
            captured["question"] = question
            captured["max_rounds"] = max_rounds
            return {"answer": "ok", "search_history": [], "total_chunks": 0}

    def factory(**kwargs):
        captured["factory_kwargs"] = kwargs
        return Agent()

    result = run_literature_agent_query(
        "How to improve LNT II production?",
        max_rounds=2,
        lexical_backfill=True,
        neighbor_backfill=True,
        rerank=True,
        selective_rerank=False,
        use_query_rewrite=True,
        multi_query_recall=True,
        agent_factory=factory,
    )

    assert result["answer"] == "ok"
    assert captured["question"] == "How to improve LNT II production?"
    assert captured["max_rounds"] == 2
    assert captured["factory_kwargs"] == {
        "lexical_backfill": True,
        "neighbor_backfill": True,
        "rerank": True,
        "selective_rerank": False,
        "rerank_top_n": 5,
        "use_query_rewrite": True,
        "multi_query_recall": True,
    }


def test_run_literature_agent_query_can_use_retrieval_profile() -> None:
    captured: dict[str, object] = {}

    class Agent:
        def query(self, question: str, *, max_rounds: int) -> dict:
            return {"answer": "ok", "search_history": [], "total_chunks": 0}

    def factory(**kwargs):
        captured.update(kwargs)
        return Agent()

    run_literature_agent_query(
        "Find evidence",
        max_rounds=1,
        retrieval_profile="agent",
        agent_factory=factory,
    )

    assert captured["retrieval_profile"] == "agent"
    assert captured["lexical_backfill"] is True
    assert captured["rerank"] is False
    assert captured["selective_rerank"] is True
    assert captured["rerank_top_n"] == 10
    assert captured["use_query_rewrite"] is False
    assert captured["multi_query_recall"] is False
