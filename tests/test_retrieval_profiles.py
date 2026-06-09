from __future__ import annotations

from types import SimpleNamespace

from src.application import search
from src.application.retrieval_profiles import get_retrieval_profile, profile_options


def test_retrieval_profiles_define_user_level_modes() -> None:
    assert profile_options()["quick"] == "快速检索"
    assert profile_options()["evidence"] == "证据检索"
    assert profile_options()["agent"] == "Agent 召回"

    quick = get_retrieval_profile("quick")
    evidence = get_retrieval_profile("evidence")
    agent = get_retrieval_profile("agent")

    assert quick.lexical_backfill is True
    assert quick.rerank is False
    assert evidence.selective_rerank is True
    assert evidence.rerank_top_n == 10
    assert evidence.query_rewrite is False
    assert agent.rerank_top_n == 10
    assert agent.query_rewrite is False
    assert agent.expand_paper_local_context is True
    assert agent.expand_neighbor_context is True


def test_search_chunks_applies_profile_without_breaking_repository_contract() -> None:
    captured: dict[str, object] = {}

    class Repository:
        def search(self, **kwargs):
            captured.update(kwargs)
            return [{"id": "p1", "score": 0.8, "payload": {"paper_id": "paper-1"}}]

    results = search.search_chunks(
        "Which table reports 3.42 g/L?",
        top_k=5,
        retrieval_profile="evidence",
        repository=Repository(),
    )

    assert captured["limit"] == 5
    assert captured["lexical_backfill"] is True
    assert captured["rerank"] is False
    assert captured["selective_rerank"] is True
    assert captured["rerank_top_n"] == 10
    assert results[0]["search_meta"]["profile"]["name"] == "evidence"
    assert results[0]["search_meta"]["profile"]["query_rewrite"] is False


def test_search_chunks_profile_still_uses_multi_query_runner_when_enabled() -> None:
    captured: dict[str, object] = {}

    class Repository:
        pass

    def fake_multi_query_runner(repository, query, **kwargs):
        captured["repository"] = repository
        captured["query"] = query
        captured.update(kwargs)
        return SimpleNamespace(
            results=[{"id": "p1", "score": 0.7, "payload": {}}],
            metadata=lambda: {"routes": [{"source": "raw"}]},
        )

    results = search.search_chunks(
        "query",
        top_k=3,
        query_rewrite=True,
        repository=Repository(),
        multi_query_runner=fake_multi_query_runner,
    )

    assert captured["limit"] == 3
    assert captured["use_query_rewrite"] is True
    assert captured["route_limit"] == 2
    assert results[0]["search_meta"]["routes"][0]["source"] == "raw"
