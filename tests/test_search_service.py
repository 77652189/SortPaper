from __future__ import annotations

from types import SimpleNamespace

from src.application import search


def test_search_chunks_merges_filters_and_formats_results() -> None:
    captured: dict[str, object] = {}

    class Repository:
        def search(self, **kwargs):
            captured.update(kwargs)
            return [
                {
                    "id": "point-1",
                    "score": 0.8,
                    "selective_rerank_reason": "evidence_query",
                    "payload": {"paper_id": "paper-1", "chunk_id": "c1"},
                }
            ]

    results = search.search_chunks(
        "lacto-N-triose II",
        paper_id="paper-1",
        top_k=7,
        filter_kwargs={"is_actionable": True},
        rerank=True,
        selective_rerank=True,
        rerank_top_n=5,
        neighbor_backfill=True,
        repository=Repository(),
    )

    assert captured["query"] == "lacto-N-triose II"
    assert captured["limit"] == 7
    assert captured["filter_kwargs"] == {"is_actionable": True, "paper_id": "paper-1"}
    assert captured["rerank"] is True
    assert captured["selective_rerank"] is True
    assert captured["rerank_top_n"] == 5
    assert captured["lexical_backfill"] is True
    assert captured["neighbor_backfill"] is True
    assert results[0]["id"] == "point-1"
    assert results[0]["reranked"] is False
    assert results[0]["search_meta"] == {}


def test_search_chunks_uses_multi_query_runner_with_adaptive_route_limit() -> None:
    captured: dict[str, object] = {}

    class Repository:
        pass

    def multi_query_runner(repository, query, **kwargs):
        captured["repository"] = repository
        captured["query"] = query
        captured.update(kwargs)
        return SimpleNamespace(
            results=[
                {
                    "id": "point-1",
                    "score": 0.9,
                    "payload": {"paper_id": "paper-1"},
                }
            ],
            metadata=lambda: {"route_policy": {"adaptive": True}},
        )

    repo = Repository()
    results = search.search_chunks(
        "Which passage reports 3.42 g/L?",
        top_k=5,
        multi_query=True,
        repository=repo,
        multi_query_runner=multi_query_runner,
    )

    assert captured["repository"] is repo
    assert captured["query"] == "Which passage reports 3.42 g/L?"
    assert captured["route_limit"] == 4
    assert captured["adaptive_route_limit"] is True
    assert captured["use_query_rewrite"] is True
    assert results[0]["search_meta"]["route_policy"]["adaptive"] is True
