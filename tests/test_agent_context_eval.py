from __future__ import annotations

from types import SimpleNamespace

from evals.agent_context_eval import (
    evaluate_context_case,
    summarize_context_results,
)
from evals.retrieval_eval import RetrievalCase
from src.retrieval.query_rewrite import QueryRewrite


class FakeStore:
    def __init__(self) -> None:
        self.expand_called = False

    def search(self, *_args, **_kwargs):
        return [
            {
                "score": 0.9,
                "payload": {
                    "paper_id": "paper-1",
                    "chunk_id": "chunk-1",
                    "global_order": 10,
                },
            }
        ]

    def expand_neighbor_context(self, results, **_kwargs):
        self.expand_called = True
        assert len(results) == 1
        return [
            {
                "score": 0.7,
                "payload": {
                    "paper_id": "paper-1",
                    "chunk_id": "chunk-2",
                    "global_order": 11,
                },
            }
        ]


def test_context_eval_counts_neighbor_context_hits() -> None:
    case = RetrievalCase(
        id="chunk:paper-1:chunk-2",
        case_type="chunk",
        query="lacto-N-triose II lgtA",
        expected_paper_ids=["paper-1"],
        expected_chunk_ids=["chunk-2"],
        expected_locations=[{
            "paper_id": "paper-1",
            "chunk_id": "chunk-2",
            "global_order": 11,
        }],
    )
    store = FakeStore()

    result = evaluate_context_case(
        store,
        case,
        search_top_k=1,
        synthesis_limit=2,
        expand_neighbor_context=True,
    )
    summary = summarize_context_results([result], ks=(1, 2))

    assert store.expand_called is True
    assert result.search_chunk_rank is None
    assert result.context_chunk_rank == 2
    assert summary["search_chunk_hit@1"] == 0.0
    assert summary["context_chunk_hit@2"] == 1.0
    assert summary["context_nearby_hit@2"] == 1.0


def test_context_eval_can_disable_neighbor_context() -> None:
    case = RetrievalCase(
        id="chunk:paper-1:chunk-2",
        case_type="chunk",
        query="lacto-N-triose II lgtA",
        expected_paper_ids=["paper-1"],
        expected_chunk_ids=["chunk-2"],
    )
    store = FakeStore()

    result = evaluate_context_case(
        store,
        case,
        search_top_k=1,
        synthesis_limit=2,
        expand_neighbor_context=False,
    )

    assert store.expand_called is False
    assert result.context_chunk_rank is None
    assert result.added_context_chunks == 0


def test_context_eval_uses_normalized_query_for_search_and_paper_local_context() -> None:
    captured: dict[str, object] = {}

    class CapturingStore:
        def search(self, query, **_kwargs):
            captured["search_query"] = query
            return [
                {
                    "score": 0.9,
                    "payload": {
                        "paper_id": "paper-1",
                        "chunk_id": "chunk-1",
                    },
                }
            ]

        def expand_paper_local_context(self, query, results, **_kwargs):
            captured["context_query"] = query
            captured["context_results"] = results
            captured["content_type_preference"] = _kwargs.get("content_type_preference")
            return []

    case = RetrievalCase(
        id="chunk:paper-1:chunk-2",
        case_type="chunk",
        query="LNT II 为什么残留过高？",
        expected_paper_ids=["paper-1"],
        expected_chunk_ids=["chunk-2"],
    )
    rewrite = QueryRewrite(
        raw_query=case.query,
        normalized_query="lacto-N-triose II accumulation engineering strategy",
        products=["LNT II"],
        genes=["lgtA"],
        metrics=["accumulation"],
        evidence_preference="table",
        aliases=["LNTII"],
        variants=["LNT II lgtA RBS"],
    )

    result = evaluate_context_case(
        CapturingStore(),
        case,
        search_top_k=1,
        synthesis_limit=2,
        expand_neighbor_context=False,
        expand_paper_local_context=True,
        rewritten_query=rewrite,
    )

    assert captured["search_query"] == "lacto-N-triose II accumulation engineering strategy"
    assert captured["context_query"] == (
        "lacto-N-triose II accumulation engineering strategy "
        "LNT II lgtA LNTII LNT II lgtA RBS"
    )
    assert len(captured["context_results"]) == 1
    assert captured["content_type_preference"] == "table"
    assert result.search_query == "lacto-N-triose II accumulation engineering strategy"
    assert result.context_query == (
        "lacto-N-triose II accumulation engineering strategy "
        "LNT II lgtA LNTII LNT II lgtA RBS"
    )
    assert result.query_rewrite == rewrite.to_dict()


def test_context_eval_passes_rerank_to_standard_search() -> None:
    captured: dict[str, object] = {}

    class CapturingStore:
        def search(self, query, **kwargs):
            captured["query"] = query
            captured.update(kwargs)
            return []

    case = RetrievalCase(
        id="chunk:paper-1:chunk-1",
        case_type="chunk",
        query="lacto-N-triose II accumulation",
        expected_paper_ids=["paper-1"],
        expected_chunk_ids=["chunk-1"],
    )

    evaluate_context_case(
        CapturingStore(),
        case,
        search_top_k=5,
        synthesis_limit=10,
        rerank=True,
        lexical_backfill=True,
        expand_neighbor_context=False,
        expand_paper_local_context=False,
    )

    assert captured["query"] == "lacto-N-triose II accumulation"
    assert captured["limit"] == 5
    assert captured["rerank"] is True
    assert captured["lexical_backfill"] is True


def test_context_eval_passes_rerank_to_multi_query(monkeypatch) -> None:
    import evals.agent_context_eval as agent_context_eval

    captured: dict[str, object] = {}
    rewrite = QueryRewrite(
        raw_query="LNT II accumulation",
        normalized_query="lacto-N-triose II accumulation",
        aliases=["LNT II"],
    )

    class CapturingStore:
        pass

    def fake_multi_query_search(store, query, **kwargs):
        captured["store"] = store
        captured["query"] = query
        captured.update(kwargs)
        return SimpleNamespace(
            results=[],
            rewrite=rewrite,
            routes=[],
        )

    monkeypatch.setattr(agent_context_eval, "multi_query_search", fake_multi_query_search)

    case = RetrievalCase(
        id="chunk:paper-1:chunk-1",
        case_type="chunk",
        query="LNT II accumulation",
        expected_paper_ids=["paper-1"],
        expected_chunk_ids=["chunk-1"],
    )
    store = CapturingStore()

    result = evaluate_context_case(
        store,
        case,
        search_top_k=5,
        synthesis_limit=10,
        rerank=True,
        lexical_backfill=True,
        rewritten_query=rewrite,
        multi_query=True,
        expand_neighbor_context=False,
        expand_paper_local_context=False,
    )

    assert captured["store"] is store
    assert captured["query"] == "LNT II accumulation"
    assert captured["limit"] == 5
    assert captured["rerank"] is True
    assert captured["lexical_backfill"] is True
    assert captured["rewritten_query"] is rewrite
    assert result.context_query == "lacto-N-triose II accumulation LNT II"
