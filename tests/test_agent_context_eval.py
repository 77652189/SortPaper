from __future__ import annotations

from evals.agent_context_eval import (
    evaluate_context_case,
    summarize_context_results,
)
from evals.retrieval_eval import RetrievalCase


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
