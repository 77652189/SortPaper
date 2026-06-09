from __future__ import annotations

from src.retrieval.multi_query import build_search_routes, multi_query_search
from src.retrieval.query_rewrite import QueryRewrite


class FakeStore:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        if query == "raw question":
            return [
                {"id": "a", "score": 0.2, "payload": {"paper_id": "p1", "chunk_id": "c1"}},
                {"id": "b", "score": 0.1, "payload": {"paper_id": "p2", "chunk_id": "c2"}},
            ]
        if query == "normalized question":
            return [
                {"id": "b", "score": 0.9, "payload": {"paper_id": "p2", "chunk_id": "c2"}},
                {"id": "c", "score": 0.4, "payload": {"paper_id": "p3", "chunk_id": "c3"}},
            ]
        return [
            {"id": "c", "score": 0.5, "payload": {"paper_id": "p3", "chunk_id": "c3"}},
        ]


def test_build_search_routes_dedupes_raw_normalized_and_variants() -> None:
    rewrite = QueryRewrite(
        raw_query="raw question",
        normalized_query="normalized question",
        variants=["normalized question", "variant one", "variant two"],
    )

    routes, used_rewrite = build_search_routes(
        "raw question",
        rewritten_query=rewrite,
        max_routes=3,
    )

    assert used_rewrite is rewrite
    assert [route.query for route in routes] == [
        "raw question",
        "normalized question",
        "variant one",
    ]
    assert [route.source for route in routes] == ["raw", "normalized", "variant"]


def test_build_search_routes_adds_structured_entity_table_and_alias_routes() -> None:
    rewrite = QueryRewrite(
        raw_query="raw question",
        normalized_query="normalized question",
        products=["LNT II"],
        organisms=["E. coli"],
        genes=["lgtA"],
        enzymes=["beta-1,3-N-acetylglucosaminyltransferase"],
        metrics=["3.42 g/L", "titer"],
        evidence_preference="table",
        aliases=["LNTII", "lacto-N-triose II"],
        variants=["variant one"],
    )

    routes, used_rewrite = build_search_routes(
        "raw question",
        rewritten_query=rewrite,
        max_routes=6,
    )

    assert used_rewrite is rewrite
    assert [route.source for route in routes] == [
        "raw",
        "normalized",
        "entity",
        "table_evidence",
        "alias",
        "variant",
    ]
    assert routes[2].query == "LNT II E. coli lgtA beta-1,3-N-acetylglucosaminyltransferase"
    assert routes[3].query == "3.42 g/L titer LNT II E. coli lgtA table"
    assert routes[4].query == "LNTII lacto-N-triose II"


def test_multi_query_search_merges_duplicate_chunks_with_rrf() -> None:
    rewrite = QueryRewrite(
        raw_query="raw question",
        normalized_query="normalized question",
        variants=["variant one"],
    )
    store = FakeStore()

    result = multi_query_search(
        store,
        "raw question",
        limit=2,
        rerank=True,
        selective_rerank=True,
        rerank_top_n=5,
        lexical_backfill=True,
        neighbor_backfill=True,
        rewritten_query=rewrite,
        route_limit=3,
        adaptive_route_limit=False,
    )

    assert [call["query"] for call in store.calls] == [
        "raw question",
        "normalized question",
        "variant one",
    ]
    assert [call["limit"] for call in store.calls] == [2, 6, 6]
    assert all(call["rerank"] is True for call in store.calls)
    assert all(call["selective_rerank"] is True for call in store.calls)
    assert all(call["rerank_top_n"] == 5 for call in store.calls)
    assert all(call["lexical_backfill"] is True for call in store.calls)
    assert all(call["neighbor_backfill"] is True for call in store.calls)
    assert [item["payload"]["chunk_id"] for item in result.results] == ["c1", "c2"]
    assert len(result.results) == 2
    merged_c2 = next(item for item in result.results if item["payload"]["chunk_id"] == "c2")
    assert merged_c2["matched_routes"] == ["raw", "normalized"]
    assert result.metadata()["rewrite"]["normalized_query"] == "normalized question"


def test_multi_query_search_adaptive_route_limit_compacts_generic_query() -> None:
    rewrite = QueryRewrite(
        raw_query="raw question",
        normalized_query="normalized question",
        variants=["variant one"],
    )
    store = FakeStore()

    result = multi_query_search(
        store,
        "raw question",
        limit=2,
        rewritten_query=rewrite,
        route_limit=4,
        adaptive_route_limit=True,
    )

    assert [call["query"] for call in store.calls] == [
        "raw question",
        "normalized question",
    ]
    assert [route.source for route in result.routes] == ["raw", "normalized"]
    assert result.metadata()["route_policy"] == {
        "adaptive": True,
        "requested_limit": 4,
        "effective_limit": 2,
        "reason": "compact_generic_query",
    }


def test_multi_query_search_adaptive_route_limit_expands_evidence_query() -> None:
    rewrite = QueryRewrite(
        raw_query="Which evidence passage reports 3.42 g/L?",
        normalized_query="3.42 g/L LNT II titer",
        products=["LNT II"],
        organisms=["E. coli"],
        genes=["lgtA"],
        metrics=["3.42 g/L"],
        evidence_preference="table",
        variants=["variant one"],
    )
    store = FakeStore()

    result = multi_query_search(
        store,
        "Which evidence passage reports 3.42 g/L?",
        limit=2,
        rewritten_query=rewrite,
        route_limit=4,
        adaptive_route_limit=True,
    )

    assert [route.source for route in result.routes] == [
        "raw",
        "normalized",
        "entity",
        "table_evidence",
    ]
    assert result.metadata()["route_policy"]["effective_limit"] == 4
    assert result.metadata()["route_policy"]["reason"] == "evidence_or_structured_query"


def test_multi_query_search_protects_raw_top_results_before_variants() -> None:
    class NoisyVariantStore:
        def search(self, query, **_kwargs):
            if query == "raw question":
                return [
                    {"id": "raw-1", "score": 0.9, "payload": {"paper_id": "p1", "chunk_id": "raw-1"}},
                    {"id": "raw-2", "score": 0.8, "payload": {"paper_id": "p1", "chunk_id": "raw-2"}},
                    {"id": "raw-3", "score": 0.7, "payload": {"paper_id": "p1", "chunk_id": "raw-3"}},
                    {"id": "raw-4", "score": 0.6, "payload": {"paper_id": "p1", "chunk_id": "raw-4"}},
                    {"id": "raw-5", "score": 0.5, "payload": {"paper_id": "p1", "chunk_id": "raw-5"}},
                ]
            return [
                {"id": "variant-1", "score": 0.99, "payload": {"paper_id": "p2", "chunk_id": "variant-1"}},
                {"id": "variant-2", "score": 0.98, "payload": {"paper_id": "p2", "chunk_id": "variant-2"}},
            ]

    rewrite = QueryRewrite(
        raw_query="raw question",
        normalized_query="normalized question",
        variants=["variant one"],
    )

    result = multi_query_search(
        NoisyVariantStore(),
        "raw question",
        limit=5,
        rewritten_query=rewrite,
        route_limit=2,
    )

    assert [item["payload"]["chunk_id"] for item in result.results[:3]] == [
        "raw-1",
        "raw-2",
        "raw-3",
    ]
    assert "variant-1" not in [item["payload"]["chunk_id"] for item in result.results]


def test_multi_query_search_allows_same_paper_tail_candidates() -> None:
    class SamePaperVariantStore:
        def search(self, query, **_kwargs):
            if query == "raw question":
                return [
                    {"id": "raw-1", "score": 0.9, "payload": {"paper_id": "p1", "chunk_id": "raw-1"}},
                    {"id": "raw-2", "score": 0.8, "payload": {"paper_id": "p1", "chunk_id": "raw-2"}},
                    {"id": "raw-3", "score": 0.7, "payload": {"paper_id": "p1", "chunk_id": "raw-3"}},
                    {"id": "raw-4", "score": 0.6, "payload": {"paper_id": "p1", "chunk_id": "raw-4"}},
                ]
            return [
                {"id": "same-paper", "score": 0.99, "payload": {"paper_id": "p1", "chunk_id": "same-paper"}},
                {"id": "other-paper", "score": 0.98, "payload": {"paper_id": "p2", "chunk_id": "other-paper"}},
            ]

    rewrite = QueryRewrite(
        raw_query="raw question",
        normalized_query="normalized question",
    )

    result = multi_query_search(
        SamePaperVariantStore(),
        "raw question",
        limit=5,
        rewritten_query=rewrite,
        route_limit=2,
    )

    chunks = [item["payload"]["chunk_id"] for item in result.results]
    assert "same-paper" in chunks
    assert "other-paper" not in chunks
    assert chunks.index("same-paper") > chunks.index("raw-4")


def test_multi_query_search_allows_cross_paper_consensus_candidates() -> None:
    class ConsensusVariantStore:
        def search(self, query, **_kwargs):
            if query == "raw question":
                return [
                    {"id": "raw-1", "score": 0.9, "payload": {"paper_id": "p1", "chunk_id": "raw-1"}},
                    {"id": "raw-2", "score": 0.8, "payload": {"paper_id": "p1", "chunk_id": "raw-2"}},
                    {"id": "raw-3", "score": 0.7, "payload": {"paper_id": "p1", "chunk_id": "raw-3"}},
                ]
            return [
                {"id": "consensus", "score": 0.99, "payload": {"paper_id": "p2", "chunk_id": "consensus"}},
            ]

    rewrite = QueryRewrite(
        raw_query="raw question",
        normalized_query="normalized question",
        variants=["variant one"],
    )

    result = multi_query_search(
        ConsensusVariantStore(),
        "raw question",
        limit=5,
        rewritten_query=rewrite,
        route_limit=3,
        adaptive_route_limit=False,
    )

    consensus = next(item for item in result.results if item["payload"]["chunk_id"] == "consensus")
    assert consensus["matched_queries"] == ["normalized question", "variant one"]
    assert result.results[-1]["payload"]["chunk_id"] == "consensus"
