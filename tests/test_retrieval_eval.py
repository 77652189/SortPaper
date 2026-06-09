from __future__ import annotations

from types import SimpleNamespace

from evals.retrieval_eval import (
    RetrievalCase,
    build_cases,
    evaluate_case,
    extract_keywords,
    first_rank,
    is_nearby_payload,
    location_from_payload,
    reciprocal_rank,
    summarize,
)


class FakeStore:
    def __init__(self, results):
        self.results = results
        self.used_strategy = ""
        self.last_kwargs = {}
        self.last_args = ()

    def search(self, *args, **kwargs):
        self.used_strategy = "standard"
        self.last_args = args
        self.last_kwargs = kwargs
        return self.results

    def search_evidence(self, *args, **kwargs):
        self.used_strategy = "two-stage"
        self.last_args = args
        self.last_kwargs = kwargs
        return self.results

    def search_paper_local_evidence(self, *args, **kwargs):
        self.used_strategy = "paper-local"
        self.last_args = args
        self.last_kwargs = kwargs
        return self.results


def test_first_rank_and_reciprocal_rank() -> None:
    assert first_rank(["a", "b", "c"], {"b"}) == 2
    assert first_rank(["a", "b", "c"], {"x"}) is None
    assert reciprocal_rank(4) == 0.25
    assert reciprocal_rank(None) == 0.0


def test_extract_keywords_ignores_common_words() -> None:
    keywords = extract_keywords(
        "The production production of lacto-N-triose II using lgtA and RBS optimization.",
        limit=4,
    )

    assert "production" not in keywords
    assert "lacto-n-triose" in keywords
    assert "lgta" in keywords


def test_build_cases_from_payloads_without_manual_labels() -> None:
    payloads = [
        {
            "paper_id": "paper-1",
            "chunk_id": "chunk-1",
            "paper_title": "Metabolic Engineering of E coli for LNT II.pdf",
            "content": "RBS optimization improved soluble lgtA expression for lacto-N-triose II biosynthesis in engineered Escherichia coli.",
            "target_products": ["lacto-N-triose II"],
            "organisms": ["Escherichia coli"],
            "category": "fermentation_experiment",
            "paper_summary": "This paper improves lacto-N-triose II biosynthesis using RBS and lgtA.",
            "page": 3,
            "global_order": 8,
            "order_in_page": 2,
            "bbox": [10, 20, 100, 80],
            "page_width": 600,
            "page_height": 800,
        }
    ]

    cases = build_cases(payloads, max_cases=5, seed=1, min_content_chars=30)

    assert cases
    assert {case.expected_paper_ids[0] for case in cases} == {"paper-1"}
    assert {case.case_type for case in cases} >= {"title", "metadata", "chunk"}
    chunk_case = next(case for case in cases if case.case_type == "chunk")
    assert chunk_case.expected_locations[0]["global_order"] == 8


def test_build_cases_hard_profile_generates_natural_metric_queries() -> None:
    payloads = [
        {
            "paper_id": "paper-1",
            "chunk_id": "chunk-1",
            "paper_title": "Engineering E coli for LNT.pdf",
            "content_type": "text",
            "content": (
                "The engineered strain produced lacto-N-tetraose with a titer of "
                "3.42 g/L after fed-batch fermentation and pathway optimization."
            ),
            "seo_terms": ["lacto-N-tetraose", "fed-batch", "titer"],
            "page": 2,
        },
    ]

    cases = build_cases(
        payloads,
        max_cases=3,
        seed=1,
        min_content_chars=40,
        content_types={"text"},
        case_profile="hard",
    )

    hard_cases = [case for case in cases if case.case_type == "hard_chunk"]
    assert hard_cases
    assert "Which evidence passage reports" in hard_cases[0].query
    assert "3.42 g/L" in hard_cases[0].query
    assert hard_cases[0].expected_chunk_ids == ["chunk-1"]


def test_nearby_payload_matches_by_global_order() -> None:
    expected = {
        "paper_id": "paper-1",
        "chunk_id": "chunk-5",
        "global_order": 10,
    }
    nearby = {
        "paper_id": "paper-1",
        "chunk_id": "chunk-7",
        "global_order": 12,
    }
    far = {
        "paper_id": "paper-1",
        "chunk_id": "chunk-20",
        "global_order": 20,
    }

    assert is_nearby_payload(nearby, expected)
    assert not is_nearby_payload(far, expected)


def test_nearby_payload_falls_back_to_page_order_and_bbox() -> None:
    expected = {
        "paper_id": "paper-1",
        "chunk_id": "chunk-5",
        "page": 2,
        "order_in_page": 4,
    }
    by_order = {
        "paper_id": "paper-1",
        "chunk_id": "chunk-6",
        "page": 2,
        "order_in_page": 5,
    }
    by_bbox = {
        "paper_id": "paper-1",
        "chunk_id": "chunk-7",
        "page": 2,
        "column": 1,
        "bbox": [102, 102, 152, 152],
        "page_width": 600,
        "page_height": 800,
    }
    expected_bbox = {
        "paper_id": "paper-1",
        "chunk_id": "chunk-5",
        "page": 2,
        "column": 1,
        "bbox": [100, 100, 150, 150],
        "page_width": 600,
        "page_height": 800,
    }

    assert is_nearby_payload(by_order, expected)
    assert is_nearby_payload(by_bbox, expected_bbox)


def test_nearby_payload_falls_back_to_chunk_id_coordinates() -> None:
    expected = {
        "paper_id": "paper-1",
        "chunk_id": "text_p4_col1_y603.6_x306.9",
    }
    nearby = {
        "paper_id": "paper-1",
        "chunk_id": "text_p4_col1_y700.0_x306.9",
    }
    other_column = {
        "paper_id": "paper-1",
        "chunk_id": "text_p4_col2_y620.0_x45.6",
    }

    assert is_nearby_payload(nearby, expected)
    assert not is_nearby_payload(other_column, expected)


def test_evaluate_case_and_summary_metrics() -> None:
    case = RetrievalCase(
        id="case-1",
        case_type="chunk",
        query="lacto-N-triose II lgtA",
        expected_paper_ids=["paper-2"],
        expected_chunk_ids=["chunk-2"],
        expected_locations=[{
            "paper_id": "paper-2",
            "chunk_id": "chunk-2",
            "global_order": 10,
        }],
    )
    store = FakeStore([
        {"score": 0.9, "payload": {"paper_id": "paper-1", "chunk_id": "chunk-1", "paper_title": "Wrong"}},
        {"score": 0.8, "payload": {"paper_id": "paper-2", "chunk_id": "chunk-near", "paper_title": "Near", "global_order": 11}},
        {"score": 0.7, "payload": {"paper_id": "paper-2", "chunk_id": "chunk-2", "paper_title": "Right", "global_order": 10}},
    ])

    result = evaluate_case(store, case, top_k=3, rerank=False)
    summary = summarize([result], ks=(1, 2, 3))

    assert result.paper_rank == 2
    assert result.chunk_rank == 3
    assert result.nearby_rank == 2
    assert summary["paper_hit@1"] == 0.0
    assert summary["paper_hit@2"] == 1.0
    assert summary["chunk_hit@2"] == 0.0
    assert summary["chunk_hit@3"] == 1.0
    assert summary["nearby_chunk_hit@2"] == 1.0


def test_evaluate_case_can_use_two_stage_strategy() -> None:
    case = RetrievalCase(
        id="case-1",
        case_type="title",
        query="demo paper",
        expected_paper_ids=["paper-1"],
        expected_chunk_ids=[],
    )
    store = FakeStore([
        {"score": 0.9, "payload": {"paper_id": "paper-1", "chunk_id": "chunk-1", "paper_title": "Right"}},
    ])

    result = evaluate_case(store, case, top_k=1, rerank=False, strategy="two-stage")

    assert store.used_strategy == "two-stage"
    assert result.paper_rank == 1


def test_evaluate_case_can_use_paper_local_strategy() -> None:
    case = RetrievalCase(
        id="case-1",
        case_type="chunk",
        query="lacto-N-triose II lgtA",
        expected_paper_ids=["paper-1"],
        expected_chunk_ids=["chunk-1"],
    )
    store = FakeStore([
        {"score": 0.9, "payload": {"paper_id": "paper-1", "chunk_id": "chunk-1", "paper_title": "Right"}},
    ])

    result = evaluate_case(store, case, top_k=1, rerank=False, strategy="paper-local")

    assert store.used_strategy == "paper-local"
    assert result.chunk_rank == 1


def test_evaluate_case_can_enable_lexical_backfill() -> None:
    case = RetrievalCase(
        id="case-1",
        case_type="chunk",
        query="lacto-N-triose II lgtA",
        expected_paper_ids=["paper-1"],
        expected_chunk_ids=["chunk-1"],
    )
    store = FakeStore([
        {"score": 0.9, "payload": {"paper_id": "paper-1", "chunk_id": "chunk-1", "paper_title": "Right"}},
    ])

    result = evaluate_case(
        store,
        case,
        top_k=1,
        rerank=False,
        lexical_backfill=True,
    )

    assert store.used_strategy == "standard"
    assert store.last_kwargs["lexical_backfill"] is True
    assert result.chunk_rank == 1
    assert result.elapsed_ms >= 0


def test_evaluate_case_passes_rerank_top_n_to_standard_search() -> None:
    case = RetrievalCase(
        id="case-1",
        case_type="chunk",
        query="lacto-N-triose II lgtA",
        expected_paper_ids=["paper-1"],
        expected_chunk_ids=["chunk-1"],
    )
    store = FakeStore([
        {"score": 0.9, "payload": {"paper_id": "paper-1", "chunk_id": "chunk-1", "paper_title": "Right"}},
    ])

    result = evaluate_case(
        store,
        case,
        top_k=50,
        rerank=True,
        rerank_top_n=10,
    )

    assert store.last_kwargs["rerank"] is True
    assert store.last_kwargs["rerank_top_n"] == 10
    assert result.chunk_rank == 1


def test_evaluate_case_passes_selective_rerank_to_standard_search() -> None:
    case = RetrievalCase(
        id="case-1",
        case_type="hard_chunk",
        query="Which evidence passage reports 3.42 g/L for LNT?",
        expected_paper_ids=["paper-1"],
        expected_chunk_ids=["chunk-1"],
    )
    store = FakeStore([
        {
            "score": 0.9,
            "selective_rerank_reason": "evidence_query",
            "payload": {"paper_id": "paper-1", "chunk_id": "chunk-1", "paper_title": "Right"},
        },
    ])

    result = evaluate_case(
        store,
        case,
        top_k=10,
        rerank=False,
        selective_rerank=True,
    )

    assert store.last_kwargs["selective_rerank"] is True
    assert result.selective_rerank_triggered is True
    assert result.selective_rerank_reasons == ["evidence_query"]


def test_evaluate_case_uses_rewritten_query() -> None:
    from src.retrieval.query_rewrite import QueryRewrite

    case = RetrievalCase(
        id="case-1",
        case_type="chunk",
        query="LNT II 为什么残留过高？",
        expected_paper_ids=["paper-1"],
        expected_chunk_ids=["chunk-1"],
    )
    store = FakeStore([
        {"score": 0.9, "payload": {"paper_id": "paper-1", "chunk_id": "chunk-1", "paper_title": "Right"}},
    ])
    rewritten = QueryRewrite(
        raw_query=case.query,
        normalized_query="lacto-N-triose II accumulation",
    )

    result = evaluate_case(
        store,
        case,
        top_k=1,
        rerank=False,
        rewritten_query=rewritten,
    )

    assert store.last_args[0] == "lacto-N-triose II accumulation"
    assert result.query_used == "lacto-N-triose II accumulation"
    assert result.query_rewrite["normalized_query"] == "lacto-N-triose II accumulation"


def test_evaluate_case_can_use_multi_query() -> None:
    from src.retrieval.query_rewrite import QueryRewrite

    case = RetrievalCase(
        id="case-1",
        case_type="chunk",
        query="raw question",
        expected_paper_ids=["paper-2"],
        expected_chunk_ids=["chunk-2"],
    )
    store = FakeStore([
        {"score": 0.9, "payload": {"paper_id": "paper-2", "chunk_id": "chunk-2", "paper_title": "Right"}},
    ])
    rewritten = QueryRewrite(
        raw_query=case.query,
        normalized_query="normalized question",
        variants=["variant one"],
    )

    result = evaluate_case(
        store,
        case,
        top_k=1,
        rerank=False,
        rerank_top_n=5,
        rewritten_query=rewritten,
        multi_query=True,
    )

    assert store.used_strategy == "standard"
    assert store.last_args[0] in {"raw question", "normalized question"}
    assert store.last_kwargs["rerank_top_n"] == 5
    assert "normalized question" in result.query_used
    assert result.search_meta["routes"][0]["source"] == "raw"
    assert result.search_meta["route_policy"]["reason"] == "compact_generic_query"
    assert result.chunk_rank == 1


def test_summarize_reports_route_policy_counts() -> None:
    from src.retrieval.query_rewrite import QueryRewrite

    case = RetrievalCase(
        id="case-1",
        case_type="title",
        query="demo paper",
        expected_paper_ids=["paper-1"],
        expected_chunk_ids=[],
    )
    result = evaluate_case(
        FakeStore([
            {
                "score": 0.9,
                "payload": {"paper_id": "paper-1", "chunk_id": "chunk-1", "paper_title": "Right"},
            }
        ]),
        case,
        top_k=1,
        rerank=False,
        rewritten_query=QueryRewrite(
            raw_query="demo paper",
            normalized_query="demo paper normalized",
        ),
        multi_query=True,
    )

    summary = summarize([result], ks=(1,))

    assert summary["route_count_avg"] == 2
    assert summary["route_policy_counts"] == {"compact_generic_query": 1}
    assert summary["route_source_counts"] == {"normalized": 1, "raw": 1}


def test_evaluate_case_can_enable_neighbor_backfill() -> None:
    case = RetrievalCase(
        id="case-1",
        case_type="chunk",
        query="lacto-N-triose II lgtA",
        expected_paper_ids=["paper-1"],
        expected_chunk_ids=["chunk-1"],
    )
    store = FakeStore([
        {"score": 0.9, "payload": {"paper_id": "paper-1", "chunk_id": "chunk-1", "paper_title": "Right"}},
    ])

    result = evaluate_case(
        store,
        case,
        top_k=1,
        rerank=False,
        neighbor_backfill=True,
    )

    assert store.used_strategy == "standard"
    assert store.last_kwargs["neighbor_backfill"] is True
    assert result.chunk_rank == 1


def test_evaluate_case_passes_fielded_lexical_weight_to_standard_search() -> None:
    case = RetrievalCase(
        id="case-1",
        case_type="chunk",
        query="lacto-N-tetraose Escherichia coli",
        expected_paper_ids=["paper-1"],
        expected_chunk_ids=["chunk-1"],
    )
    store = FakeStore([
        {"score": 0.9, "payload": {"paper_id": "paper-1", "chunk_id": "chunk-1", "paper_title": "Right"}},
    ])

    result = evaluate_case(
        store,
        case,
        top_k=1,
        rerank=False,
        fielded_lexical_weight=0.1,
    )

    assert store.used_strategy == "standard"
    assert store.last_kwargs["fielded_lexical_weight"] == 0.1
    assert result.chunk_rank == 1
