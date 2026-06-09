from __future__ import annotations

from evals.retrieval_report_analysis import classify_failures, compare_reports


def _result(
    *,
    case_id: str,
    paper_rank: int | None,
    chunk_rank: int | None,
    nearby_rank: int | None,
) -> dict:
    return {
        "case": {
            "id": case_id,
            "query": f"query {case_id}",
            "source_title": "paper.pdf",
            "expected_paper_ids": ["paper-1"],
            "expected_chunk_ids": [f"chunk-{case_id}"],
        },
        "paper_rank": paper_rank,
        "chunk_rank": chunk_rank,
        "nearby_rank": nearby_rank,
        "top_papers": ["paper-1", "paper-2"],
        "top_chunks": ["chunk-a", "chunk-b"],
    }


def test_classify_failures_splits_chunk_bottlenecks() -> None:
    report = {
        "results": [
            _result(case_id="paper-miss", paper_rank=None, chunk_rank=None, nearby_rank=None),
            _result(case_id="nearby-hit", paper_rank=1, chunk_rank=None, nearby_rank=4),
            _result(case_id="all-miss", paper_rank=1, chunk_rank=None, nearby_rank=None),
            _result(case_id="deep-hit", paper_rank=1, chunk_rank=80, nearby_rank=20),
            _result(case_id="top-hit", paper_rank=1, chunk_rank=3, nearby_rank=3),
        ],
    }

    analysis = classify_failures(report, k=100, example_limit=2)

    assert analysis["chunk_cases"] == 5
    assert analysis["paper_hit"] == 4
    assert analysis["exact_hit"] == 2
    assert analysis["nearby_hit"] == 3
    buckets = {bucket["name"]: bucket for bucket in analysis["buckets"]}
    assert buckets["paper_miss"]["count"] == 1
    assert buckets["paper_hit_exact_miss_nearby_hit"]["count"] == 1
    assert buckets["paper_hit_exact_and_nearby_miss"]["count"] == 1
    assert buckets["exact_hit_only_after_top10"]["count"] == 1
    assert buckets["paper_miss"]["examples"][0]["id"] == "paper-miss"


def test_compare_reports_returns_metric_deltas() -> None:
    baseline = {
        "summary": {
            "paper_hit@10": 0.8,
            "chunk_hit@10": 0.4,
            "nearby_chunk_hit@10": 0.5,
            "elapsed_ms_p50": 1000.0,
        },
    }
    candidate = {
        "summary": {
            "paper_hit@10": 0.9,
            "chunk_hit@10": 0.55,
            "nearby_chunk_hit@10": 0.7,
            "elapsed_ms_p50": 1400.0,
        },
    }

    comparison = compare_reports(baseline, candidate, ks=[10])

    assert comparison["deltas"] == {
        "paper_hit@10": 0.1,
        "chunk_hit@10": 0.15,
        "nearby_chunk_hit@10": 0.2,
        "elapsed_ms_p50": 400.0,
    }
