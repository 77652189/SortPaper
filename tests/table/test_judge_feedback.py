from __future__ import annotations

from src.judge.table_judge import (
    RegionTextContext,
    adjusted_region_for_action,
    build_auto_decision,
    build_bbox_candidates,
    build_storage_decision,
    build_table_embedding_text,
    classify_table_result,
    compare_table_candidate_scores,
    score_table_candidate,
)
from src.parsers.table.models import TableEvidence, TableRegion
from tools import table_region_benchmark


def _region(confidence: float = 0.8) -> TableRegion:
    return TableRegion(
        page=1,
        bbox=(10, 20, 100, 120),
        confidence=confidence,
        evidence=[TableEvidence("caption", 0.35, "near_table_caption", details={"text": "Table 1. Example"})],
    )


def test_successful_low_consistency_table_requests_text_retry() -> None:
    feedback = classify_table_result(
        region=_region(),
        chunk_metadata={
            "rows": 6,
            "cols": 4,
            "structural_quality": {"fallback_reasons": ["low_consistency"]},
        },
        parser_succeeded=True,
    )

    assert feedback.failure_category == "header_or_span_complexity"
    assert feedback.recommended_action == "retry_text_strategy"
    assert feedback.needs_llm_judge is True


def test_no_output_with_almost_no_text_expands_bbox() -> None:
    feedback = classify_table_result(
        region=_region(),
        chunk_metadata=None,
        parser_succeeded=False,
        text_context=RegionTextContext(word_count=2),
    )

    assert feedback.failure_category == "bbox_too_small"
    assert feedback.recommended_action == "expand_bbox"


def test_body_text_heavy_region_requests_shrink() -> None:
    feedback = classify_table_result(
        region=_region(),
        chunk_metadata=None,
        parser_succeeded=False,
        text_context=RegionTextContext(word_count=50, long_line_count=3, body_word_hits=12),
    )

    assert feedback.failure_category == "body_text_contamination"
    assert feedback.recommended_action == "shrink_bbox"


def test_adjusted_region_for_expand_increases_bbox() -> None:
    region = _region()

    adjusted = adjusted_region_for_action(region, "expand_bbox", page_width=200, page_height=300)

    assert adjusted is not None
    assert adjusted.bbox[0] < region.bbox[0]
    assert adjusted.bbox[3] > region.bbox[3]


def test_bbox_candidates_include_body_tail_and_wider_expansion_variants() -> None:
    region = _region()

    shrink_names = {candidate.name for candidate in build_bbox_candidates(region, "shrink_bbox", page_width=200, page_height=300)}
    expand_names = {candidate.name for candidate in build_bbox_candidates(region, "expand_bbox", page_width=200, page_height=300)}

    assert "trim_body_text_tail" in shrink_names
    assert "shrink_to_upper_dense_band" in shrink_names
    assert "expand_down_large" in expand_names
    assert "expand_width_preserve_y" in expand_names


def test_llm_sampling_prefers_failed_unique_regions() -> None:
    report = {
        "pdf_reports": [
            {
                "results": [
                    {
                        "pdf": "a.pdf",
                        "region_id": "r1",
                        "parser": "region_guided",
                        "needs_llm_judge": True,
                        "retry_succeeded": False,
                        "rule_category": "no_parser_output",
                    },
                    {
                        "pdf": "a.pdf",
                        "region_id": "r1",
                        "parser": "pdfplumber",
                        "needs_llm_judge": True,
                        "retry_succeeded": False,
                        "rule_category": "no_parser_output",
                    },
                    {
                        "pdf": "a.pdf",
                        "region_id": "r2",
                        "parser": "region_guided",
                        "needs_llm_judge": True,
                        "retry_succeeded": True,
                        "rule_category": "body_text_contamination",
                    },
                ]
            }
        ]
    }

    selected = table_region_benchmark.select_llm_judge_rows(report, max_regions=2)

    assert [(row["region_id"], row["parser"]) for row in selected] == [
        ("r1", "region_guided"),
        ("r2", "region_guided"),
    ]

    failures_only = table_region_benchmark.select_llm_judge_rows(
        report,
        max_regions=2,
        only_failures=True,
    )
    assert [(row["region_id"], row["parser"]) for row in failures_only] == [
        ("r1", "region_guided"),
        ("r2", "region_guided"),
    ]


def test_llm_sampling_only_failures_backfills_suspicious_categories() -> None:
    report = {
        "pdf_reports": [
            {
                "results": [
                    {
                        "pdf": "a.pdf",
                        "region_id": "r1",
                        "parser": "region_guided",
                        "needs_llm_judge": True,
                        "retry_succeeded": False,
                        "rule_category": "unknown",
                    },
                    {
                        "pdf": "a.pdf",
                        "region_id": "r2",
                        "parser": "region_guided",
                        "needs_llm_judge": False,
                        "retry_succeeded": True,
                        "rule_category": "body_text_contamination",
                    },
                    {
                        "pdf": "a.pdf",
                        "region_id": "r3",
                        "parser": "region_guided",
                        "needs_llm_judge": False,
                        "retry_succeeded": True,
                        "rule_category": "valid_table",
                    },
                ]
            }
        ]
    }

    selected = table_region_benchmark.select_llm_judge_rows(
        report,
        max_regions=3,
        only_failures=True,
    )

    assert [(row["region_id"], row["rule_category"]) for row in selected] == [
        ("r1", "unknown"),
        ("r2", "body_text_contamination"),
    ]


def test_rule_llm_agree_is_true_when_category_and_action_match(monkeypatch) -> None:
    class FakeJudgeResult:
        def to_dict(self) -> dict:
            return {
                "failure_category": "no_parser_output",
                "recommended_action": "retry_text_strategy",
                "confidence": 0.8,
                "reason": "same as rule",
            }

    monkeypatch.setattr(
        table_region_benchmark,
        "judge_table_failure_with_llm",
        lambda **kwargs: FakeJudgeResult(),
    )

    payload = table_region_benchmark.judge_result_row_with_llm(
        row={
            "pdf": "a.pdf",
            "page": 1,
            "region_bbox": [0, 0, 10, 10],
            "parser": "pdfplumber",
            "succeeded": False,
            "rows": 0,
            "cols": 0,
            "rule_category": "no_parser_output",
            "recommended_action": "retry_text_strategy",
            "needs_llm_judge": True,
            "rule_confidence": 0.6,
        },
        text_context={},
        model="deepseek-chat",
    )

    assert payload["agree"] is True
    assert payload["failure_category"] == "no_parser_output"


def test_llm_judge_summary_counts_agreement_and_outputs() -> None:
    report = {
        "pdf_reports": [
            {
                "results": [
                    {
                        "llm_judge_attempted": True,
                        "rule_llm_agree": True,
                        "rule_category": "no_parser_output",
                        "recommended_action": "retry_text_strategy",
                        "llm_failure_category": "no_parser_output",
                        "llm_recommended_action": "retry_text_strategy",
                        "llm_error": "",
                    },
                    {
                        "llm_judge_attempted": True,
                        "rule_llm_agree": False,
                        "rule_category": "unknown",
                        "recommended_action": "needs_llm_judge",
                        "llm_failure_category": "body_text_contamination",
                        "llm_recommended_action": "shrink_bbox",
                        "llm_error": "ValueError: bad schema",
                    },
                    {"llm_judge_attempted": False},
                ]
            }
        ]
    }

    summary = table_region_benchmark.build_llm_judge_summary(report)

    assert summary["attempted_rows"] == 2
    assert summary["agree_count"] == 1
    assert summary["agreement_rate"] == 0.5
    assert summary["errors"] == {"ValueError: bad schema": 1}
    assert summary["llm_actions"]["shrink_bbox"] == 1


def test_auto_decision_allows_safe_actions_by_default() -> None:
    decision = build_auto_decision(
        llm_recommended_action="retry_text_strategy",
        llm_confidence=0.8,
    )

    assert decision["would_retry"] is True
    assert decision["safe_auto_action"] is True
    assert decision["auto_action_allowed"] is True
    assert decision["human_review_required"] is False


def test_auto_decision_marks_drop_candidate_without_deleting() -> None:
    safe = build_auto_decision(
        llm_recommended_action="shrink_bbox",
        llm_confidence=0.8,
    )
    drop = build_auto_decision(
        llm_recommended_action="drop_candidate",
        llm_confidence=0.99,
    )

    assert safe["auto_action_allowed"] is True
    assert safe["would_adjust_bbox"] is True
    assert drop["auto_action_allowed"] is True
    assert drop["human_review_required"] is False
    assert drop["would_drop_candidate"] is True


def test_table_candidate_score_uses_portable_quality_signals() -> None:
    score = score_table_candidate(
        metadata={
            "rows": 8,
            "cols": 4,
            "structural_quality": {
                "consistency_score": 0.9,
                "fill_rate": 0.82,
                "prose_cell_ratio": 0.02,
            },
        },
        bbox=(10, 20, 200, 180),
        region_bbox=(8, 18, 202, 182),
        text_context=RegionTextContext(word_count=40, numeric_tokens=16),
    )

    assert score.tier == "strong"
    assert score.score >= 0.78
    assert score.reasons == ["passed_self_check"]


def test_table_candidate_score_flags_body_text_without_labels() -> None:
    score = score_table_candidate(
        metadata={
            "rows": 5,
            "cols": 2,
            "structural_quality": {
                "consistency_score": 0.45,
                "fill_rate": 0.50,
                "prose_cell_ratio": 0.40,
                "fallback_reasons": ["prose_like_cells"],
            },
        },
        bbox=(10, 20, 300, 260),
        region_bbox=(10, 20, 180, 160),
        text_context=RegionTextContext(word_count=80, long_line_count=3, body_word_hits=10),
    )

    assert score.tier in {"suspicious", "failed"}
    assert "prose_contamination" in score.reasons
    assert "bbox_much_larger_than_region" in score.reasons


def test_compare_table_candidate_scores_reports_direction() -> None:
    original = score_table_candidate(
        metadata={"rows": 3, "cols": 2, "structural_quality": {"consistency_score": 0.45, "fill_rate": 0.4}},
    )
    candidate = score_table_candidate(
        metadata={"rows": 6, "cols": 4, "structural_quality": {"consistency_score": 0.85, "fill_rate": 0.8}},
    )

    comparison = compare_table_candidate_scores(original, candidate)

    assert comparison["verdict"] == "improved"
    assert comparison["score_delta"] > 0


def test_storage_decision_excludes_failed_unparsed_table_without_deleting() -> None:
    decision = build_storage_decision(
        content_type="table",
        metadata={
            "table_candidate_score": 0.2,
            "table_candidate_tier": "failed",
            "table_region_unparsed": True,
            "rows": 0,
            "cols": 0,
            "rule_failure_category": "no_parser_output",
            "rule_recommended_action": "accept",
        },
    )

    assert decision["excluded_from_storage"] is True
    assert decision["storage_decision_source"] == "failed_self_check_no_structure"


def test_storage_decision_keeps_suspicious_repairable_table_candidate() -> None:
    decision = build_storage_decision(
        content_type="table",
        metadata={
            "table_candidate_score": 0.45,
            "table_candidate_tier": "suspicious",
            "table_region_unparsed": False,
            "rows": 4,
            "cols": 3,
            "rule_failure_category": "header_or_span_complexity",
            "rule_recommended_action": "retry_text_strategy",
        },
    )

    assert decision["excluded_from_storage"] is False


def test_storage_decision_excludes_body_heavy_low_column_candidate() -> None:
    decision = build_storage_decision(
        content_type="table",
        metadata={
            "table_candidate_score": 0.69,
            "table_candidate_tier": "usable",
            "table_candidate_reasons": ["text_context_body_heavy"],
            "rows": 5,
            "cols": 2,
            "table_label": "",
            "table_caption": "",
            "rule_failure_category": "ok",
            "rule_recommended_action": "accept",
        },
    )

    assert decision["excluded_from_storage"] is True
    assert decision["storage_decision_source"] == "body_heavy_low_column_candidate"


def test_storage_decision_keeps_captioned_body_heavy_low_column_table() -> None:
    decision = build_storage_decision(
        content_type="table",
        metadata={
            "table_candidate_score": 0.65,
            "table_candidate_tier": "usable",
            "table_candidate_reasons": ["text_context_body_heavy"],
            "rows": 8,
            "cols": 2,
            "table_label": "Table 3",
            "table_caption": "Table 3. Comparison of cell factories",
            "rule_failure_category": "header_or_span_complexity",
            "rule_recommended_action": "retry_text_strategy",
        },
    )

    assert decision["excluded_from_storage"] is False


def test_storage_decision_corrects_legacy_captioned_body_heavy_exclusion() -> None:
    decision = build_storage_decision(
        content_type="table",
        metadata={
            "excluded_from_storage": True,
            "storage_decision_source": "body_heavy_low_column_candidate",
            "table_label": "Table 3",
            "table_candidate_score": 0.65,
            "table_candidate_tier": "usable",
            "table_candidate_reasons": ["text_context_body_heavy"],
            "rows": 8,
            "cols": 2,
        },
    )

    assert decision["excluded_from_storage"] is False
    assert decision["storage_decision_source"] == "captioned_table_overrides_body_heavy_low_column"


def test_table_embedding_text_removes_markdown_noise_and_keeps_caption() -> None:
    text = build_table_embedding_text(
        raw_content=(
            "| strain | titer |\n"
            "| --- | --- |\n"
            "| B1 | 0.37 (cid:8) 0.05 |\n"
            "|  |  |\n"
        ),
        metadata={"table_caption": "Table 1. Shake flask titers"},
    )

    assert "Table 1. Shake flask titers" in text
    assert "---" not in text
    assert "|" not in text
    assert "B1; 0.37 ± 0.05" in text


def test_llm_judge_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(table_region_benchmark, "load_project_env", lambda: None)

    try:
        table_region_benchmark.main(["missing.pdf", "--llm-judge"])
    except SystemExit as exc:
        assert "DEEPSEEK_API_KEY" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_self_check_summary_reports_unlabeled_health_metrics() -> None:
    report = {
        "pdf_reports": [
            {
                "pdf": "a.pdf",
                "caption_scan": {"anchored_count": 1, "broad_count": 1},
                "detect_ms": 120,
                "regions": [{}, {}],
                "parser_summaries": {
                    "pdfplumber": {"duration_ms": 300, "chunks": 2},
                },
                "results": [
                    {
                        "succeeded": True,
                        "candidate_score": 0.82,
                        "candidate_tier": "strong",
                        "candidate_reasons": "passed_self_check",
                        "rule_category": "ok",
                        "recommended_action": "accept",
                    },
                    {
                        "succeeded": False,
                        "candidate_score": 0.25,
                        "candidate_tier": "failed",
                        "candidate_reasons": "no_parser_output | low_consistency",
                        "rule_category": "no_parser_output",
                        "recommended_action": "retry_text_strategy",
                        "needs_llm_judge": True,
                        "retry_attempted": True,
                        "retry_succeeded": True,
                        "retry_score_verdict": "improved",
                    },
                ],
            }
        ]
    }

    summary = table_region_benchmark.build_self_check_summary(report)

    assert summary["rows"] == 2
    assert summary["success_rate"] == 0.5
    assert summary["candidate_tiers"] == {"strong": 1, "failed": 1}
    assert summary["retry_verdicts"] == {"improved": 1}
    assert summary["risk_flags"]["needs_llm_judge"] == 1
    assert summary["zero_region_pdfs"] == 0
    assert summary["slow_pdfs"][0]["pdf"] == "a.pdf"


def test_self_check_summary_reports_zero_region_caption_mismatch() -> None:
    report = {
        "pdf_reports": [
            {
                "pdf": "missing_region.pdf",
                "detect_ms": 10,
                "caption_scan": {"anchored_count": 2, "broad_count": 2},
                "regions": [],
                "parser_summaries": {},
                "results": [],
            }
        ]
    }

    summary = table_region_benchmark.build_self_check_summary(report)

    assert summary["zero_region_pdfs"] == 1
    assert summary["zero_region_with_caption"][0]["pdf"] == "missing_region.pdf"
