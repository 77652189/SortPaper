from __future__ import annotations

from src.domain.table_candidate_policy import compare_table_candidate_scores, score_table_candidate


def test_candidate_policy_scores_strong_structural_table() -> None:
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
        text_context={"word_count": 40, "numeric_tokens": 16},
    )

    assert score.tier == "strong"
    assert score.score >= 0.78
    assert score.reasons == ["passed_self_check"]


def test_candidate_policy_flags_body_text_and_reports_improvement() -> None:
    original = score_table_candidate(
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
        text_context={"word_count": 80, "long_line_count": 3, "body_word_hits": 10},
    )
    candidate = score_table_candidate(
        metadata={"rows": 6, "cols": 4, "structural_quality": {"consistency_score": 0.85, "fill_rate": 0.8}},
    )

    comparison = compare_table_candidate_scores(original, candidate)

    assert original.tier in {"suspicious", "failed"}
    assert "prose_contamination" in original.reasons
    assert "bbox_much_larger_than_region" in original.reasons
    assert comparison["verdict"] == "improved"
