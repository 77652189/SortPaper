from __future__ import annotations

from src.domain.table_storage_policy import build_storage_decision


def test_table_storage_policy_excludes_drop_candidate() -> None:
    decision = build_storage_decision(
        content_type="table",
        metadata={"llm_recommended_action": "drop_candidate"},
    )

    assert decision["excluded_from_storage"] is True
    assert decision["storage_decision_source"] == "judge_drop_candidate"


def test_table_storage_policy_allows_non_table() -> None:
    decision = build_storage_decision(content_type="text", metadata={})

    assert decision == {
        "excluded_from_storage": False,
        "storage_exclusion_reason": "",
        "storage_decision_source": "not_table",
    }


def test_table_storage_policy_corrects_captioned_legacy_body_heavy_exclusion() -> None:
    decision = build_storage_decision(
        content_type="table",
        metadata={
            "excluded_from_storage": True,
            "storage_decision_source": "body_heavy_low_column_candidate",
            "table_caption": "Table 1. Fermentation results",
        },
    )

    assert decision["excluded_from_storage"] is False
    assert decision["storage_decision_source"] == "captioned_table_overrides_body_heavy_low_column"
