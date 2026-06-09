from __future__ import annotations

from src.application.table_debug import table_storage_view_state


def test_table_storage_view_state_marks_usable_table() -> None:
    state = table_storage_view_state(
        content_type="table",
        metadata={"rows": 3, "cols": 2},
    )

    assert state["excluded_from_storage"] is False
    assert state["storage_exclusion_reason"] == ""
    assert state["parsed"] is True
    assert state["usable_table"] is True
    assert state["table_status"] == "可用表格"


def test_table_storage_view_state_marks_unparsed_candidate() -> None:
    state = table_storage_view_state(
        content_type="table",
        metadata={"rows": 3, "cols": 2, "table_region_unparsed": True},
    )

    assert state["excluded_from_storage"] is False
    assert state["parsed"] is False
    assert state["usable_table"] is False
    assert state["table_status"] == "未解析候选"


def test_table_storage_view_state_forces_false_positive_exclusion() -> None:
    state = table_storage_view_state(
        content_type="table",
        metadata={"rows": 3, "cols": 2},
        issue_type="false_positive",
    )

    assert state["excluded_from_storage"] is True
    assert state["usable_table"] is False
    assert state["table_status"] == "不入库候选"
