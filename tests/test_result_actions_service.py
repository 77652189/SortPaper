from __future__ import annotations

from src.application import result_actions


def test_resolve_storage_state_uses_session_state_before_qdrant_count() -> None:
    state = result_actions.resolve_storage_state(
        {"paper_id": "paper-1"},
        session_stored_state={"stored": 2},
        paper_count_lookup=lambda _paper_id: 0,
    )

    assert state["stored_key"] == "stored_paper-1"
    assert state["is_stored"] is True
    assert state["qdrant_count"] == 0


def test_resolve_storage_state_uses_qdrant_count_and_tolerates_lookup_error() -> None:
    counted = result_actions.resolve_storage_state(
        {"paper_id": "paper-1"},
        session_stored_state=None,
        paper_count_lookup=lambda _paper_id: 2,
    )

    def broken_lookup(_paper_id: str) -> int:
        raise RuntimeError("qdrant unavailable")

    fallback = result_actions.resolve_storage_state(
        {"paper_id": "paper-1"},
        session_stored_state=None,
        paper_count_lookup=broken_lookup,
    )

    assert counted["is_stored"] is True
    assert counted["qdrant_count"] == 2
    assert fallback["is_stored"] is False
    assert fallback["qdrant_count"] == 0


def test_save_result_snapshot_returns_pending_when_file_exists() -> None:
    saved: list[tuple[dict, str]] = []
    result = {"paper_id": "paper-1"}

    state = result_actions.save_result_snapshot(
        result,
        "paper.pdf",
        snapshot_builder=lambda parsed, filename: {"snapshot": parsed, "filename": filename},
        result_saver=lambda snapshot, filename: saved.append((snapshot, filename)) or "",
    )

    assert state["saved"] is False
    assert state["pending_save_filename"] == "paper.pdf"
    assert state["success_message"] == ""
    assert saved == [({"snapshot": result, "filename": "paper.pdf"}, "paper.pdf")]


def test_save_and_overwrite_result_snapshot_success_messages() -> None:
    result = {"paper_id": "paper-1"}

    saved_state = result_actions.save_result_snapshot(
        result,
        "paper.pdf",
        snapshot_builder=lambda parsed, filename: {"snapshot": parsed, "filename": filename},
        result_saver=lambda _snapshot, _filename: "saved/path/paper.json",
    )
    overwritten_state = result_actions.overwrite_result_snapshot(
        result,
        "paper.pdf",
        snapshot_builder=lambda parsed, filename: {"snapshot": parsed, "filename": filename},
        result_overwriter=lambda _snapshot, _filename: None,
    )

    assert saved_state["saved"] is True
    assert saved_state["pending_save_filename"] == ""
    assert "saved/path/paper.json" in saved_state["success_message"]
    assert overwritten_state["pending_save_filename"] == ""
    assert "paper.pdf" in overwritten_state["success_message"]


def test_store_result_chunks_uses_injected_store_runner() -> None:
    result = {"paper_id": "paper-1"}

    state = result_actions.store_result_chunks(
        result,
        store_runner=lambda parsed: {"stored": 2, "paper_id": parsed["paper_id"]},
    )

    assert state == {"stored": 2, "paper_id": "paper-1"}


def test_enrich_result_quality_uses_paper_id_and_updates_result() -> None:
    result = {"paper_id": "paper-1", "quality": {}}
    calls: list[str] = []

    state = result_actions.enrich_result_quality(
        result,
        enrich_runner=lambda paper_id: calls.append(paper_id) or {"category": "high"},
    )

    assert calls == ["paper-1"]
    assert state["quality"] == {"category": "high"}
    assert state["result"] is result
    assert result["quality"] == {"category": "high"}


def test_quality_helpers_identify_pipeline_and_apply_quality() -> None:
    result = {"mode": "legacy_pipeline", "quality": {}}

    assert result_actions.is_pipeline_result(result) is True
    assert result_actions.has_quality(result) is False
    assert result_actions.apply_quality_result(result, {"category": "high"}) is result
    assert result_actions.has_quality(result) is True
