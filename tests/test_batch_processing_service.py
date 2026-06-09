from __future__ import annotations

from app_modes import MODE_LEGACY_PIPELINE, MODE_MINERU, MODE_MINERU_AUTO_STORE, MODE_MINERU_FULL
from src.application import batch_processing


def test_precheck_batch_file_prepares_first_file_and_skips_batch_duplicate() -> None:
    seen: dict[str, dict] = {}

    first = batch_processing.precheck_batch_file(
        no=1,
        filename="a.pdf",
        file_bytes=b"a",
        mode=MODE_MINERU,
        seen_paper_ids=seen,
        paper_id_builder=lambda _bytes: "paper-1",
        existing_reason_lookup=lambda _paper_id: "",
    )
    duplicate = batch_processing.precheck_batch_file(
        no=2,
        filename="b.pdf",
        file_bytes=b"b",
        mode=MODE_MINERU,
        seen_paper_ids=seen,
        paper_id_builder=lambda _bytes: "paper-1",
        existing_reason_lookup=lambda _paper_id: "",
    )

    assert first["kind"] == "prepared"
    assert first["job"]["paper_id"] == "paper-1"
    assert duplicate["kind"] == "skip_batch_duplicate"
    assert duplicate["row"]["duplicate_of"] == "1. a.pdf"


def test_precheck_batch_file_skips_existing_only_for_auto_store_mode() -> None:
    existing = batch_processing.precheck_batch_file(
        no=1,
        filename="a.pdf",
        file_bytes=b"a",
        mode=MODE_MINERU_AUTO_STORE,
        seen_paper_ids={},
        paper_id_builder=lambda _bytes: "paper-1",
        existing_reason_lookup=lambda _paper_id: "已入库 2 条记录",
    )
    manual = batch_processing.precheck_batch_file(
        no=1,
        filename="a.pdf",
        file_bytes=b"a",
        mode=MODE_MINERU,
        seen_paper_ids={},
        paper_id_builder=lambda _bytes: "paper-1",
        existing_reason_lookup=lambda _paper_id: "已入库 2 条记录",
    )

    assert existing["kind"] == "skip_existing"
    assert existing["row"]["skip_type"] == "已入库"
    assert manual["kind"] == "prepared"


def test_precheck_failure_and_embedding_failure_rows_are_stable() -> None:
    row = batch_processing.precheck_failure_row(1, "bad.pdf", RuntimeError("broken"))
    failures = batch_processing.embedding_preflight_failure_rows(
        [{"no": 2, "file": "a.pdf", "paper_id": "paper-1"}],
        "missing key",
    )

    assert row["status"] == "fail"
    assert "预检查失败" in row["reason"]
    assert failures == [
        {"no": 2, "file": "a.pdf", "paper_id": "paper-1", "status": "fail", "reason": "missing key"}
    ]


def test_completed_batch_result_builds_row_and_preview_entry() -> None:
    result = batch_processing.completed_batch_result(
        {"no": 1, "file": "a.pdf", "paper_id": "paper-1"},
        {
            "status": "ok",
            "category": "mineru_stored",
            "chunks": 3,
            "stored": 2,
            "parse_seconds": 0.05,
            "store_seconds": 0.3,
            "total_seconds": 0.4,
            "preview_result": {"paper_id": "paper-1"},
        },
    )

    assert result["counter"] == "success"
    assert result["preview_entry"] == {"no": 1, "file": "a.pdf", "result": {"paper_id": "paper-1"}}
    assert result["row"]["parse"] == "<0.1s"
    assert result["row"]["store"] == "0.3s"
    assert result["row"]["category"] == "mineru_stored"


def test_process_batch_job_mineru_modes_use_expected_figure_vision() -> None:
    calls: list[tuple[str, dict]] = []

    def preview(_bytes, _paper_id, _filename, **kwargs):
        calls.append(("preview", kwargs))
        return {"merged_chunks": [], "worker_timing": {"total": 1.0}}

    def ingest(_bytes, _paper_id, _filename, **kwargs):
        calls.append(("ingest", kwargs))
        return {
            "merged_chunks": [],
            "store_result": {"stored": 1},
            "worker_timing": {"total": 2.0, "store": 0.5},
        }

    common = {
        "run_preview": lambda *_args, **_kwargs: {},
        "run_pipeline": lambda *_args, **_kwargs: {},
        "run_mineru_preview": preview,
        "run_mineru_ingest": ingest,
        "fallback_runner": lambda *_args: {},
        "snapshot_builder": lambda result, filename: result,
        "result_saver": lambda snapshot, filename: None,
    }

    fast = batch_processing.process_batch_job(b"%PDF", "paper-fast", "fast.pdf", MODE_MINERU, **common)
    full = batch_processing.process_batch_job(b"%PDF", "paper-full", "full.pdf", MODE_MINERU_FULL, **common)
    stored = batch_processing.process_batch_job(
        b"%PDF",
        "paper-store",
        "store.pdf",
        MODE_MINERU_AUTO_STORE,
        **common,
    )

    assert calls[0] == ("preview", {})
    assert calls[1] == ("preview", {"describe_figure_groups": True})
    assert calls[2] == ("ingest", {"describe_figure_groups": True})
    assert fast["category"] == "mineru_preview"
    assert full["category"] == "mineru_full"
    assert stored["category"] == "mineru_stored"
    assert stored["stored"] == 1


def test_process_batch_job_legacy_pipeline_saves_snapshot() -> None:
    saved: list[tuple[dict, str]] = []

    def pipeline(_bytes, paper_id, filename):
        return {"paper_id": paper_id, "filename": filename, "merged_chunks": [{"chunk_id": "1"}]}

    result = batch_processing.process_batch_job(
        b"%PDF",
        "paper-1",
        "paper.pdf",
        MODE_LEGACY_PIPELINE,
        run_preview=lambda *_args, **_kwargs: {},
        run_pipeline=pipeline,
        run_mineru_preview=lambda *_args, **_kwargs: {},
        run_mineru_ingest=lambda *_args, **_kwargs: {},
        fallback_runner=lambda *_args: {},
        snapshot_builder=lambda parsed, filename: {"snapshot": parsed, "filename": filename},
        result_saver=lambda snapshot, filename: saved.append((snapshot, filename)),
    )

    assert result["category"] == "legacy_parsed"
    assert result["chunks"] == 1
    assert result["preview_result"]["mode"] == "legacy_pipeline"
    assert saved[0][0]["filename"] == "paper.pdf"
    assert saved[0][1] == "paper.pdf"


def test_format_duration_handles_skipped_and_subsecond_values() -> None:
    assert batch_processing.format_duration(0.05) == "<0.1s"
    assert batch_processing.format_duration(1.234) == "1.2s"
    assert batch_processing.format_duration(10, skipped=True) == "未执行"
