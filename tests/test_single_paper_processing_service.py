from __future__ import annotations

import time

from app_modes import (
    MODE_LEGACY_AUTO_STORE,
    MODE_LEGACY_PIPELINE,
    MODE_LEGACY_PREVIEW,
    MODE_MINERU,
    MODE_MINERU_AUTO_STORE,
    MODE_MINERU_FULL,
)
from src.application import single_paper_processing


def test_run_direct_single_parse_dispatches_non_pipeline_modes() -> None:
    calls: list[tuple[str, dict]] = []

    def mineru_preview(_bytes, _paper_id, _filename, **kwargs):
        calls.append(("mineru_preview", kwargs))
        return {"status": "preview"}

    def mineru_ingest(_bytes, _paper_id, _filename, **kwargs):
        calls.append(("mineru_ingest", kwargs))
        return {"status": "stored"}

    def legacy_preview(_bytes):
        calls.append(("legacy_preview", {}))
        return {"status": "legacy"}

    common = {
        "run_mineru_preview": mineru_preview,
        "run_mineru_ingest": mineru_ingest,
        "run_preview": legacy_preview,
    }

    assert single_paper_processing.run_direct_single_parse(
        b"%PDF", "paper-fast", "fast.pdf", MODE_MINERU, **common
    )["status"] == "preview"
    assert single_paper_processing.run_direct_single_parse(
        b"%PDF", "paper-full", "full.pdf", MODE_MINERU_FULL, **common
    )["status"] == "preview"
    assert single_paper_processing.run_direct_single_parse(
        b"%PDF", "paper-store", "store.pdf", MODE_MINERU_AUTO_STORE, **common
    )["status"] == "stored"
    legacy = single_paper_processing.run_direct_single_parse(
        b"%PDF", "paper-legacy", "legacy.pdf", MODE_LEGACY_PREVIEW, **common
    )
    assert legacy["mode"] == "legacy_preview"
    assert single_paper_processing.run_direct_single_parse(
        b"%PDF", "paper-pipeline", "pipeline.pdf", MODE_LEGACY_PIPELINE, **common
    ) is None

    assert calls == [
        ("mineru_preview", {}),
        ("mineru_preview", {"describe_figure_groups": True}),
        ("mineru_ingest", {"describe_figure_groups": True}),
        ("legacy_preview", {}),
    ]


def test_duplicate_reparse_decision_prompts_only_for_existing_manual_parse() -> None:
    decision = single_paper_processing.duplicate_reparse_decision(
        paper_id="paper-1",
        mode=MODE_MINERU,
        existing_count=3,
        force_reparse_enabled=False,
    )

    assert decision["force_key"] == "force_reparse_paper-1"
    assert decision["existing_count"] == 3
    assert decision["should_prompt"] is True
    assert "paper-1" in decision["message"]
    assert "3" in decision["message"]


def test_duplicate_reparse_decision_skips_auto_store_and_forced_reparse() -> None:
    auto_store = single_paper_processing.duplicate_reparse_decision(
        paper_id="paper-1",
        mode=MODE_MINERU_AUTO_STORE,
        existing_count=3,
        force_reparse_enabled=False,
    )
    forced = single_paper_processing.duplicate_reparse_decision(
        paper_id="paper-1",
        mode=MODE_MINERU,
        existing_count=3,
        force_reparse_enabled=True,
    )
    new_paper = single_paper_processing.duplicate_reparse_decision(
        paper_id="paper-1",
        mode=MODE_MINERU,
        existing_count=0,
        force_reparse_enabled=False,
    )

    assert auto_store["should_prompt"] is False
    assert forced["should_prompt"] is False
    assert new_paper["should_prompt"] is False
    assert auto_store["message"] == ""
    assert forced["message"] == ""
    assert new_paper["message"] == ""


def test_single_parse_execution_plan_describes_direct_and_legacy_modes() -> None:
    fast = single_paper_processing.single_parse_execution_plan(
        MODE_MINERU,
        timeout_seconds=600,
        embedding_api_key_env="DASHSCOPE_API_KEY",
    )
    legacy = single_paper_processing.single_parse_execution_plan(
        MODE_LEGACY_AUTO_STORE,
        timeout_seconds=600,
        embedding_api_key_env="DASHSCOPE_API_KEY",
    )

    assert fast == {"kind": "direct", "spinner": "MinerU 快速预览中..."}
    assert legacy["kind"] == "legacy_pipeline"
    assert "历史解析链路运行中" in legacy["timeout_intro"]
    assert "DASHSCOPE_API_KEY" in legacy["timeout_error"]
    assert "历史解析 + 入库准备" in legacy["spinner"]


def test_mark_legacy_pipeline_result_sets_compat_mode() -> None:
    result = {"paper_id": "paper-1"}

    assert single_paper_processing.mark_legacy_pipeline_result(result) is result
    assert result["mode"] == "legacy_pipeline"


def test_finalize_single_parse_result_sets_mode_and_elapsed() -> None:
    legacy_auto = single_paper_processing.finalize_single_parse_result(
        {"mode": "legacy_pipeline"},
        mode=MODE_LEGACY_AUTO_STORE,
        elapsed=1.2,
    )
    mineru_auto = single_paper_processing.finalize_single_parse_result(
        {},
        mode=MODE_MINERU_AUTO_STORE,
        elapsed=2.3,
    )
    pipeline = single_paper_processing.finalize_single_parse_result(
        {},
        mode=MODE_LEGACY_PIPELINE,
        elapsed=3.4,
    )

    assert legacy_auto["mode"] == "auto"
    assert legacy_auto["_elapsed"] == 1.2
    assert mineru_auto["mode"] == "mineru_auto_store"
    assert pipeline["mode"] == "legacy_pipeline"


def test_store_legacy_auto_result_stores_and_saves_snapshot() -> None:
    saved: list[tuple[dict, str]] = []
    result = {"paper_id": "paper-1"}

    store_result = single_paper_processing.store_legacy_auto_result(
        result,
        "paper.pdf",
        store_runner=lambda parsed: {"stored": 1, "paper_id": parsed["paper_id"]},
        snapshot_builder=lambda parsed, filename: {"snapshot": parsed, "filename": filename},
        result_saver=lambda snapshot, filename: saved.append((snapshot, filename)),
    )

    assert store_result == {"stored": 1, "paper_id": "paper-1"}
    assert saved == [({"snapshot": result, "filename": "paper.pdf"}, "paper.pdf")]


def test_execute_single_parse_runs_direct_path_and_finalizes_result() -> None:
    ticks = iter([10.0, 12.5])

    state = single_paper_processing.execute_single_parse(
        b"%PDF",
        "paper-1",
        "paper.pdf",
        MODE_MINERU,
        timeout_seconds=600,
        embedding_api_key_env="DASHSCOPE_API_KEY",
        run_mineru_preview=lambda *_args, **_kwargs: {"paper_id": "paper-1"},
        run_mineru_ingest=lambda *_args, **_kwargs: {"unused": True},
        run_preview=lambda *_args, **_kwargs: {"unused": True},
        run_pipeline=lambda *_args, **_kwargs: {"unused": True},
        clock=lambda: next(ticks),
    )

    assert state["status"] == "ok"
    assert state["elapsed"] == 2.5
    assert state["result"]["paper_id"] == "paper-1"
    assert state["result"]["_elapsed"] == 2.5


def test_execute_single_parse_runs_legacy_pipeline_and_marks_mode() -> None:
    ticks = iter([1.0, 4.0])

    state = single_paper_processing.execute_single_parse(
        b"%PDF",
        "paper-1",
        "paper.pdf",
        MODE_LEGACY_PIPELINE,
        timeout_seconds=600,
        embedding_api_key_env="DASHSCOPE_API_KEY",
        run_mineru_preview=lambda *_args, **_kwargs: {"unused": True},
        run_mineru_ingest=lambda *_args, **_kwargs: {"unused": True},
        run_preview=lambda *_args, **_kwargs: {"unused": True},
        run_pipeline=lambda *_args, **_kwargs: {"paper_id": "paper-1"},
        clock=lambda: next(ticks),
    )

    assert state["status"] == "ok"
    assert state["elapsed"] == 3.0
    assert state["result"]["mode"] == "legacy_pipeline"


def test_execute_single_parse_reports_legacy_timeout() -> None:
    ticks = iter([1.0, 2.0])

    def slow_pipeline(*_args, **_kwargs):
        time.sleep(0.05)
        return {"paper_id": "paper-1"}

    state = single_paper_processing.execute_single_parse(
        b"%PDF",
        "paper-1",
        "paper.pdf",
        MODE_LEGACY_PIPELINE,
        timeout_seconds=0.001,
        embedding_api_key_env="DASHSCOPE_API_KEY",
        run_mineru_preview=lambda *_args, **_kwargs: {"unused": True},
        run_mineru_ingest=lambda *_args, **_kwargs: {"unused": True},
        run_preview=lambda *_args, **_kwargs: {"unused": True},
        run_pipeline=slow_pipeline,
        clock=lambda: next(ticks),
    )

    assert state["status"] == "timeout"
    assert state["elapsed"] == 1.0
    assert state["result"] is None


def test_auto_store_feedback_handles_legacy_and_mineru_results() -> None:
    legacy = single_paper_processing.auto_store_feedback(
        mode=MODE_LEGACY_AUTO_STORE,
        paper_id="paper-1",
        store_result={"stored": 2},
    )
    mineru_success = single_paper_processing.auto_store_feedback(
        mode=MODE_MINERU_AUTO_STORE,
        paper_id="paper-1",
        result={"store_result": {"stored": 2, "degraded": 1}},
    )
    mineru_duplicate = single_paper_processing.auto_store_feedback(
        mode=MODE_MINERU_AUTO_STORE,
        paper_id="paper-1",
        result={"store_result": {"duplicate": True, "existing_count": 3}},
    )
    mineru_error = single_paper_processing.auto_store_feedback(
        mode=MODE_MINERU_AUTO_STORE,
        paper_id="paper-1",
        result={"store_result": {"error": "boom"}},
    )

    assert legacy["stored_key"] == "stored_paper-1"
    assert legacy["level"] == "success"
    assert legacy["store_result"] == {"stored": 2}
    assert mineru_success["level"] == "success"
    assert "stored=2" in mineru_success["message"]
    assert mineru_duplicate["level"] == "warning"
    assert "3" in mineru_duplicate["message"]
    assert mineru_error["level"] == "error"
    assert "boom" in mineru_error["message"]
