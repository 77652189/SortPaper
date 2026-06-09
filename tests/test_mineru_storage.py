from __future__ import annotations

from types import SimpleNamespace


class FakeClient:
    def count(self, **_kwargs):
        return SimpleNamespace(count=0)


class FakeStore:
    collection = "papers"
    added: list[dict] = []

    def __init__(self) -> None:
        self.client = FakeClient()

    def add(self, **kwargs) -> None:
        self.added.append(kwargs)

    def save(self, _path: str) -> None:
        return None


def test_default_mineru_verdicts_allow_storage(monkeypatch) -> None:
    import src.store.qdrant_store as qdrant_store
    from app_pipeline import _default_mineru_verdicts, store_parsed_chunks

    FakeStore.added = []
    monkeypatch.setattr(qdrant_store, "QdrantStore", FakeStore)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    chunks = [
        {
            "chunk_id": "text-1",
            "content_type": "text",
            "raw_content": "LNT biosynthesis text.",
            "page": 1,
            "bbox": [1, 2, 3, 4],
            "metadata": {"parser": "mineru_vlm", "mineru_type": "text"},
        },
        {
            "chunk_id": "image-1",
            "content_type": "image",
            "raw_content": "[Vision reparse needed]",
            "page": 2,
            "bbox": [10, 20, 200, 260],
            "metadata": {
                "parser": "mineru_vlm",
                "mineru_type": "chart",
                "figure_caption": "Figure 2. LC-MS analysis of LNT biosynthesis.",
                "mineru_visual_content": "LC-MS peaks for LNT II and LNT.",
                "vision_group_description": "Grouped VL description of Figure 2 panels.",
            },
        },
    ]
    result = {
        "paper_id": "paper-mineru",
        "filename": "mineru.pdf",
        "mode": "mineru_preview",
        "merged_chunks": chunks,
        "verdicts": _default_mineru_verdicts(chunks),
    }

    stored = store_parsed_chunks(result)

    assert stored["stored"] == 2
    assert stored["missing_verdicts"] == 0
    assert len(FakeStore.added) == 2
    image_add = FakeStore.added[1]
    assert image_add["content"] == "[Vision reparse needed]"
    assert "Grouped VL description" in image_add["embed_content"]
    assert "LC-MS analysis" in image_add["embed_content"]
    assert "LC-MS peaks" in image_add["embed_content"]


def test_default_mineru_verdicts_are_clean_and_pending() -> None:
    from app_pipeline import _default_mineru_verdicts

    verdicts = _default_mineru_verdicts([
        {"chunk_id": "chunk-1", "metadata": {"parser": "mineru_vlm"}},
        {"chunk_id": "", "metadata": {"parser": "mineru_vlm"}},
    ])

    assert list(verdicts) == ["chunk-1"]
    assert verdicts["chunk-1"]["passed"] is True
    assert verdicts["chunk-1"]["score"] == 1.0
    assert verdicts["chunk-1"]["issue_type"] == "mineru_parser_trusted"
    assert verdicts["chunk-1"]["judge_scope"] == "mineru_default"


def test_run_mineru_ingest_reuses_preview_and_store(monkeypatch) -> None:
    import app_pipeline

    preview_result = {
        "paper_id": "paper-mineru",
        "filename": "mineru.pdf",
        "mode": "mineru_preview",
        "status": "mineru_preview",
        "merged_chunks": [{"chunk_id": "text-1", "content_type": "text", "raw_content": "text"}],
        "worker_timing": {"mineru_api": 0.1, "adapter": 0.2, "total": 0.3},
        "mineru": {"summary": {"chunks": 1}},
    }
    calls: list[tuple] = []

    def fake_preview(*args, **kwargs):
        calls.append(("preview", args, kwargs))
        return dict(preview_result, worker_timing=dict(preview_result["worker_timing"]), mineru=dict(preview_result["mineru"]))

    def fake_store(result):
        calls.append(("store", result, {}))
        return {"stored": 1, "degraded": 0, "failed": 0, "attempted": 1, "error": ""}

    monkeypatch.setattr(app_pipeline, "run_mineru_preview", fake_preview)
    monkeypatch.setattr(app_pipeline, "store_parsed_chunks", fake_store)

    result = app_pipeline.run_mineru_ingest(
        b"%PDF",
        "paper-mineru",
        "mineru.pdf",
        model_version="vlm",
        describe_figure_groups=True,
        vision_model="vision-test",
        force_refresh=True,
    )

    assert calls[0][0] == "preview"
    assert calls[0][2]["model_version"] == "vlm"
    assert calls[0][2]["describe_figure_groups"] is True
    assert calls[0][2]["vision_model"] == "vision-test"
    assert calls[0][2]["force_refresh"] is True
    assert calls[1][0] == "store"
    assert result["mode"] == "mineru_auto_store"
    assert result["status"] == "mineru_stored"
    assert result["store_result"]["stored"] == 1
    assert result["mineru"]["store_result"]["stored"] == 1
    assert "store" in result["worker_timing"]


def test_app_modes_keep_legacy_aliases() -> None:
    from app_modes import (
        MODE_LEGACY_AUTO_STORE,
        MODE_LEGACY_PIPELINE,
        MODE_LEGACY_PREVIEW,
        MODE_MINERU,
        MODE_MINERU_AUTO_STORE,
        MODE_MINERU_FULL,
        LEGACY_MODES,
        PRIMARY_MODES,
        is_auto_store_mode,
        is_mineru_auto_store_mode,
        is_mineru_full_mode,
        is_mineru_mode,
        normalize_mode,
    )

    assert PRIMARY_MODES == (MODE_MINERU, MODE_MINERU_FULL, MODE_MINERU_AUTO_STORE)
    assert LEGACY_MODES == (MODE_LEGACY_PREVIEW, MODE_LEGACY_PIPELINE, MODE_LEGACY_AUTO_STORE)
    assert normalize_mode("\u5feb\u901f\u9884\u89c8") == MODE_LEGACY_PREVIEW
    assert normalize_mode("\u5b8c\u6574\u6d41\u6c34\u7ebf") == MODE_LEGACY_PIPELINE
    assert normalize_mode("\u4e00\u952e\u5165\u5e93") == MODE_LEGACY_AUTO_STORE
    assert is_mineru_mode(MODE_MINERU)
    assert is_mineru_full_mode(MODE_MINERU_FULL)
    assert is_mineru_auto_store_mode(MODE_MINERU_AUTO_STORE)
    assert is_auto_store_mode(MODE_MINERU_AUTO_STORE)
    assert is_auto_store_mode(MODE_LEGACY_AUTO_STORE)


def test_batch_mineru_modes_use_expected_figure_vision(monkeypatch) -> None:
    import app
    from app_modes import MODE_MINERU, MODE_MINERU_AUTO_STORE, MODE_MINERU_FULL

    calls: list[tuple[str, dict]] = []

    def fake_preview(*args, **kwargs):
        calls.append(("preview", kwargs))
        figure_seconds = 3.0 if kwargs.get("describe_figure_groups") else 0.0
        mode = "mineru_full" if kwargs.get("describe_figure_groups") else "mineru_preview"
        return {
            "mode": mode,
            "status": mode,
            "merged_chunks": [],
            "worker_timing": {
                "mineru_api": 1.0,
                "adapter": 2.0,
                "figure_vision": figure_seconds,
                "total": 3.0 + figure_seconds,
            },
        }

    def fake_ingest(*args, **kwargs):
        calls.append(("ingest", kwargs))
        return {
            "mode": "mineru_auto_store",
            "status": "mineru_stored",
            "merged_chunks": [],
            "worker_timing": {
                "mineru_api": 1.0,
                "adapter": 2.0,
                "figure_vision": 3.0,
                "store": 4.0,
                "total": 10.0,
            },
            "store_result": {"stored": 0, "degraded": 0, "failed": 0, "attempted": 0, "excluded": 0},
        }

    monkeypatch.setattr(app, "run_mineru_preview", fake_preview)
    monkeypatch.setattr(app, "run_mineru_ingest", fake_ingest)

    fast = app._process_batch_job(b"%PDF", "paper-fast", "fast.pdf", MODE_MINERU)
    full = app._process_batch_job(b"%PDF", "paper-full", "full.pdf", MODE_MINERU_FULL)
    stored = app._process_batch_job(b"%PDF", "paper-store", "store.pdf", MODE_MINERU_AUTO_STORE)

    assert calls[0] == ("preview", {})
    assert calls[1] == ("preview", {"describe_figure_groups": True})
    assert calls[2] == ("ingest", {"describe_figure_groups": True})
    assert fast["figure_vision_seconds"] == 0.0
    assert full["figure_vision_seconds"] == 3.0
    assert stored["figure_vision_seconds"] == 3.0
    assert stored["store_seconds"] == 4.0
    assert stored["parse_seconds"] == 6.0


def test_mineru_batch_timing_preserves_subsecond_values() -> None:
    import app

    fields = app._mineru_batch_timing_fields(
        {
            "worker_timing": {
                "mineru_api": 21.6,
                "adapter": 0.034,
                "figure_vision": 0.049,
                "store": 0.041,
                "total": 21.724,
            }
        },
        fallback_elapsed=99,
    )

    assert fields["adapter_seconds"] == 0.034
    assert fields["figure_vision_seconds"] == 0.049
    assert fields["store_seconds"] == 0.041
    assert fields["parse_seconds"] == 21.683
    assert app._format_duration(fields["adapter_seconds"]) == "<0.1s"
    assert app._format_duration(0, skipped=True) == "未执行"
