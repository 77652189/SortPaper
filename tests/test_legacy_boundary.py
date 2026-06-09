from __future__ import annotations


def test_app_pipeline_legacy_preview_wrapper(monkeypatch) -> None:
    import app_pipeline
    import src.legacy

    calls: list[bytes] = []

    def fake_preview(pdf_bytes: bytes) -> dict:
        calls.append(pdf_bytes)
        return {"mode": "preview", "status": "ok"}

    monkeypatch.setattr(src.legacy, "run_legacy_preview", fake_preview)

    assert app_pipeline.run_preview(b"%PDF") == {"mode": "preview", "status": "ok"}
    assert calls == [b"%PDF"]


def test_app_pipeline_legacy_pipeline_wrapper(monkeypatch) -> None:
    import app_pipeline
    import src.legacy

    calls: list[tuple[bytes, str, str]] = []

    def fake_pipeline(pdf_bytes: bytes, paper_id: str, filename: str = "") -> dict:
        calls.append((pdf_bytes, paper_id, filename))
        return {"mode": "pipeline", "status": "done"}

    monkeypatch.setattr(src.legacy, "run_legacy_pipeline", fake_pipeline)

    assert app_pipeline.run_pipeline(b"%PDF", "paper-1", "paper.pdf") == {
        "mode": "pipeline",
        "status": "done",
    }
    assert calls == [(b"%PDF", "paper-1", "paper.pdf")]
