from __future__ import annotations

from src.application import vector_library


def test_paper_count_uses_repository() -> None:
    calls: list[str] = []

    class Repository:
        def paper_count(self, paper_id: str) -> int:
            calls.append(paper_id)
            return 3

    assert vector_library.paper_count("paper-1", repository=Repository()) == 3
    assert calls == ["paper-1"]


def test_embedding_preflight_error_reports_missing_key(monkeypatch) -> None:
    monkeypatch.setattr(vector_library, "EMBEDDING_PROVIDER", "dashscope")
    monkeypatch.setattr(vector_library, "EMBEDDING_API_KEY_ENV", "MISSING_KEY")
    monkeypatch.delenv("MISSING_KEY", raising=False)

    assert vector_library.embedding_preflight_error() == "缺少 MISSING_KEY，无法生成向量入库"


def test_vector_library_wraps_common_repository_operations() -> None:
    calls: list[str] = []

    class Repository:
        count = 7

        def list_papers(self):
            calls.append("list")
            return [{"paper_id": "paper-1"}]

        def delete_by_paper_id(self, paper_id):
            calls.append(f"delete:{paper_id}")
            return 2

        def clear(self):
            calls.append("clear")

    repo = Repository()

    assert vector_library.total_count(repository=repo) == 7
    assert vector_library.list_papers(repository=repo) == [{"paper_id": "paper-1"}]
    assert vector_library.delete_paper("paper-1", repository=repo) == 2
    assert vector_library.clear_library(repository=repo) == 7
    assert calls == ["list", "delete:paper-1", "clear"]


def test_enrich_paper_uses_injected_runner() -> None:
    calls: list[str] = []

    result = vector_library.enrich_paper(
        "paper-1",
        enrich_runner=lambda paper_id: calls.append(paper_id) or {"category": "ok"},
    )

    assert calls == ["paper-1"]
    assert result == {"category": "ok"}


def test_embedding_preflight_error_reports_embed_failure(monkeypatch) -> None:
    monkeypatch.setattr(vector_library, "EMBEDDING_PROVIDER", "qwen3_local")

    class Repository:
        def embed(self, _text: str):
            raise RuntimeError("boom")

    error = vector_library.embedding_preflight_error(repository=Repository())

    assert error.startswith("Embedding 预检失败：RuntimeError: boom")


def test_enrich_pending_papers_tracks_success_failures_and_progress() -> None:
    papers = [
        {"paper_id": "ok", "paper_title": "OK", "enrichment_status": "pending"},
        {"paper_id": "err", "paper_title": "ERR", "enrichment_status": "pending"},
        {"paper_id": "boom", "paper_title": "BOOM", "enrichment_status": "pending"},
        {"paper_id": "done", "paper_title": "DONE", "enrichment_status": "done"},
    ]
    progress: list[tuple[int, int, str, int, int]] = []

    def enrich_runner(paper_id: str) -> dict:
        if paper_id == "err":
            return {"error": "failed"}
        if paper_id == "boom":
            raise RuntimeError("raised")
        return {"category": "ok"}

    result = vector_library.enrich_pending_papers(
        papers,
        enrich_runner=enrich_runner,
        progress_callback=lambda index, total, paper, success, fail: progress.append(
            (index, total, paper["paper_id"], success, fail)
        ),
    )

    assert result["total"] == 3
    assert result["success_n"] == 1
    assert result["fail_n"] == 2
    assert result["failures"] == ["ERR: failed", "BOOM: raised"]
    assert progress == [
        (1, 3, "ok", 1, 0),
        (2, 3, "err", 1, 1),
        (3, 3, "boom", 1, 2),
    ]


def test_enrich_pending_papers_uses_default_quality_enricher(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        vector_library,
        "default_quality_enricher",
        lambda paper_id: calls.append(paper_id) or {"category": "ok"},
    )

    result = vector_library.enrich_pending_papers([
        {"paper_id": "paper-1", "paper_title": "Paper", "enrichment_status": "pending"},
    ])

    assert calls == ["paper-1"]
    assert result["success_n"] == 1
    assert result["fail_n"] == 0
