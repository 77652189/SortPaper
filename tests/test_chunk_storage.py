from __future__ import annotations

from types import SimpleNamespace


class FakeClient:
    def __init__(self, existing_count: int = 0) -> None:
        self.existing_count = existing_count

    def count(self, **_kwargs):
        return SimpleNamespace(count=self.existing_count)


class FakeStore:
    collection = "papers"
    added: list[dict] = []
    existing_count = 0

    def __init__(self) -> None:
        self.client = FakeClient(self.existing_count)

    def add(self, **kwargs) -> None:
        self.added.append(kwargs)

    def save(self, _path: str) -> None:
        return None


def _patch_store(monkeypatch, existing_count: int = 0):
    import src.store.qdrant_store as qdrant_store

    FakeStore.added = []
    FakeStore.existing_count = existing_count
    monkeypatch.setattr(qdrant_store, "QdrantStore", FakeStore)
    return FakeStore


def _store(result: dict) -> dict:
    from src.store.chunk_storage import store_parsed_result_chunks

    return store_parsed_result_chunks(
        result,
        embedding_api_key_env="TEST_EMBEDDING_KEY",
        embed_content_for_chunk=lambda chunk: f"embed::{chunk.get('raw_content', '')}",
    )


def test_chunk_storage_stores_clean_chunk(monkeypatch) -> None:
    fake_store = _patch_store(monkeypatch)
    monkeypatch.setenv("TEST_EMBEDDING_KEY", "test-key")

    result = {
        "paper_id": "paper-1",
        "filename": "demo.pdf",
        "verdicts": {"text-1": {"passed": True, "score": 0.95}},
        "merged_chunks": [{
            "chunk_id": "text-1",
            "content_type": "text",
            "raw_content": "text body",
            "page": 1,
            "bbox": [0, 0, 1, 1],
            "metadata": {},
        }],
    }

    stored = _store(result)

    assert stored["stored"] == 1
    assert stored["degraded"] == 0
    assert fake_store.added[0]["content"] == "text body"
    assert fake_store.added[0]["embed_content"] == "embed::text body"
    assert fake_store.added[0]["metadata"]["paper_title"] == "demo.pdf"
    assert fake_store.added[0]["metadata"]["enrichment_status"] == "pending"


def test_chunk_storage_result_and_metadata_contract(monkeypatch) -> None:
    fake_store = _patch_store(monkeypatch)
    monkeypatch.setenv("TEST_EMBEDDING_KEY", "test-key")

    result = {
        "paper_id": "paper-1",
        "filename": "demo.pdf",
        "verdicts": {"text-1": {"passed": True, "score": 0.95}},
        "merged_chunks": [
            {
                "chunk_id": "text-1",
                "content_type": "text",
                "raw_content": "text body",
                "page": 1,
                "bbox": [0, 0, 1, 1],
                "column": 0,
                "order_in_page": 2,
                "global_order": 7,
                "metadata": {"parser": "fake"},
            },
            {
                "chunk_id": "text-2",
                "content_type": "text",
                "raw_content": "without verdict",
                "metadata": {},
            },
        ],
    }

    stored = _store(result)

    assert {
        "stored",
        "degraded",
        "failed",
        "excluded",
        "excluded_tables",
        "excluded_reasons",
        "attempted",
        "missing_verdicts",
        "error",
    }.issubset(stored.keys())

    metadata = fake_store.added[0]["metadata"]
    assert {
        "page",
        "bbox",
        "column",
        "order_in_page",
        "global_order",
        "chunk_id",
        "content_type",
        "table_quality",
        "score",
        "paper_title",
        "raw_content",
        "enrichment_status",
    }.issubset(metadata.keys())
    assert metadata["parser"] == "fake"
    assert stored["missing_verdicts"] == 1


def test_chunk_storage_stores_degraded_table(monkeypatch) -> None:
    fake_store = _patch_store(monkeypatch)
    monkeypatch.setenv("TEST_EMBEDDING_KEY", "test-key")

    result = {
        "paper_id": "paper-1",
        "verdicts": {"table-1": {"passed": False, "score": 0.4, "issue_type": "structure_error", "feedback": "bad rows"}},
        "merged_chunks": [{
            "chunk_id": "table-1",
            "content_type": "table",
            "raw_content": "| A | B |",
            "metadata": {},
        }],
    }

    stored = _store(result)

    assert stored["stored"] == 0
    assert stored["degraded"] == 1
    metadata = fake_store.added[0]["metadata"]
    assert metadata["table_quality"] == "degraded"
    assert metadata["judge_feedback"] == "bad rows"


def test_chunk_storage_excludes_false_positive_table(monkeypatch) -> None:
    fake_store = _patch_store(monkeypatch)
    monkeypatch.setenv("TEST_EMBEDDING_KEY", "test-key")

    result = {
        "paper_id": "paper-1",
        "verdicts": {"table-1": {"passed": False, "score": 0.2, "issue_type": "false_positive"}},
        "merged_chunks": [{
            "chunk_id": "table-1",
            "content_type": "table",
            "raw_content": "not a table",
            "metadata": {},
        }],
    }

    stored = _store(result)

    assert stored["excluded"] == 1
    assert stored["excluded_tables"] == 1
    assert fake_store.added == []


def test_chunk_storage_excludes_metadata_marked_chunk(monkeypatch) -> None:
    fake_store = _patch_store(monkeypatch)
    monkeypatch.setenv("TEST_EMBEDDING_KEY", "test-key")

    result = {
        "paper_id": "paper-1",
        "verdicts": {"table-1": {"passed": True, "score": 0.9}},
        "merged_chunks": [{
            "chunk_id": "table-1",
            "content_type": "table",
            "raw_content": "| A |",
            "metadata": {
                "excluded_from_storage": True,
                "storage_exclusion_reason": "already excluded",
            },
        }],
    }

    stored = _store(result)

    assert stored["excluded"] == 1
    assert stored["excluded_reasons"] == {"already excluded": 1}
    assert fake_store.added == []


def test_chunk_storage_counts_missing_verdict(monkeypatch) -> None:
    fake_store = _patch_store(monkeypatch)
    monkeypatch.setenv("TEST_EMBEDDING_KEY", "test-key")

    result = {
        "paper_id": "paper-1",
        "verdicts": {},
        "merged_chunks": [{
            "chunk_id": "text-1",
            "content_type": "text",
            "raw_content": "text body",
            "metadata": {},
        }],
    }

    stored = _store(result)

    assert stored["missing_verdicts"] == 1
    assert fake_store.added == []


def test_chunk_storage_missing_embedding_key_reports_attempted(monkeypatch) -> None:
    fake_store = _patch_store(monkeypatch)
    monkeypatch.delenv("TEST_EMBEDDING_KEY", raising=False)

    result = {
        "paper_id": "paper-1",
        "verdicts": {
            "text-1": {"passed": True, "score": 0.9},
            "table-1": {"passed": False, "issue_type": "structure_error"},
            "table-2": {"passed": False, "issue_type": "false_positive"},
        },
        "merged_chunks": [
            {"chunk_id": "text-1", "content_type": "text", "raw_content": "text", "metadata": {}},
            {"chunk_id": "table-1", "content_type": "table", "raw_content": "table", "metadata": {}},
            {"chunk_id": "table-2", "content_type": "table", "raw_content": "bad", "metadata": {}},
        ],
    }

    stored = _store(result)

    assert stored["failed"] == 2
    assert stored["attempted"] == 2
    assert "TEST_EMBEDDING_KEY" in stored["error"]
    assert fake_store.added == []
