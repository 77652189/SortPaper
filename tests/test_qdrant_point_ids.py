from __future__ import annotations

from types import SimpleNamespace

from src.store.qdrant_store import openai_embedding_client_kwargs, qdrant_point_id


def test_qdrant_point_id_includes_paper_id() -> None:
    chunk_id = "text_p1_col2_y100.0_x20.0"

    assert qdrant_point_id("paper-a", chunk_id) != qdrant_point_id("paper-b", chunk_id)
    assert qdrant_point_id("paper-a", chunk_id) == qdrant_point_id("paper-a", chunk_id)


def test_openai_embedding_client_uses_embedding_base_url() -> None:
    kwargs = openai_embedding_client_kwargs("test-key")

    assert kwargs["api_key"] == "test-key"
    assert kwargs["base_url"] == "https://api.openai.com/v1"


def test_list_papers_marks_legacy_without_quality_as_pending() -> None:
    from src.store.qdrant_store import QdrantStore

    class FakeClient:
        def scroll(self, **kwargs):
            assert kwargs["offset"] is None
            return [
                SimpleNamespace(payload={
                    "paper_id": "legacy-raw",
                    "paper_title": "Legacy Raw",
                }),
                SimpleNamespace(payload={
                    "paper_id": "legacy-quality",
                    "paper_title": "Legacy Quality",
                    "category": "fermentation_experiment",
                }),
                SimpleNamespace(payload={
                    "paper_id": "new-pending",
                    "paper_title": "New Pending",
                    "enrichment_status": "pending",
                }),
            ], None

    store = QdrantStore.__new__(QdrantStore)
    store.client = FakeClient()
    store.collection = "papers"

    papers = {paper["paper_id"]: paper for paper in store.list_papers()}

    assert papers["legacy-raw"]["enrichment_status"] == "pending"
    assert papers["legacy-raw"]["enrichment_status_note"] == "旧数据缺少质量分析标记"
    assert papers["legacy-quality"]["enrichment_status"] == "done"
    assert papers["new-pending"]["enrichment_status"] == "pending"
