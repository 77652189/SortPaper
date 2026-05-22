from __future__ import annotations

from types import SimpleNamespace


def test_store_parsed_chunks_writes_pending_without_quality(monkeypatch) -> None:
    import app_pipeline
    import src.store.qdrant_store as qdrant_store

    added: list[dict] = []

    class FakeClient:
        def count(self, **_kwargs):
            return SimpleNamespace(count=0)

    class FakeStore:
        collection = "papers"

        def __init__(self) -> None:
            self.client = FakeClient()

        def add(self, **kwargs) -> None:
            added.append(kwargs)

        def save(self, _path: str) -> None:
            return None

    monkeypatch.setenv(app_pipeline.EMBEDDING_API_KEY_ENV, "test-key")
    monkeypatch.setattr(qdrant_store, "QdrantStore", FakeStore)

    result = {
        "paper_id": "paper-1",
        "filename": "demo.pdf",
        "quality": {},
        "verdicts": {"c1": {"passed": True, "score": 0.9}},
        "merged_chunks": [
            {
                "chunk_id": "c1",
                "content_type": "text",
                "raw_content": "raw chunk text",
                "page": 1,
                "bbox": [0, 0, 1, 1],
                "metadata": {},
            }
        ],
    }

    store_result = app_pipeline.store_parsed_chunks(result)

    assert store_result["stored"] == 1
    assert added[0]["content"] == "raw chunk text"
    assert added[0]["embed_content"] == "raw chunk text"
    metadata = added[0]["metadata"]
    assert metadata["enrichment_status"] == "pending"
    assert metadata["paper_title"] == "demo.pdf"
    assert metadata["raw_content"] == "raw chunk text"


def test_evaluate_and_enrich_from_qdrant_uses_raw_content_and_updates_payload(monkeypatch) -> None:
    import app_pipeline
    import src.judge.paper_evaluator as paper_evaluator
    import src.store.qdrant_store as qdrant_store

    point = SimpleNamespace(
        id=101,
        payload={
            "paper_id": "paper-1",
            "paper_title": "Stored title",
            "chunk_id": "c1",
            "raw_content": "original chunk",
            "content": "old display content",
            "content_type": "text",
            "global_order": 1,
        },
    )
    paper_payload_updates: list[dict] = []
    chunk_payload_updates: list[dict] = []

    class FakeClient:
        def set_payload(self, collection_name, payload, points):
            chunk_payload_updates.append({
                "collection_name": collection_name,
                "payload": payload,
                "points": points,
            })

    class FakeStore:
        collection = "papers"

        def __init__(self) -> None:
            self.client = FakeClient()

        def scroll_by_paper_id(self, paper_id):
            assert paper_id == "paper-1"
            return [point]

        def update_paper_payload(self, paper_id, payload, point_ids=None):
            assert paper_id == "paper-1"
            assert point_ids == [101]
            paper_payload_updates.append(payload)
            return len(point_ids)

    class FakeEvaluator:
        def evaluate(self, chunks, title=""):
            assert title == "Stored title"
            assert [chunk.raw_content for chunk in chunks] == ["original chunk"]
            return {
                "category": "fermentation_experiment",
                "credibility": 0.82,
                "fermentation_relevance": 0.9,
                "is_actionable": True,
                "paper_title": "Evaluated title",
                "paper_summary": "summary",
                "target_products": ["LNT"],
                "organisms": ["E. coli"],
                "classify_reason": "reason",
                "classify_status": "ok",
                "chunk_contexts": {"c1": "context"},
            }

    monkeypatch.setattr(qdrant_store, "QdrantStore", FakeStore)
    monkeypatch.setattr(paper_evaluator, "PaperQualityEvaluator", FakeEvaluator)

    quality = app_pipeline.evaluate_and_enrich_from_qdrant("paper-1")

    assert quality["updated_points"] == 1
    assert paper_payload_updates[0]["enrichment_status"] == "done"
    assert paper_payload_updates[0]["paper_title"] == "Evaluated title"
    assert chunk_payload_updates[0]["payload"]["content"] == "context\n\noriginal chunk"
    assert chunk_payload_updates[0]["payload"]["raw_content"] == "original chunk"


def test_scroll_by_paper_id_reads_all_pages() -> None:
    from src.store.qdrant_store import QdrantStore

    calls: list[object] = []

    class FakeClient:
        def scroll(self, **kwargs):
            calls.append(kwargs.get("offset"))
            if kwargs.get("offset") is None:
                return [SimpleNamespace(id=1, payload={})], "next"
            return [SimpleNamespace(id=2, payload={})], None

    store = QdrantStore.__new__(QdrantStore)
    store.client = FakeClient()
    store.collection = "papers"

    points = store.scroll_by_paper_id("paper-1", page_size=1)

    assert [point.id for point in points] == [1, 2]
    assert calls == [None, "next"]


def test_run_preview_skips_text_llm_judge(monkeypatch) -> None:
    import app_pipeline
    import src.judge.llm_judge as llm_judge
    import src.parsers.pymupdf_parser as pymupdf_parser
    import src.parsers.table_parser as table_parser
    from src.parsers.layout_chunk import LayoutChunk

    text_chunk = LayoutChunk(
        content_type="text",
        raw_content="This is a useful paragraph with enough length to avoid the preview noise filter.",
        page=1,
        bbox=(0, 0, 100, 20),
        column=0,
        order_in_page=0,
        chunk_id="text-1",
    )
    table_chunk = LayoutChunk(
        content_type="table",
        raw_content="| A | B |\n| 1 | 2 |",
        page=1,
        bbox=(0, 30, 100, 80),
        column=0,
        order_in_page=1,
        chunk_id="table-1",
        metadata={},
    )
    calls: list[str] = []

    class FakeTextParser:
        def __init__(self, _path) -> None:
            pass

        def parse(self):
            return [text_chunk]

    class FakeTableParser:
        def __init__(self, _path, enable_vision_fallback=False) -> None:
            pass

        def parse(self, strategy="all"):
            return [table_chunk]

    class FakeJudge:
        def judge(self, worker_type, content, pdf_path):
            calls.append(worker_type)
            return SimpleNamespace(
                passed=True,
                score=0.91,
                feedback="ok",
                issue_type="none",
            )

    monkeypatch.setattr(pymupdf_parser, "PyMuPDFParser", FakeTextParser)
    monkeypatch.setattr(table_parser, "TableParser", FakeTableParser)
    monkeypatch.setattr(llm_judge, "LLMJudge", FakeJudge)
    monkeypatch.setattr(app_pipeline, "_extract_image_placeholders", lambda _path: ([], []))

    result = app_pipeline.run_preview(b"%PDF fake")

    assert calls == ["table"]
    assert result["verdicts"]["text-1"]["issue_type"] == "preview_text_not_judged"
    assert result["verdicts"]["table-1"]["score"] == 0.91
