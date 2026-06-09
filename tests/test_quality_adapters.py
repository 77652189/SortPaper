from __future__ import annotations

from types import SimpleNamespace

from src.adapters.quality import PaperQualityEvaluatorAdapter, default_quality_enrichment_dependencies
from src.adapters.store import defaults as store_defaults
from src.adapters.store.qdrant import QdrantQualityEnrichmentStore, QdrantVectorLibraryRepository


def test_default_store_repositories_build_qdrant_adapters(monkeypatch) -> None:
    import src.adapters.store.qdrant as qdrant_adapters

    class FakeSearchRepository:
        pass

    class FakeVectorLibraryRepository:
        pass

    monkeypatch.setattr(qdrant_adapters, "QdrantSearchRepository", FakeSearchRepository)
    monkeypatch.setattr(qdrant_adapters, "QdrantVectorLibraryRepository", FakeVectorLibraryRepository)

    assert isinstance(store_defaults.default_search_repository(), FakeSearchRepository)
    assert isinstance(store_defaults.default_vector_library_repository(), FakeVectorLibraryRepository)


def test_default_quality_enrichment_dependencies_builds_adapter_bundle(monkeypatch) -> None:
    import src.adapters.quality as quality_adapters
    import src.adapters.store.qdrant as qdrant_adapters

    class FakeStore:
        pass

    class FakeEvaluator:
        pass

    monkeypatch.setattr(qdrant_adapters, "QdrantQualityEnrichmentStore", FakeStore)
    monkeypatch.setattr(quality_adapters, "PaperQualityEvaluatorAdapter", FakeEvaluator)

    dependencies = default_quality_enrichment_dependencies()

    assert isinstance(dependencies.store, FakeStore)
    assert isinstance(dependencies.evaluator, FakeEvaluator)
    assert callable(dependencies.search_text_builder)


def test_paper_quality_evaluator_adapter_delegates_to_evaluator() -> None:
    calls: list[tuple[list, str]] = []

    class Evaluator:
        def evaluate(self, chunks: list, title: str = "") -> dict:
            calls.append((chunks, title))
            return {"category": "ok"}

    adapter = PaperQualityEvaluatorAdapter(Evaluator())

    assert adapter.evaluate(["chunk"], title="Paper") == {"category": "ok"}
    assert calls == [(["chunk"], "Paper")]


def test_qdrant_quality_enrichment_store_sets_chunk_payload_and_invalidates() -> None:
    calls: list[dict] = []
    invalidated: list[bool] = []

    class Client:
        def set_payload(self, **kwargs):
            calls.append(kwargs)

    class Store:
        collection = "papers"

        def __init__(self) -> None:
            self.client = Client()

        def _invalidate_lexical_cache(self) -> None:
            invalidated.append(True)

    adapter = QdrantQualityEnrichmentStore(Store())

    adapter.set_chunk_payload(101, {"search_text": "text"})
    adapter.invalidate_lexical_cache()

    assert calls == [{
        "collection_name": "papers",
        "payload": {"search_text": "text"},
        "points": [101],
    }]
    assert invalidated == [True]


def test_qdrant_vector_library_repository_counts_by_paper_id() -> None:
    calls: list[dict] = []

    class Client:
        def count(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(count=3)

    class Store:
        collection = "papers"

        def __init__(self) -> None:
            self.client = Client()

    repo = QdrantVectorLibraryRepository(Store())

    assert repo.paper_count("paper-1") == 3
    assert calls[0]["collection_name"] == "papers"
    condition = calls[0]["count_filter"].must[0]
    assert condition.key == "paper_id"
    assert condition.match.value == "paper-1"
