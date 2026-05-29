from __future__ import annotations


def test_qdrant_search_passes_lexical_backfill_and_filters(monkeypatch) -> None:
    import src.store.qdrant_store as qdrant_store
    from app_utils import qdrant_search

    captured: dict[str, object] = {}

    class FakeStore:
        def search(self, **kwargs):
            captured.update(kwargs)
            return [
                {
                    "id": "point-1",
                    "score": 0.8,
                    "payload": {"paper_id": "paper-1", "chunk_id": "chunk-1"},
                }
            ]

    monkeypatch.setattr(qdrant_store, "QdrantStore", lambda: FakeStore())

    results = qdrant_search(
        "lacto-N-triose II lgtA",
        paper_id="paper-1",
        top_k=7,
        rerank=True,
        filter_kwargs={"is_actionable": True},
        lexical_backfill=True,
    )

    assert captured["query"] == "lacto-N-triose II lgtA"
    assert captured["limit"] == 7
    assert captured["rerank"] is True
    assert captured["lexical_backfill"] is True
    assert captured["filter_kwargs"] == {
        "is_actionable": True,
        "paper_id": "paper-1",
    }
    assert results[0]["id"] == "point-1"


def test_qdrant_search_enables_lexical_backfill_by_default(monkeypatch) -> None:
    import src.store.qdrant_store as qdrant_store
    from app_utils import qdrant_search

    captured: dict[str, object] = {}

    class FakeStore:
        def search(self, **kwargs):
            captured.update(kwargs)
            return []

    monkeypatch.setattr(qdrant_store, "QdrantStore", lambda: FakeStore())

    qdrant_search("lacto-N-triose II", top_k=3)

    assert captured["lexical_backfill"] is True
    assert captured["filter_kwargs"] is None


def test_qdrant_search_returns_top10_by_default(monkeypatch) -> None:
    import src.store.qdrant_store as qdrant_store
    from app_utils import qdrant_search

    captured: dict[str, object] = {}

    class FakeStore:
        def search(self, **kwargs):
            captured.update(kwargs)
            return []

    monkeypatch.setattr(qdrant_store, "QdrantStore", lambda: FakeStore())

    qdrant_search("lacto-N-triose II")

    assert captured["limit"] == 10


def test_qdrant_search_can_disable_lexical_backfill(monkeypatch) -> None:
    import src.store.qdrant_store as qdrant_store
    from app_utils import qdrant_search

    captured: dict[str, object] = {}

    class FakeStore:
        def search(self, **kwargs):
            captured.update(kwargs)
            return []

    monkeypatch.setattr(qdrant_store, "QdrantStore", lambda: FakeStore())

    qdrant_search("lacto-N-triose II", top_k=3, lexical_backfill=False)

    assert captured["lexical_backfill"] is False
