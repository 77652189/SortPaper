from __future__ import annotations

from types import SimpleNamespace


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
                    "selective_rerank_reason": "evidence_query",
                    "payload": {"paper_id": "paper-1", "chunk_id": "chunk-1"},
                }
            ]

    monkeypatch.setattr(qdrant_store, "QdrantStore", lambda: FakeStore())

    results = qdrant_search(
        "lacto-N-triose II lgtA",
        paper_id="paper-1",
        top_k=7,
        rerank=True,
        selective_rerank=True,
        rerank_top_n=5,
        filter_kwargs={"is_actionable": True},
        lexical_backfill=True,
        neighbor_backfill=True,
    )

    assert captured["query"] == "lacto-N-triose II lgtA"
    assert captured["limit"] == 7
    assert captured["rerank"] is True
    assert captured["selective_rerank"] is True
    assert captured["rerank_top_n"] == 5
    assert captured["lexical_backfill"] is True
    assert captured["neighbor_backfill"] is True
    assert captured["filter_kwargs"] == {
        "is_actionable": True,
        "paper_id": "paper-1",
    }
    assert results[0]["id"] == "point-1"
    assert results[0]["selective_rerank_reason"] == "evidence_query"


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
    assert captured["neighbor_backfill"] is False
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


def test_qdrant_search_passes_adaptive_route_limit_to_multi_query(monkeypatch) -> None:
    import src.retrieval.multi_query as multi_query
    import src.store.qdrant_store as qdrant_store
    from app_utils import qdrant_search

    captured: dict[str, object] = {}

    class FakeStore:
        pass

    def fake_multi_query_search(store, query, **kwargs):
        captured["store"] = store
        captured["query"] = query
        captured.update(kwargs)
        return SimpleNamespace(
            results=[
                {
                    "id": "point-1",
                    "score": 0.8,
                    "payload": {"paper_id": "paper-1", "chunk_id": "chunk-1"},
                }
            ],
            metadata=lambda: {"route_policy": {"adaptive": True}},
        )

    monkeypatch.setattr(qdrant_store, "QdrantStore", lambda: FakeStore())
    monkeypatch.setattr(multi_query, "multi_query_search", fake_multi_query_search)

    results = qdrant_search("Which passage reports 3.42 g/L?", top_k=5, multi_query=True)

    assert captured["query"] == "Which passage reports 3.42 g/L?"
    assert captured["route_limit"] == 4
    assert captured["adaptive_route_limit"] is True
    assert results[0]["search_meta"]["route_policy"]["adaptive"] is True
