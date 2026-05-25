from __future__ import annotations

from types import SimpleNamespace


def test_backfill_search_text_dry_run_does_not_write(monkeypatch) -> None:
    import tools.backfill_search_text as backfill

    class FakeClient:
        def __init__(self) -> None:
            self.writes = []

        def scroll(self, **_kwargs):
            return [
                SimpleNamespace(
                    id="point-1",
                    payload={
                        "paper_title": "demo.pdf",
                        "content": "E. coli improves LNTII through lgtA.",
                    },
                )
            ], None

        def set_payload(self, **kwargs):
            self.writes.append(kwargs)

    class FakeStore:
        collection = "papers"

        def __init__(self) -> None:
            self.client = FakeClient()

    monkeypatch.setattr(backfill, "QdrantStore", FakeStore)

    result = backfill.backfill_search_text(
        apply=False,
        batch_size=10,
        max_points=None,
        force=False,
    )

    assert result["scanned"] == 1
    assert result["updated"] == 1
    assert result["applied"] == 0


def test_backfill_search_text_apply_writes_payload(monkeypatch) -> None:
    import tools.backfill_search_text as backfill

    fake_instances = []

    class FakeClient:
        def __init__(self) -> None:
            self.writes = []

        def scroll(self, **_kwargs):
            return [
                SimpleNamespace(
                    id="point-1",
                    payload={
                        "paper_title": "demo.pdf",
                        "content": "E. coli improves LNTII through lgtA.",
                    },
                )
            ], None

        def set_payload(self, **kwargs):
            self.writes.append(kwargs)

    class FakeStore:
        collection = "papers"

        def __init__(self) -> None:
            self.client = FakeClient()
            self._lexical_document_cache = ["stale"]
            self._lexical_index_cache = {"stale": True}
            fake_instances.append(self)

    monkeypatch.setattr(backfill, "QdrantStore", FakeStore)

    result = backfill.backfill_search_text(
        apply=True,
        batch_size=10,
        max_points=None,
        force=False,
    )

    store = fake_instances[0]
    assert result["applied"] == 1
    assert len(store.client.writes) == 1
    assert "search_text" in store.client.writes[0]["payload"]
    assert store._lexical_document_cache is None
    assert store._lexical_index_cache is None
