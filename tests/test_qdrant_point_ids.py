from __future__ import annotations

import sys
from types import SimpleNamespace

from src.store.qdrant_store import (
    QdrantStore,
    dashscope_rerank_api_key,
    openai_embedding_client_kwargs,
    qdrant_point_id,
)


def test_qdrant_point_id_includes_paper_id() -> None:
    chunk_id = "text_p1_col2_y100.0_x20.0"

    assert qdrant_point_id("paper-a", chunk_id) != qdrant_point_id("paper-b", chunk_id)
    assert qdrant_point_id("paper-a", chunk_id) == qdrant_point_id("paper-a", chunk_id)


def test_openai_embedding_client_uses_embedding_base_url() -> None:
    kwargs = openai_embedding_client_kwargs("test-key")

    assert kwargs["api_key"] == "test-key"
    assert kwargs["base_url"] == "https://api.openai.com/v1"


def test_dashscope_rerank_api_key_accepts_bom_env(monkeypatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("\ufeffDASHSCOPE_API_KEY", "rerank-key")

    assert dashscope_rerank_api_key() == "rerank-key"


def test_rerank_passes_explicit_dashscope_api_key(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeTextReRank:
        @staticmethod
        def call(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                status_code=200,
                output={"results": [{"index": 0, "relevance_score": 0.9}]},
                usage={},
            )

    monkeypatch.setenv("DASHSCOPE_API_KEY", "rerank-key")
    monkeypatch.setitem(sys.modules, "dashscope", SimpleNamespace(TextReRank=FakeTextReRank))

    store = QdrantStore.__new__(QdrantStore)
    candidates = [{"id": 1, "score": 0.1, "payload": {"content": "candidate"}}]

    reranked = store._rerank("query", candidates, top_n=1)

    assert captured["api_key"] == "rerank-key"
    assert reranked[0]["reranked"] is True


def test_rerank_preserves_original_index_scores(monkeypatch) -> None:
    class FakeTextReRank:
        @staticmethod
        def call(**kwargs):
            return SimpleNamespace(
                status_code=200,
                output={"results": [
                    {"index": 2, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.42},
                    {"index": 1, "relevance_score": 0.11},
                ]},
                usage={},
            )

    monkeypatch.setenv("DASHSCOPE_API_KEY", "rerank-key")
    monkeypatch.setitem(sys.modules, "dashscope", SimpleNamespace(TextReRank=FakeTextReRank))

    store = QdrantStore.__new__(QdrantStore)
    candidates = [
        {"id": "a", "score": 0.1, "payload": {"content": "first"}},
        {"id": "b", "score": 0.2, "payload": {"content": "second"}},
        {"id": "c", "score": 0.3, "payload": {"content": "third"}},
    ]

    reranked = store._rerank("query", candidates, top_n=3)

    assert [(item["id"], item["score"]) for item in reranked] == [
        ("c", 0.91),
        ("a", 0.42),
        ("b", 0.11),
    ]


def test_lnt_query_requires_exact_anchor_for_lexical_boost() -> None:
    query = "如何解决 LNT II 中间产物残留过高?"

    assert QdrantStore._lexical_match_score(
        query,
        "Engineered Escherichia coli improved metabolic carbon flux.",
    ) == 0.0
    assert QdrantStore._lexical_match_score(
        query,
        "RBS optimization improved lgtA expression for lacto-N-triose II biosynthesis.",
    ) > 0.0


def test_lnt_query_boost_promotes_specific_mechanism_chunk() -> None:
    query = "大肠杆菌中如何解决 LNT II 中间产物残留过高?"
    broad = {
        "id": "broad",
        "score": 0.8,
        "payload": {"content": "LNT II is one of the human milk oligosaccharide intermediates."},
    }
    mechanism = {
        "id": "mechanism",
        "score": 0.6,
        "payload": {
            "category": "fermentation_experiment",
            "organisms": ["Escherichia coli"],
            "content": (
                "For lacto-N-triose II (LNT II) production, RBS optimization regulated "
                "lgtA translation and improved soluble expression."
            ),
        },
    }

    boosted = QdrantStore._apply_query_relevance_boost(query, [broad, mechanism])

    assert boosted[0]["id"] == "mechanism"
    assert boosted[0]["lexical_score"] > boosted[1]["lexical_score"]


def test_list_papers_marks_legacy_without_quality_as_pending() -> None:
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
