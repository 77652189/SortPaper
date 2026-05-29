from __future__ import annotations

import sys
from types import SimpleNamespace

from src.store.qdrant_store import (
    QdrantStore,
    dashscope_rerank_api_key,
    DENSE_VECTOR_NAME,
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


def test_dashscope_embedding_empty_response_reports_api_status(monkeypatch) -> None:
    class FakeTextEmbedding:
        @staticmethod
        def call(**_kwargs):
            return SimpleNamespace(
                status_code=401,
                code="InvalidApiKey",
                message="invalid api key",
                request_id="req-1",
                output={"embeddings": []},
            )

    monkeypatch.setenv("DASHSCOPE_API_KEY", "embedding-key")
    monkeypatch.setitem(sys.modules, "dashscope", SimpleNamespace(TextEmbedding=FakeTextEmbedding))

    store = QdrantStore.__new__(QdrantStore)

    try:
        store._embed_dashscope("lacto-N-triose II")
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected empty DashScope embedding response to fail")

    assert "No embedding returned from DashScope" in message
    assert "status_code=401" in message
    assert "code=InvalidApiKey" in message
    assert "message=invalid api key" in message
    assert "request_id=req-1" in message


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


def test_rerank_document_text_cleans_payload_noise_and_keeps_domain_terms() -> None:
    payload = {
        "paper_title": "2021-Example_2′-FL_paper(科研通-ablesci.com).pdf",
        "category": "fermentation_experiment",
        "target_products": ["2′-FL"],
        "organisms": ["E. coli"],
        "paper_summary": "[分类] biosynthesis_review | text | chunk 1/2",
        "content": "[标题] demo.pdf [位置] text | chunk 3/9 E. coli produces 2′-FL.",
    }

    text = QdrantStore._rerank_document_text(payload).lower()

    assert "pdf" not in text
    assert "text | chunk" not in text
    assert "biosynthesis_review" not in text
    assert "fermentation_experiment" not in text
    assert "2-fucosyllactose" in text
    assert "escherichia coli" in text


def test_add_writes_search_text_payload(monkeypatch) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.points = []

        def get_collection(self, _collection):
            return SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(
                        vectors={DENSE_VECTOR_NAME: SimpleNamespace(size=2)}
                    )
                )
            )

        def upsert(self, **kwargs):
            self.points.extend(kwargs["points"])

    store = QdrantStore.__new__(QdrantStore)
    store.client = FakeClient()
    store.collection = "papers"
    store._lexical_document_cache = None
    monkeypatch.setattr(QdrantStore, "_uses_sparse_vectors", property(lambda _self: False))
    monkeypatch.setattr(store, "embed", lambda _text: ([0.1, 0.2], {}))

    store.add(
        paper_id="paper-1",
        worker_type="text",
        content="E. coli improves LNTII through lgtA.",
        metadata={
            "chunk_id": "chunk-1",
            "paper_title": "demo.pdf",
            "target_products": ["LNTII"],
            "organisms": ["E. coli"],
        },
    )

    payload = store.client.points[0].payload

    assert "search_text" in payload
    assert "lacto-N-triose II" in payload["search_text"]
    assert "Escherichia coli" in payload["search_text"]


def test_lexical_entries_prefer_search_text_over_rerank_text() -> None:
    class FakeClient:
        def scroll(self, **_kwargs):
            return [
                SimpleNamespace(
                    id="point-1",
                    payload={
                        "paper_id": "paper-1",
                        "chunk_id": "chunk-1",
                        "content": "unrelated content",
                        "search_text": "RBS lgtA lacto-N-triose II",
                    },
                )
            ], None

    store = QdrantStore.__new__(QdrantStore)
    store.client = FakeClient()
    store.collection = "papers"

    candidates = store._lexical_candidates(
        "lacto-N-triose II lgtA",
        qdrant_filter=None,
        limit=5,
    )

    assert candidates[0]["id"] == "point-1"


def test_lexical_candidates_cache_unfiltered_scroll_results() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.scroll_calls = 0

        def scroll(self, **_kwargs):
            self.scroll_calls += 1
            return [
                SimpleNamespace(
                    id="point-1",
                    payload={
                        "paper_id": "paper-1",
                        "chunk_id": "chunk-1",
                        "content": (
                            "RBS optimization improved lgtA expression for "
                            "lacto-N-triose II biosynthesis."
                        ),
                    },
                )
            ], None

    store = QdrantStore.__new__(QdrantStore)
    store.client = FakeClient()
    store.collection = "papers"

    first = store._lexical_candidates(
        "lacto-N-triose II lgtA",
        qdrant_filter=None,
        limit=5,
    )
    second = store._lexical_candidates(
        "lacto-N-triose II lgtA",
        qdrant_filter=None,
        limit=5,
    )

    assert store.client.scroll_calls == 1
    assert first[0]["id"] == "point-1"
    assert second[0]["id"] == "point-1"


def test_lexical_candidates_use_index_to_score_only_matching_entries(monkeypatch) -> None:
    class FakeClient:
        def scroll(self, **_kwargs):
            return [
                SimpleNamespace(
                    id="point-target",
                    payload={
                        "paper_id": "paper-1",
                        "chunk_id": "chunk-target",
                        "content": "RBS optimization improved lgtA expression.",
                    },
                ),
                SimpleNamespace(
                    id="point-noise",
                    payload={
                        "paper_id": "paper-2",
                        "chunk_id": "chunk-noise",
                        "content": "A distant discussion about unrelated glycan yields.",
                    },
                ),
            ], None

    store = QdrantStore.__new__(QdrantStore)
    store.client = FakeClient()
    store.collection = "papers"

    original = QdrantStore._lexical_match_score_from_terms.__func__
    scored_texts: list[str] = []

    def wrapped(cls, normalized_text, *, query_terms, requires_strict_anchor):
        scored_texts.append(normalized_text)
        return original(
            cls,
            normalized_text,
            query_terms=query_terms,
            requires_strict_anchor=requires_strict_anchor,
        )

    monkeypatch.setattr(
        QdrantStore,
        "_lexical_match_score_from_terms",
        classmethod(wrapped),
    )

    candidates = store._lexical_candidates(
        "lgtA",
        qdrant_filter=None,
        limit=5,
    )

    assert [item["id"] for item in candidates] == ["point-target"]
    assert len(scored_texts) == 1
    assert "lgta" in scored_texts[0]


def test_neighbor_candidates_add_adjacent_chunks_from_same_paper() -> None:
    class FakeClient:
        def scroll(self, **_kwargs):
            return [
                SimpleNamespace(
                    id="point-near",
                    payload={
                        "paper_id": "paper-1",
                        "chunk_id": "chunk-near",
                        "global_order": 11,
                        "content": "Nearby evidence chunk.",
                    },
                ),
                SimpleNamespace(
                    id="point-far",
                    payload={
                        "paper_id": "paper-1",
                        "chunk_id": "chunk-far",
                        "global_order": 20,
                        "content": "Far chunk.",
                    },
                ),
                SimpleNamespace(
                    id="point-other-paper",
                    payload={
                        "paper_id": "paper-2",
                        "chunk_id": "chunk-other",
                        "global_order": 11,
                        "content": "Other paper chunk.",
                    },
                ),
            ], None

    store = QdrantStore.__new__(QdrantStore)
    store.client = FakeClient()
    store.collection = "papers"

    results = [{
        "id": "point-seed",
        "score": 0.8,
        "payload": {
            "paper_id": "paper-1",
            "chunk_id": "chunk-seed",
            "global_order": 10,
        },
    }]

    merged = store._merge_neighbor_candidates(
        results=results,
        qdrant_filter=None,
        limit=5,
        window=2,
    )

    assert [item["id"] for item in merged] == ["point-seed", "point-near"]
    assert merged[1]["neighbor_source_chunk_id"] == "chunk-seed"


def test_expand_neighbor_context_returns_only_context_chunks() -> None:
    class FakeClient:
        def scroll(self, **_kwargs):
            return [
                SimpleNamespace(
                    id="point-seed",
                    payload={
                        "paper_id": "paper-1",
                        "chunk_id": "chunk-seed",
                        "global_order": 10,
                        "content": "Seed chunk.",
                    },
                ),
                SimpleNamespace(
                    id="point-near",
                    payload={
                        "paper_id": "paper-1",
                        "chunk_id": "chunk-near",
                        "global_order": 11,
                        "content": "Nearby context chunk.",
                    },
                ),
            ], None

    store = QdrantStore.__new__(QdrantStore)
    store.client = FakeClient()
    store.collection = "papers"

    context = store.expand_neighbor_context(
        [{
            "id": "point-seed",
            "score": 0.8,
            "payload": {
                "paper_id": "paper-1",
                "chunk_id": "chunk-seed",
                "global_order": 10,
            },
        }],
        total_limit=5,
    )

    assert [item["id"] for item in context] == ["point-near"]
    assert context[0]["context_expansion"] is True


def test_normalize_search_text_expands_common_hmo_aliases() -> None:
    normalized = QdrantStore._normalize_search_text(
        "LNTII, LNnT, 2′-FL, and E. coli"
    )

    assert "lacto n triose ii" in normalized
    assert "lacto n neotetraose" in normalized
    assert "2 fucosyllactose" in normalized
    assert "escherichia coli" in normalized


def test_rank_papers_from_results_keeps_first_unique_order() -> None:
    results = [
        {"payload": {"paper_id": "paper-a"}},
        {"payload": {"paper_id": "paper-a"}},
        {"payload": {"paper_id": "paper-b"}},
        {"payload": {"paper_id": ""}},
        {"payload": {"paper_id": "paper-c"}},
    ]

    assert QdrantStore._rank_papers_from_results(results, limit=2) == ["paper-a", "paper-b"]


def test_interleave_ranked_groups_and_deduplicate_results() -> None:
    groups = [
        [
            {"id": "a1", "payload": {"paper_id": "paper-a", "chunk_id": "1"}},
            {"id": "a2", "payload": {"paper_id": "paper-a", "chunk_id": "2"}},
        ],
        [
            {"id": "b1", "payload": {"paper_id": "paper-b", "chunk_id": "1"}},
        ],
    ]

    interleaved = QdrantStore._interleave_ranked_groups(groups)
    assert [item["id"] for item in interleaved] == ["a1", "b1", "a2"]

    deduped = QdrantStore._deduplicate_results(interleaved + [interleaved[0]])
    assert [item["id"] for item in deduped] == ["a1", "b1", "a2"]


def test_prioritize_evidence_groups_keeps_representatives_then_lead_depth() -> None:
    groups = [
        [{"id": "a1"}, {"id": "a2"}, {"id": "a3"}],
        [{"id": "b1"}, {"id": "b2"}],
        [{"id": "c1"}, {"id": "c2"}],
    ]

    prioritized = QdrantStore._prioritize_evidence_groups(groups)

    assert [item["id"] for item in prioritized] == [
        "a1", "b1", "c1", "a2", "a3", "b2", "c2",
    ]


def test_score_rank_evidence_groups_prefers_best_evidence_across_papers() -> None:
    groups = [
        [
            {"id": "a1", "score": 0.7, "lexical_score": 0.7, "paper_rank": 1},
            {"id": "a2", "score": 1.2, "lexical_score": 1.2, "paper_rank": 1},
        ],
        [{"id": "b1", "score": 0.8, "lexical_score": 0.8, "paper_rank": 2}],
        [{"id": "c1", "score": 0.6, "lexical_score": 0.6, "paper_rank": 3}],
    ]

    ranked = QdrantStore._score_rank_evidence_groups(groups)

    assert [item["id"] for item in ranked] == ["a2", "b1", "a1", "c1"]


def test_content_type_preference_bonus_only_boosts_matching_type() -> None:
    assert QdrantStore._content_type_preference_bonus(
        {"content_type": "table"},
        content_type_preference="table",
    ) == 0.08
    assert QdrantStore._content_type_preference_bonus(
        {"worker_type": "text"},
        content_type_preference="text",
    ) == 0.08
    assert QdrantStore._content_type_preference_bonus(
        {"content_type": "text"},
        content_type_preference="table",
    ) == 0.0
    assert QdrantStore._content_type_preference_bonus(
        {"content_type": "table"},
        content_type_preference="any",
    ) == 0.0


def test_restore_anchor_candidates_keeps_raw_hits_in_final_page() -> None:
    anchors = [
        {"id": "raw-1", "score": 0.9, "payload": {"paper_id": "p1", "chunk_id": "raw-1"}},
        {"id": "raw-2", "score": 0.8, "payload": {"paper_id": "p1", "chunk_id": "raw-2"}},
        {"id": "raw-3", "score": 0.7, "payload": {"paper_id": "p1", "chunk_id": "raw-3"}},
        {"id": "raw-4", "score": 0.6, "payload": {"paper_id": "p1", "chunk_id": "raw-4"}},
        {"id": "raw-5", "score": 0.5, "payload": {"paper_id": "p1", "chunk_id": "raw-5"}},
    ]
    boosted = anchors[:3] + [
        {"id": f"lex-{i}", "score": 1.0 - i * 0.01, "payload": {"paper_id": "p1", "chunk_id": f"lex-{i}"}}
        for i in range(7)
    ]

    restored = QdrantStore._restore_anchor_candidates(boosted, anchors, limit=10)

    chunk_ids = [item["payload"]["chunk_id"] for item in restored]
    assert "raw-4" in chunk_ids
    assert "raw-5" in chunk_ids
    assert len(restored) == 10
    assert restored[-2]["anchor_restored"] is True
    assert restored[-1]["anchor_restored"] is True


def test_restore_anchor_candidates_does_not_duplicate_existing_hits() -> None:
    anchors = [
        {"id": "raw-1", "payload": {"paper_id": "p1", "chunk_id": "raw-1"}},
        {"id": "raw-2", "payload": {"paper_id": "p1", "chunk_id": "raw-2"}},
    ]
    ranked = [anchors[1], anchors[0], anchors[0]]

    restored = QdrantStore._restore_anchor_candidates(ranked, anchors, limit=3)

    assert [item["payload"]["chunk_id"] for item in restored] == ["raw-2", "raw-1"]


def test_backfill_anchor_count_keeps_top10_guard_narrow() -> None:
    assert QdrantStore._backfill_anchor_count(3) == 3
    assert QdrantStore._backfill_anchor_count(5) == 3
    assert QdrantStore._backfill_anchor_count(10) == 5
    assert QdrantStore._backfill_anchor_count(20) == 10


def test_search_paper_local_evidence_rescores_chunks_inside_ranked_papers() -> None:
    class FakeClient:
        def scroll(self, **_kwargs):
            return [
                SimpleNamespace(
                    id="point-target",
                    payload={
                        "paper_id": "paper-1",
                        "chunk_id": "chunk-target",
                        "content": "RBS optimization improved lgtA expression for lacto-N-triose II.",
                    },
                ),
                SimpleNamespace(
                    id="point-noise",
                    payload={
                        "paper_id": "paper-1",
                        "chunk_id": "chunk-noise",
                        "content": "A generic introduction about human milk oligosaccharides.",
                    },
                ),
                SimpleNamespace(
                    id="point-other-paper",
                    payload={
                        "paper_id": "paper-2",
                        "chunk_id": "chunk-other",
                        "content": "RBS optimization improved lgtA expression for lacto-N-triose II.",
                    },
                ),
            ], None

    store = QdrantStore.__new__(QdrantStore)
    store.client = FakeClient()
    store.collection = "papers"
    store.search = lambda **_kwargs: [
        {
            "id": "point-representative",
            "score": 0.9,
            "payload": {
                "paper_id": "paper-1",
                "chunk_id": "chunk-representative",
                "content": "Paper-level representative hit.",
            },
        }
    ]

    results = store.search_paper_local_evidence(
        "lacto-N-triose II lgtA",
        limit=3,
        lexical_backfill=True,
    )

    assert results[0]["id"] == "point-target"
    assert results[0]["evidence_stage"] == "paper_local"
    assert "point-other-paper" not in [item["id"] for item in results]


def test_expand_paper_local_context_adds_deeper_evidence_from_hit_papers() -> None:
    class FakeClient:
        def scroll(self, **_kwargs):
            return [
                SimpleNamespace(
                    id="point-existing",
                    payload={
                        "paper_id": "paper-1",
                        "chunk_id": "chunk-existing",
                        "content": "Representative lacto-N-triose II paper hit.",
                    },
                ),
                SimpleNamespace(
                    id="point-target",
                    payload={
                        "paper_id": "paper-1",
                        "chunk_id": "chunk-target",
                        "content": "RBS optimization improved lgtA expression for lacto-N-triose II.",
                    },
                ),
                SimpleNamespace(
                    id="point-other",
                    payload={
                        "paper_id": "paper-2",
                        "chunk_id": "chunk-other",
                        "content": "RBS optimization improved lgtA expression for lacto-N-triose II.",
                    },
                ),
            ], None

    store = QdrantStore.__new__(QdrantStore)
    store.client = FakeClient()
    store.collection = "papers"
    results = [{
        "id": "point-existing",
        "score": 0.9,
        "payload": {"paper_id": "paper-1", "chunk_id": "chunk-existing"},
    }]

    context = store.expand_paper_local_context(
        "lacto-N-triose II lgtA RBS",
        results,
        total_limit=2,
    )

    assert [item["id"] for item in context] == ["point-target"]
    assert context[0]["paper_local_context"] is True
    assert context[0]["context_expansion"] is True


def test_lexical_idf_score_prefers_rare_matching_terms() -> None:
    lexical_index = {
        "entries": [{}, {}, {}, {}],
        "inverted": {
            "common": {0, 1, 2, 3},
            "rare": {0},
        },
    }
    query_terms = {"common": 0.2, "rare": 0.2}

    rare_score = QdrantStore._lexical_idf_match_score_from_terms(
        "common rare",
        query_terms=query_terms,
        requires_strict_anchor=False,
        lexical_index=lexical_index,
    )
    common_score = QdrantStore._lexical_idf_match_score_from_terms(
        "common",
        query_terms=query_terms,
        requires_strict_anchor=False,
        lexical_index=lexical_index,
    )

    assert rare_score > common_score


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
