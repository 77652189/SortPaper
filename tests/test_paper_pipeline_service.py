from __future__ import annotations

from types import SimpleNamespace

from src.application import paper_pipeline


def test_store_parsed_chunks_uses_injected_store_port() -> None:
    calls: list[dict] = []

    class Store:
        def store(self, result: dict, *, embedding_api_key_env: str, embed_content_for_chunk):
            chunk = result["merged_chunks"][0]
            calls.append({
                "result": result,
                "embedding_api_key_env": embedding_api_key_env,
                "embed_content": embed_content_for_chunk(chunk),
            })
            return {"stored": 1, "degraded": 0, "failed": 0, "attempted": 1, "error": ""}

    result = {
        "paper_id": "paper-1",
        "merged_chunks": [
            {
                "chunk_id": "c1",
                "content_type": "text",
                "raw_content": "LNT biosynthesis text.",
                "metadata": {},
            }
        ],
    }

    stored = paper_pipeline.store_parsed_chunks(
        result,
        embedding_api_key_env="DASHSCOPE_API_KEY",
        parsed_chunk_store=Store(),
    )

    assert stored["stored"] == 1
    assert calls == [{
        "result": result,
        "embedding_api_key_env": "DASHSCOPE_API_KEY",
        "embed_content": "LNT biosynthesis text.",
    }]


def test_point_to_quality_chunk_prefers_raw_content_and_stable_fields() -> None:
    point = SimpleNamespace(
        id=101,
        payload={
            "chunk_id": "c1",
            "raw_content": "raw text",
            "content": "display text",
            "worker_type": "table",
            "page": 2,
            "global_order": 5,
        },
    )

    chunk = paper_pipeline.point_to_quality_chunk(point)

    assert chunk.chunk_id == "c1"
    assert chunk.raw_content == "raw text"
    assert chunk.content_type == "table"
    assert chunk.page == 2
    assert chunk.bbox == [0, 0, 0, 0]
    assert chunk.global_order == 5
    assert chunk.metadata is point.payload


def test_paper_payload_from_quality_preserves_quality_contract_defaults() -> None:
    payload = paper_pipeline.paper_payload_from_quality(
        {
            "category": "fermentation_experiment",
            "credibility": 0.8,
            "paper_summary": "summary",
        },
        title="Stored title",
        paper_id="paper-1",
    )

    assert payload["category"] == "fermentation_experiment"
    assert payload["credibility"] == 0.8
    assert payload["paper_title"] == "Stored title"
    assert payload["target_products"] == []
    assert payload["organisms"] == []
    assert payload["classify_reason"] == ""
    assert payload["enrichment_status"] == "done"


def test_chunk_payload_update_for_quality_adds_context_and_search_text() -> None:
    payload = {"raw_content": "original chunk", "content": "old content"}
    paper_payload = {"paper_summary": "summary", "paper_title": "title"}

    update = paper_pipeline.chunk_payload_update_for_quality(
        payload,
        paper_payload=paper_payload,
        context="context",
        search_text_builder=lambda updated: f"{updated['paper_title']}|{updated['content']}",
    )
    no_context = paper_pipeline.chunk_payload_update_for_quality(
        payload,
        paper_payload=paper_payload,
        context="",
        search_text_builder=lambda updated: f"{updated['paper_title']}|{updated['raw_content']}",
    )

    assert update["content"] == "context\n\noriginal chunk"
    assert update["raw_content"] == "original chunk"
    assert update["context"] == "context"
    assert update["search_text"] == "title|context\n\noriginal chunk"
    assert no_context == {"search_text": "title|original chunk"}


def test_evaluate_and_enrich_paper_uses_injected_ports() -> None:
    point = SimpleNamespace(
        id=101,
        payload={
            "paper_title": "Stored title",
            "chunk_id": "c1",
            "raw_content": "original chunk",
            "content_type": "text",
            "global_order": 1,
        },
    )
    updates: list[tuple[str, dict, list[int]]] = []
    chunk_updates: list[tuple[int, dict]] = []
    invalidated: list[bool] = []

    class Store:
        def scroll_by_paper_id(self, paper_id: str):
            assert paper_id == "paper-1"
            return [point]

        def update_paper_payload(self, paper_id: str, payload: dict, *, point_ids=None):
            updates.append((paper_id, payload, point_ids))
            return len(point_ids)

        def set_chunk_payload(self, point_id, payload: dict) -> None:
            chunk_updates.append((point_id, payload))

        def invalidate_lexical_cache(self) -> None:
            invalidated.append(True)

    class Evaluator:
        def evaluate(self, chunks: list, title: str = "") -> dict:
            assert title == "Stored title"
            assert [chunk.raw_content for chunk in chunks] == ["original chunk"]
            return {
                "category": "fermentation_experiment",
                "credibility": 0.9,
                "paper_title": "Evaluated title",
                "chunk_contexts": {"c1": "context"},
            }

    quality = paper_pipeline.evaluate_and_enrich_paper(
        "paper-1",
        store=Store(),
        evaluator=Evaluator(),
        search_text_builder=lambda payload: f"{payload['paper_title']}|{payload['content']}",
    )

    assert quality["updated_points"] == 1
    assert updates[0][0] == "paper-1"
    assert updates[0][1]["paper_title"] == "Evaluated title"
    assert updates[0][2] == [101]
    assert chunk_updates == [(101, {
        "context": "context",
        "content": "context\n\noriginal chunk",
        "raw_content": "original chunk",
        "search_text": "Evaluated title|context\n\noriginal chunk",
    })]
    assert invalidated == [True]
