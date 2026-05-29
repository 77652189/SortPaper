from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


class FakeStore:
    def search(self, **_kwargs):
        return []

    def expand_neighbor_context(self, *_args, **_kwargs):
        return []


def test_dashscope_agent_api_key_accepts_bom_env(monkeypatch) -> None:
    from src.agent.literature_agent import dashscope_agent_api_key

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("\ufeffDASHSCOPE_API_KEY", "agent-key")

    assert dashscope_agent_api_key() == "agent-key"


def test_agent_generation_passes_explicit_dashscope_api_key(monkeypatch) -> None:
    import src.agent.literature_agent as literature_agent

    captured: dict[str, object] = {}

    class FakeGeneration:
        @staticmethod
        def call(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                status_code=200,
                output=SimpleNamespace(choices=[SimpleNamespace(message={})]),
                message="",
            )

    monkeypatch.setenv("DASHSCOPE_API_KEY", "agent-key")
    monkeypatch.setattr(literature_agent, "QdrantStore", lambda: FakeStore())
    monkeypatch.setitem(sys.modules, "dashscope", SimpleNamespace(Generation=FakeGeneration))

    agent = literature_agent.LiteratureAgent()
    result = agent.query("How to improve LNT II production?", max_rounds=1)

    assert captured["api_key"] == "agent-key"
    assert captured["model"] == "qwen-plus"
    assert result["total_chunks"] == 0


def test_agent_generation_missing_dashscope_key_fails_early(monkeypatch) -> None:
    import src.agent.literature_agent as literature_agent

    class FakeGeneration:
        @staticmethod
        def call(**_kwargs):
            raise AssertionError("DashScope should not be called without an API key")

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("\ufeffDASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr(literature_agent, "QdrantStore", lambda: FakeStore())
    monkeypatch.setitem(sys.modules, "dashscope", SimpleNamespace(Generation=FakeGeneration))

    agent = literature_agent.LiteratureAgent()

    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        agent.query("How to improve LNT II production?", max_rounds=1)


def test_agent_search_passes_lexical_backfill_to_store(monkeypatch) -> None:
    import src.agent.literature_agent as literature_agent

    captured: dict[str, object] = {}

    class CapturingStore:
        def search(self, **kwargs):
            captured.update(kwargs)
            return [
                {
                    "score": 0.9,
                    "payload": {
                        "content": "RBS optimization improved lgtA expression.",
                        "paper_title": "LNT II paper",
                        "page": 3,
                        "content_type": "text",
                    },
                }
            ]

        def expand_neighbor_context(self, *_args, **_kwargs):
            return []

    monkeypatch.setattr(literature_agent, "QdrantStore", lambda: CapturingStore())
    monkeypatch.setattr(
        literature_agent.LiteratureAgent,
        "_synthesize",
        lambda self, query, chunks: "answer",
    )

    agent = literature_agent.LiteratureAgent(lexical_backfill=True)
    result = agent._execute_tool(
        "search_literature",
        {
            "query": "lacto-N-triose II lgtA",
            "category": "fermentation_experiment",
            "credibility_gte": 0.7,
        },
    )

    assert captured["query"] == "lacto-N-triose II lgtA"
    assert captured["rerank"] is True
    assert captured["lexical_backfill"] is True
    assert captured["filter_kwargs"] == {
        "category": "fermentation_experiment",
        "credibility_gte": 0.7,
    }
    assert result[0]["paper_title"] == "LNT II paper"


def test_agent_search_enables_lexical_backfill_by_default(monkeypatch) -> None:
    import src.agent.literature_agent as literature_agent

    captured: dict[str, object] = {}

    class CapturingStore:
        def search(self, **kwargs):
            captured.update(kwargs)
            return []

        def expand_neighbor_context(self, *_args, **_kwargs):
            return []

    monkeypatch.setattr(literature_agent, "QdrantStore", lambda: CapturingStore())

    agent = literature_agent.LiteratureAgent()
    agent._execute_tool("search_literature", {"query": "lacto-N-triose II"})

    assert captured["lexical_backfill"] is True


def test_agent_search_can_disable_lexical_backfill(monkeypatch) -> None:
    import src.agent.literature_agent as literature_agent

    captured: dict[str, object] = {}

    class CapturingStore:
        def search(self, **kwargs):
            captured.update(kwargs)
            return []

        def expand_neighbor_context(self, *_args, **_kwargs):
            return []

    monkeypatch.setattr(literature_agent, "QdrantStore", lambda: CapturingStore())

    agent = literature_agent.LiteratureAgent(lexical_backfill=False)
    agent._execute_tool("search_literature", {"query": "lacto-N-triose II"})

    assert captured["lexical_backfill"] is False


def test_agent_search_can_use_query_rewrite_multi_query(monkeypatch) -> None:
    import src.agent.literature_agent as literature_agent

    captured: dict[str, object] = {}

    class CapturingStore:
        def search(self, **_kwargs):
            return []

        def expand_neighbor_context(self, *_args, **_kwargs):
            return []

    def fake_multi_query_search(store, query, **kwargs):
        captured["store"] = store
        captured["query"] = query
        captured.update(kwargs)
        return SimpleNamespace(
            results=[
                {
                    "score": 0.9,
                    "matched_routes": ["raw", "normalized"],
                    "payload": {
                        "content": "RBS optimization improved lgtA expression.",
                        "paper_title": "LNT II paper",
                        "page": 3,
                        "content_type": "text",
                    },
                }
            ],
            metadata=lambda: {"routes": [{"source": "raw"}, {"source": "normalized"}]},
        )

    monkeypatch.setattr(literature_agent, "QdrantStore", lambda: CapturingStore())
    monkeypatch.setattr(literature_agent, "multi_query_search", fake_multi_query_search)

    agent = literature_agent.LiteratureAgent(
        lexical_backfill=True,
        use_query_rewrite=True,
        multi_query_recall=True,
    )
    result = agent._execute_tool("search_literature", {"query": "LNT II accumulation"})

    assert captured["query"] == "LNT II accumulation"
    assert captured["rerank"] is True
    assert captured["lexical_backfill"] is True
    assert captured["use_query_rewrite"] is True
    assert captured["route_limit"] == 4
    assert result[0]["matched_routes"] == ["raw", "normalized"]


def test_agent_expands_neighbor_context_without_returning_as_search_hits(monkeypatch) -> None:
    import src.agent.literature_agent as literature_agent

    captured_context: dict[str, object] = {}

    class CapturingStore:
        def search(self, **_kwargs):
            return [
                {
                    "score": 0.9,
                    "payload": {
                        "paper_id": "paper-1",
                        "chunk_id": "chunk-1",
                        "content": "Primary evidence chunk.",
                        "paper_title": "LNT II paper",
                        "page": 3,
                        "content_type": "text",
                    },
                }
            ]

        def expand_neighbor_context(self, results, **kwargs):
            captured_context.update(kwargs)
            assert len(results) == 1
            return [
                {
                    "neighbor_source_chunk_id": "chunk-1",
                    "neighbor_distance": 1,
                    "payload": {
                        "paper_id": "paper-1",
                        "chunk_id": "chunk-2",
                        "content": "Adjacent explanatory chunk.",
                        "paper_title": "LNT II paper",
                        "page": 3,
                        "content_type": "text",
                    },
                }
            ]

    monkeypatch.setattr(literature_agent, "QdrantStore", lambda: CapturingStore())

    agent = literature_agent.LiteratureAgent(expand_neighbor_context=True)
    result = agent._execute_tool("search_literature", {"query": "lacto-N-triose II"})

    assert len(result) == 1
    assert result[0]["content"] == "Primary evidence chunk."
    assert len(agent._all_chunks) == 2
    assert agent._all_chunks[1]["_context_expansion"] is True
    assert agent._all_chunks[1]["_neighbor_source_chunk_id"] == "chunk-1"
    assert captured_context["per_result_limit"] == 2
    assert captured_context["total_limit"] == 5


def test_agent_adds_paper_local_context_before_neighbor_context(monkeypatch) -> None:
    import src.agent.literature_agent as literature_agent

    class CapturingStore:
        def search(self, **_kwargs):
            return [
                {
                    "score": 0.9,
                    "payload": {
                        "paper_id": "paper-1",
                        "chunk_id": "chunk-1",
                        "content": "Primary evidence chunk.",
                        "paper_title": "LNT II paper",
                        "page": 3,
                        "content_type": "text",
                    },
                }
            ]

        def expand_paper_local_context(self, query, results, **kwargs):
            assert query == "lacto-N-triose II"
            assert len(results) == 1
            assert kwargs["total_limit"] == 5
            assert kwargs["per_paper_limit"] == 3
            return [
                {
                    "paper_local_context": True,
                    "payload": {
                        "paper_id": "paper-1",
                        "chunk_id": "chunk-local",
                        "content": "Paper-local deeper evidence.",
                    },
                }
            ]

        def expand_neighbor_context(self, results, **kwargs):
            assert len(results) == 2
            assert kwargs["total_limit"] == 4
            return [
                {
                    "neighbor_source_chunk_id": "chunk-1",
                    "neighbor_distance": 1,
                    "payload": {
                        "paper_id": "paper-1",
                        "chunk_id": "chunk-neighbor",
                        "content": "Adjacent explanatory chunk.",
                    },
                }
            ]

    monkeypatch.setattr(literature_agent, "QdrantStore", lambda: CapturingStore())

    agent = literature_agent.LiteratureAgent(
        expand_neighbor_context=True,
        expand_paper_local_context=True,
    )
    result = agent._execute_tool("search_literature", {"query": "lacto-N-triose II"})

    assert len(result) == 1
    assert [chunk["chunk_id"] for chunk in agent._all_chunks] == [
        "chunk-1",
        "chunk-local",
        "chunk-neighbor",
    ]
    assert agent._all_chunks[1]["_paper_local_context"] is True
    assert agent._all_chunks[2]["_neighbor_source_chunk_id"] == "chunk-1"


def test_agent_can_disable_neighbor_context_expansion(monkeypatch) -> None:
    import src.agent.literature_agent as literature_agent

    class CapturingStore:
        def search(self, **_kwargs):
            return [
                {
                    "score": 0.9,
                    "payload": {
                        "content": "Primary evidence chunk.",
                        "paper_title": "LNT II paper",
                        "page": 3,
                        "content_type": "text",
                    },
                }
            ]

        def expand_neighbor_context(self, *_args, **_kwargs):
            raise AssertionError("Neighbor context expansion should be disabled")

    monkeypatch.setattr(literature_agent, "QdrantStore", lambda: CapturingStore())

    agent = literature_agent.LiteratureAgent(expand_neighbor_context=False)
    agent._execute_tool("search_literature", {"query": "lacto-N-triose II"})

    assert len(agent._all_chunks) == 1
