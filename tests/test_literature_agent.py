from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


class FakeStore:
    def search(self, **_kwargs):
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
