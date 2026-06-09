from __future__ import annotations


def test_rewrite_image_desc_uses_deepseek_adapter(monkeypatch) -> None:
    import src.graph.pipeline_graph as pipeline_graph

    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs

        def complete(self, **kwargs):
            captured.update(kwargs)
            return "rewritten image description"

    monkeypatch.setattr(pipeline_graph, "DeepSeekChatClient", FakeClient)

    result = pipeline_graph._rewrite_image_desc("original figure text", "too vague")

    assert result == "rewritten image description"
    assert captured["client_kwargs"] == {"timeout": 30.0}
    assert captured["system_prompt"] == ""
    assert captured["model"] == "deepseek-chat"
    assert captured["max_tokens"] == 2048
    assert captured["temperature"] == 0.2
    assert captured["label"] == "deepseek-image-rewrite"
    assert "original figure text" in str(captured["user_prompt"])
    assert "too vague" in str(captured["user_prompt"])


def test_rewrite_image_desc_returns_empty_on_adapter_failure(monkeypatch) -> None:
    import src.graph.pipeline_graph as pipeline_graph

    class FailingClient:
        def __init__(self, **_kwargs):
            pass

        def complete(self, **_kwargs):
            raise RuntimeError("offline")

    monkeypatch.setattr(pipeline_graph, "DeepSeekChatClient", FailingClient)

    assert pipeline_graph._rewrite_image_desc("original", "feedback") == ""
