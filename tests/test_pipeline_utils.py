from __future__ import annotations

from src.application.pipeline_utils import generate_chunk_description


def test_generate_chunk_description_uses_injected_llm_client() -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def complete(self, **kwargs):
            captured.update(kwargs)
            return "A table about lacto-N-tetraose yield."

    result = generate_chunk_description(
        "table",
        "yield values by strain",
        llm_client=FakeClient(),
    )

    assert result == "A table about lacto-N-tetraose yield."
    assert captured["system_prompt"] == ""
    assert captured["model"] == "deepseek-chat"
    assert captured["max_tokens"] == 80
    assert captured["temperature"] == 0.3
    assert captured["label"] == "deepseek-chunk-description"
    assert "yield values by strain" in str(captured["user_prompt"])


def test_generate_chunk_description_returns_empty_on_llm_failure() -> None:
    class FailingClient:
        def complete(self, **_kwargs):
            raise RuntimeError("offline")

    assert generate_chunk_description("image", "chart", llm_client=FailingClient()) == ""
