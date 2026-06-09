from __future__ import annotations

from src.judge.llm_judge import LLMJudge


def test_llm_judge_uses_injected_chat_client() -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def complete(self, **kwargs):
            captured.update(kwargs)
            return '{"passed": true, "score": 0.92, "feedback": "looks good", "issue_type": "none"}'

    judge = LLMJudge(deepseek_model="deepseek-chat", threshold=0.7, llm_client=FakeClient())

    verdict = judge.judge("text", "paper content", "paper.pdf")

    assert verdict.passed is True
    assert verdict.score == 0.92
    assert verdict.feedback == "looks good"
    assert captured["system_prompt"] == ""
    assert captured["model"] == "deepseek-chat"
    assert captured["temperature"] == 0.2
    assert captured["max_tokens"] == 2048
    assert captured["label"] == "deepseek-judge"
    assert "paper content" in str(captured["user_prompt"])


def test_llm_judge_returns_failed_verdict_on_client_error() -> None:
    class FailingClient:
        def complete(self, **_kwargs):
            raise RuntimeError("offline")

    verdict = LLMJudge(llm_client=FailingClient()).judge("table", "content", "paper.pdf")

    assert verdict.passed is False
    assert verdict.score == 0.0
    assert "DeepSeek call failed" in verdict.feedback
