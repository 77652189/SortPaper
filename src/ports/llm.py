from __future__ import annotations

from typing import Any, Protocol


class ChatCompletionClient(Protocol):
    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        response_format: dict[str, Any] | None = None,
        label: str = "llm-chat",
    ) -> str:
        """Return the assistant text for a chat completion request."""
