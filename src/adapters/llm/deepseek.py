from __future__ import annotations

import os
from typing import Any

from src.judge.llm_runtime import run_llm_call


DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


class DeepSeekChatClient:
    """OpenAI-compatible DeepSeek chat adapter."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEEPSEEK_BASE_URL,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else _env_value("DEEPSEEK_API_KEY")
        self.base_url = base_url
        self.timeout = timeout

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        response_format: dict[str, Any] | None = None,
        label: str = "deepseek-chat",
    ) -> str:
        from openai import OpenAI

        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is not configured")

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        response = run_llm_call(
            lambda: client.chat.completions.create(**kwargs),
            label=label,
        )
        return response.choices[0].message.content or ""


def _env_value(name: str, default: str = "") -> str:
    return (os.getenv(name) or os.getenv(f"\ufeff{name}") or default).strip()
