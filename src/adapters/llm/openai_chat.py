from __future__ import annotations

import os
from typing import Any

from src.application.settings import OPENAI_CHAT_BASE_URL
from src.judge.llm_runtime import run_llm_call


class OpenAIChatClient:
    """OpenAI chat adapter that implements the ChatCompletionClient port."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = OPENAI_CHAT_BASE_URL,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else _env_value("OPENAI_API_KEY")
        self.base_url = base_url
        self.timeout = timeout

    def configuration_error(self) -> str:
        if not self.api_key:
            return "OPENAI_API_KEY 未配置，无法生成中文摘要翻译。"
        if _looks_like_placeholder_key(self.api_key) and "api.openai.com" in self.base_url:
            return (
                "OPENAI_API_KEY 看起来是本地占位 key，但当前 OPENAI_CHAT_BASE_URL 指向 OpenAI 官方接口。"
                " 请在 .env 中配置真实 OpenAI API key，或把 OPENAI_CHAT_BASE_URL 改成对应的兼容服务地址。"
            )
        return ""

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        response_format: dict[str, Any] | None = None,
        label: str = "openai-chat",
    ) -> str:
        from openai import OpenAI

        if error := self.configuration_error():
            raise ValueError(error)

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


def _looks_like_placeholder_key(value: str) -> bool:
    text = str(value or "").strip().lower()
    return (
        not text
        or text in {"your-openai-key", "your-openai-api-key"}
        or text.startswith("sk-local")
    )
