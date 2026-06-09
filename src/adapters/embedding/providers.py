from __future__ import annotations

import logging
import os
import time
from typing import Any

from src.application.settings import env_value

logger = logging.getLogger(__name__)

_QWEN3_LOCAL_MODEL: Any | None = None
_QWEN3_LOCAL_TOKENIZER: Any | None = None


def response_debug_summary(response: Any) -> str:
    parts: list[str] = []
    for field in ("status_code", "code", "message", "request_id"):
        value = (
            response.get(field)
            if isinstance(response, dict)
            else getattr(response, field, None)
        )
        if value not in (None, ""):
            parts.append(f"{field}={value}")
    return ", ".join(parts) if parts else "no status details"


def openai_embedding_client_kwargs(api_key: str, *, base_url: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"api_key": api_key, "timeout": 60.0}
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        *,
        api_key_env: str,
        model: str,
        dimensions: int,
        base_url: str,
    ) -> None:
        self.api_key_env = api_key_env
        self.model = model
        self.dimensions = dimensions
        self.base_url = base_url

    def embed(self, text: str) -> list[float]:
        from openai import OpenAI
        from src.judge.llm_runtime import run_llm_call

        api_key = env_value(self.api_key_env)
        if not api_key:
            raise ValueError(f"{self.api_key_env} is not configured")

        client = OpenAI(**openai_embedding_client_kwargs(api_key, base_url=self.base_url))
        response = run_llm_call(
            lambda: client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions,
            ),
            label="openai-embedding",
        )
        if not response.data:
            raise ValueError("No embedding returned from OpenAI")
        return [float(value) for value in response.data[0].embedding]


class LocalQwen3EmbeddingProvider:
    def __init__(
        self,
        *,
        model_path: str,
        max_length: int,
        query_instruct: str,
    ) -> None:
        self.model_path = model_path
        self.max_length = max_length
        self.query_instruct = query_instruct

    def embed(self, text: str, *, is_query: bool = False) -> list[float]:
        import torch
        import torch.nn.functional as F

        tokenizer, model = self._load_model()
        input_text = (
            f"Instruct: {self.query_instruct}\nQuery:{text}"
            if is_query
            else text
        )
        batch = tokenizer(
            [input_text],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(model.device) for key, value in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            attention_mask = batch["attention_mask"]
            left_padding = bool(attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                pooled = outputs.last_hidden_state[:, -1]
            else:
                lengths = attention_mask.sum(dim=1) - 1
                pooled = outputs.last_hidden_state[
                    torch.arange(attention_mask.shape[0], device=model.device),
                    lengths,
                ]
            embedding = F.normalize(pooled, p=2, dim=1)[0].detach().cpu().tolist()
        return [float(value) for value in embedding]

    def _load_model(self) -> tuple[Any, Any]:
        global _QWEN3_LOCAL_MODEL, _QWEN3_LOCAL_TOKENIZER
        if _QWEN3_LOCAL_MODEL is not None and _QWEN3_LOCAL_TOKENIZER is not None:
            return _QWEN3_LOCAL_TOKENIZER, _QWEN3_LOCAL_MODEL

        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side="left",
            local_files_only=True,
        )
        model = AutoModel.from_pretrained(self.model_path, local_files_only=True)
        model.eval()
        _QWEN3_LOCAL_TOKENIZER = tokenizer
        _QWEN3_LOCAL_MODEL = model
        logger.info("Loaded local Qwen3 embedding model: %s", self.model_path)
        return tokenizer, model


class DashScopeEmbeddingProvider:
    def __init__(
        self,
        *,
        api_key_env: str,
        model: str,
    ) -> None:
        self.api_key_env = api_key_env
        self.model = model

    def embed(self, text: str) -> tuple[list[float], dict[int, float]]:
        from dashscope import TextEmbedding
        from src.judge.llm_runtime import run_llm_call

        api_key = env_value(self.api_key_env)
        if not api_key:
            raise ValueError(f"{self.api_key_env} is not configured")
        os.environ.setdefault(self.api_key_env, api_key)

        for attempt in range(3):
            try:
                response = run_llm_call(
                    lambda: TextEmbedding.call(
                        model=self.model,
                        input=text,
                        output_type="dense&sparse",
                        api_key=api_key,
                    ),
                    label="dashscope-embedding",
                )
                break
            except Exception as exc:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Embedding failed after 3 attempts: {exc}") from exc

        output = response.output
        if isinstance(output, dict):
            embeddings = output.get("embeddings", [])
        else:
            embeddings = getattr(output, "embeddings", None) or []
        if not embeddings:
            raise ValueError(
                "No embedding returned from DashScope "
                f"({response_debug_summary(response)})"
            )

        first = embeddings[0]
        if isinstance(first, dict):
            dense = first.get("embedding")
        else:
            dense = getattr(first, "embedding", None)
        if not dense:
            raise ValueError("Embedding payload missing dense vector")

        sparse_raw = (
            first.get("sparse_embedding")
            if isinstance(first, dict)
            else getattr(first, "sparse_embedding", None)
        ) or []
        sparse_dict: dict[int, float] = {
            int(item["index"]): float(item["value"]) for item in sparse_raw
        }

        return ([float(value) for value in dense], sparse_dict)
