from __future__ import annotations

import hashlib
import logging

from src.adapters.llm.deepseek import DeepSeekChatClient
from src.ports.llm import ChatCompletionClient

logger = logging.getLogger(__name__)


def build_paper_id(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


def chunk_to_dict(chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "content_type": chunk.content_type,
        "page": chunk.page,
        "column": chunk.column,
        "global_order": chunk.global_order,
        "raw_content": chunk.raw_content,
        "bbox": chunk.bbox,
        "metadata": chunk.metadata or {},
    }


def generate_chunk_description(
    content_type: str,
    raw_content: str,
    *,
    llm_client: ChatCompletionClient | None = None,
) -> str:
    """Generate a compact English description for table/image embedding prefixes."""
    type_label = "table" if content_type == "table" else "figure/image"
    prompt = (
        f"Write ONE short English sentence summarizing what this {type_label} is about. "
        f"Focus on key concepts, entities, and data dimensions. Be specific.\n\n"
        f"Content:\n{raw_content[:2000]}"
    )

    try:
        client = llm_client or DeepSeekChatClient(base_url="https://api.deepseek.com")
        return client.complete(
            system_prompt="",
            user_prompt=prompt,
            model="deepseek-chat",
            max_tokens=80,
            temperature=0.3,
            label="deepseek-chunk-description",
        ).strip()
    except Exception as exc:
        logger.warning("Failed to generate %s description: %s", content_type, exc)
        return ""
