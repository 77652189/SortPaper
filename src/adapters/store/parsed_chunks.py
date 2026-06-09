from __future__ import annotations

from typing import Any


class LegacyParsedChunkStore:
    """Adapter for the existing parsed-result chunk storage implementation."""

    def store(
        self,
        result: dict[str, Any],
        *,
        embedding_api_key_env: str,
        embed_content_for_chunk,
    ) -> dict[str, Any]:
        from src.store.chunk_storage import store_parsed_result_chunks

        return store_parsed_result_chunks(
            result,
            embedding_api_key_env=embedding_api_key_env,
            embed_content_for_chunk=embed_content_for_chunk,
        )
