from __future__ import annotations

import pytest

from src.domain import text_chunking
from src.store.qdrant_store import QdrantStore


def test_chunk_text_keeps_positions_and_relative_metadata() -> None:
    chunks = text_chunking.chunk_text("alpha\n\nbeta", chunk_size=20, chunk_overlap=0)

    assert len(chunks) == 1
    assert chunks[0]["content"] == "alpha\n\nbeta"
    assert chunks[0]["char_start"] == 0
    assert chunks[0]["char_end"] == len("alpha\n\nbeta")
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["chunk_count"] == 1
    assert chunks[0]["page"] is None
    assert chunks[0]["section_title"] is None


def test_qdrant_store_chunk_text_delegates_to_domain_policy() -> None:
    store = QdrantStore.__new__(QdrantStore)

    chunks = store._chunk_text("abcdef", chunk_size=4, chunk_overlap=1)

    assert [chunk["content"] for chunk in chunks] == ["abcd", "def"]


def test_chunk_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="chunk_overlap must be smaller"):
        text_chunking.chunk_text("abcdef", chunk_size=4, chunk_overlap=4)
