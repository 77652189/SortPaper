from __future__ import annotations

from src.store.qdrant_store import qdrant_point_id


def test_qdrant_point_id_includes_paper_id() -> None:
    chunk_id = "text_p1_col2_y100.0_x20.0"

    assert qdrant_point_id("paper-a", chunk_id) != qdrant_point_id("paper-b", chunk_id)
    assert qdrant_point_id("paper-a", chunk_id) == qdrant_point_id("paper-a", chunk_id)
