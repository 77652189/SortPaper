from __future__ import annotations

from src.domain.point_identity import qdrant_point_id, stable_hash


def test_stable_hash_is_deterministic() -> None:
    assert stable_hash("paper:chunk") == stable_hash("paper:chunk")
    assert stable_hash("paper:chunk") != stable_hash("paper:other")


def test_qdrant_point_id_includes_paper_id_and_fits_int63() -> None:
    chunk_id = "text_p1_col2_y100.0_x20.0"

    assert qdrant_point_id("paper-a", chunk_id) != qdrant_point_id("paper-b", chunk_id)
    assert qdrant_point_id("paper-a", chunk_id) == qdrant_point_id("paper-a", chunk_id)
    assert 0 <= qdrant_point_id("paper-a", chunk_id) < 2 ** 63
