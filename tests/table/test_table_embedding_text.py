from __future__ import annotations

from src.domain.table_embedding_text import build_table_embedding_text


def test_table_embedding_text_removes_markdown_noise_and_keeps_caption() -> None:
    text = build_table_embedding_text(
        raw_content=(
            "| strain | titer |\n"
            "| --- | --- |\n"
            "| B1 | 0.37 (cid:8) 0.05 |\n"
            "|  |  |\n"
        ),
        metadata={"table_caption": "Table 1. Shake flask titers"},
    )

    assert "Table 1. Shake flask titers" in text
    assert "---" not in text
    assert "|" not in text
    assert "B1; 0.37 \u00b1 0.05" in text


def test_table_embedding_text_deduplicates_label_already_in_caption() -> None:
    text = build_table_embedding_text(
        raw_content="| A | B |\n| --- | --- |\n| 1 | 2 |",
        metadata={"table_label": "Table 2", "table_caption": "Table 2. Results"},
    )

    assert text.count("Table 2") == 1
    assert "A; B" in text
