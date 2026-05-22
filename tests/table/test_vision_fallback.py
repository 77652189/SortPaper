from __future__ import annotations

from src.parsers.table.vision_fallback import markdown_table_to_rows


def test_markdown_table_to_rows_skips_separator() -> None:
    markdown = """
| A | B |
| --- | --- |
| 1 | 2 |
| 3 | 4 |
"""

    assert markdown_table_to_rows(markdown) == [
        ["A", "B"],
        ["1", "2"],
        ["3", "4"],
    ]


def test_markdown_table_to_rows_ignores_non_table_text() -> None:
    markdown = """
Here is the table:

| Product | Yield |
| :--- | ---: |
| X | 80% |
"""

    assert markdown_table_to_rows(markdown) == [
        ["Product", "Yield"],
        ["X", "80%"],
    ]
