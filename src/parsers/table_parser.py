from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ParseResult:
    content: str
    metadata: dict[str, Any]


class TableParser:
    def __init__(self, pdf_path: str | Path) -> None:
        self.pdf_path = str(pdf_path)

    def parse(self, feedback: str | None = None) -> ParseResult:
        import pdfplumber

        markdown_tables: list[str] = []
        table_count = 0
        settings = self._table_settings(feedback)

        with pdfplumber.open(self.pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables(table_settings=settings)
                for table_index, table in enumerate(tables, start=1):
                    if not table or len(table) < 2:
                        continue
                    normalized = self._normalize_table(table)
                    if not normalized or len(normalized[0]) < 2:
                        continue
                    markdown = self._to_markdown(normalized)
                    markdown_tables.append(f"## Page {page_index} Table {table_index}\n{markdown}")
                    table_count += 1

        return ParseResult(
            content="\n\n".join(markdown_tables),
            metadata={
                "table_count": table_count,
                "parser": "pdfplumber",
                "feedback_used": bool(feedback),
            },
        )

    @staticmethod
    def _normalize_table(table: list[list[str | None]]) -> list[list[str]]:
        normalized: list[list[str]] = []
        max_cols = max((len(row) for row in table if row), default=0)
        if max_cols < 2:
            return []

        for row in table:
            cleaned = [(cell or "").replace("\n", " ").strip() for cell in row]
            if len(cleaned) < max_cols:
                cleaned.extend([""] * (max_cols - len(cleaned)))
            if any(cell for cell in cleaned):
                normalized.append(cleaned)
        return normalized

    @staticmethod
    def _to_markdown(table: list[list[str]]) -> str:
        header = table[0]
        separator = ["---"] * len(header)
        rows = [header, separator, *table[1:]]
        return "\n".join("| " + " | ".join(row) + " |" for row in rows)

    @staticmethod
    def _table_settings(feedback: str | None) -> dict[str, Any]:
        settings: dict[str, Any] = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
            "join_tolerance": 3,
        }
        if feedback:
            settings["snap_tolerance"] = 5
            settings["join_tolerance"] = 5
        return settings
