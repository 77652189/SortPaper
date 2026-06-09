from __future__ import annotations

import re
from typing import Any


def build_table_embedding_text(
    *,
    raw_content: str,
    metadata: dict[str, Any] | None = None,
    max_chars: int = 5000,
) -> str:
    """Build cleaner embedding text for tables while preserving raw display text."""
    metadata = metadata or {}
    parts: list[str] = []
    caption = str(metadata.get("table_caption") or metadata.get("caption") or "").strip()
    label = str(metadata.get("table_label") or "").strip()
    if label and caption and label.lower() not in caption.lower():
        parts.append(label)
    if caption:
        parts.append(caption)

    for line in str(raw_content or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if _is_markdown_separator_row(line):
            continue
        if "|" in line:
            cells = [
                _clean_embedding_cell(cell)
                for cell in line.strip("|").split("|")
            ]
            cells = [cell for cell in cells if cell]
            if cells:
                parts.append("; ".join(cells))
        else:
            cleaned = _clean_embedding_cell(line)
            if cleaned:
                parts.append(cleaned)

    text = "\n".join(_dedupe_preserve_order(parts))
    return text[:max_chars].strip()


def _is_markdown_separator_row(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell or "---") for cell in cells)


def _clean_embedding_cell(cell: str) -> str:
    text = re.sub(r"\s+", " ", str(cell).strip())
    text = text.replace("(cid:8)", "\u00b1").replace("(cid:2)", "-").replace("(cid:3)", "-")
    if text in {"---", "-", "鈥?"}:
        return ""
    return text


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result
