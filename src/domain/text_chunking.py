from __future__ import annotations

from typing import Any


def chunk_text(content: str, chunk_size: int, chunk_overlap: int) -> list[dict[str, Any]]:
    text = content.strip()
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text_len = len(text)
    paragraphs_with_pos = split_paragraphs_with_pos(text)

    chunks: list[dict[str, Any]] = []
    current = ""
    current_start = 0

    for paragraph, paragraph_start in paragraphs_with_pos:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= chunk_size:
            if not current:
                current_start = paragraph_start
            current = candidate
            continue

        if current:
            chunks.append({
                "content": current,
                "char_start": current_start,
                "char_end": current_start + len(current),
            })
            current = ""

        if len(paragraph) <= chunk_size:
            current = paragraph
            current_start = paragraph_start
            continue

        for item in chunk_long_text(paragraph, chunk_size, chunk_overlap):
            chunks.append({
                "content": item["content"],
                "char_start": paragraph_start + item["char_start"],
                "char_end": paragraph_start + item["char_end"],
            })

    if current:
        chunks.append({
            "content": current,
            "char_start": current_start,
            "char_end": current_start + len(current),
        })

    chunk_count = len(chunks)
    for index, chunk in enumerate(chunks):
        chunk["chunk_index"] = index
        chunk["chunk_count"] = chunk_count
        chunk["relative_position"] = round(index / max(chunk_count - 1, 1), 6)
        chunk["relative_char_start"] = round(chunk["char_start"] / max(text_len, 1), 6)
        chunk["relative_char_end"] = round(chunk["char_end"] / max(text_len, 1), 6)
        chunk["page"] = None
        chunk["section_title"] = None
    return chunks


def split_paragraphs_with_pos(text: str) -> list[tuple[str, int]]:
    results: list[tuple[str, int]] = []
    cursor = 0
    for part in text.split("\n\n"):
        stripped = part.strip()
        if not stripped:
            cursor += len(part) + 2
            continue
        start = text.find(stripped, cursor)
        if start < 0:
            start = cursor
        results.append((stripped, start))
        cursor = start + len(stripped)
    if not results:
        cleaned = text.strip()
        return [(cleaned, text.find(cleaned))] if cleaned else []
    return results


def chunk_long_text(text: str, chunk_size: int, chunk_overlap: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        target_end = min(start + chunk_size, text_len)
        end = target_end
        if end < text_len:
            boundary = text.rfind("\n", start, target_end)
            if boundary < 0:
                boundary = max(
                    text.rfind("\u3002", start, target_end),
                    text.rfind(".", start, target_end),
                )
            if boundary > start + int(chunk_size * 0.6):
                end = boundary + 1
        chunk_content = text[start:end].strip()
        if chunk_content:
            chunks.append({"content": chunk_content, "char_start": start, "char_end": end})
        if end <= start:
            break
        if end >= text_len:
            break
        start = end - chunk_overlap
        if start < 0:
            start = 0
    return chunks
