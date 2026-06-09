from __future__ import annotations

import re
from pathlib import Path
from typing import Any


NOISE_PATTERNS = (
    r"\[(?:鏍囬|鍒嗙被|浣嶇疆|鎽樿|鍒ゆ柇渚濇嵁|浜х墿|鑿屾牚)\]",
    r"\b(?:text|table|image)\s*\|\s*chunk\s+\d+(?:/\d+)?",
    r"\bbiosynthesis_review\b|\bfermentation_experiment\b",
    r"\b(?:pdf|text|chunk)\b",
    r"\(绉戠爺閫?ablesci\.com\)",
)


def build_search_text(payload: dict[str, Any], *, max_chars: int = 6000) -> str:
    """Build clean lexical search text from Qdrant-compatible payload fields."""
    parts: list[str] = []
    parts.append(_clean_title(str(payload.get("paper_title") or "")))
    parts.extend(_normalize_list(payload.get("target_products")))
    parts.extend(_normalize_list(payload.get("organisms")))
    parts.extend(_normalize_list(payload.get("genes")))
    parts.extend(_normalize_list(payload.get("enzymes")))
    parts.extend(_normalize_list(payload.get("metrics")))
    parts.extend(_normalize_list(payload.get("aliases")))
    parts.extend(_normalize_list(payload.get("seo_terms")))
    parts.extend(_normalize_list(payload.get("seo_metric_terms")))
    parts.append(str(payload.get("seo_anchor_text") or ""))
    parts.append(str(payload.get("seo_section") or ""))
    parts.append(str(payload.get("paper_summary") or ""))
    parts.append(str(payload.get("context") or ""))
    parts.extend(_caption_texts(payload))
    parts.extend(_visual_texts(payload))

    content = str(payload.get("content") or payload.get("raw_content") or "")
    if str(payload.get("content_type") or payload.get("worker_type") or "") == "table":
        content = _table_text(content, payload)
    parts.append(content)

    text = "\n".join(_dedupe_preserve_order(_clean_field(part) for part in parts))
    return text[:max_chars].strip()


def _clean_title(value: str) -> str:
    text = Path(str(value)).stem
    text = text.replace("_", " ")
    return _clean_field(text)


def _normalize_list(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        items = re.split(r"[,;|/]\s*", value)
    elif isinstance(value, (list, tuple, set)):
        items = [str(item) for item in value]
    else:
        items = [str(value)]
    return [_clean_field(item) for item in items if str(item).strip()]


def _table_text(content: str, payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = payload
    try:
        from src.domain.table_embedding_text import build_table_embedding_text

        return build_table_embedding_text(
            raw_content=content,
            metadata=metadata,
        ) or content
    except Exception:
        return content


def _caption_texts(payload: dict[str, Any]) -> list[str]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = payload
    values: list[str] = []
    for key in (
        "table_caption",
        "table_footnote",
        "image_caption",
        "chart_caption",
        "figure_caption",
        "figure_label",
        "attached_heading",
    ):
        value = metadata.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        elif value:
            values.append(str(value))
    return values


def _visual_texts(payload: dict[str, Any]) -> list[str]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = payload
    return [
        str(metadata.get(key) or "")
        for key in (
            "vision_group_description",
            "mineru_visual_content",
            "image_body",
            "chart_body",
        )
        if str(metadata.get(key) or "").strip()
    ]


def _clean_field(value: Any) -> str:
    text = str(value or "")
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    text = _expand_scientific_aliases(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _expand_scientific_aliases(value: str) -> str:
    text = str(value)
    text = re.sub(r"2\s*[\u2032\u2019']?\s*-?\s*fl\b", "2'-FL 2-FL 2-fucosyllactose", text, flags=re.IGNORECASE)
    text = re.sub(r"3\s*[\u2032\u2019']?\s*-?\s*fl\b", "3-FL 3-fucosyllactose", text, flags=re.IGNORECASE)
    text = re.sub(r"\blnt\s*ii\b|\blntii\b", "LNT II lacto-N-triose II", text, flags=re.IGNORECASE)
    text = re.sub(r"\blnnt\b", "LNnT lacto-N-neotetraose", text, flags=re.IGNORECASE)
    text = re.sub(r"\blnt\b(?!\s*ii)", "LNT lacto-N-tetraose", text, flags=re.IGNORECASE)
    text = re.sub(r"\be\.\s*coli\b", "E. coli Escherichia coli", text, flags=re.IGNORECASE)
    text = re.sub(r"\budp\s*-\s*gal\b", "UDP-Gal UDP-galactose", text, flags=re.IGNORECASE)
    text = re.sub(r"\budp\s*-\s*glcnac\b", "UDP-GlcNAc UDP-N-acetylglucosamine", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhmos?\b", "HMO human milk oligosaccharide human milk oligosaccharides", text, flags=re.IGNORECASE)
    return text


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = str(item or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result
