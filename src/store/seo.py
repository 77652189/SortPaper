from __future__ import annotations

import re
from typing import Any


METRIC_PATTERNS = (
    r"\b\d+(?:\.\d+)?\s*(?:g/L|mg/L|g l-1|mg l-1|mM|uM|µM|OD600|h|hr|hours?)\b",
    r"\b(?:titer|titre|yield|productivity|conversion|concentration|OD600|optical density)\b",
)


def build_chunk_seo_metadata(chunk: dict[str, Any], paper_title: str) -> dict[str, Any]:
    """Build retrieval-only metadata for chunk SEO without changing raw evidence."""
    metadata = dict(chunk.get("metadata", {}) or {})
    content_type = str(chunk.get("content_type") or "text")
    raw_content = str(chunk.get("raw_content") or "")
    captions = _caption_terms(metadata)
    visual_terms = _visual_terms(metadata)
    section = _first_text(
        metadata.get("attached_heading"),
        metadata.get("section_title"),
        metadata.get("heading"),
    )
    evidence_type = _evidence_type(content_type, metadata)
    terms = _dedupe(
        [
            paper_title,
            section,
            metadata.get("figure_label"),
            metadata.get("figure_group_id"),
            evidence_type,
            *captions,
            *visual_terms,
            *_metric_terms(raw_content),
            *_metric_terms(" ".join(captions + visual_terms)),
        ]
    )
    return {
        "seo_section": section,
        "seo_evidence_type": evidence_type,
        "seo_terms": terms,
        "seo_metric_terms": _dedupe(_metric_terms(raw_content) + _metric_terms(" ".join(captions + visual_terms))),
        "seo_anchor_text": _anchor_text(content_type, metadata, section),
        "seo_weight": _evidence_weight(content_type, metadata, raw_content),
    }


def _caption_terms(metadata: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in (
        "table_caption",
        "table_footnote",
        "image_caption",
        "chart_caption",
        "figure_caption",
    ):
        value = metadata.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        elif value:
            values.append(str(value))
    return [item.strip() for item in values if item.strip()]


def _visual_terms(metadata: dict[str, Any]) -> list[str]:
    return [
        str(metadata.get(key) or "").strip()
        for key in (
            "vision_group_description",
            "mineru_visual_content",
            "image_body",
            "chart_body",
        )
        if str(metadata.get(key) or "").strip()
    ]


def _metric_terms(text: str) -> list[str]:
    terms: list[str] = []
    for pattern in METRIC_PATTERNS:
        terms.extend(match.group(0) for match in re.finditer(pattern, text, flags=re.IGNORECASE))
    return terms


def _evidence_type(content_type: str, metadata: dict[str, Any]) -> str:
    mineru_type = str(metadata.get("mineru_type") or "")
    if content_type == "table":
        return "table numeric evidence"
    if content_type == "image":
        if mineru_type == "chart":
            return "figure chart evidence"
        return "figure visual evidence"
    return "text evidence"


def _evidence_weight(content_type: str, metadata: dict[str, Any], raw_content: str) -> float:
    weight = 1.0
    if content_type == "table":
        weight += 0.18
    elif content_type == "image":
        weight += 0.08
        if metadata.get("vision_group_description"):
            weight += 0.12
    if _metric_terms(raw_content):
        weight += 0.12
    section = str(metadata.get("attached_heading") or metadata.get("section_title") or "").lower()
    if any(token in section for token in ("result", "discussion", "fermentation", "production")):
        weight += 0.08
    return round(min(weight, 1.5), 3)


def _anchor_text(content_type: str, metadata: dict[str, Any], section: str) -> str:
    parts = [
        section,
        metadata.get("figure_label"),
        metadata.get("figure_caption"),
        metadata.get("table_caption"),
        _evidence_type(content_type, metadata),
    ]
    return " | ".join(str(part).strip() for part in parts if str(part or "").strip())


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _dedupe(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result
