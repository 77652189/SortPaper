from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from src.parsers.config import VISION
from src.parsers.layout_chunk import LayoutChunk, LayoutMerger, infer_column


DEFAULT_INCLUDED_TYPES = {
    "text",
    "ref_text",
    "list",
    "aside_text",
    "table",
    "image",
    "chart",
}


class MinerUZipParser:
    """Convert MinerU's result zip into PaperSort LayoutChunk objects."""

    def __init__(
        self,
        zip_path: str | Path,
        *,
        include_types: set[str] | None = None,
        merge: bool = True,
    ) -> None:
        self.zip_path = Path(zip_path)
        self.include_types = set(include_types or DEFAULT_INCLUDED_TYPES)
        self.merge = merge

    def parse(self) -> list[LayoutChunk]:
        with zipfile.ZipFile(self.zip_path) as archive:
            content_items = _load_content_list(archive)
        page_sizes = _infer_page_sizes(content_items)
        chunks: list[LayoutChunk] = []
        order_by_page: dict[int, int] = {}
        for item_index, item in enumerate(content_items):
            if not isinstance(item, dict):
                continue
            source_type = str(item.get("type") or "").strip()
            if source_type not in self.include_types:
                continue
            if source_type in {"image", "chart"} and not _keep_mineru_visual_item(item):
                continue
            raw_content = _raw_content_for_item(item)
            if not raw_content.strip():
                continue
            bbox = _normalize_bbox(item.get("bbox"))
            if bbox is None:
                continue
            page = int(item.get("page_idx") or 0) + 1
            page_width, page_height = page_sizes.get(page, _bbox_page_size(bbox))
            order = order_by_page.get(page, 0)
            order_by_page[page] = order + 1
            content_type = _content_type_for_mineru_type(source_type)
            chunks.append(
                LayoutChunk(
                    content_type=content_type,
                    raw_content=raw_content,
                    page=page,
                    bbox=bbox,
                    column=infer_column(page_width, bbox[0], bbox[2]),
                    order_in_page=order,
                    metadata=_metadata_for_item(
                        item,
                        item_index=item_index,
                        page_width=page_width,
                        page_height=page_height,
                    ),
                )
            )
        if self.merge:
            chunks = LayoutMerger.merge(chunks)
        chunks = merge_mineru_section_headings(chunks)
        return attach_mineru_figure_groups(chunks)


def merge_mineru_section_headings(chunks: list[LayoutChunk]) -> list[LayoutChunk]:
    """Attach MinerU section heading chunks to the following text chunk."""
    result: list[LayoutChunk] = []
    index = 0
    while index < len(chunks):
        current = chunks[index]
        if (
            _is_mineru_heading_chunk(current)
            and index + 1 < len(chunks)
            and _can_attach_heading_to_next(current, chunks[index + 1])
        ):
            next_chunk = chunks[index + 1]
            merged = LayoutChunk(
                content_type="text",
                raw_content=f"{current.raw_content.strip()}\n\n{next_chunk.raw_content.strip()}",
                page=next_chunk.page,
                bbox=(
                    min(current.bbox[0], next_chunk.bbox[0]),
                    min(current.bbox[1], next_chunk.bbox[1]),
                    max(current.bbox[2], next_chunk.bbox[2]),
                    max(current.bbox[3], next_chunk.bbox[3]),
                ),
                column=next_chunk.column,
                order_in_page=current.order_in_page,
                metadata={
                    **next_chunk.metadata,
                    "attached_heading": current.raw_content.strip(),
                    "attached_heading_chunk_id": current.chunk_id,
                    "attached_heading_bbox": list(current.bbox),
                },
            )
            result.append(merged)
            index += 2
            continue
        result.append(current)
        index += 1
    for global_order, chunk in enumerate(result):
        chunk.global_order = global_order
        chunk.order_in_page = sum(1 for previous in result[:global_order] if previous.page == chunk.page)
    return result


def _is_mineru_heading_chunk(chunk: LayoutChunk) -> bool:
    if chunk.content_type != "text":
        return False
    if chunk.metadata.get("parser") != "mineru_vlm":
        return False
    text = chunk.raw_content.strip()
    if not text:
        return False
    if text.endswith(":") and len(text.split()) > 4:
        return False
    text_level = chunk.metadata.get("text_level")
    if isinstance(text_level, int) and text_level >= 1:
        return len(text) <= 120
    words = re.findall(r"[A-Za-z][A-Za-z-]*", text)
    if not words or len(words) > 8:
        return False
    letters = re.sub(r"[^A-Za-z]+", "", text)
    return bool(letters) and letters.upper() == letters and len(letters) >= 4


def _can_attach_heading_to_next(heading: LayoutChunk, next_chunk: LayoutChunk) -> bool:
    if next_chunk.content_type != "text":
        return False
    if heading.page != next_chunk.page:
        return False
    if next_chunk.metadata.get("mineru_type") in {"aside_text"}:
        return False
    return True


def attach_mineru_figure_groups(chunks: list[LayoutChunk]) -> list[LayoutChunk]:
    """Group same-page MinerU image/chart chunks that belong to the same figure."""
    result = list(chunks)
    visual_indexes_by_page: dict[int, list[int]] = {}
    for index, chunk in enumerate(result):
        if _is_mineru_visual_chunk(chunk):
            visual_indexes_by_page.setdefault(chunk.page, []).append(index)

    for page, visual_indexes in visual_indexes_by_page.items():
        pending: list[int] = []
        groups: list[dict[str, Any]] = []
        active_group: dict[str, Any] | None = None
        for index in visual_indexes:
            chunk = result[index]
            caption = _figure_caption_for_chunk(chunk)
            figure_label = _figure_label(caption)
            if figure_label:
                group = {
                    "member_indexes": [*pending, index],
                    "caption_owner_index": index,
                    "caption": caption,
                    "figure_label": figure_label,
                }
                groups.append(group)
                active_group = group
                pending = []
                continue
            if active_group is not None and _visual_belongs_to_group(chunk, [result[i] for i in active_group["member_indexes"]]):
                active_group["member_indexes"].append(index)
            else:
                pending.append(index)

        for group_counter, group in enumerate(groups, start=1):
            member_indexes = list(group["member_indexes"])
            group_id = f"figure_p{page}_{group_counter:03d}"
            group_bbox = _union_bbox([result[i].bbox for i in member_indexes])
            child_ids = [result[i].chunk_id for i in member_indexes]
            for member_index in member_indexes:
                member = result[member_index]
                member.metadata.update({
                    "figure_group_id": group_id,
                    "figure_label": group["figure_label"],
                    "figure_caption": group["caption"],
                    "figure_group_bbox": list(group_bbox),
                    "figure_group_child_ids": child_ids,
                    "figure_group_role": "caption_owner" if member_index == group["caption_owner_index"] else "part",
                    "figure_group_index": group_counter,
                })

        grouped_indexes = {
            member_index
            for group in groups
            for member_index in group["member_indexes"]
        }
        for index in visual_indexes:
            if index in grouped_indexes:
                continue
            chunk = result[index]
            chunk.metadata.setdefault("figure_group_id", f"figure_p{page}_ungrouped_{index:03d}")
            chunk.metadata.setdefault("figure_group_role", "single")
            chunk.metadata.setdefault("figure_group_bbox", list(chunk.bbox))
            chunk.metadata.setdefault("figure_group_child_ids", [chunk.chunk_id])
    return result


def _is_mineru_visual_chunk(chunk: LayoutChunk) -> bool:
    return (
        chunk.content_type == "image"
        and chunk.metadata.get("parser") == "mineru_vlm"
        and chunk.metadata.get("mineru_type") in {"image", "chart"}
    )


def _figure_caption_for_chunk(chunk: LayoutChunk) -> str:
    captions: list[str] = []
    for key in ("image_caption", "chart_caption"):
        captions.extend(_string_list(chunk.metadata.get(key)))
    return " ".join(captions).strip()


def _figure_label(caption: str) -> str:
    match = re.search(r"\b(Figure|Fig\.)\s*([0-9]+[A-Za-z]?)", caption, flags=re.IGNORECASE)
    if not match:
        return ""
    return f"Figure {match.group(2)}"


def _visual_belongs_to_group(chunk: LayoutChunk, group_members: list[LayoutChunk]) -> bool:
    group_bbox = _union_bbox([member.bbox for member in group_members])
    vertical_overlap = min(chunk.bbox[3], group_bbox[3]) - max(chunk.bbox[1], group_bbox[1])
    if vertical_overlap > 0:
        return True
    vertical_gap = min(abs(chunk.bbox[1] - group_bbox[3]), abs(group_bbox[1] - chunk.bbox[3]))
    return vertical_gap <= 80


def _union_bbox(bboxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )


def summarize_mineru_zip(zip_path: str | Path, chunks: list[LayoutChunk] | None = None) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path) as archive:
        content_items = _load_content_list(archive)
    original_counts = Counter(
        str(item.get("type") or "")
        for item in content_items
        if isinstance(item, dict)
    )
    visual_items = [
        item for item in content_items
        if isinstance(item, dict) and str(item.get("type") or "") in {"image", "chart"}
    ]
    kept_visual = sum(1 for item in visual_items if _keep_mineru_visual_item(item))
    summary: dict[str, Any] = {
        "content_items": len(content_items),
        "original_type_counts": dict(original_counts),
        "visual_items": len(visual_items),
        "kept_visual_items": kept_visual,
        "filtered_visual_items": len(visual_items) - kept_visual,
    }
    if chunks is not None:
        summary["chunks"] = len(chunks)
        summary["chunk_type_counts"] = dict(Counter(chunk.content_type for chunk in chunks))
        summary["mineru_chunk_type_counts"] = dict(
            Counter(str(chunk.metadata.get("mineru_type") or "") for chunk in chunks)
        )
        summary["figure_groups"] = len({
            str(chunk.metadata.get("figure_group_id") or "")
            for chunk in chunks
            if chunk.metadata.get("figure_group_id")
        })
        summary["pages"] = sorted({chunk.page for chunk in chunks})
    return summary


def _load_content_list(archive: zipfile.ZipFile) -> list[dict[str, Any]]:
    names = archive.namelist()
    candidates = [
        name for name in names
        if name.endswith("_content_list.json") and not name.endswith("_content_list_v2.json")
    ]
    if not candidates and "content_list.json" in names:
        candidates = ["content_list.json"]
    if not candidates:
        raise FileNotFoundError("MinerU zip does not contain content_list.json")
    data = json.loads(archive.read(candidates[0]).decode("utf-8"))
    if not isinstance(data, list):
        raise ValueError("MinerU content_list.json must be a list")
    return data


def _infer_page_sizes(items: list[dict[str, Any]]) -> dict[int, tuple[float, float]]:
    page_sizes: dict[int, tuple[float, float]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        bbox = _normalize_bbox(item.get("bbox"))
        if bbox is None:
            continue
        page = int(item.get("page_idx") or 0) + 1
        width, height = page_sizes.get(page, (0.0, 0.0))
        page_sizes[page] = (max(width, bbox[2]), max(height, bbox[3]))
    return page_sizes


def _bbox_page_size(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return (max(bbox[2], 1.0), max(bbox[3], 1.0))


def _normalize_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _content_type_for_mineru_type(source_type: str):
    if source_type == "table":
        return "table"
    if source_type in {"image", "chart"}:
        return "image"
    return "text"


def _keep_mineru_visual_item(item: dict[str, Any]) -> bool:
    bbox = _normalize_bbox(item.get("bbox"))
    if bbox is None:
        return False
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    if area < VISION.MIN_AREA_PT2:
        return False
    source_type = str(item.get("type") or "").strip()
    if source_type == "chart":
        return True
    has_semantic_content = bool(
        str(item.get("content") or "").strip()
        or _string_list(item.get("image_caption"))
        or _string_list(item.get("image_footnote"))
    )
    return has_semantic_content or area >= VISION.MIN_AREA_PT2 * 4


def _raw_content_for_item(item: dict[str, Any]) -> str:
    source_type = str(item.get("type") or "").strip()
    if source_type == "table":
        return _table_content(item)
    if source_type == "chart":
        return _image_like_content(
            item,
            content_key="content",
            caption_key="chart_caption",
            footnote_key="chart_footnote",
        )
    if source_type == "image":
        return _image_like_content(
            item,
            content_key="content",
            caption_key="image_caption",
            footnote_key="image_footnote",
        )
    if source_type == "list":
        return "\n".join(str(value).strip() for value in item.get("list_items") or [] if str(value).strip())
    return str(item.get("text") or "").strip()


def _table_content(item: dict[str, Any]) -> str:
    parts: list[str] = []
    captions = _string_list(item.get("table_caption"))
    if captions:
        parts.append("\n".join(captions))
    body = str(item.get("table_body") or "").strip()
    if body:
        parts.append(html_table_to_markdown(body) if "<table" in body.lower() else body)
    footnotes = _string_list(item.get("table_footnote"))
    if footnotes:
        parts.append("\n".join(footnotes))
    return "\n\n".join(part for part in parts if part.strip()).strip()


def _image_like_content(
    item: dict[str, Any],
    *,
    content_key: str,
    caption_key: str,
    footnote_key: str,
) -> str:
    parts = ["[Vision reparse needed] MinerU visual extraction is used only as a placeholder."]
    if item.get("img_path"):
        parts.append(f"[image: {item['img_path']}]")
    return "\n\n".join(part for part in parts if part.strip()).strip()


def _metadata_for_item(
    item: dict[str, Any],
    *,
    item_index: int,
    page_width: float,
    page_height: float,
) -> dict[str, Any]:
    source_type = str(item.get("type") or "").strip()
    metadata: dict[str, Any] = {
        "parser": "mineru_vlm",
        "mineru_type": source_type,
        "mineru_item_index": item_index,
        "page_idx": item.get("page_idx"),
        "page_width": page_width,
        "page_height": page_height,
    }
    for key in (
        "text_level",
        "sub_type",
        "img_path",
        "table_caption",
        "table_footnote",
        "image_caption",
        "image_footnote",
        "chart_caption",
        "chart_footnote",
    ):
        if key in item:
            metadata[key] = item[key]
    if source_type == "table":
        metadata["table_html"] = str(item.get("table_body") or "")
    if source_type in {"image", "chart"}:
        metadata["vision_needed"] = True
        metadata["mineru_visual_content"] = str(item.get("content") or "")
    return metadata


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._current_row = []
        elif tag in {"td", "th"} and self._current_row is not None:
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._current_cell is not None and self._current_row is not None:
            self._current_row.append(_clean_cell_text("".join(self._current_cell)))
            self._current_cell = None
        elif tag == "tr" and self._current_row is not None:
            if any(cell.strip() for cell in self._current_row):
                self.rows.append(self._current_row)
            self._current_row = None


def html_table_to_markdown(html: str) -> str:
    parser = _TableHTMLParser()
    parser.feed(html)
    rows = parser.rows
    if not rows:
        return re.sub(r"\s+", " ", html).strip()
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = normalized[0]
    separator = ["---"] * width
    body = normalized[1:]
    return "\n".join(_markdown_row(row) for row in [header, separator, *body])


def _markdown_row(row: list[str]) -> str:
    return "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"


def _clean_cell_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
