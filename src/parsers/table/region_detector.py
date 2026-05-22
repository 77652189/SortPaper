from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass

from src.parsers.config import TABLE_PARSER as CFG
from src.parsers.table.diagnostics import bbox_iou, bbox_overlap_ratio
from src.parsers.table.models import BBox, TableEvidence, TableRegion

logger = logging.getLogger(__name__)

TABLE_CAPTION_RE = re.compile(
    r"^(?:\d+\s+)?(?:table|tab\.)\s*s?\d+(?=\b|[A-Z])"
    r"|^(?:\d+\s+)?supplementary\s*table\s*s?\d+(?=\b|[A-Z])"
    r"|^(?:\d+\s+)?table\s*\d+[.:]"
    r"|^(?:\d+\s+)?表\s*[S号]?\s*\d+\b"
    r"|^(?:\d+\s+)?附表\s*[S号]?\s*\d+\b",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?%?|[<>]=?\s*\d+|p\s*[<=>]\s*0?\.\d+", re.IGNORECASE)
ROW_LABEL_RE = re.compile(r"^[A-Za-z]{0,4}\d+[A-Za-z]?$")
BODY_WORDS = {
    "abstract",
    "introduction",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "references",
}
COMMON_PROSE_WORDS = {
    "the",
    "of",
    "and",
    "was",
    "were",
    "that",
    "this",
    "from",
    "with",
    "for",
    "thereby",
    "respectively",
    "synthesis",
    "pathway",
    "broth",
    "strain",
    "metabolic",
    "genes",
    "yield",
    "introduced",
    "strengthened",
}


@dataclass(frozen=True)
class _TextLine:
    text: str
    bbox: BBox
    words: list[dict]


class TableRegionDetector:
    """Generate recall-oriented candidate table regions from deterministic evidence."""

    MIN_KEEP_SCORE = CFG.REGION_MIN_KEEP_SCORE
    MERGE_IOU = CFG.REGION_MERGE_IOU
    LINE_SNAP = CFG.REGION_LINE_SNAP
    CAPTION_SCAN_DEPTH = CFG.REGION_CAPTION_SCAN_DEPTH

    def __init__(self, pdf_path: str) -> None:
        self.pdf_path = pdf_path

    def detect(self) -> list[TableRegion]:
        import pdfplumber

        regions: list[TableRegion] = []
        candidate_pages = self._prefilter_caption_pages()
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                if candidate_pages is not None and page_index not in candidate_pages:
                    continue
                try:
                    regions.extend(self.detect_page(page, page_index))
                except Exception:
                    logger.warning("table region detection failed on page %d", page_index, exc_info=True)
        return self._suppress_cross_page_caption_duplicates(regions)

    def _prefilter_caption_pages(self) -> set[int] | None:
        """Return 1-based pages likely to contain table captions, or None to scan all.

        pdfplumber's extract_words can be very slow on some PDFs. When the table
        pipeline is keyword-anchored, a fast PyMuPDF text pass can skip pages with
        no Table/Tab. caption before paying pdfplumber's page parsing cost.
        """
        if not CFG.KEYWORD_ENTRY_ENABLED or not getattr(CFG, "TABLE_PAGE_PREFILTER_ENABLED", True):
            return None
        try:
            import fitz
        except Exception:
            return None

        pages: set[int] = set()
        try:
            with fitz.open(self.pdf_path) as doc:
                broad_pages: set[int] = set()
                for index, page in enumerate(doc, start=1):
                    text = page.get_text("text") or ""
                    anchored, broad = self._caption_page_matches(text)
                    if broad:
                        broad_pages.add(index)
                    if anchored:
                        pages.add(index)
        except Exception:
            logger.debug("table page prefilter failed; scanning all pages", exc_info=True)
            return None
        return pages or broad_pages

    @staticmethod
    def _contains_table_caption_text(text: str) -> bool:
        anchored, broad = TableRegionDetector._caption_page_matches(text)
        return anchored or broad

    @staticmethod
    def _caption_page_matches(text: str) -> tuple[bool, bool]:
        anchored = any(TABLE_CAPTION_RE.search(line.strip()) for line in text.splitlines())
        broad = bool(re.search(
            r"\b(?:table|tab\.)\s*s?\d+(?=\b|[A-Z])"
            r"|\bsupplementary\s*table\s*s?\d+(?=\b|[A-Z])"
            r"|表\s*[S号]?\s*\d+\b"
            r"|附表\s*[S号]?\s*\d+\b",
            text,
            re.IGNORECASE,
        ))
        return anchored, broad

    def detect_page(self, page, page_index: int) -> list[TableRegion]:
        page_width = float(getattr(page, "width", 0) or 0)
        page_height = float(getattr(page, "height", 0) or 0)
        if page_width <= 0 or page_height <= 0:
            return []

        words = page.extract_words() or []
        lines = self._build_lines(words)
        candidates: list[BBox] = []
        pre_scored: list[TableRegion] = []
        caption_candidates = self._caption_candidates(page, lines, page_width, page_height)
        candidates.extend(caption_candidates)
        page_bbox = tuple(float(value) for value in getattr(page, "bbox", (0.0, 0.0, page_width, page_height)))
        pre_scored.extend(
            self._clip_regions_to_page(
                self._rotated_table_regions(page, page_index, page_width, page_height),
                page_bbox,
            )
        )
        if not CFG.KEYWORD_ENTRY_ENABLED:
            candidates.extend(self._line_candidates(page, page_width, page_height))
            candidates.extend(self._alignment_candidates(lines, page_width, page_height))

        scored: list[TableRegion] = []
        clipped_candidates = [
            clipped
            for bbox in candidates
            if (clipped := self._clip_bbox_to_page(bbox, page_bbox)) is not None
        ]
        for bbox in self._merge_candidates(clipped_candidates):
            evidence = self._score_candidate(page, lines, bbox, page_width)
            score = round(max(0.0, min(1.0, sum(item.score for item in evidence))), 3)
            if score < self.MIN_KEEP_SCORE:
                continue
            scored.append(TableRegion(
                page=page_index,
                bbox=bbox,
                confidence=score,
                evidence=evidence,
            ))

        scored = self._suppress_overlapping_regions(scored + pre_scored)
        scored.sort(key=lambda region: (region.page, region.bbox[1], region.bbox[0]))
        return scored

    @staticmethod
    def _clip_bbox_to_page(bbox: BBox, page_bbox: BBox) -> BBox | None:
        px0, py0, px1, py1 = page_bbox
        x0, y0, x1, y1 = bbox
        clipped = (
            max(px0, float(x0)),
            max(py0, float(y0)),
            min(px1, float(x1)),
            min(py1, float(y1)),
        )
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            return None
        return clipped

    @classmethod
    def _clip_regions_to_page(
        cls,
        regions: list[TableRegion],
        page_bbox: BBox,
    ) -> list[TableRegion]:
        clipped_regions: list[TableRegion] = []
        for region in regions:
            clipped_bbox = cls._clip_bbox_to_page(region.bbox, page_bbox)
            if clipped_bbox is None:
                continue
            clipped_regions.append(TableRegion(
                page=region.page,
                bbox=clipped_bbox,
                confidence=region.confidence,
                evidence=region.evidence,
                source=region.source,
            ))
        return clipped_regions

    def _caption_candidates(
        self,
        page,
        lines: list[_TextLine],
        page_width: float,
        page_height: float,
    ) -> list[BBox]:
        candidates: list[BBox] = []
        for line in lines:
            if not TABLE_CAPTION_RE.search(line.text.strip()):
                continue
            table_lines = self._table_lines_after_caption(line, lines)
            wide_edges = self._wide_edges_after_caption(page, line, page_width, page_height)
            row_bounds = self._stable_table_line_bounds(table_lines)

            if row_bounds:
                content_x0 = min(line.bbox[0], row_bounds[0])
                content_x1 = max(line.bbox[2], row_bounds[1])
                if wide_edges:
                    edge_x0 = min(float(edge.get("x0", content_x0)) for edge in wide_edges)
                    edge_x1 = max(float(edge.get("x1", content_x1)) for edge in wide_edges)
                    row_width = max(1.0, row_bounds[1] - row_bounds[0])
                    edge_width = max(1.0, edge_x1 - edge_x0)
                    content_x0 = min(content_x0, edge_x0)
                    if edge_width <= row_width * CFG.HORIZONTAL_LINE_BBOX_WIDTH_RATIO:
                        content_x1 = max(content_x1, edge_x1)
            elif wide_edges:
                content_x0 = min([line.bbox[0]] + [float(edge.get("x0", line.bbox[0])) for edge in wide_edges])
                content_x1 = max([line.bbox[2]] + [float(edge.get("x1", line.bbox[2])) for edge in wide_edges])
            else:
                content_x0 = min([item.bbox[0] for item in table_lines] + [line.bbox[0]])
                content_x1 = max([item.bbox[2] for item in table_lines] + [line.bbox[2], page_width * 0.55])

            x0 = max(0.0, content_x0 - 8)
            y0 = max(0.0, line.bbox[1] - 4)
            x1 = min(page_width, content_x1 + 8)
            y1 = self._caption_candidate_bottom(line, table_lines, wide_edges, page_height)
            candidates.append((x0, y0, x1, y1))
        return candidates

    def _table_lines_after_caption(
        self,
        caption: _TextLine,
        lines: list[_TextLine],
    ) -> list[_TextLine]:
        nearby = [
            line for line in lines
            if line.bbox[1] > caption.bbox[1]
            and line.bbox[1] <= caption.bbox[3] + self.CAPTION_SCAN_DEPTH
        ]

        table_lines: list[_TextLine] = []
        last_bottom = caption.bbox[3]
        for line in nearby:
            gap = line.bbox[1] - last_bottom
            is_data_row = self._is_table_data_row(line)
            if table_lines and gap > 95 and not is_data_row:
                break

            if is_data_row or (
                self._is_table_like_line(line)
                and not self._is_prose_heavy_line(line)
            ):
                table_lines.append(line)
                last_bottom = line.bbox[3]
                continue

            first_word = line.text.strip().split()[0].strip(":.").lower() if line.text.strip() else ""
            if table_lines and first_word in BODY_WORDS:
                break
            if table_lines and self._is_prose_heavy_line(line):
                break

        return table_lines

    def _wide_edges_after_caption(
        self,
        page,
        caption: _TextLine,
        page_width: float,
        page_height: float,
    ) -> list[dict]:
        return [
            edge for edge in getattr(page, "edges", [])
            if edge.get("orientation") == "h"
            and caption.bbox[3] - 8 <= float(edge.get("top", 0)) <= min(
                page_height,
                caption.bbox[3] + self.CAPTION_SCAN_DEPTH,
            )
            and float(edge.get("x1", 0)) - float(edge.get("x0", 0)) >= page_width * 0.20
        ]

    def _caption_candidate_bottom(
        self,
        caption: _TextLine,
        table_lines: list[_TextLine],
        wide_edges: list[dict],
        page_height: float,
    ) -> float:
        bottoms = [caption.bbox[3] + 120]
        if table_lines:
            bottoms.append(max(line.bbox[3] for line in table_lines) + 24)
        if wide_edges:
            bottoms.append(max(float(edge.get("top", caption.bbox[3])) for edge in wide_edges) + 60)
        return min(page_height, max(bottoms))

    def _line_candidates(self, page, page_width: float, page_height: float) -> list[BBox]:
        wide_edges = [
            edge for edge in getattr(page, "edges", [])
            if edge.get("orientation") == "h"
            and float(edge.get("x1", 0)) - float(edge.get("x0", 0)) >= page_width * 0.20
        ]
        if not wide_edges:
            return []

        wide_edges.sort(key=lambda edge: float(edge.get("top", 0)))
        groups: list[list[dict]] = []
        for edge in wide_edges:
            edge_top = float(edge.get("top", 0))
            placed = False
            for group in reversed(groups):
                if abs(edge_top - float(group[-1].get("top", 0))) <= 140:
                    group.append(edge)
                    placed = True
                    break
            if not placed:
                groups.append([edge])

        candidates: list[BBox] = []
        for group in groups:
            if len(group) < 2:
                continue
            x0 = max(0.0, min(float(edge.get("x0", 0)) for edge in group) - 4)
            x1 = min(page_width, max(float(edge.get("x1", 0)) for edge in group) + 4)
            y0 = max(0.0, min(float(edge.get("top", 0)) for edge in group) - 16)
            y1 = min(page_height, max(float(edge.get("top", 0)) for edge in group) + 100)
            candidates.append((x0, y0, x1, y1))
        return candidates

    def _alignment_candidates(
        self,
        lines: list[_TextLine],
        page_width: float,
        page_height: float,
    ) -> list[BBox]:
        table_like_runs: list[list[_TextLine]] = []
        current: list[_TextLine] = []

        for line in lines:
            if self._line_alignment_score(line) >= 2:
                current.append(line)
                continue
            if len(current) >= 3:
                table_like_runs.append(current)
            current = []
        if len(current) >= 3:
            table_like_runs.append(current)

        candidates: list[BBox] = []
        for run in table_like_runs:
            row_bounds = self._stable_table_line_bounds(run)
            if row_bounds:
                raw_x0, raw_x1 = row_bounds
            else:
                raw_x0 = min(line.bbox[0] for line in run)
                raw_x1 = max(line.bbox[2] for line in run)
            x0 = max(0.0, raw_x0 - 8)
            y0 = max(0.0, min(line.bbox[1] for line in run) - 8)
            x1 = min(page_width, raw_x1 + 8)
            y1 = min(page_height, max(line.bbox[3] for line in run) + 8)
            candidates.append((x0, y0, x1, y1))
        return candidates

    def _rotated_table_regions(
        self,
        page,
        page_index: int,
        page_width: float,
        page_height: float,
    ) -> list[TableRegion]:
        chars = list(getattr(page, "chars", []) or [])
        rotated = [char for char in chars if self._is_rotated_char(char)]
        if len(rotated) < 20:
            return []

        groups = self._group_rotated_chars(rotated)
        caption_groups: list[tuple[int, str]] = []
        for index, group in enumerate(groups):
            text = self._rotated_group_text(group).strip()
            if TABLE_CAPTION_RE.search(text):
                caption_groups.append((index, text))
        if not caption_groups:
            return []

        regions: list[TableRegion] = []
        for caption_index, caption_text in caption_groups:
            cluster = self._rotated_table_cluster(groups, caption_index)
            if sum(len(group) for group in cluster) < 20:
                continue
            bbox = self._chars_bbox([char for group in cluster for char in group], page_width, page_height, margin=8.0)
            if bbox is None:
                continue
            evidence = [
                TableEvidence(
                    "caption",
                    0.35,
                    "rotated_table_caption",
                    self._chars_bbox(groups[caption_index], page_width, page_height, margin=2.0),
                    {"text": caption_text},
                ),
                TableEvidence(
                    "rotation",
                    0.30,
                    "rotated_text_block",
                    bbox,
                    {"chars": sum(len(group) for group in cluster)},
                ),
            ]
            score = round(max(0.0, min(1.0, sum(item.score for item in evidence))), 3)
            regions.append(TableRegion(
                page=page_index,
                bbox=bbox,
                confidence=score,
                evidence=evidence,
            ))
        return regions

    @staticmethod
    def _is_rotated_char(char: dict) -> bool:
        matrix = char.get("matrix", [1, 0, 0, 1, 0, 0])
        try:
            return abs(float(matrix[1])) > 0.1 or abs(float(matrix[2])) > 0.1
        except (TypeError, ValueError, IndexError):
            return False

    @staticmethod
    def _group_rotated_chars(chars: list[dict]) -> list[list[dict]]:
        sorted_chars = sorted(chars, key=lambda char: float(char.get("x0", 0)))
        groups: list[list[dict]] = []
        for char in sorted_chars:
            x0 = float(char.get("x0", 0))
            for group in reversed(groups):
                group_x = sum(float(item.get("x0", 0)) for item in group) / len(group)
                if abs(x0 - group_x) <= 6.0:
                    group.append(char)
                    break
            else:
                groups.append([char])
        return groups

    @staticmethod
    def _rotated_group_text(group: list[dict]) -> str:
        return "".join(
            str(char.get("text", ""))
            for char in sorted(group, key=lambda char: -float(char.get("top", 0)))
        )

    def _rotated_table_cluster(self, groups: list[list[dict]], caption_index: int) -> list[list[dict]]:
        start = caption_index
        end = caption_index
        for direction in (-1, 1):
            index = caption_index + direction
            previous = caption_index
            while 0 <= index < len(groups):
                gap = abs(self._group_center_x(groups[index]) - self._group_center_x(groups[previous]))
                if gap > 90.0:
                    break
                if len(groups[index]) >= 2:
                    start = min(start, index)
                    end = max(end, index)
                    previous = index
                index += direction
        return groups[start:end + 1]

    @staticmethod
    def _group_center_x(group: list[dict]) -> float:
        return sum(float(char.get("x0", 0)) for char in group) / max(1, len(group))

    @staticmethod
    def _chars_bbox(
        chars: list[dict],
        page_width: float,
        page_height: float,
        margin: float = 0.0,
    ) -> BBox | None:
        if not chars:
            return None
        x0 = max(0.0, min(float(char.get("x0", 0)) for char in chars) - margin)
        y0 = max(0.0, min(float(char.get("top", char.get("y0", 0))) for char in chars) - margin)
        x1 = min(page_width, max(float(char.get("x1", char.get("x0", 0))) for char in chars) + margin)
        y1 = min(page_height, max(float(char.get("bottom", char.get("top", 0))) for char in chars) + margin)
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)

    def _score_candidate(
        self,
        page,
        lines: list[_TextLine],
        bbox: BBox,
        page_width: float,
    ) -> list[TableEvidence]:
        evidence: list[TableEvidence] = []
        in_lines = [line for line in lines if bbox_iou(line.bbox, bbox) > 0 or self._center_in_bbox(line.bbox, bbox)]
        text = " ".join(line.text for line in in_lines).strip()

        caption_lines = [
            line for line in lines
            if TABLE_CAPTION_RE.search(line.text.strip())
            and line.bbox[3] <= bbox[3]
            and abs(line.bbox[3] - bbox[1]) <= 180
        ]
        if caption_lines:
            evidence.append(TableEvidence(
                "caption",
                0.35,
                "near_table_caption",
                caption_lines[0].bbox,
                {"text": caption_lines[0].text},
            ))

        wide_h = [
            edge for edge in getattr(page, "edges", [])
            if edge.get("orientation") == "h"
            and bbox[1] - 3 <= float(edge.get("top", 0)) <= bbox[3] + 3
            and float(edge.get("x1", 0)) - float(edge.get("x0", 0)) >= page_width * 0.20
        ]
        if len(wide_h) >= 3:
            evidence.append(TableEvidence("lines", 0.30, "multiple_wide_horizontal_lines", bbox, {"count": len(wide_h)}))
        elif len(wide_h) >= 2:
            evidence.append(TableEvidence("lines", 0.20, "wide_horizontal_lines", bbox, {"count": len(wide_h)}))

        aligned_lines = [line for line in in_lines if self._line_alignment_score(line) >= 2]
        if len(aligned_lines) >= 3:
            evidence.append(TableEvidence("alignment", 0.30, "repeated_x_alignment", bbox, {"lines": len(aligned_lines)}))
        elif len(aligned_lines) >= 2:
            evidence.append(TableEvidence("alignment", 0.15, "some_x_alignment", bbox, {"lines": len(aligned_lines)}))

        tokens = text.split()
        if tokens:
            numeric_ratio = sum(1 for token in tokens if NUMBER_RE.search(token)) / len(tokens)
            short_ratio = sum(1 for token in tokens if len(token.strip(".,;:()[]")) <= 12) / len(tokens)
            if numeric_ratio >= 0.20:
                evidence.append(TableEvidence("content", 0.15, "high_numeric_density", bbox, {"ratio": round(numeric_ratio, 3)}))
            if short_ratio >= 0.70 and len(tokens) >= 8:
                evidence.append(TableEvidence("content", 0.10, "short_cell_like_tokens", bbox, {"ratio": round(short_ratio, 3)}))

        long_lines = [line for line in in_lines if len(line.text.split()) >= 18]
        if len(long_lines) >= 3:
            evidence.append(TableEvidence("negative", -0.30, "long_sentence_block", bbox, {"lines": len(long_lines)}))

        first_words = {line.text.strip().split()[0].strip(":.").lower() for line in in_lines if line.text.strip()}
        if first_words & BODY_WORDS:
            evidence.append(TableEvidence("negative", -0.20, "section_like_body_region", bbox))

        return evidence

    def _build_lines(self, words: list[dict]) -> list[_TextLine]:
        if not words:
            return []
        sorted_words = sorted(words, key=lambda word: (float(word.get("top", 0)), float(word.get("x0", 0))))
        rows: list[list[dict]] = []
        for word in sorted_words:
            top = float(word.get("top", 0))
            for row in reversed(rows):
                if abs(top - float(row[0].get("top", 0))) <= self.LINE_SNAP:
                    row.append(word)
                    break
            else:
                rows.append([word])

        lines: list[_TextLine] = []
        for row in rows:
            row = sorted(row, key=lambda word: float(word.get("x0", 0)))
            text = " ".join(str(word.get("text", "")) for word in row).strip()
            if not text:
                continue
            lines.append(_TextLine(
                text=text,
                bbox=(
                    min(float(word.get("x0", 0)) for word in row),
                    min(float(word.get("top", 0)) for word in row),
                    max(float(word.get("x1", word.get("x0", 0))) for word in row),
                    max(float(word.get("bottom", word.get("top", 0))) for word in row),
                ),
                words=row,
            ))
        return lines

    @staticmethod
    def _line_alignment_score(line: _TextLine) -> int:
        if len(line.words) < 2:
            return 0
        bins = Counter(round(float(word.get("x0", 0)) / 12) for word in line.words)
        return len(bins)

    def _is_table_like_line(self, line: _TextLine) -> bool:
        if self._line_alignment_score(line) >= 2:
            return True
        tokens = line.text.split()
        if not tokens:
            return False
        numeric_ratio = sum(1 for token in tokens if NUMBER_RE.search(token)) / len(tokens)
        short_ratio = sum(1 for token in tokens if len(token.strip(".,;:()[]")) <= 12) / len(tokens)
        return numeric_ratio >= 0.20 or (short_ratio >= 0.70 and len(tokens) >= 3)

    @staticmethod
    def _is_table_data_row(line: _TextLine) -> bool:
        tokens = [token.strip(".,;:()[]") for token in line.text.split() if token.strip(".,;:()[]")]
        if len(tokens) < 2:
            return False
        first = tokens[0]
        numeric_count = sum(1 for token in tokens[1:] if NUMBER_RE.search(token))
        return bool(ROW_LABEL_RE.match(first)) and numeric_count >= 2

    @staticmethod
    def _is_prose_heavy_line(line: _TextLine) -> bool:
        tokens = [token.strip(".,;:()[]").lower() for token in line.text.split()]
        if len(tokens) < 14:
            return False
        prose_words = sum(1 for token in tokens if token in COMMON_PROSE_WORDS)
        return prose_words >= 5

    def _stable_table_line_bounds(self, lines: list[_TextLine]) -> tuple[float, float] | None:
        bounds = [
            bbox for line in lines
            if self._is_table_data_row(line)
            for bbox in [self._trimmed_table_row_bbox(line)]
            if bbox is not None
        ]
        if len(bounds) < 2:
            return None

        x0_values = sorted(bbox[0] for bbox in bounds)
        x1_values = sorted(bbox[2] for bbox in bounds)
        x0 = x0_values[max(0, len(x0_values) // 4)]
        # Use the upper-middle value instead of max so one untrimmed prose row cannot
        # widen the whole region into the adjacent column.
        x1 = x1_values[min(len(x1_values) - 1, int(len(x1_values) * 0.75))]
        if x1 <= x0:
            return None
        return x0, x1

    @staticmethod
    def _trimmed_table_row_bbox(line: _TextLine) -> BBox | None:
        words = sorted(line.words, key=lambda word: float(word.get("x0", 0)))
        if not words:
            return None

        kept: list[dict] = []
        numeric_seen = 0
        for index, word in enumerate(words):
            text = str(word.get("text", "")).strip(".,;:()[]")
            if NUMBER_RE.search(text):
                numeric_seen += 1
            kept.append(word)
            if index + 1 >= len(words):
                continue
            next_word = words[index + 1]
            next_text = str(next_word.get("text", "")).strip(".,;:()[]").lower()
            gap = float(next_word.get("x0", 0)) - float(word.get("x1", word.get("x0", 0)))
            if (
                numeric_seen >= 4
                and gap >= 16
                and next_text in COMMON_PROSE_WORDS
            ):
                break

        if len(kept) < 2:
            return None
        return (
            min(float(word.get("x0", 0)) for word in kept),
            min(float(word.get("top", 0)) for word in kept),
            max(float(word.get("x1", word.get("x0", 0))) for word in kept),
            max(float(word.get("bottom", word.get("top", 0))) for word in kept),
        )

    @staticmethod
    def _center_in_bbox(inner: BBox, outer: BBox) -> bool:
        cx = (inner[0] + inner[2]) / 2
        cy = (inner[1] + inner[3]) / 2
        return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]

    def _merge_candidates(self, candidates: list[BBox]) -> list[BBox]:
        merged: list[BBox] = []
        for bbox in candidates:
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                continue
            for index, existing in enumerate(merged):
                mutual_overlap = min(
                    bbox_overlap_ratio(bbox, existing),
                    bbox_overlap_ratio(existing, bbox),
                )
                if bbox_iou(bbox, existing) >= self.MERGE_IOU or mutual_overlap >= self.MERGE_IOU:
                    merged[index] = (
                        min(existing[0], bbox[0]),
                        min(existing[1], bbox[1]),
                        max(existing[2], bbox[2]),
                        max(existing[3], bbox[3]),
                    )
                    break
            else:
                merged.append(bbox)
        return merged

    @staticmethod
    def _candidate_overlap(a: BBox, b: BBox) -> float:
        return max(
            bbox_iou(a, b),
            bbox_overlap_ratio(a, b),
            bbox_overlap_ratio(b, a),
        )

    def _suppress_overlapping_regions(self, regions: list[TableRegion]) -> list[TableRegion]:
        """Drop duplicate detector regions while preserving the best diagnostic candidate."""
        kept: list[TableRegion] = []
        for region in sorted(regions, key=self._region_rank):
            if any(self._same_region_family(region, existing) for existing in kept):
                continue
            kept.append(region)
        return kept

    def _suppress_cross_page_caption_duplicates(self, regions: list[TableRegion]) -> list[TableRegion]:
        """Drop repeated caption hits across pages without removing explicit continuations."""
        kept: list[TableRegion] = []
        for region in sorted(regions, key=self._region_rank):
            if any(self._is_cross_page_duplicate_caption_region(region, existing) for existing in kept):
                continue
            kept.append(region)
        kept.sort(key=lambda item: (item.page, item.bbox[1], item.bbox[0]))
        return kept

    def _is_cross_page_duplicate_caption_region(self, region: TableRegion, existing: TableRegion) -> bool:
        if region.page == existing.page:
            return False
        if self._caption_key(region) != self._caption_key(existing):
            return False

        region_text = self._caption_text(region)
        existing_text = self._caption_text(existing)
        if not region_text or not existing_text:
            return False
        if self._caption_marks_continuation(region_text) or self._caption_marks_continuation(existing_text):
            return False

        return self._caption_title_compatible(region_text, existing_text)

    def _same_region_family(self, region: TableRegion, existing: TableRegion) -> bool:
        overlap = self._candidate_overlap(region.bbox, existing.bbox)
        if overlap < 0.80:
            if (
                not self._is_vertically_nested_same_table(region.bbox, existing.bbox)
                and not self._is_near_duplicate_caption_region(region, existing)
            ):
                return False
        confidence_gap = abs(region.confidence - existing.confidence)
        return confidence_gap <= 0.30 or min(
            self._bbox_area(region.bbox),
            self._bbox_area(existing.bbox),
        ) <= max(self._bbox_area(region.bbox), self._bbox_area(existing.bbox)) * 0.80

    def _is_near_duplicate_caption_region(self, region: TableRegion, existing: TableRegion) -> bool:
        region_key = self._caption_key(region)
        existing_key = self._caption_key(existing)
        if not region_key or region_key != existing_key:
            return False

        ax0, ay0, ax1, ay1 = region.bbox
        bx0, by0, bx1, by1 = existing.bbox
        x_overlap = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        min_width = min(max(0.0, ax1 - ax0), max(0.0, bx1 - bx0))
        if min_width <= 0 or x_overlap / min_width < 0.60:
            return False

        vertical_gap = max(0.0, max(ay0, by0) - min(ay1, by1))
        return vertical_gap <= 120.0

    @staticmethod
    def _caption_key(region: TableRegion) -> str:
        for item in region.evidence:
            if item.source != "caption":
                continue
            text = str(item.details.get("text", "")).strip().lower()
            match = re.search(r"(?:supplementary\s+table|table|tab\.)\s*(s?\d+)", text, re.IGNORECASE)
            if match:
                return match.group(1).lower()
            if text:
                return re.sub(r"\s+", " ", text)[:80]
        return ""

    @staticmethod
    def _caption_text(region: TableRegion) -> str:
        for item in region.evidence:
            if item.source == "caption":
                return str(item.details.get("text", "")).strip()
        return ""

    @staticmethod
    def _caption_marks_continuation(text: str) -> bool:
        return bool(re.search(r"\bcontinued\b|\bcontinuation\b|\bcont\.\b", text, re.IGNORECASE))

    @staticmethod
    def _caption_title_compatible(first: str, second: str) -> bool:
        def normalize(text: str) -> str:
            text = re.sub(
                r"^(?:supplementary\s+table|table|tab\.)\s*s?\d+[.:]?\s*",
                "",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
            return re.sub(r"\s+", " ", text)

        first_norm = normalize(first)
        second_norm = normalize(second)
        if not first_norm or not second_norm:
            return True
        if first_norm == second_norm:
            return True
        shorter, longer = sorted((first_norm, second_norm), key=len)
        return len(shorter) >= 12 and longer.startswith(shorter)

    @staticmethod
    def _is_vertically_nested_same_table(a: BBox, b: BBox) -> bool:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        a_h = max(0.0, ay1 - ay0)
        b_h = max(0.0, by1 - by0)
        a_w = max(0.0, ax1 - ax0)
        b_w = max(0.0, bx1 - bx0)
        if min(a_h, b_h, a_w, b_w) <= 0:
            return False

        y_overlap = max(0.0, min(ay1, by1) - max(ay0, by0))
        x_overlap = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        return (
            y_overlap / min(a_h, b_h) >= 0.90
            and x_overlap / min(a_w, b_w) >= 0.85
        )

    @staticmethod
    def _region_rank(region: TableRegion) -> tuple[float, ...]:
        width = max(0.0, region.bbox[2] - region.bbox[0])
        height = max(0.0, region.bbox[3] - region.bbox[1])
        evidence_count = len(region.evidence)
        evidence_sources = {item.source for item in region.evidence}
        has_caption = 1.0 if "caption" in evidence_sources else 0.0
        has_lines = 1.0 if "lines" in evidence_sources else 0.0
        return (-has_lines, -has_caption, -region.confidence, -evidence_count, -height, width, region.bbox[1])

    @staticmethod
    def _bbox_area(bbox: BBox) -> float:
        x0, y0, x1, y1 = bbox
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)
