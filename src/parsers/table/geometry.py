from __future__ import annotations

import logging

from src.parsers.config import TABLE_PARSER as CFG

logger = logging.getLogger(__name__)


def refine_table_bbox(
    pdf_page,
    bbox: tuple,
    page_width: float,
    page_height: float,
) -> tuple[float, float, float, float]:
    words = pdf_page.extract_words()
    title_y = bbox[1] - page_height * CFG.REFINE_BBOX_SEARCH_UPWARD_RATIO
    for word in words:
        if word["text"].strip().lower().startswith(("table", "tab.")):
            word_y = float(word["bottom"])
            if word_y < bbox[1] and word_y > title_y:
                title_y = word_y

    region_top = max(0.0, title_y)
    region_bot, clip_x0, clip_x1 = detect_columnar_table_bounds(
        pdf_page, region_top, page_width, page_height,
    )
    if region_bot < bbox[3]:
        region_bot = max(bbox[3], min(page_height, region_top + 400))
    return (
        min(bbox[0], clip_x0),
        min(bbox[1], region_top),
        max(bbox[2], clip_x1),
        max(bbox[3], region_bot),
    )


def detect_columnar_table_bounds(
    pdf_page,
    region_top: float,
    page_width: float,
    page_height: float,
) -> tuple[float, float, float]:
    words = pdf_page.extract_words()
    if not words:
        return min(region_top + 100, page_height), 0.0, page_width

    below = [word for word in words if word["top"] >= region_top]
    if not below:
        return min(region_top + 100, page_height), 0.0, page_width

    below.sort(key=lambda word: (word["top"], word["x0"]))
    lines: list[list[dict]] = []
    for word in below:
        if not lines or abs(word["top"] - lines[-1][0]["top"]) > 3:
            lines.append([word])
        else:
            lines[-1].append(word)

    if len(lines) < 2:
        return min(region_top + 100, page_height), 0.0, page_width

    x_tolerance = CFG.COLUMN_ALIGN_X_TOLERANCE
    min_shared_cols = CFG.COLUMN_ALIGN_MIN_SHARED

    def _xsnap(line: list[dict]) -> set[int]:
        return {
            round(word["x0"] / x_tolerance) * x_tolerance
            for word in line
        }

    prev_xs = _xsnap(lines[0])
    table_lines: list[list[dict]] = [lines[0]]
    gap_count = 0

    for line in lines[1:]:
        cur_xs = _xsnap(line)
        shared = sum(
            1 for px in prev_xs
            if any(abs(px - cx) <= x_tolerance for cx in cur_xs)
        )
        if shared >= min_shared_cols:
            table_lines.append(line)
            prev_xs = cur_xs
            gap_count = 0
        elif gap_count == 0 and table_lines and len(table_lines) >= 2:
            table_lines.append(line)
            gap_count = 1
        elif table_lines and len(table_lines) >= 2:
            break
        else:
            prev_xs = cur_xs
            table_lines = [line]

    if len(table_lines) < 2:
        logger.debug(
            "detect_columnar_table_bounds: only %d lines -> fallback",
            len(table_lines),
        )
        return min(region_top + 100, page_height), 0.0, page_width

    table_bottom = max(word["bottom"] for line in table_lines for word in line)
    all_x0 = [word["x0"] for line in table_lines for word in line]
    all_x1 = [word["x1"] for line in table_lines for word in line]
    table_x0 = max(0.0, min(all_x0) - 5)
    table_x1 = min(page_width, max(all_x1) + 5)

    logger.debug(
        "detect_columnar_table_bounds: region_top=%.0f table_bottom=%.0f lines=%d height=%.0f",
        region_top, table_bottom, len(table_lines), table_bottom - region_top,
    )
    return min(table_bottom + 3, page_height), table_x0, table_x1


def count_threeline_lines(
    pdf_page,
    bbox: tuple[float, float, float, float],
) -> tuple[int, list[float], int, float, float]:
    x0, top, x1, bottom = bbox
    page_width = float(pdf_page.width) if pdf_page.width else 500.0
    min_h_width = max((x1 - x0) * 0.25, page_width * 0.15)
    bbox_height = bottom - top
    v_height_min = max(20.0, bbox_height * 0.10)

    seen_y_bins: dict[int, float] = {}
    v_count = 0
    h_x0_min = float("inf")
    h_x1_max = float("-inf")

    for edge in pdf_page.edges:
        orientation = edge.get("orientation", "")
        ey0 = float(edge.get("top", 0))
        ey1 = float(edge.get("bottom", ey0))
        ex0 = float(edge.get("x0", 0))
        ex1 = float(edge.get("x1", ex0))

        if orientation == "h":
            ey = ey0
            if not (top - 2 <= ey <= bottom + 2):
                continue
            if abs(ex1 - ex0) < min_h_width:
                continue
            y_bin = round(ey / 10)
            if y_bin not in seen_y_bins:
                seen_y_bins[y_bin] = ey
            h_x0_min = min(h_x0_min, ex0)
            h_x1_max = max(h_x1_max, ex1)

        elif orientation == "v":
            if not (x0 - 2 <= ex0 <= x1 + 2):
                continue
            v_top = min(ey0, ey1)
            v_bottom = max(ey0, ey1)
            if v_top > bottom or v_bottom < top:
                continue
            if (v_bottom - v_top) < v_height_min:
                continue
            v_count += 1

    h_ys = sorted(seen_y_bins.values())
    if h_x0_min == float("inf"):
        h_x0_min, h_x1_max = x0, x1
    return len(h_ys), h_ys, v_count, h_x0_min, h_x1_max
