from __future__ import annotations

import logging
import re
from collections import defaultdict
from statistics import median

from src.parsers.config import TABLE_PARSER as CFG

logger = logging.getLogger(__name__)


def find_col_ranges(
    words: list[dict],
    min_gap_pt: float = 8.0,
) -> list[tuple[float, float]]:
    if not words:
        return []

    sorted_words = sorted(words, key=lambda word: float(word.get("x0", 0)))
    spans = [
        (float(word.get("x0", 0)), float(word.get("x1", 0)))
        for word in sorted_words
    ]

    merged: list[tuple[float, float]] = []
    cur_x0, cur_x1 = spans[0]
    for span_x0, span_x1 in spans[1:]:
        if span_x0 - cur_x1 < min_gap_pt:
            cur_x1 = max(cur_x1, span_x1)
        else:
            merged.append((cur_x0, cur_x1))
            cur_x0, cur_x1 = span_x0, span_x1
    merged.append((cur_x0, cur_x1))
    return merged


def word_to_col(word: dict, col_ranges: list[tuple[float, float]]) -> int:
    if not col_ranges:
        return -1

    word_center = (
        float(word.get("x0", 0)) + float(word.get("x1", 0))
    ) / 2
    best_idx = 0
    best_dist = float("inf")

    for idx, (col_x0, col_x1) in enumerate(col_ranges):
        if col_x0 <= word_center <= col_x1:
            return idx
        dist = min(abs(word_center - col_x0), abs(word_center - col_x1))
        if dist < best_dist:
            best_dist = dist
            best_idx = idx

    return best_idx


def build_data_rows(
    words: list[dict],
    col_ranges: list[tuple[float, float]],
    row_gap_pt: float = 6.0,
) -> list[list[str]]:
    if not words:
        return []

    def _top(word: dict) -> float:
        return float(word.get("top", word.get("y0", 0)))

    sorted_words = sorted(words, key=lambda word: (_top(word), float(word.get("x0", 0))))
    rows_raw: list[list[dict]] = []
    current_row = [sorted_words[0]]
    current_y = _top(sorted_words[0])

    for word in sorted_words[1:]:
        word_y = _top(word)
        if word_y - current_y > row_gap_pt:
            rows_raw.append(current_row)
            current_row = [word]
            current_y = word_y
        else:
            current_row.append(word)
    rows_raw.append(current_row)

    result: list[list[str]] = []
    num_cols = len(col_ranges)
    for row_words in rows_raw:
        row = [""] * num_cols
        for word in row_words:
            col_idx = word_to_col(word, col_ranges)
            if 0 <= col_idx < num_cols:
                text = word.get("text", "")
                row[col_idx] = (row[col_idx] + " " + text).strip()
        result.append(row)
    return result


NUMBER_TEXT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)%?$")
PLUS_MINUS_TEXT = {"±", "+/-", "卤"}


def _group_words_by_row(words: list[dict], row_gap_pt: float = 4.0) -> list[list[dict]]:
    if not words:
        return []

    def _top(word: dict) -> float:
        return float(word.get("top", word.get("y0", 0)))

    sorted_words = sorted(words, key=lambda word: (_top(word), float(word.get("x0", 0))))
    rows: list[list[dict]] = []
    current = [sorted_words[0]]
    current_y = _top(sorted_words[0])
    for word in sorted_words[1:]:
        word_y = _top(word)
        if abs(word_y - current_y) > row_gap_pt:
            rows.append(current)
            current = [word]
            current_y = word_y
        else:
            current.append(word)
            current_y = (current_y * (len(current) - 1) + word_y) / len(current)
    rows.append(current)
    return [sorted(row, key=lambda word: float(word.get("x0", 0))) for row in rows]


def _is_number_token(text: str) -> bool:
    return bool(NUMBER_TEXT_RE.match(text.strip()))


def _is_plus_minus_token(text: str) -> bool:
    return text.strip() in PLUS_MINUS_TEXT


def _word_text(word: dict) -> str:
    return str(word.get("text", "")).strip()


def _cell_from_words(words: list[dict]) -> tuple[str, tuple[float, float]]:
    text = " ".join(_word_text(word) for word in words if _word_text(word)).strip()
    x0 = min(float(word.get("x0", 0.0)) for word in words)
    x1 = max(float(word.get("x1", word.get("x0", 0.0))) for word in words)
    return text, (x0, x1)


def _numeric_error_cells(row_words: list[dict]) -> list[tuple[str, tuple[float, float]]] | None:
    """Return row cells for common scientific rows like key | mean ± sd | mean ± sd."""
    if len(row_words) < 2:
        return None

    cells: list[tuple[str, tuple[float, float]]] = []
    index = 0
    first_text = _word_text(row_words[0])
    if first_text and not _is_number_token(first_text) and not _is_plus_minus_token(first_text):
        cells.append(_cell_from_words([row_words[0]]))
        index = 1

    while index < len(row_words):
        word = row_words[index]
        text = _word_text(word)
        if (
            _is_number_token(text)
            and index + 2 < len(row_words)
            and _is_plus_minus_token(_word_text(row_words[index + 1]))
            and _is_number_token(_word_text(row_words[index + 2]))
        ):
            cells.append(_cell_from_words(row_words[index:index + 3]))
            index += 3
            continue
        if _is_number_token(text):
            cells.append(_cell_from_words([word]))
        index += 1

    numeric_cells = sum(
        1 for text, _span in cells
        if any(_is_number_token(part) for part in text.replace("±", " ± ").replace("卤", " ± ").split())
    )
    if len(cells) >= CFG.MIN_COLS and numeric_cells >= 1:
        return cells
    return None


def _extract_numeric_error_table(
    header_words: list[dict],
    data_words: list[dict],
) -> list[list[str]]:
    row_cells = [
        cells for row in _group_words_by_row(data_words)
        if (cells := _numeric_error_cells(row)) is not None
    ]
    if len(row_cells) < CFG.MIN_DATA_ROWS:
        return []

    widths = [len(cells) for cells in row_cells]
    mode_width = max(set(widths), key=lambda width: (widths.count(width), width))
    usable_rows = [cells for cells in row_cells if len(cells) == mode_width]
    if len(usable_rows) < CFG.MIN_DATA_ROWS or mode_width < CFG.MIN_COLS:
        return []

    col_ranges: list[tuple[float, float]] = []
    for col_idx in range(mode_width):
        x0s = [cells[col_idx][1][0] for cells in usable_rows]
        x1s = [cells[col_idx][1][1] for cells in usable_rows]
        col_ranges.append((float(median(x0s)), float(median(x1s))))

    header_row = [""] * mode_width
    for word in header_words:
        col_idx = word_to_col(word, col_ranges)
        if 0 <= col_idx < mode_width:
            header_row[col_idx] = (header_row[col_idx] + " " + _word_text(word)).strip()

    if sum(1 for cell in header_row if cell) < 2:
        return []
    return [header_row] + [[text for text, _span in cells] for cells in usable_rows]


def merge_multiline_cells(data_rows: list[list[str]]) -> list[list[str]]:
    if not data_rows:
        return data_rows
    n_cols = len(data_rows[0]) if data_rows else 0

    if n_cols >= 4 and n_cols % 2 == 0:
        half = n_cols // 2
        result: list[list[str]] = []
        left_key_idx = -1
        right_key_idx = -1
        left_last_key = ""
        right_last_key = ""

        def _is_pseudo(key: str, char: str, source: str, prev_key: str) -> bool:
            if char:
                return False
            if key.startswith("("):
                return True
            if prev_key.endswith("-"):
                return True
            if source.startswith("("):
                return True
            return False

        def _apply_pseudo(
            prev_row: list[str],
            offset: int,
            key: str,
            char: str,
            source: str,
            prev_key: str,
        ) -> None:
            if key.startswith("("):
                if source:
                    prev_row[offset + 2] = (prev_row[offset + 2] + " " + source).strip()
            elif prev_key.endswith("-"):
                prev_row[offset] = (prev_key + key).strip()
                if char:
                    prev_row[offset + 1] = (prev_row[offset + 1] + " " + char).strip()
                if source:
                    prev_row[offset + 2] = (prev_row[offset + 2] + " " + source).strip()
            else:
                if key:
                    prev_row[offset + 1] = (prev_row[offset + 1] + " " + key).strip()
                if source:
                    prev_row[offset + 2] = (prev_row[offset + 2] + " " + source).strip()

        for row in data_rows:
            left = row[:half]
            right = row[half:]
            left_key = left[0].strip()
            right_key = right[0].strip()
            left_char = left[1].strip() if half > 1 else ""
            right_char = right[1].strip() if half > 1 else ""
            left_source = left[2].strip() if half > 2 else ""
            right_source = right[2].strip() if half > 2 else ""
            left_has = any(cell.strip() for cell in left)
            right_has = any(cell.strip() for cell in right)

            left_pseudo = bool(left_key) and _is_pseudo(left_key, left_char, left_source, left_last_key)
            right_pseudo = bool(right_key) and _is_pseudo(right_key, right_char, right_source, right_last_key)
            left_is_cont = not left_key or left_pseudo
            right_is_cont = not right_key or right_pseudo

            if not left_is_cont and not right_is_cont:
                result.append(list(row))
                left_key_idx = right_key_idx = len(result) - 1
                left_last_key = left_key
                right_last_key = right_key
            elif not left_is_cont and right_is_cont:
                result.append(list(left) + [""] * half)
                left_key_idx = len(result) - 1
                left_last_key = left_key
                if right_has and right_key_idx >= 0:
                    prev = result[right_key_idx]
                    if right_pseudo:
                        _apply_pseudo(prev, half, right_key, right_char, right_source, right_last_key)
                        right_last_key = prev[half]
                    else:
                        for idx, cell in enumerate(right):
                            if cell.strip():
                                prev[half + idx] = (prev[half + idx] + " " + cell).strip()
            elif left_is_cont and not right_is_cont:
                if left_has and left_key_idx >= 0:
                    prev = result[left_key_idx]
                    if left_pseudo:
                        _apply_pseudo(prev, 0, left_key, left_char, left_source, left_last_key)
                        left_last_key = prev[0]
                    else:
                        for idx, cell in enumerate(left):
                            if cell.strip():
                                prev[idx] = (prev[idx] + " " + cell).strip()
                result.append([""] * half + list(right))
                right_key_idx = len(result) - 1
                right_last_key = right_key
            else:
                if left_has and left_key_idx >= 0:
                    prev = result[left_key_idx]
                    if left_pseudo:
                        _apply_pseudo(prev, 0, left_key, left_char, left_source, left_last_key)
                        left_last_key = prev[0]
                    else:
                        for idx, cell in enumerate(left):
                            if cell.strip():
                                prev[idx] = (prev[idx] + " " + cell).strip()
                if right_has and right_key_idx >= 0:
                    prev = result[right_key_idx]
                    if right_pseudo:
                        _apply_pseudo(prev, half, right_key, right_char, right_source, right_last_key)
                        right_last_key = prev[half]
                    else:
                        for idx, cell in enumerate(right):
                            if cell.strip():
                                prev[half + idx] = (prev[half + idx] + " " + cell).strip()
        return result

    result: list[list[str]] = []
    for row in data_rows:
        if not row[0].strip() and result:
            prev = result[-1]
            for idx, cell in enumerate(row):
                if cell.strip() and idx < len(prev):
                    prev[idx] = (prev[idx] + " " + cell).strip()
        else:
            result.append(list(row))
    return result


def extract_threeline(
    pdf_page,
    bbox: tuple[float, float, float, float],
    h_ys: list[float],
    lines_x0: float = 0.0,
    lines_x1: float = 0.0,
) -> list[list[str]]:
    x0, _top, x1, _bottom = bbox
    margin = 1.5
    bbox_width = max(1.0, x1 - x0)
    line_width = max(0.0, lines_x1 - lines_x0)
    if lines_x1 > 0 and line_width <= bbox_width * 1.35:
        eff_x0 = min(x0, lines_x0) - margin
        eff_x1 = max(x1, lines_x1) + margin
    else:
        eff_x0, eff_x1 = x0, x1

    header_bbox = (eff_x0, h_ys[0] + margin, eff_x1, h_ys[1] - margin)
    data_bbox = (eff_x0, h_ys[1] + margin, eff_x1, h_ys[2] - margin)

    try:
        header_words = pdf_page.within_bbox(header_bbox).extract_words()
        data_words = pdf_page.within_bbox(data_bbox).extract_words()
    except Exception:
        logger.debug("extract_threeline: within_bbox failed", exc_info=True)
        return []

    if not header_words:
        return []

    numeric_table = _extract_numeric_error_table(header_words, data_words)
    if numeric_table:
        logger.debug(
            "extract_threeline: numeric-error strategy detected %d cols, data_rows=%d",
            len(numeric_table[0]), len(numeric_table) - 1,
        )
        return numeric_table

    col_ranges = find_col_ranges(header_words, min_gap_pt=CFG.THREELINE_COL_GAP_PT)
    if len(col_ranges) < 2 and data_words:
        col_ranges = find_col_ranges(
            header_words + data_words,
            min_gap_pt=CFG.THREELINE_COL_GAP_PT,
        )
    if len(col_ranges) < CFG.MIN_COLS:
        return []

    num_cols = len(col_ranges)
    header_row = [""] * num_cols
    for word in header_words:
        col_idx = word_to_col(word, col_ranges)
        if 0 <= col_idx < num_cols:
            header_row[col_idx] = (header_row[col_idx] + " " + word.get("text", "")).strip()

    data_rows = build_data_rows(data_words, col_ranges)
    data_rows = merge_multiline_cells(data_rows)
    logger.debug(
        "extract_threeline: detected %d cols, header=%s, data_rows=%d",
        num_cols, header_row, len(data_rows),
    )
    return [header_row] + data_rows


def extract_headerbox(
    pdf_page,
    bbox: tuple[float, float, float, float],
    h_ys: list[float],
    lines_x0: float = 0.0,
    lines_x1: float = 0.0,
) -> list[list[str]]:
    x0, _top, x1, _bottom = bbox
    margin = 1.5
    page_height = float(pdf_page.height)
    data_y_bottom = page_height * 0.95

    if lines_x1 > 0:
        eff_x0 = min(x0, lines_x0) - margin
        eff_x1 = max(x1, lines_x1) + margin
    else:
        eff_x0, eff_x1 = x0 - margin, x1 + margin

    above_lines = [
        edge for edge in pdf_page.edges
        if edge.get("orientation") == "h"
        and edge.get("top", 0) < h_ys[0] - margin
        and h_ys[0] - edge.get("top", 0) < 150
        and edge.get("x0", float("inf")) <= x0 + 3
        and edge.get("x1", 0.0) >= x1 - 3
    ]
    if above_lines:
        top_line = min(above_lines, key=lambda edge: h_ys[0] - edge.get("top", 0))
        eff_x0 = min(eff_x0, top_line.get("x0", eff_x0) - margin)
        eff_x1 = max(eff_x1, top_line.get("x1", eff_x1) + margin)

        below_lines = [
            edge for edge in pdf_page.edges
            if edge.get("orientation") == "h"
            and edge.get("top", 0) > h_ys[1] + margin
            and edge.get("x0", float("inf")) <= eff_x0 + 10
            and edge.get("x1", 0.0) >= eff_x1 - 10
        ]
        if below_lines:
            bottom_line = min(below_lines, key=lambda edge: edge.get("top", 0))
            data_y_bottom = bottom_line.get("top", data_y_bottom) - margin

    header_bbox = (eff_x0, h_ys[0] + margin, eff_x1, h_ys[1] - margin)
    data_bbox = (eff_x0, h_ys[1] + margin, eff_x1, data_y_bottom)

    try:
        header_words = pdf_page.within_bbox(header_bbox).extract_words()
        data_words = pdf_page.within_bbox(data_bbox).extract_words()
    except Exception:
        logger.debug("extract_headerbox: within_bbox failed", exc_info=True)
        return []

    if not header_words and not data_words:
        return []

    if above_lines and data_words:
        col_ranges = find_col_ranges(
            (header_words or []) + data_words,
            min_gap_pt=CFG.THREELINE_COL_GAP_PT,
        )
    else:
        col_ranges = find_col_ranges(header_words, min_gap_pt=CFG.THREELINE_COL_GAP_PT)
        if len(col_ranges) < 2 and data_words:
            col_ranges = find_col_ranges(
                header_words + data_words,
                min_gap_pt=CFG.THREELINE_COL_GAP_PT,
            )

    if len(col_ranges) < CFG.MIN_COLS:
        return []

    num_cols = len(col_ranges)
    header_row = [""] * num_cols
    for word in (header_words or []):
        col_idx = word_to_col(word, col_ranges)
        if 0 <= col_idx < num_cols:
            header_row[col_idx] = (header_row[col_idx] + " " + word.get("text", "")).strip()

    data_rows = build_data_rows(data_words, col_ranges)
    data_rows = merge_multiline_cells(data_rows)
    logger.debug(
        "extract_headerbox: detected %d cols, header=%s, data_rows=%d",
        num_cols, header_row, len(data_rows),
    )
    return [header_row] + data_rows


def detect_rotated_table(pdf_page) -> list[list[str]]:
    header_keywords = frozenset({
        "host", "characteristic", "medium", "titer", "reference",
        "strain", "enzyme", "substrate", "product", "yield",
        "condition", "temperature", "concentration",
    })
    min_rotated = 30
    min_gap_pt = 10.0
    cluster_tolerance = 8.0
    header_gap = 15.0

    rotated = [
        char for char in pdf_page.chars
        if abs(char.get("matrix", [1, 0, 0, 1, 0, 0])[1]) > 0.1
    ]
    if len(rotated) < min_rotated:
        return []

    groups: dict[float, list] = defaultdict(list)
    for char in rotated:
        key = round(char["x0"] * 2) / 2
        groups[key].append(char)
    sorted_keys = sorted(groups.keys())
    if len(sorted_keys) < 3:
        return []

    title_key: float | None = None
    for key in sorted_keys[:2]:
        text = "".join(
            char["text"] for char in sorted(groups[key], key=lambda char: -char["top"])
        ).lower()
        if text.startswith("table"):
            title_key = key
            break

    header_key: float | None = None
    for key in sorted_keys:
        if key == title_key:
            continue
        text = "".join(
            char["text"] for char in sorted(groups[key], key=lambda char: -char["top"])
        ).lower()
        if sum(1 for keyword in header_keywords if keyword in text) >= 3:
            header_key = key
            break
    if header_key is None:
        return []

    header_chars = sorted(groups[header_key], key=lambda char: -char["top"])
    header_col_groups: list[list] = []
    current = [header_chars[0]]
    for idx in range(1, len(header_chars)):
        gap = header_chars[idx - 1]["top"] - header_chars[idx]["top"]
        if gap > header_gap:
            header_col_groups.append(current)
            current = []
        current.append(header_chars[idx])
    header_col_groups.append(current)

    n_cols = len(header_col_groups)
    if n_cols < 2:
        return []

    header_row = [
        "".join(char["text"] for char in group)
        for group in header_col_groups
    ]

    data_keys = [key for key in sorted_keys if key not in (title_key, header_key)]
    if not data_keys:
        return []

    last_col_max_top = max(char["top"] for char in header_col_groups[-1])
    main_threshold = last_col_max_top + 10.0
    main_keys = [
        key for key in data_keys
        if min(char["top"] for char in groups[key]) <= main_threshold
    ]
    continuation_keys = [key for key in data_keys if key not in main_keys]
    if not main_keys:
        return []

    row_groups: dict[float, list[float]] = {key: [key] for key in main_keys}
    for continuation_key in continuation_keys:
        nearest = min(main_keys, key=lambda main_key: abs(main_key - continuation_key))
        row_groups[nearest].append(continuation_key)

    gap_pairs: list[tuple[float, float]] = []
    for main_key in main_keys:
        chars = sorted(groups[main_key], key=lambda char: -char["top"])
        tops = [char["top"] for char in chars]
        for idx in range(len(tops) - 1):
            gap = tops[idx] - tops[idx + 1]
            if gap > min_gap_pt:
                gap_pairs.append((tops[idx], tops[idx + 1]))

    if len(gap_pairs) < n_cols - 1:
        return []

    used: set[int] = set()
    clusters: list[tuple[float, list[float], list[float]]] = []
    for idx, (before_top, after_top) in enumerate(gap_pairs):
        if idx in used:
            continue
        cluster_before = [before_top]
        cluster_after = [after_top]
        for next_idx in range(idx + 1, len(gap_pairs)):
            if next_idx in used:
                continue
            next_before, next_after = gap_pairs[next_idx]
            if abs(after_top - next_after) <= cluster_tolerance:
                cluster_before.append(next_before)
                cluster_after.append(next_after)
                used.add(next_idx)
        used.add(idx)
        boundary = (min(cluster_before) + max(cluster_after)) / 2.0
        clusters.append((boundary, cluster_before, cluster_after))

    clusters.sort(key=lambda cluster: -cluster[0])
    if len(clusters) < n_cols - 1:
        return []

    boundaries = sorted(
        [cluster[0] for cluster in clusters[: n_cols - 1]],
        reverse=True,
    )

    def _top_to_col(top: float) -> int:
        return sum(1 for boundary in boundaries if top <= boundary)

    result: list[list[str]] = [header_row]
    for main_key in sorted(main_keys):
        col_subrows: list[list[list]] = [[] for _ in range(n_cols)]
        for key in sorted(row_groups[main_key]):
            subrows: list[list] = [[] for _ in range(n_cols)]
            for char in groups[key]:
                col_idx = _top_to_col(char["top"])
                if 0 <= col_idx < n_cols:
                    subrows[col_idx].append(char)
            for col_idx in range(n_cols):
                if subrows[col_idx]:
                    col_subrows[col_idx].append(subrows[col_idx])

        cols = [""] * n_cols
        for col_idx, subrows in enumerate(col_subrows):
            if not subrows:
                continue
            if len(subrows) == 1:
                cols[col_idx] = "".join(
                    char["text"] for char in sorted(subrows[0], key=lambda char: -char["top"])
                )
                continue

            ranges = [
                (min(char["top"] for char in subrow), max(char["top"] for char in subrow))
                for subrow in subrows
            ]

            def _overlaps(first: tuple, second: tuple) -> bool:
                first_low, first_high = first
                second_low, second_high = second
                return first_high > second_low - 5 and second_high > first_low - 5

            has_wrap = any(
                _overlaps(ranges[idx], ranges[next_idx])
                for idx in range(len(ranges))
                for next_idx in range(idx + 1, len(ranges))
            )
            if has_wrap:
                cols[col_idx] = "".join(
                    "".join(
                        char["text"] for char in sorted(subrow, key=lambda char: -char["top"])
                    )
                    for subrow in subrows
                )
            else:
                all_chars = [char for subrow in subrows for char in subrow]
                cols[col_idx] = "".join(
                    char["text"] for char in sorted(all_chars, key=lambda char: -char["top"])
                )

        if any(col.strip() for col in cols):
            result.append(cols)

    return result if len(result) >= 2 else []
