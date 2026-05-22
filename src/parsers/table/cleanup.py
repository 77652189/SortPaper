from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

from src.parsers.config import TABLE_PARSER as CFG

logger = logging.getLogger(__name__)

HEADER_KEYWORDS: tuple[str, ...] = (
    "product", "yield", "reference", "substrate",
    "enzyme", "key enzyme", "reaction",
    "host", "strain", "characteristic", "cultivation",
    "company", "brand", "brands", "country",
)

BODY_WORDS: tuple[str, ...] = (
    "the", "of", "and", "was", "were", "that", "this", "from",
    "with", "for", "are", "not", "but", "had", "has", "have", "been",
    "which", "their", "also", "between", "after", "during", "over",
    "into", "than", "about", "more", "some", "these", "those", "each",
    "both", "such", "only", "other", "they", "them", "then", "there",
    "therefore", "however", "although", "because", "while", "where",
    "when", "upon", "within", "without", "through", "under",
    "significantly", "respectively", "additionally", "interestingly",
    "pathway", "synthesis", "enzymes", "increasing", "yield", "metabolic",
    "genes", "effects", "production", "introduced", "strengthened",
)


def find_header_row(normalized: list[list[str]]) -> int:
    for idx, row in enumerate(normalized):
        content = " ".join(cell.lower() for cell in row)
        non_empty = sum(1 for cell in row if cell.strip())
        if non_empty >= 2 and any(keyword in content for keyword in HEADER_KEYWORDS):
            return idx

    for idx, row in enumerate(normalized):
        if "table " in " ".join(cell.lower() for cell in row):
            if idx + 1 < len(normalized):
                next_row = normalized[idx + 1]
                non_empty = sum(1 for cell in next_row if cell.strip())
                if non_empty >= 2:
                    return idx + 1
            return idx

    return 0


def adjust_table(normalized: list[list[str]]) -> list[list[str]]:
    header_row_idx = find_header_row(normalized)
    if header_row_idx > 0:
        return normalized[header_row_idx:] + normalized[:header_row_idx]
    return normalized


def looks_garbled(cell: str) -> bool:
    s = cell.strip()
    alpha = [char for char in s if char.isalpha()]
    if len(alpha) < 5:
        return False
    vowels = set("aeiouyAEIOUY")
    if any(char in vowels for char in s):
        return False
    has_upper = any(char.isupper() for char in s)
    has_lower = any(char.islower() for char in s)
    if has_upper and has_lower:
        return False
    return True


def is_valid_table(normalized: list[list[str]]) -> bool:
    if not normalized or len(normalized) < CFG.MIN_DATA_ROWS + 1:
        return False

    header_row_idx = find_header_row(normalized)
    header = normalized[header_row_idx]
    data_rows = normalized[header_row_idx + 1:]
    num_cols = len(header)

    if num_cols < CFG.MIN_COLS:
        return False

    if len(data_rows) < CFG.MIN_DATA_ROWS:
        return False

    has_data = any(any(cell.strip() for cell in row) for row in data_rows)
    if not has_data:
        return False

    sample_rows = (normalized[:header_row_idx] if header_row_idx > 0 else []) + [header] + data_rows[:3]
    sample_cells = [cell for row in sample_rows for cell in row if cell.strip()]
    if sample_cells:
        garbled = sum(1 for cell in sample_cells if looks_garbled(cell))
        if garbled / len(sample_cells) > CFG.GARBLED_CELL_RATIO_REJECT:
            return False

    header_nonempty = sum(1 for cell in header if cell.strip())
    header_text = " ".join(cell.lower() for cell in header)
    has_header_signal = (
        any(keyword in header_text for keyword in HEADER_KEYWORDS)
        or "table " in header_text
    ) and not is_prose_mixed_row(header)
    prose_rows = sum(1 for row in data_rows if is_prose_mixed_row(row))
    if (
        not has_header_signal
        and prose_rows >= 2
        and prose_rows / max(1, len(data_rows)) >= CFG.PROSE_MIXED_ROW_RATIO_REJECT
    ):
        logger.debug(
            "is_valid_table: rejected prose-mixed pseudo table prose_rows=%d/%d",
            prose_rows, len(data_rows),
        )
        return False

    if header_nonempty <= 2:
        long_cells = sum(
            1 for cell in sample_cells if len(cell) >= CFG.BODY_TEXT_CELL_CHARS
        )
        if long_cells >= CFG.BODY_TEXT_MIN_CELLS:
            logger.debug(
                "is_valid_table: rejected body-text pseudo table header_cols=%d long_cells=%d/%d",
                header_nonempty, long_cells, len(sample_cells),
            )
            return False

    return True


def structural_quality(normalized: list[list[str]]) -> dict[str, Any]:
    if not normalized or len(normalized) < 2:
        return {
            "consistency_score": 0.0,
            "fill_rate": 0.0,
            "fallback_to_vision": True,
            "fallback_reasons": ["empty_or_too_short"],
        }

    col_counts = [sum(1 for cell in row if cell.strip()) for row in normalized]
    max_cols = max(col_counts)
    if max_cols < 2:
        return {
            "consistency_score": 0.0,
            "fill_rate": 0.0,
            "fallback_to_vision": True,
            "fallback_reasons": ["too_few_columns"],
        }

    data_rows = len(normalized) - 1
    if data_rows < CFG.MIN_DATA_ROWS + 1:
        logger.debug(
            "structural_quality: too few data rows rows=%d -> fallback",
            len(normalized),
        )
        return {
            "consistency_score": 1.0,
            "fill_rate": 1.0,
            "fallback_to_vision": True,
            "fallback_reasons": ["too_few_data_rows"],
        }

    mode_count = Counter(col_counts).most_common(1)[0][1]
    consistency_score = mode_count / len(col_counts)

    total_cells = len(normalized) * max_cols
    empty_cells = sum(1 for row in normalized for cell in row if not cell.strip())
    fill_rate = 1.0 - empty_cells / total_cells if total_cells > 0 else 0.0

    fallback_reasons = []
    if consistency_score < CFG.VISION_FALLBACK_MIN_CONSISTENCY:
        fallback_reasons.append("low_consistency")
    if fill_rate < CFG.VISION_FALLBACK_MIN_FILL_RATE:
        fallback_reasons.append("low_fill_rate")

    prose_cells = 0
    nonempty_cells = 0
    for row in normalized:
        for cell in row:
            text = cell.strip()
            if not text:
                continue
            nonempty_cells += 1
            if (
                len(text) >= CFG.BODY_TEXT_CELL_CHARS
                and count_body_words(text) >= CFG.TAIL_TRIM_MIN_BODY_WORDS
            ):
                prose_cells += 1
    prose_cell_ratio = prose_cells / nonempty_cells if nonempty_cells else 0.0
    allows_descriptive_columns = (
        consistency_score >= CFG.DESCRIPTIVE_COLUMN_MIN_CONSISTENCY
        and fill_rate >= CFG.DESCRIPTIVE_COLUMN_MIN_FILL_RATE
        and len(normalized) >= CFG.DESCRIPTIVE_COLUMN_MIN_ROWS
        and max_cols >= CFG.DESCRIPTIVE_COLUMN_MIN_COLS
    )
    if (
        not allows_descriptive_columns
        and (
            prose_cells >= CFG.BODY_TEXT_MIN_CELLS
            or prose_cell_ratio >= CFG.PROSE_CELL_RATIO_REPARSE
        )
    ):
        fallback_reasons.append("prose_like_cells")

    return {
        "consistency_score": round(consistency_score, 2),
        "fill_rate": round(fill_rate, 2),
        "prose_cell_ratio": round(prose_cell_ratio, 2),
        "allows_descriptive_columns": allows_descriptive_columns,
        "fallback_to_vision": CFG.VISION_FALLBACK_ENABLED and bool(fallback_reasons),
        "fallback_reasons": fallback_reasons,
    }


def count_body_words(text: str) -> int:
    text_lower = text.lower()
    count = 0
    for word in BODY_WORDS:
        idx = 0
        while True:
            idx = text_lower.find(word, idx)
            if idx == -1:
                break
            count += 1
            idx += len(word)
    return count


def is_prose_mixed_row(row: list[str]) -> bool:
    nonempty = [cell.strip() for cell in row if cell.strip()]
    if len(nonempty) < 3:
        return False
    text = " ".join(nonempty)
    if len(text) < 70:
        return False
    numeric_cells = sum(1 for cell in nonempty if re.search(r"\d", cell))
    if numeric_cells < 2:
        return False
    return count_body_words(text) >= 4


def trim_tail_body_text(normalized: list[list[str]]) -> list[list[str]]:
    if len(normalized) < 3:
        return normalized

    col_counts = [sum(1 for cell in row if cell.strip()) for row in normalized]
    mode_cols = Counter(col_counts).most_common(1)[0][0]
    if mode_cols < CFG.TAIL_TRIM_MIN_MODE_COLS:
        return normalized

    while len(normalized) >= 3:
        last = normalized[-1]
        last_nonempty = sum(1 for cell in last if cell.strip())
        if last_nonempty > mode_cols * 0.5:
            break
        has_long = any(
            len(cell.strip()) >= CFG.TAIL_TRIM_LONG_CELL_CHARS
            and count_body_words(cell) >= CFG.TAIL_TRIM_MIN_BODY_WORDS
            for cell in last
        )
        if has_long or last_nonempty <= 2:
            normalized.pop()
        else:
            break
    return normalized


def truncate_at_body_text(normalized: list[list[str]]) -> list[list[str]]:
    if not normalized:
        return normalized

    max_nonempty = 0
    logger.debug("truncate_at_body_text: rows=%d", len(normalized))

    for idx, row in enumerate(normalized):
        row_nonempty = sum(1 for cell in row if cell.strip())
        if row_nonempty > max_nonempty:
            max_nonempty = row_nonempty

        if any(len(cell) > CFG.MAX_CELL_CHARS for cell in row):
            logger.debug(
                "truncated body-text row=%d due to long cell: %.60s",
                idx, max(row, key=len),
            )
            return trim_tail_body_text(normalized[:idx])

        if (
            max_nonempty >= CFG.TRUNCATE_SPARSE_COL_THRESHOLD
            and row_nonempty <= 2
            and idx >= 2
        ):
            text = " ".join(cell.strip() for cell in row if cell.strip())
            if re.search(r'[+\-*†‡§¶#−‐‑‒–—―]|^[a-z]\)|^[a-z]\s', text):
                logger.debug(
                    "skipped table footnote row=%d cols=%d->%d",
                    idx, max_nonempty, row_nonempty,
                )
                continue
            else:
                logger.debug(
                    "truncated body-text row=%d due to sparse columns: %.60s",
                    idx, text,
                )
                return trim_tail_body_text(normalized[:idx])

        for cell in row:
            if len(cell) < CFG.TRUNCATE_BODY_WORD_MIN_LEN:
                continue
            matches = count_body_words(cell)
            if matches >= CFG.TRUNCATE_BODY_WORD_MIN_COUNT:
                logger.debug(
                    "truncated body-text row=%d due to prose cell: %.60s",
                    idx, cell,
                )
                return trim_tail_body_text(normalized[:idx])

    return trim_tail_body_text(normalized)


def repair_table_structure(normalized: list[list[str]]) -> tuple[list[list[str]], list[str]]:
    """Apply conservative, data-free repairs to common PDF table artifacts.

    The goal is not to infer missing content, only to undo parser artifacts that
    are visible from the table matrix itself: empty columns, ``value | ± sd``
    splits, and word-fragment columns caused by over-aggressive text strategy.
    """
    if not normalized:
        return normalized, []

    repaired = [list(row) for row in normalized]
    actions: list[str] = []

    repaired, changed = _drop_empty_columns(repaired)
    if changed:
        actions.append("drop_empty_columns")

    repaired, pm_count = _merge_plus_minus_columns(repaired)
    if pm_count:
        actions.append(f"merge_plus_minus_columns:{pm_count}")
        repaired, changed = _drop_empty_columns(repaired)
        if changed and "drop_empty_columns" not in actions:
            actions.append("drop_empty_columns")

    repaired, fragment_count = _merge_fragment_columns(repaired)
    if fragment_count:
        actions.append(f"merge_fragment_columns:{fragment_count}")
        repaired, changed = _drop_empty_columns(repaired)
        if changed and "drop_empty_columns" not in actions:
            actions.append("drop_empty_columns")

    return repaired, actions


def _drop_empty_columns(rows: list[list[str]]) -> tuple[list[list[str]], bool]:
    if not rows:
        return rows, False
    width = max((len(row) for row in rows), default=0)
    padded = [row + [""] * (width - len(row)) for row in rows]
    keep = [
        idx for idx in range(width)
        if any(row[idx].strip() for row in padded)
    ]
    if len(keep) == width:
        return padded, False
    return [[row[idx] for idx in keep] for row in padded], True


def _merge_plus_minus_columns(rows: list[list[str]]) -> tuple[list[list[str]], int]:
    if not rows or len(rows[0]) < 3:
        return rows, 0

    merged = [list(row) for row in rows]
    merge_count = 0
    idx = 1
    while idx < len(merged[0]):
        col = [row[idx].strip() for row in merged]
        nonempty = [cell for cell in col if cell]
        if not nonempty:
            idx += 1
            continue
        pm_ratio = sum(1 for cell in nonempty if _is_pm_fragment(cell)) / len(nonempty)
        if pm_ratio < 0.45:
            idx += 1
            continue
        for row in merged:
            row[idx - 1] = _join_cells(row[idx - 1], row[idx])
            del row[idx]
        merge_count += 1
    return merged, merge_count


def _merge_fragment_columns(rows: list[list[str]]) -> tuple[list[list[str]], int]:
    if not rows or len(rows[0]) < 5:
        return rows, 0

    merged = [list(row) for row in rows]
    merge_count = 0
    idx = 1
    while idx < len(merged[0]):
        if not _should_merge_fragment_column(merged, idx):
            idx += 1
            continue
        for row in merged:
            row[idx - 1] = _join_cells(row[idx - 1], row[idx], compact=True)
            del row[idx]
        merge_count += 1
    return merged, merge_count


def _should_merge_fragment_column(rows: list[list[str]], idx: int) -> bool:
    if idx < 2:
        return False
    right = [row[idx].strip() for row in rows]
    left = [row[idx - 1].strip() for row in rows]
    right_nonempty = [cell for cell in right if cell]
    if len(right_nonempty) < 2:
        return False
    right_fragment_ratio = sum(1 for cell in right_nonempty if _is_word_fragment(cell)) / len(right_nonempty)
    if right_fragment_ratio < 0.60:
        return False

    paired = [
        (left_cell, right_cell)
        for left_cell, right_cell in zip(left, right)
        if left_cell and right_cell
    ]
    if len(paired) < max(2, len(right_nonempty) // 2):
        return False
    left_wordish = sum(1 for left_cell, _ in paired if _is_wordish(left_cell)) / len(paired)
    numeric_right = sum(1 for cell in right_nonempty if re.search(r"\d", cell)) / len(right_nonempty)
    return left_wordish >= 0.50 and numeric_right <= 0.20


def _is_pm_fragment(cell: str) -> bool:
    text = cell.strip()
    return bool(re.match(r"^(?:±|\+/-|\+|−|-)\s*\d", text))


def _is_word_fragment(cell: str) -> bool:
    text = cell.strip().strip(".,;:()[]")
    if not text or re.search(r"\d", text):
        return False
    if len(text) > 6:
        return False
    if not any(char.isalpha() for char in text):
        return False
    lowered = text.lower()
    if lowered in BODY_WORDS or lowered in HEADER_KEYWORDS:
        return False
    return True


def _is_wordish(cell: str) -> bool:
    text = cell.strip()
    return bool(text) and any(char.isalpha() for char in text) and not re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", text)


def _join_cells(left: str, right: str, *, compact: bool = False) -> str:
    left = left.strip()
    right = right.strip()
    if not left:
        return right
    if not right:
        return left
    if compact and _is_wordish(left) and _is_word_fragment(right):
        return f"{left}{right}"
    return f"{left} {right}".strip()


def unsplit_twin_columns(normalized: list[list[str]]) -> list[list[str]]:
    if not normalized:
        return normalized
    n_cols = len(normalized[0])
    if n_cols < 4 or n_cols % 2 != 0:
        return normalized

    half = n_cols // 2
    scan = min(4, len(normalized))

    skip_start = 0
    for row_index, row in enumerate(normalized[:2]):
        left = [cell.strip() for cell in row[:half]]
        right = [cell.strip() for cell in row[half:]]
        if not any(right) and sum(1 for cell in left if cell) == 1:
            skip_start = row_index + 1
        else:
            break

    header_rows_raw: list[list[str]] = []
    data_start = skip_start
    for row_index, row in enumerate(normalized[skip_start:skip_start + scan]):
        left = [cell.strip() for cell in row[:half]]
        right = [cell.strip() for cell in row[half:]]
        if left == right and any(left):
            header_rows_raw.append(list(row[:half]))
            data_start = skip_start + row_index + 1
        else:
            break

    if not header_rows_raw:
        inferred_header = _infer_twin_column_header(normalized, half)
        if inferred_header is None:
            return normalized
        header_row, data_start, right_seed = inferred_header
    elif data_start >= len(normalized):
        return normalized
    else:
        if len(header_rows_raw) > 1:
            header_row = [
                " ".join(row[col].strip() for row in header_rows_raw if row[col].strip())
                for col in range(half)
            ]
        else:
            header_row = header_rows_raw[0]
        right_seed = None

    left_rows: list[list[str]] = []
    right_rows: list[list[str]] = []
    if right_seed and any(cell.strip() for cell in right_seed):
        right_rows.append(right_seed)
    for row in normalized[data_start:]:
        left_part = list(row[:half])
        right_part = list(row[half:])
        if any(cell.strip() for cell in left_part):
            left_rows.append(left_part)
        if any(cell.strip() for cell in right_part):
            right_rows.append(right_part)

    result: list[list[str]] = [header_row] + left_rows + right_rows

    logger.debug(
        "unsplit_twin_columns: %d cols -> %d cols, %d rows",
        n_cols, half, len(result),
    )
    return result


def _infer_twin_column_header(
    normalized: list[list[str]],
    half: int,
) -> tuple[list[str], int, list[str] | None] | None:
    if not normalized or half < 2:
        return None

    first = normalized[0]
    left = [cell.strip() for cell in first[:half]]
    right = [cell.strip() for cell in first[half:]]
    left_text = " ".join(left).lower()
    right_text = " ".join(right).lower()
    if (
        any(keyword in left_text for keyword in HEADER_KEYWORDS)
        and any(cell.strip() for cell in right)
        and not any(keyword in right_text for keyword in HEADER_KEYWORDS)
    ):
        return left, 1, right

    paired_rows = 0
    for row in normalized[: min(8, len(normalized))]:
        left_key = row[0].strip() if row else ""
        right_key = row[half].strip() if len(row) > half else ""
        if left_key and right_key:
            paired_rows += 1
    if paired_rows >= 3:
        return [f"column {idx + 1}" for idx in range(half)], 0, None

    return None


def normalize_table(table: list[list[str | None]]) -> list[list[str]]:
    normalized: list[list[str]] = []
    max_cols = max((len(row) for row in table if row), default=0)
    if max_cols < 1:
        return []

    for row in table:
        cleaned = [(cell or "").replace("\n", " ").strip() for cell in row]
        if len(cleaned) < max_cols:
            cleaned.extend([""] * (max_cols - len(cleaned)))
        if any(cell for cell in cleaned):
            normalized.append(cleaned)
    return normalized


def to_markdown(table: list[list[str]]) -> str:
    if not table:
        return ""
    header_row_idx = find_header_row(table)
    header = table[header_row_idx]
    data = table[header_row_idx + 1:]
    separator = ["---"] * len(header)
    rows = [header, separator] + data
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)
