from __future__ import annotations

from src.parsers.table import structure_extractors
from src.parsers.table.pdfplumber_extractor import PdfplumberTableExtractor


def _word(text: str, x0: float, x1: float, top: float) -> dict:
    return {"text": text, "x0": x0, "x1": x1, "top": top, "bottom": top + 6}


class FakeCrop:
    def __init__(self, words: list[dict], bbox: tuple[float, float, float, float]) -> None:
        self.words = [
            word for word in words
            if bbox[0] <= word["x0"] <= bbox[2] and bbox[1] <= word["top"] <= bbox[3]
        ]

    def extract_words(self) -> list[dict]:
        return self.words


class FakePage:
    width = 320
    height = 240

    def __init__(self) -> None:
        self.words = [
            _word("strain", 40, 58, 106),
            _word("LNT", 105, 121, 106),
            _word("II", 124, 131, 106),
            _word("titer", 134, 151, 106),
            _word("(g/L)", 154, 178, 106),
            _word("LNT", 205, 221, 106),
            _word("titer", 224, 241, 106),
            _word("(g/L)", 244, 268, 106),
        ]
        for row, top, first, second in [
            ("B1", 126, "0.37", "0.22"),
            ("B2", 138, "1.67", "0.62"),
            ("B3", 150, "0.79", "0.85"),
        ]:
            self.words.extend([
                _word(row, 42, 52, top),
                _word(first, 116, 134, top),
                _word("±", 138, 143, top),
                _word("0.05", 147, 165, top),
                _word(second, 218, 236, top),
                _word("±", 240, 245, top),
                _word("0.03", 249, 267, top),
            ])

    def within_bbox(self, bbox: tuple[float, float, float, float]) -> FakeCrop:
        return FakeCrop(self.words, bbox)


def test_extract_threeline_uses_numeric_error_rows_to_rebuild_columns() -> None:
    rows = structure_extractors.extract_threeline(
        FakePage(),
        (30, 95, 290, 200),
        [100, 118, 190],
        30,
        290,
    )

    assert rows == [
        ["strain", "LNT II titer (g/L)", "LNT titer (g/L)"],
        ["B1", "0.37 ± 0.05", "0.22 ± 0.03"],
        ["B2", "1.67 ± 0.05", "0.62 ± 0.03"],
        ["B3", "0.79 ± 0.05", "0.85 ± 0.03"],
    ]


def test_select_threeline_ignores_caption_and_unrelated_horizontal_lines() -> None:
    selected = PdfplumberTableExtractor._select_threeline_ys(
        [320.6, 354.6, 367.5, 504.9, 580.1]
    )

    assert selected == [354.6, 367.5, 580.1]
