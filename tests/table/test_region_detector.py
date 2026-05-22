from __future__ import annotations

from src.parsers.table.models import TableEvidence, TableRegion
from src.parsers.table.region_detector import TableRegionDetector


class FakePage:
    width = 600
    height = 800

    def __init__(self) -> None:
        self.edges = [
            {"orientation": "h", "x0": 70, "x1": 530, "top": 130},
            {"orientation": "h", "x0": 70, "x1": 530, "top": 170},
            {"orientation": "h", "x0": 70, "x1": 530, "top": 250},
        ]

    def extract_words(self) -> list[dict]:
        return [
            _word("Table", 70, 100),
            _word("1.", 105, 100),
            _word("Results", 130, 100),
            _word("Group", 75, 145),
            _word("Value", 190, 145),
            _word("P", 310, 145),
            _word("A", 75, 190),
            _word("12.5", 190, 190),
            _word("0.01", 310, 190),
            _word("B", 75, 215),
            _word("13.1", 190, 215),
            _word("0.03", 310, 215),
        ]


class CaptionOnlyPage:
    width = 600
    height = 800
    edges: list[dict] = []

    def extract_words(self) -> list[dict]:
        return [
            _word("Table", 70, 100),
            _word("2.", 105, 100),
            _word("Characteristics", 130, 100),
        ]


class GeometryOnlyPage:
    width = 600
    height = 800

    def __init__(self) -> None:
        self.edges = [
            {"orientation": "h", "x0": 70, "x1": 530, "top": 130},
            {"orientation": "h", "x0": 70, "x1": 530, "top": 170},
            {"orientation": "h", "x0": 70, "x1": 530, "top": 250},
        ]

    def extract_words(self) -> list[dict]:
        return [
            _word("Group", 75, 145),
            _word("Value", 190, 145),
            _word("P", 310, 145),
            _word("A", 75, 190),
            _word("12.5", 190, 190),
            _word("0.01", 310, 190),
            _word("B", 75, 215),
            _word("13.1", 190, 215),
            _word("0.03", 310, 215),
        ]


class RotatedTablePage:
    width = 600
    height = 800
    edges: list[dict] = []

    def __init__(self) -> None:
        self.chars = []
        self.chars.extend(_rotated_line("Table 3. Titers", 120, 420))
        self.chars.extend(_rotated_line("strain titer yield", 150, 420))
        self.chars.extend(_rotated_line("A1 1.2 80", 180, 420))
        self.chars.extend(_rotated_line("A2 1.5 85", 210, 420))

    def extract_words(self) -> list[dict]:
        return []


class LongCaptionTableWithAdjacentTextPage:
    width = 600
    height = 800

    def __init__(self) -> None:
        self.edges = [
            {"orientation": "h", "x0": 48, "x1": 555, "top": 130},
            {"orientation": "h", "x0": 48, "x1": 555, "top": 170},
            {"orientation": "h", "x0": 48, "x1": 555, "top": 620},
        ]

    def extract_words(self) -> list[dict]:
        words = [
            _word("Table", 50, 100),
            _word("2.", 86, 100),
            _word("Different", 110, 100),
            _word("Engineering", 168, 100),
            _word("Strain", 50, 145),
            _word("LNT", 130, 145),
            _word("II", 158, 145),
            _word("LNT", 230, 145),
        ]
        for idx in range(1, 20):
            top = 180 + idx * 20
            words.extend([
                _word(f"B{idx}", 55, top),
                _word(f"{idx / 10:.2f}", 130, top),
                _word("±", 166, top),
                _word("0.05", 184, top),
                _word(f"{idx / 5:.2f}", 232, top),
            ])
            if idx >= 12:
                words.extend([
                    _word("the", 330, top),
                    _word("pathway", 356, top),
                    _word("enzymes", 410, top),
                    _word("thereby", 462, top),
                    _word("increasing", 514, top),
                ])
        return words


def _word(text: str, x0: float, top: float) -> dict:
    return {
        "text": text,
        "x0": x0,
        "x1": x0 + max(8, len(text) * 6),
        "top": top,
        "bottom": top + 10,
    }


def _rotated_line(text: str, x0: float, top: float) -> list[dict]:
    return [
        {
            "text": char,
            "x0": x0,
            "x1": x0 + 5,
            "top": top - index * 9,
            "bottom": top - index * 9 + 7,
            "matrix": [0, 1, -1, 0, 0, 0],
        }
        for index, char in enumerate(text)
    ]


def test_detector_keeps_recall_oriented_caption_and_geometry_region() -> None:
    detector = TableRegionDetector("paper.pdf")

    regions = detector.detect_page(FakePage(), 1)

    assert regions
    region = regions[0]
    assert region.page == 1
    assert region.confidence >= 0.65
    assert region.band == "strong"
    assert {"caption", "lines", "alignment"} <= {item.source for item in region.evidence}
    assert region.bbox[3] < 360


def test_region_metadata_exposes_evidence_and_band() -> None:
    region = TableRegion(
        page=2,
        bbox=(10, 20, 100, 120),
        confidence=0.5,
        evidence=[TableEvidence("caption", 0.35, "near_table_caption")],
    )

    metadata = region.to_metadata()

    assert metadata["band"] == "gray"
    assert metadata["confidence"] == 0.5
    assert metadata["evidence"][0]["source"] == "caption"


def test_caption_candidate_keeps_recall_window_without_table_lines() -> None:
    detector = TableRegionDetector("paper.pdf")

    regions = detector.detect_page(CaptionOnlyPage(), 1)

    assert len(regions) == 1
    assert regions[0].bbox[1] < 100
    assert regions[0].bbox[3] >= 220


def test_detector_skips_geometry_only_regions_without_table_keyword() -> None:
    detector = TableRegionDetector("paper.pdf")

    regions = detector.detect_page(GeometryOnlyPage(), 1)

    assert regions == []


def test_detector_finds_rotated_caption_table_region() -> None:
    detector = TableRegionDetector("paper.pdf")

    regions = detector.detect_page(RotatedTablePage(), 3)

    assert len(regions) == 1
    region = regions[0]
    assert region.page == 3
    assert region.confidence >= 0.65
    assert region.bbox[0] < 120
    assert region.bbox[2] > 210
    assert {"caption", "rotation"} <= {item.source for item in region.evidence}


def test_caption_candidate_extends_long_table_without_adjacent_body_text() -> None:
    detector = TableRegionDetector("paper.pdf")

    regions = detector.detect_page(LongCaptionTableWithAdjacentTextPage(), 4)

    assert regions
    assert len(regions) == 1
    region = regions[0]
    assert region.bbox[3] >= 620
    assert region.bbox[2] < 330


def test_detector_suppresses_nested_duplicate_regions() -> None:
    detector = TableRegionDetector("paper.pdf")
    broad = TableRegion(
        page=1,
        bbox=(50, 90, 550, 500),
        confidence=0.70,
        evidence=[TableEvidence("caption", 0.35, "near_table_caption")],
    )
    tight = TableRegion(
        page=1,
        bbox=(70, 130, 530, 250),
        confidence=0.70,
        evidence=[
            TableEvidence("lines", 0.30, "multiple_wide_horizontal_lines"),
            TableEvidence("alignment", 0.30, "repeated_x_alignment"),
        ],
    )

    regions = detector._suppress_overlapping_regions([broad, tight])

    assert len(regions) == 1
    assert regions[0].bbox == tight.bbox


def test_detector_suppresses_near_duplicate_same_caption_regions() -> None:
    detector = TableRegionDetector("paper.pdf")
    first = TableRegion(
        page=1,
        bbox=(50, 90, 550, 300),
        confidence=0.65,
        evidence=[TableEvidence("caption", 0.35, "near_table_caption", details={"text": "Table 1. Results"})],
    )
    duplicate = TableRegion(
        page=1,
        bbox=(55, 260, 545, 430),
        confidence=0.60,
        evidence=[TableEvidence("caption", 0.35, "near_table_caption", details={"text": "Table 1. Results"})],
    )

    regions = detector._suppress_overlapping_regions([first, duplicate])

    assert len(regions) == 1


def test_detector_suppresses_cross_page_duplicate_caption_regions() -> None:
    detector = TableRegionDetector("paper.pdf")
    weak_duplicate = TableRegion(
        page=2,
        bbox=(50, 90, 550, 220),
        confidence=0.35,
        evidence=[TableEvidence("caption", 0.35, "near_table_caption", details={"text": "Table 1. Results"})],
    )
    table_region = TableRegion(
        page=3,
        bbox=(50, 90, 550, 500),
        confidence=0.95,
        evidence=[
            TableEvidence("caption", 0.35, "near_table_caption", details={"text": "Table 1. Results"}),
            TableEvidence("lines", 0.30, "multiple_wide_horizontal_lines"),
            TableEvidence("alignment", 0.30, "repeated_x_alignment"),
        ],
    )

    regions = detector._suppress_cross_page_caption_duplicates([weak_duplicate, table_region])

    assert len(regions) == 1
    assert regions[0].page == 3


def test_detector_keeps_cross_page_caption_continuations() -> None:
    detector = TableRegionDetector("paper.pdf")
    first_page = TableRegion(
        page=3,
        bbox=(50, 90, 550, 760),
        confidence=0.95,
        evidence=[TableEvidence("caption", 0.35, "near_table_caption", details={"text": "Table 1. Results"})],
    )
    continuation = TableRegion(
        page=4,
        bbox=(50, 60, 550, 420),
        confidence=0.75,
        evidence=[TableEvidence("caption", 0.35, "near_table_caption", details={"text": "Table 1 continued"})],
    )

    regions = detector._suppress_cross_page_caption_duplicates([first_page, continuation])

    assert len(regions) == 2


def test_caption_text_prefilter_matches_table_labels() -> None:
    assert TableRegionDetector._contains_table_caption_text("Table 2. Results")
    assert TableRegionDetector._contains_table_caption_text("Table1Comparison of models")
    assert TableRegionDetector._contains_table_caption_text("Supplementary Table S1 primers")
    assert TableRegionDetector._contains_table_caption_text("56 Supplementary Table S1 primers")
    assert TableRegionDetector._contains_table_caption_text("SupplementaryTableS1.")
    assert TableRegionDetector._contains_table_caption_text("Tab. 3 data")
    assert TableRegionDetector._contains_table_caption_text("表 1 菌株和质粒")
    assert TableRegionDetector._contains_table_caption_text("附表 S2 引物列表")
    assert not TableRegionDetector._contains_table_caption_text("stable isotope labeling")


def test_detector_clips_candidate_bbox_to_pdf_page_bbox() -> None:
    clipped = TableRegionDetector._clip_bbox_to_page(
        (0, 20, 620, 820),
        (9, -9, 594, 774),
    )

    assert clipped == (9, 20, 594, 774)
