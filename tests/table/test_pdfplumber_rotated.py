from __future__ import annotations

from src.parsers.layout_chunk import LayoutChunk
from src.parsers.table.models import TableEvidence, TableRegion
from src.parsers.table.pdfplumber_extractor import PdfplumberTableExtractor


def _rotated_line(text: str, x0: float, top: float) -> list[dict]:
    chars: list[dict] = []
    cursor = top
    for char in text:
        if char == " ":
            cursor -= 20
            continue
        chars.append({
            "text": char,
            "x0": x0,
            "x1": x0 + 5,
            "top": cursor,
            "bottom": cursor + 7,
            "matrix": [0, 1, -1, 0, 0, 0],
        })
        cursor -= 9
    return chars


def _rotated_columns(cells: list[str], x0: float, tops: list[float]) -> list[dict]:
    chars: list[dict] = []
    for cell, top in zip(cells, tops):
        chars.extend(_rotated_line(cell, x0, top))
    return chars


class FakeRotatedPage:
    width = 600
    height = 800
    bbox = (0, 0, 600, 800)
    edges: list[dict] = []

    def __init__(self) -> None:
        self.chars = []
        self.chars.extend(_rotated_line("Table 3. Titers", 120, 420))
        self.chars.extend(_rotated_columns(["strain", "titer", "yield"], 150, [420, 340, 260]))
        self.chars.extend(_rotated_columns(["A1", "1.2", "80"], 180, [420, 340, 260]))
        self.chars.extend(_rotated_columns(["A2", "1.5", "85"], 210, [420, 340, 260]))


class FakePdf:
    def __init__(self, pages: list[FakeRotatedPage]) -> None:
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_region_path_extracts_rotated_table_and_keeps_region_metadata(monkeypatch) -> None:
    import pdfplumber

    page = FakeRotatedPage()
    monkeypatch.setattr(pdfplumber, "open", lambda path: FakePdf([page]))

    def validate_and_fallback(
        normalized,
        bbox,
        page_index,
        page_width,
        page_height,
        *,
        parser_name,
        pdf_page=None,
        table_index=0,
        vision_clip_bbox=None,
    ):
        return LayoutChunk(
            content_type="table",
            raw_content="\n".join("\t".join(row) for row in normalized),
            page=page_index,
            bbox=bbox,
            column=0,
            order_in_page=table_index,
            metadata={
                "parser": parser_name,
                "rows": len(normalized),
                "cols": len(normalized[0]),
            },
        )

    extractor = PdfplumberTableExtractor(
        pdf_path="paper.pdf",
        post_process_table=lambda *args, **kwargs: None,
        validate_and_fallback=validate_and_fallback,
    )
    region = TableRegion(
        page=1,
        bbox=(100, 250, 230, 430),
        confidence=0.72,
        evidence=[
            TableEvidence("caption", 0.35, "rotated_table_caption"),
            TableEvidence("rotation", 0.30, "rotated_text_block"),
        ],
    )

    chunks = extractor.parse_regions([region])

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.metadata["parser"] == "pdfplumber_rotated"
    assert chunk.metadata["rotated"] is True
    assert chunk.metadata["source_region_id"] == region.region_id
    assert chunk.metadata["table_region"]["region_id"] == region.region_id
    assert chunk.metadata["table_region_match"] == "matched"
    assert chunk.metadata["rows"] >= 3
    assert chunk.bbox[0] >= 100
    assert chunk.bbox[1] < chunk.bbox[3]
    assert chunk.bbox[3] <= 430
