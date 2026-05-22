from __future__ import annotations

from typing import Any

from src.parsers.table import cleanup
from src.parsers.table.post_processor import TablePostProcessor


def _identity(rows: list[list[str | None]]) -> list[list[str]]:
    return [["" if cell is None else str(cell) for cell in row] for row in rows]


def _valid(rows: list[list[str]]) -> bool:
    return bool(rows and len(rows) >= 2 and len(rows[0]) >= 2)


def _markdown(rows: list[list[str]]) -> str:
    header = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join("---" for _ in rows[0]) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, sep, *body])


def _make_processor(
    *,
    allow_vision_fallback: bool,
    structural_quality,
    refine_table_bbox=None,
    parse_table_with_vision=None,
) -> TablePostProcessor:
    return TablePostProcessor(
        pdf_path="paper.pdf",
        normalize_table=_identity,
        unsplit_twin_columns=lambda rows: rows,
        truncate_at_body_text=lambda rows: rows,
        is_valid_table=_valid,
        structural_quality=structural_quality,
        refine_table_bbox=refine_table_bbox or (lambda page, bbox, width, height: bbox),
        parse_table_with_vision=parse_table_with_vision or (lambda path, page, bbox: []),
        to_markdown=_markdown,
        allow_vision_fallback=allow_vision_fallback,
    )


def test_preview_marks_structure_reparse_without_calling_vision() -> None:
    def quality(rows: list[list[str]]) -> dict[str, Any]:
        return {
            "consistency_score": 0.5,
            "fill_rate": 0.35,
            "fallback_to_vision": True,
            "fallback_reasons": ["low_consistency", "low_fill_rate"],
        }

    def forbidden_vision(path: str, page: int, bbox: tuple[float, float, float, float]):
        raise AssertionError("preview must not call vision")

    processor = _make_processor(
        allow_vision_fallback=False,
        structural_quality=quality,
        parse_table_with_vision=forbidden_vision,
    )

    chunk = processor.post_process(
        [["A", "B"], ["1", ""]],
        (10, 20, 110, 120),
        1,
        200,
        300,
        parser_name="unit",
        table_index=2,
    )

    assert chunk is not None
    assert chunk.raw_content == "| A | B |\n| --- | --- |\n| 1 |  |"
    assert "vision_fallback_needed" not in chunk.metadata
    assert chunk.metadata["structure_reparse_needed"] is True
    assert chunk.metadata["vision_fallback_disabled"] is True
    assert chunk.metadata["manual_review_needed"] is True
    assert chunk.metadata["vision_fallback_attempted"] is False
    assert chunk.metadata["vision_fallback_succeeded"] is False
    assert chunk.metadata["vision_fallback_reasons"] == ["low_consistency", "low_fill_rate"]
    assert chunk.metadata["extraction_attempt"]["parser"] == "unit"
    assert chunk.metadata["extraction_attempt"]["succeeded"] is True
    assert chunk.metadata["original_bbox"] == [10.0, 20.0, 110.0, 120.0]
    assert "vision_clip_bbox" not in chunk.metadata


def test_full_fallback_uses_refined_bbox_and_replaces_rows() -> None:
    calls: list[tuple[str, int, tuple[float, float, float, float]]] = []

    def quality(rows: list[list[str]]) -> dict[str, Any]:
        return {
            "consistency_score": 0.4,
            "fill_rate": 0.7,
            "fallback_to_vision": True,
            "fallback_reasons": ["low_consistency"],
        }

    def refine(page, bbox, width, height):
        assert page == "pdf-page"
        assert width == 200
        assert height == 300
        return (5.0, 15.0, 150.0, 160.0)

    def parse_vision(path: str, page: int, bbox: tuple[float, float, float, float]):
        calls.append((path, page, bbox))
        return [["H1", "H2"], ["v1", "v2"], ["v3", "v4"]]

    processor = _make_processor(
        allow_vision_fallback=True,
        structural_quality=quality,
        refine_table_bbox=refine,
        parse_table_with_vision=parse_vision,
    )

    chunk = processor.post_process(
        [["A", "B"], ["1", ""]],
        (10, 20, 110, 120),
        2,
        200,
        300,
        parser_name="unit",
        pdf_page="pdf-page",
        table_index=1,
    )

    assert chunk is not None
    assert calls == [("paper.pdf", 1, (5.0, 15.0, 150.0, 160.0))]
    assert chunk.raw_content == "| H1 | H2 |\n| --- | --- |\n| v1 | v2 |\n| v3 | v4 |"
    assert chunk.metadata["vision_fallback_attempted"] is True
    assert chunk.metadata["vision_fallback_succeeded"] is True
    assert "vision_fallback_needed" not in chunk.metadata
    assert chunk.metadata["original_bbox"] == [10.0, 20.0, 110.0, 120.0]
    assert chunk.metadata["refined_bbox"] == [5.0, 15.0, 150.0, 160.0]
    assert chunk.metadata["vision_clip_bbox"] == [5.0, 15.0, 150.0, 160.0]
    assert chunk.metadata["rows"] == 3
    assert chunk.metadata["cols"] == 2


def test_full_fallback_can_use_region_clip_override() -> None:
    calls: list[tuple[str, int, tuple[float, float, float, float]]] = []

    def quality(rows: list[list[str]]) -> dict[str, Any]:
        return {
            "consistency_score": 0.4,
            "fill_rate": 0.7,
            "fallback_to_vision": True,
            "fallback_reasons": ["low_consistency"],
        }

    def parse_vision(path: str, page: int, bbox: tuple[float, float, float, float]):
        calls.append((path, page, bbox))
        return [["H1", "H2"], ["v1", "v2"]]

    processor = _make_processor(
        allow_vision_fallback=True,
        structural_quality=quality,
        refine_table_bbox=lambda page, bbox, width, height: (1, 2, 3, 4),
        parse_table_with_vision=parse_vision,
    )

    chunk = processor.post_process(
        [["A", "B"], ["1", ""]],
        (10, 20, 110, 120),
        2,
        200,
        300,
        parser_name="unit",
        pdf_page="pdf-page",
        table_index=1,
        vision_clip_bbox=(5, 15, 180, 220),
    )

    assert chunk is not None
    assert calls == [("paper.pdf", 1, (5.0, 15.0, 180.0, 220.0))]
    assert chunk.metadata["vision_clip_bbox"] == [5.0, 15.0, 180.0, 220.0]


def test_structural_quality_exposes_reasons_for_fallback() -> None:
    short_quality = cleanup.structural_quality([["A", "B"], ["1", "2"]])
    assert short_quality["fallback_to_vision"] is True
    assert short_quality["fallback_reasons"] == ["too_few_data_rows"]

    inconsistent_quality = cleanup.structural_quality(
        [
            ["A", "B", "C"],
            ["1", "2"],
            ["3"],
            ["4", "5"],
            ["6"],
        ]
    )
    assert inconsistent_quality["fallback_to_vision"] is True
    assert "low_consistency" in inconsistent_quality["fallback_reasons"]

    prose_quality = cleanup.structural_quality(
        [
            ["Measure", "Finding"],
            [
                "A",
                "The value was measured from the samples and therefore should be interpreted with caution.",
            ],
            [
                "B",
                "This result was also observed after treatment and within the control population.",
            ],
            [
                "C",
                "However the authors noted that these measurements were not significantly different.",
            ],
            ["D", "12.5"],
        ]
    )
    assert prose_quality["fallback_to_vision"] is True
    assert "prose_like_cells" in prose_quality["fallback_reasons"]
    assert prose_quality["prose_cell_ratio"] > 0


def test_rejects_headerless_rows_mixed_with_adjacent_body_text() -> None:
    rows = [
        ["B12", "0.86 ± 0.04", "2.30 ± 0.21", "broth of strain B1 with the result from the samples"],
        ["B13", "0.87 ± 0.03", "3.42 ± 0.09", ""],
        [
            "B14",
            "0.91 ± 0.12",
            "1.63 ± 0.05",
            "the pathway enzymes thereby increasing the yield of LNT II",
        ],
        [
            "B15",
            "0.50 ± 0.04",
            "2.51 ± 0.09",
            "metabolic pathway genes may have different effects on synthesis",
        ],
    ]

    assert cleanup.is_valid_table(rows) is False
