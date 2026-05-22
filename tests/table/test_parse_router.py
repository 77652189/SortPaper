from __future__ import annotations

import pytest

from src.parsers.layout_chunk import LayoutChunk
from src.parsers.table.models import TableEvidence, TableRegion
from src.parsers.table.parse_router import TableParseRouter


def _chunk(name: str) -> LayoutChunk:
    return LayoutChunk(
        content_type="table",
        raw_content=name,
        page=1,
        bbox=(0, 0, 10, 10),
        column=0,
        order_in_page=0,
        metadata={"parser": name},
    )


def test_all_strategy_runs_extractors_in_priority_order() -> None:
    calls: list[str] = []

    router = TableParseRouter(
        keyword_guided=lambda: calls.append("keyword") or [_chunk("keyword")],
        pdfplumber=lambda feedback: calls.append(f"pdf:{feedback}") or [_chunk("pdf")],
        deduplicate=lambda chunks: chunks,
    )

    chunks = router.parse(strategy="all", feedback="tune")

    assert calls == ["keyword", "pdf:tune"]
    assert [c.raw_content for c in chunks] == ["keyword", "pdf"]


def test_all_strategy_uses_region_extractors_without_broad_scan_when_regions_exist() -> None:
    calls: list[str] = []
    region = TableRegion(
        page=2,
        bbox=(0, 0, 10, 10),
        confidence=0.7,
        evidence=[TableEvidence("caption", 0.35, "near_table_caption")],
    )

    router = TableParseRouter(
        keyword_guided=lambda: calls.append("keyword") or [],
        keyword_regions=lambda regions: calls.append(f"region_keyword:{len(regions)}") or [
            _chunk("region_keyword")
        ],
        pdfplumber=lambda feedback: calls.append("pdf") or [_chunk("pdf")],
        pdfplumber_regions=lambda regions, feedback: calls.append(
            f"region_pdf:{len(regions)}:{feedback}"
        ) or [_chunk("region_pdf")],
        deduplicate=lambda chunks: chunks,
    )

    chunks = router.parse(
        strategy="all",
        feedback="tune",
        regions=[region],
    )

    assert calls == [
        "region_keyword:1",
        "region_pdf:1:tune",
    ]
    assert [c.raw_content for c in chunks] == [
        "region_keyword",
        "region_pdf",
    ]


def test_single_engine_strategies_use_expected_extractors() -> None:
    calls: list[str] = []

    router = TableParseRouter(
        keyword_guided=lambda: calls.append("keyword") or [_chunk("keyword")],
        pdfplumber=lambda feedback: calls.append("pdf") or [_chunk("pdf")],
        deduplicate=lambda chunks: chunks,
    )

    assert [c.raw_content for c in router.parse(
        strategy="pdfplumber_only",
        feedback=None,
    )] == ["keyword", "pdf"]
    assert calls == ["keyword", "pdf"]


def test_unknown_strategy_fails_fast() -> None:
    router = TableParseRouter(
        keyword_guided=lambda: [],
        pdfplumber=lambda feedback: [],
        deduplicate=lambda chunks: chunks,
    )

    with pytest.raises(ValueError):
        router.parse(strategy="bad", feedback=None)
