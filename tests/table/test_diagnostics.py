from __future__ import annotations

from pathlib import Path

from src.parsers.layout_chunk import LayoutChunk
from src.parsers.table.diagnostics import (
    attach_region_attempts,
    attach_region_diagnostics,
    bbox_iou,
    build_region_attempts,
)
from src.parsers.table.models import TableEvidence, TableRegion
from src.parsers.table.parser import TableParser
from src.parsers.config import TABLE_PARSER
from src.parsers.table import parser as table_parser_module
from src.judge.table_judge import RegionTextContext


def test_bbox_iou() -> None:
    assert bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)) == 25 / 175


def test_attach_region_diagnostics_to_matching_table_chunk() -> None:
    chunk = LayoutChunk(
        content_type="table",
        raw_content="table",
        page=1,
        bbox=(10, 10, 100, 100),
        column=0,
        order_in_page=0,
        metadata={},
    )
    region = TableRegion(
        page=1,
        bbox=(8, 8, 105, 105),
        confidence=0.7,
        evidence=[TableEvidence("caption", 0.35, "near_table_caption")],
    )

    attach_region_diagnostics([chunk], [region])

    assert chunk.metadata["table_region_match"] == "matched"
    assert chunk.metadata["table_region"]["band"] == "strong"


def test_region_discovery_chunk_exposes_table_label_and_caption() -> None:
    parser = TableParser("paper.pdf")
    region = TableRegion(
        page=1,
        bbox=(10, 20, 100, 120),
        confidence=0.7,
        evidence=[
            TableEvidence(
                "caption",
                0.35,
                "near_table_caption",
                details={"text": "Table S1. Strains used in this study"},
            )
        ],
    )

    chunks = parser._region_discovery_chunks([region])

    assert len(chunks) == 1
    assert chunks[0].metadata["table_label"] == "Table S1"
    assert chunks[0].metadata["table_caption"] == "Table S1. Strains used in this study"


def test_unparsed_strong_region_becomes_placeholder() -> None:
    parser = TableParser("paper.pdf", enable_vision_fallback=False)
    region = TableRegion(
        page=1,
        bbox=(10, 20, 200, 220),
        confidence=0.66,
        evidence=[TableEvidence("caption", 0.35, "near_table_caption")],
    )

    attempts = build_region_attempts([], [region])
    placeholders = parser._unparsed_region_chunks([], [region], attempts)

    assert len(placeholders) == 1
    chunk = placeholders[0]
    assert chunk.metadata["table_region_match"] == "unparsed"
    assert chunk.metadata["table_region_unparsed"] is True
    assert chunk.metadata["manual_review_needed"] is True
    assert chunk.metadata["structure_extraction_pending"] is True
    assert "vision_fallback_needed" not in chunk.metadata
    assert {
        attempt["parser"] for attempt in chunk.metadata["region_extraction_attempts"]
    } == {"region_guided", "pdfplumber_region"}
    assert all(
        attempt["succeeded"] is False
        for attempt in chunk.metadata["region_extraction_attempts"]
    )


def test_region_discovery_mode_emits_bbox_only_chunk() -> None:
    parser = TableParser("paper.pdf", enable_vision_fallback=False)
    region = TableRegion(
        page=2,
        bbox=(10, 20, 200, 220),
        confidence=0.72,
        evidence=[
            TableEvidence(
                "caption",
                0.35,
                "near_table_caption",
                details={"text": "Table 1. Example"},
            )
        ],
    )

    chunks = parser._region_discovery_chunks([region])

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.raw_content.startswith("[Table region detected]")
    assert "Table 1. Example" in chunk.raw_content
    assert chunk.metadata["parser"] == "table_region_detector"
    assert chunk.metadata["table_region_match"] == "region_only"
    assert chunk.metadata["table_region_discovery_only"] is True
    assert chunk.metadata["structure_extraction_pending"] is True
    assert chunk.metadata["vision_fallback_disabled"] is True
    assert "vision_fallback_needed" not in chunk.metadata


def test_llm_judge_disabled_adds_rule_feedback_without_calling_llm(monkeypatch) -> None:
    monkeypatch.setattr(TABLE_PARSER, "LLM_JUDGE_ENABLED", False)
    parser = TableParser("paper.pdf", enable_vision_fallback=False)
    region = TableRegion(
        page=2,
        bbox=(10, 20, 200, 220),
        confidence=0.72,
        evidence=[
            TableEvidence(
                "caption",
                0.35,
                "near_table_caption",
                details={"text": "Table 1. Example"},
            )
        ],
    )
    monkeypatch.setattr(parser, "_detect_regions", lambda: [region])
    monkeypatch.setattr(
        parser,
        "_collect_region_text_contexts",
        lambda regions: {region.region_id: RegionTextContext(word_count=2)},
    )

    chunks = parser.parse(strategy="all")

    assert len(chunks) == 1
    assert chunks[0].metadata["rule_failure_category"] == "bbox_too_small"
    assert chunks[0].metadata["rule_recommended_action"] == "expand_bbox"
    assert chunks[0].metadata["llm_decision_mode"] == "disabled"
    assert "llm_recommended_action" not in chunks[0].metadata


def test_llm_judge_writes_metadata_and_keeps_original_when_bbox_candidate_not_adopted(monkeypatch) -> None:
    class FakeJudgeResult:
        def to_dict(self) -> dict:
            return {
                "failure_category": "bbox_too_small",
                "recommended_action": "expand_bbox",
                "confidence": 0.8,
                "reason": "bbox likely too small",
            }

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(TABLE_PARSER, "LLM_JUDGE_ENABLED", True)
    monkeypatch.setattr(TABLE_PARSER, "BBOX_CANDIDATE_ENABLED", False)
    monkeypatch.setattr(
        table_parser_module,
        "judge_table_failure_with_llm",
        lambda **kwargs: FakeJudgeResult(),
    )
    parser = TableParser("paper.pdf", enable_vision_fallback=False)
    region = TableRegion(
        page=2,
        bbox=(10, 20, 200, 220),
        confidence=0.72,
        evidence=[
            TableEvidence(
                "caption",
                0.35,
                "near_table_caption",
                details={"text": "Table 1. Example"},
            )
        ],
    )
    monkeypatch.setattr(parser, "_detect_regions", lambda: [region])
    monkeypatch.setattr(
        parser,
        "_collect_region_text_contexts",
        lambda regions: {region.region_id: RegionTextContext(word_count=2)},
    )

    chunks = parser.parse(strategy="all")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.raw_content.startswith("[Table region detected]")
    assert chunk.bbox == region.bbox
    assert chunk.metadata["llm_decision_mode"] == "default"
    assert chunk.metadata["llm_recommended_action"] == "expand_bbox"
    assert chunk.metadata["safe_auto_action"] is True
    assert chunk.metadata["auto_action_allowed"] is True
    assert chunk.metadata["auto_action_attempted"] is True
    assert chunk.metadata["bbox_candidate_adopted"] is False
    assert chunk.metadata["rule_llm_agree"] is True


def test_llm_judge_replaces_chunk_only_for_safe_retry(monkeypatch) -> None:
    class FakeJudgeResult:
        def to_dict(self) -> dict:
            return {
                "failure_category": "bbox_too_small",
                "recommended_action": "retry_text_strategy",
                "confidence": 0.8,
                "reason": "retry text extraction",
            }

    class FakePdfplumberExtractor:
        def parse_regions(self, regions, feedback=None):
            assert feedback == "retry_text_strategy"
            region = regions[0]
            return [
                LayoutChunk(
                    content_type="table",
                    raw_content="| A | B |\n|---|---|\n| 1 | 2 |",
                    page=region.page,
                    bbox=region.bbox,
                    column=2,
                    order_in_page=0,
                    metadata={"parser": "pdfplumber_region_text", "rows": 2, "cols": 2},
                )
            ]

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(TABLE_PARSER, "LLM_JUDGE_ENABLED", True)
    monkeypatch.setattr(
        table_parser_module,
        "judge_table_failure_with_llm",
        lambda **kwargs: FakeJudgeResult(),
    )
    parser = TableParser("paper.pdf", enable_vision_fallback=False)
    parser._pdfplumber_extractor = FakePdfplumberExtractor()
    region = TableRegion(
        page=2,
        bbox=(10, 20, 200, 220),
        confidence=0.72,
        evidence=[
            TableEvidence(
                "caption",
                0.35,
                "near_table_caption",
                details={"text": "Table 1. Example"},
            )
        ],
    )
    monkeypatch.setattr(parser, "_detect_regions", lambda: [region])
    monkeypatch.setattr(
        parser,
        "_collect_region_text_contexts",
        lambda regions: {region.region_id: RegionTextContext(word_count=2)},
    )

    chunks = parser.parse(strategy="all")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.raw_content.startswith("| A | B |")
    assert chunk.metadata["auto_action_allowed"] is True
    assert chunk.metadata["llm_assistive_succeeded"] is True
    assert chunk.metadata["llm_assistive_action"] == "retry_text_strategy"


def test_drop_candidate_marks_excluded_without_deleting() -> None:
    parser = TableParser("paper.pdf", enable_vision_fallback=False)
    region = TableRegion(
        page=1,
        bbox=(10, 20, 200, 220),
        confidence=0.72,
        evidence=[TableEvidence("caption", 0.35, "near_table_caption")],
    )
    chunk = LayoutChunk(
        content_type="table",
        raw_content="not a table",
        page=1,
        bbox=region.bbox,
        column=2,
        order_in_page=0,
        metadata={"llm_recommended_action": "drop_candidate"},
    )

    replacement = parser._assistive_retry_chunk(chunk, region)

    assert replacement is None
    assert chunk.metadata["excluded_from_storage"] is True
    assert chunk.metadata["auto_action_succeeded"] is True


def test_mark_unparseable_marks_reason_without_deleting() -> None:
    parser = TableParser("paper.pdf", enable_vision_fallback=False)
    region = TableRegion(
        page=1,
        bbox=(10, 20, 200, 220),
        confidence=0.72,
        evidence=[TableEvidence("caption", 0.35, "near_table_caption")],
    )
    chunk = LayoutChunk(
        content_type="table",
        raw_content="fragmented text",
        page=1,
        bbox=region.bbox,
        column=2,
        order_in_page=0,
        metadata={
            "llm_recommended_action": "mark_unparseable",
            "llm_reason": "text layer is too fragmented",
        },
    )

    replacement = parser._assistive_retry_chunk(chunk, region)

    assert replacement is None
    assert chunk.metadata["excluded_from_storage"] is True
    assert chunk.metadata["unparseable_reason"] == "text layer is too fragmented"


def test_table_parser_uses_decoupled_judge_module() -> None:
    parser_source = Path("src/parsers/table/parser.py").read_text(encoding="utf-8")

    assert "from src.judge.table_judge import" in parser_source
    assert "src.parsers.table.llm_feedback_judge" not in parser_source
    assert "LLMJudge" not in parser_source


def test_keyword_entry_without_regions_falls_back_to_broad_parser(monkeypatch) -> None:
    parser = TableParser("paper.pdf", enable_vision_fallback=False)
    broad_chunk = LayoutChunk(
        content_type="table",
        raw_content="| A | B |\n|---|---|\n| 1 | 2 |",
        page=1,
        bbox=(10, 20, 100, 120),
        column=2,
        order_in_page=0,
        metadata={"parser": "pdfplumber", "rows": 2, "cols": 2},
    )
    monkeypatch.setattr(parser, "_detect_regions", lambda: [])
    monkeypatch.setattr(parser._router, "_keyword_guided", lambda: [])
    monkeypatch.setattr(parser._router, "_pdfplumber", lambda feedback=None: [broad_chunk])

    chunks = parser.parse(strategy="all")

    assert chunks == [broad_chunk]


def test_region_detection_failure_falls_back_to_broad_parser(monkeypatch) -> None:
    parser = TableParser("paper.pdf", enable_vision_fallback=False)
    broad_chunk = LayoutChunk(
        content_type="table",
        raw_content="table",
        page=1,
        bbox=(10, 20, 100, 120),
        column=2,
        order_in_page=0,
        metadata={"parser": "pdfplumber", "rows": 2, "cols": 2},
    )

    def fail_detect():
        raise RuntimeError("detector broke")

    monkeypatch.setattr(parser._region_detector, "detect", fail_detect)
    monkeypatch.setattr(parser._router, "_keyword_guided", lambda: [])
    monkeypatch.setattr(parser._router, "_pdfplumber", lambda feedback=None: [broad_chunk])

    chunks = parser.parse(strategy="all")

    assert chunks == [broad_chunk]


def test_structure_extraction_failure_without_regions_is_not_silenced(monkeypatch) -> None:
    parser = TableParser("paper.pdf", enable_vision_fallback=False)
    monkeypatch.setattr(parser, "_detect_regions", lambda: [])

    def fail_parse(*, strategy, feedback, regions):
        raise RuntimeError("router broke")

    monkeypatch.setattr(parser._router, "parse", fail_parse)

    try:
        parser.parse(strategy="all")
    except RuntimeError as exc:
        assert "router broke" in str(exc)
    else:
        raise AssertionError("router exception should propagate when no regions exist")


def test_unparsed_region_without_caption_needs_very_high_confidence_for_placeholder() -> None:
    parser = TableParser("paper.pdf", enable_vision_fallback=False)
    false_positive_like = TableRegion(
        page=1,
        bbox=(0, 144, 563, 280),
        confidence=0.70,
        evidence=[
            TableEvidence("lines", 0.30, "multiple_wide_horizontal_lines"),
            TableEvidence("alignment", 0.30, "repeated_x_alignment"),
            TableEvidence("content", 0.10, "short_cell_like_tokens"),
        ],
    )
    very_strong = TableRegion(
        page=1,
        bbox=(0, 300, 563, 430),
        confidence=0.90,
        evidence=[
            TableEvidence("lines", 0.30, "multiple_wide_horizontal_lines"),
            TableEvidence("alignment", 0.30, "repeated_x_alignment"),
            TableEvidence("content", 0.30, "high_numeric_density"),
        ],
    )

    placeholders = parser._unparsed_region_chunks(
        [],
        [false_positive_like, very_strong],
        build_region_attempts([], [false_positive_like, very_strong]),
    )

    assert len(placeholders) == 1
    assert placeholders[0].metadata["table_region"]["bbox"] == [0.0, 300.0, 563.0, 430.0]


def test_region_attempts_include_success_and_missing_extractors() -> None:
    region = TableRegion(
        page=1,
        bbox=(10, 20, 200, 220),
        confidence=0.7,
        evidence=[TableEvidence("caption", 0.35, "near_table_caption")],
    )
    chunk = LayoutChunk(
        content_type="table",
        raw_content="table",
        page=1,
        bbox=(12, 22, 198, 218),
        column=0,
        order_in_page=0,
        metadata={
            "parser": "pdfplumber_region_text",
            "rows": 3,
            "cols": 2,
            "source_region_id": region.region_id,
        },
    )

    attempts = build_region_attempts([chunk], [region])
    attach_region_attempts([chunk], attempts)

    region_attempts = chunk.metadata["region_extraction_attempts"]
    assert any(
        attempt["parser"] == "pdfplumber_region_text" and attempt["succeeded"] is True
        for attempt in region_attempts
    )
    assert any(
        attempt["parser"] == "region_guided" and attempt["succeeded"] is False
        for attempt in region_attempts
    )


def test_table_dedup_keeps_higher_quality_duplicate() -> None:
    weak = LayoutChunk(
        content_type="table",
        raw_content="weak",
        page=1,
        bbox=(10, 10, 100, 100),
        column=0,
        order_in_page=0,
        metadata={
            "parser": "region_guided_text",
            "rows": 2,
            "cols": 2,
            "vision_fallback_needed": True,
            "structural_quality": {
                "consistency_score": 0.5,
                "fill_rate": 0.4,
                "fallback_to_vision": True,
            },
        },
    )
    strong = LayoutChunk(
        content_type="table",
        raw_content="strong",
        page=1,
        bbox=(12, 12, 102, 102),
        column=0,
        order_in_page=1,
        metadata={
            "parser": "pdfplumber",
            "rows": 6,
            "cols": 4,
            "structural_quality": {
                "consistency_score": 1.0,
                "fill_rate": 0.9,
                "fallback_to_vision": False,
            },
        },
    )

    result = TableParser._deduplicate_chunks([weak, strong])

    assert len(result) == 1
    assert result[0].raw_content == "strong"


def test_table_dedup_removes_contained_subset() -> None:
    parent = LayoutChunk(
        content_type="table",
        raw_content="parent",
        page=1,
        bbox=(10, 10, 210, 210),
        column=0,
        order_in_page=0,
        metadata={
            "parser": "pdfplumber",
            "rows": 8,
            "cols": 4,
            "structural_quality": {
                "consistency_score": 1.0,
                "fill_rate": 0.95,
                "fallback_to_vision": False,
            },
        },
    )
    subset = LayoutChunk(
        content_type="table",
        raw_content="subset",
        page=1,
        bbox=(40, 40, 160, 120),
        column=0,
        order_in_page=1,
        metadata={
            "parser": "contained_candidate",
            "rows": 3,
            "cols": 4,
            "structural_quality": {
                "consistency_score": 1.0,
                "fill_rate": 0.9,
                "fallback_to_vision": False,
            },
        },
    )

    result = TableParser._deduplicate_chunks([parent, subset])

    assert len(result) == 1
    assert result[0].raw_content == "parent"
    assert result[0].metadata["dedup_replaced"][0]["reason"] in {"contained", "contained_subset"}


def test_table_dedup_removes_loose_contained_subset_marked_for_vision() -> None:
    parent = LayoutChunk(
        content_type="table",
        raw_content="parent",
        page=1,
        bbox=(10, 10, 210, 210),
        column=0,
        order_in_page=0,
        metadata={
            "parser": "pdfplumber",
            "rows": 8,
            "cols": 4,
            "structural_quality": {
                "consistency_score": 1.0,
                "fill_rate": 0.95,
                "fallback_to_vision": False,
            },
        },
    )
    loose_subset = LayoutChunk(
        content_type="table",
        raw_content="loose_subset",
        page=1,
        bbox=(70, 70, 230, 150),
        column=0,
        order_in_page=1,
        metadata={
            "parser": "loose_candidate",
            "rows": 2,
            "cols": 3,
            "vision_fallback_needed": True,
            "structural_quality": {
                "consistency_score": 0.5,
                "fill_rate": 0.4,
                "fallback_to_vision": True,
            },
        },
    )

    result = TableParser._deduplicate_chunks([parent, loose_subset])

    assert len(result) == 1
    assert result[0].raw_content == "parent"
    assert result[0].metadata["dedup_replaced"][0]["reason"] == "contained_subset"


def test_table_dedup_can_keep_better_contained_candidate() -> None:
    parent = LayoutChunk(
        content_type="table",
        raw_content="parent",
        page=1,
        bbox=(10, 10, 210, 210),
        column=0,
        order_in_page=0,
        metadata={
            "parser": "table_region_detector",
            "rows": 0,
            "cols": 0,
            "table_region_unparsed": True,
        },
    )
    extracted = LayoutChunk(
        content_type="table",
        raw_content="extracted",
        page=1,
        bbox=(40, 40, 160, 120),
        column=0,
        order_in_page=1,
        metadata={
            "parser": "pdfplumber_region_text",
            "rows": 4,
            "cols": 3,
            "structural_quality": {
                "consistency_score": 1.0,
                "fill_rate": 0.9,
                "fallback_to_vision": False,
            },
        },
    )

    result = TableParser._deduplicate_chunks([parent, extracted])

    assert len(result) == 1
    assert result[0].raw_content == "extracted"
    assert result[0].metadata["dedup_replaced"][0]["parser"] == "table_region_detector"


def test_table_dedup_removes_subset_that_spills_into_adjacent_text() -> None:
    parent = LayoutChunk(
        content_type="table",
        raw_content="parent",
        page=4,
        bbox=(47, 304, 296, 650),
        column=0,
        order_in_page=0,
        metadata={
            "parser": "pdfplumber_region_text",
            "rows": 20,
            "cols": 3,
            "structural_quality": {
                "consistency_score": 0.9,
                "fill_rate": 0.9,
                "fallback_to_vision": False,
            },
        },
    )
    spilling_subset = LayoutChunk(
        content_type="table",
        raw_content="subset plus body",
        page=4,
        bbox=(71, 489, 563, 640),
        column=2,
        order_in_page=1,
        metadata={
            "parser": "pdfplumber_region_text",
            "rows": 8,
            "cols": 8,
            "vision_fallback_needed": True,
            "structural_quality": {
                "consistency_score": 0.35,
                "fill_rate": 0.85,
                "fallback_to_vision": True,
            },
        },
    )

    result = TableParser._deduplicate_chunks([parent, spilling_subset])

    assert len(result) == 1
    assert result[0].raw_content == "parent"
    assert result[0].metadata["dedup_replaced"][0]["reason"] == "overlap_subset"


def test_dedup_keeps_split_fragments_from_same_region() -> None:
    top_fragment = LayoutChunk(
        content_type="table",
        raw_content="top",
        page=4,
        bbox=(43, 300, 563, 640),
        column=2,
        order_in_page=0,
        metadata={
            "parser": "pdfplumber_region_text",
            "source_region_id": "table_region_p4",
            "rows": 10,
            "cols": 7,
            "vision_fallback_needed": True,
            "vision_clip_bbox": [43.0, 300.0, 563.0, 640.0],
            "structural_quality": {
                "consistency_score": 0.7,
                "fill_rate": 0.72,
                "fallback_to_vision": True,
            },
        },
    )
    lower_fragment = LayoutChunk(
        content_type="table",
        raw_content="lower",
        page=4,
        bbox=(71, 489, 555, 678),
        column=2,
        order_in_page=1,
        metadata={
            "parser": "pdfplumber_region_text",
            "source_region_id": "table_region_p4",
            "rows": 8,
            "cols": 8,
            "vision_fallback_needed": True,
            "structural_quality": {
                "consistency_score": 0.35,
                "fill_rate": 0.85,
                "fallback_to_vision": True,
            },
        },
    )

    result = TableParser._deduplicate_chunks([top_fragment, lower_fragment])

    assert [chunk.raw_content for chunk in result] == ["top", "lower"]


def test_table_dedup_keeps_same_region_candidates_when_spatial_overlap_is_not_duplicate() -> None:
    broken = LayoutChunk(
        content_type="table",
        raw_content="broken six cols",
        page=3,
        bbox=(40, 70, 1040, 870),
        column=2,
        order_in_page=0,
        metadata={
            "parser": "pdfplumber_region_text",
            "source_region_id": "table_region_p3",
            "rows": 20,
            "cols": 6,
            "vision_fallback_needed": True,
            "structural_quality": {
                "consistency_score": 0.52,
                "fill_rate": 0.71,
                "fallback_to_vision": True,
            },
        },
    )
    folded = LayoutChunk(
        content_type="table",
        raw_content="folded three cols",
        page=3,
        bbox=(315, 73, 1000, 870),
        column=1,
        order_in_page=1,
        metadata={
            "parser": "pdfplumber_region_text",
            "source_region_id": "table_region_p3",
            "rows": 40,
            "cols": 3,
            "structural_quality": {
                "consistency_score": 1.0,
                "fill_rate": 0.95,
                "fallback_to_vision": False,
            },
        },
    )

    result = TableParser._deduplicate_chunks([broken, folded])

    assert [chunk.raw_content for chunk in result] == ["broken six cols", "folded three cols"]


def test_table_dedup_keeps_distinct_fragments_from_same_region() -> None:
    upper = LayoutChunk(
        content_type="table",
        raw_content="upper",
        page=3,
        bbox=(40, 70, 340, 300),
        column=0,
        order_in_page=0,
        metadata={
            "parser": "pdfplumber_region_text",
            "source_region_id": "table_region_p3",
            "rows": 20,
            "cols": 3,
            "structural_quality": {
                "consistency_score": 0.8,
                "fill_rate": 0.8,
                "fallback_to_vision": False,
            },
        },
    )
    lower = LayoutChunk(
        content_type="table",
        raw_content="lower",
        page=3,
        bbox=(315, 74, 555, 458),
        column=1,
        order_in_page=1,
        metadata={
            "parser": "pdfplumber",
            "source_region_id": "table_region_p3",
            "rows": 15,
            "cols": 3,
            "structural_quality": {
                "consistency_score": 0.8,
                "fill_rate": 0.8,
                "fallback_to_vision": False,
            },
        },
    )

    result = TableParser._deduplicate_chunks([upper, lower])

    assert [chunk.raw_content for chunk in result] == ["upper", "lower"]


def test_table_dedup_removes_exact_duplicate_from_same_region() -> None:
    first = LayoutChunk(
        content_type="table",
        raw_content="first",
        page=4,
        bbox=(29.59, 574.53, 565.66, 793.70),
        column=2,
        order_in_page=0,
        metadata={
            "parser": "region_guided_geometry",
            "source_region_id": "table_region_p4",
            "rows": 29,
            "cols": 4,
            "structural_quality": {
                "consistency_score": 1.0,
                "fill_rate": 0.95,
                "fallback_to_vision": False,
            },
        },
    )
    duplicate = LayoutChunk(
        content_type="table",
        raw_content="duplicate",
        page=4,
        bbox=(29.59, 574.53, 565.66, 793.70),
        column=2,
        order_in_page=1,
        metadata={
            "parser": "pdfplumber_region_geometry",
            "source_region_id": "table_region_p4",
            "rows": 29,
            "cols": 4,
            "structural_quality": {
                "consistency_score": 1.0,
                "fill_rate": 0.95,
                "fallback_to_vision": False,
            },
        },
    )

    result = TableParser._deduplicate_chunks([first, duplicate])

    assert len(result) == 1
    assert result[0].metadata["dedup_replaced"][0]["reason"] == "same_region_exact"


def test_region_fragments_expand_vision_clip_without_removing_chunks() -> None:
    upper = LayoutChunk(
        content_type="table",
        raw_content="upper",
        page=3,
        bbox=(40, 70, 340, 300),
        column=0,
        order_in_page=0,
        metadata={
            "parser": "pdfplumber_region_text",
            "source_region_id": "table_region_p3",
            "vision_fallback_needed": True,
            "vision_clip_bbox": [40.0, 70.0, 340.0, 300.0],
        },
    )
    lower = LayoutChunk(
        content_type="table",
        raw_content="lower",
        page=3,
        bbox=(315, 74, 555, 458),
        column=1,
        order_in_page=1,
        metadata={
            "parser": "pdfplumber",
            "source_region_id": "table_region_p3",
        },
    )

    result = TableParser._expand_vision_clips_by_region([upper, lower])

    assert [chunk.raw_content for chunk in result] == ["upper", "lower"]
    assert result[0].bbox == (40, 70, 340, 300)
    assert result[0].metadata["vision_clip_bbox"] == [40.0, 70.0, 555.0, 458.0]
    assert result[0].metadata["region_fragment_count_for_vision_clip"] == 2


def test_same_region_adjacent_fragments_are_kept_for_recall() -> None:
    first = LayoutChunk(
        content_type="table",
        raw_content=(
            "| strains | characteristics | source |\n"
            "| --- | --- | --- |\n"
            "| DH5α | left desc | Invitrogen |\n"
            "| B8 | desc | thisstudy |"
        ),
        page=3,
        bbox=(53, 94, 545, 298),
        column=2,
        order_in_page=0,
        metadata={
            "parser": "pdfplumber_region_text",
            "source_region_id": "table_region_p3",
            "rows": 4,
            "cols": 3,
            "vision_fallback_needed": True,
            "vision_clip_bbox": [53.0, 94.0, 545.0, 298.0],
        },
    )
    second = LayoutChunk(
        content_type="table",
        raw_content=(
            "| column 1 | column 2 | column 3 |\n"
            "| --- | --- | --- |\n"
            "| B9 | desc | thisstudy |\n"
            "| pET-galE | desc | thisstudy |"
        ),
        page=3,
        bbox=(53, 297, 358, 392),
        column=2,
        order_in_page=1,
        metadata={
            "parser": "pdfplumber_region_text",
            "source_region_id": "table_region_p3",
            "rows": 3,
            "cols": 3,
            "vision_fallback_needed": True,
        },
    )

    result = TableParser._deduplicate_chunks([first, second])

    assert [chunk.raw_content for chunk in result] == [first.raw_content, second.raw_content]
