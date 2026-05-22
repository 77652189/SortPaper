from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parsers.layout_chunk import LayoutChunk
from src.parsers.table.diagnostics import bbox_iou, bbox_overlap_ratio
from src.judge.table_judge import (
    TableJudgeFeedback,
    build_bbox_candidates,
    build_auto_decision,
    build_text_context,
    build_storage_decision,
    classify_table_result,
    compare_table_candidate_scores,
    judge_table_failure_with_llm,
    score_table_candidate,
)
from src.parsers.table.models import TableRegion
from src.parsers.table.parser import TableParser
from src.parsers.table.region_detector import TableRegionDetector


DEFAULT_PARSERS = ("region_guided", "pdfplumber")

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency guard
    load_dotenv = None


def safe_print(text: str) -> None:
    encoded = text.encode(sys.stdout.encoding or "utf-8", errors="backslashreplace")
    print(encoded.decode(sys.stdout.encoding or "utf-8", errors="replace"))


def load_project_env() -> None:
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")


@dataclass
class ParserResult:
    pdf: str
    page: int
    table_label: str
    caption: str
    region_id: str
    region_bbox: list[float]
    region_confidence: float
    region_band: str
    parser: str
    succeeded: bool
    duration_ms: int
    rows: int = 0
    cols: int = 0
    chunk_bbox: list[float] | None = None
    overlap: float = 0.0
    rule_category: str = ""
    recommended_action: str = ""
    needs_llm_judge: bool = False
    rule_confidence: float = 0.0
    rule_reasons: str = ""
    retry_attempted: bool = False
    retry_succeeded: bool = False
    retry_action: str = ""
    retry_rows: int = 0
    retry_cols: int = 0
    retry_overlap: float = 0.0
    retry_failure_reason: str = ""
    llm_judge_attempted: bool = False
    llm_failure_category: str = ""
    llm_recommended_action: str = ""
    llm_confidence: float = 0.0
    llm_reason: str = ""
    llm_error: str = ""
    rule_llm_agree: bool = False
    would_retry: bool = False
    would_adjust_bbox: bool = False
    would_drop_candidate: bool = False
    safe_auto_action: bool = False
    human_review_required: bool = False
    auto_action_allowed: bool = False
    auto_decision_reason: str = ""
    candidate_score: float = 0.0
    candidate_tier: str = ""
    candidate_reasons: str = ""
    table_repair_actions: str = ""
    retry_candidate_score: float = 0.0
    retry_score_delta: float = 0.0
    retry_score_verdict: str = ""
    retry_adoptable: bool = False
    excluded_from_storage: bool = False
    storage_exclusion_reason: str = ""
    storage_decision_source: str = ""
    failure_reason: str = ""
    preview: str = ""


def iter_pdfs(paths: Iterable[str], recursive: bool) -> list[Path]:
    pdfs: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.suffix.lower() == ".pdf":
            pdfs.append(path)
        elif path.is_dir():
            pattern = "**/*.pdf" if recursive else "*.pdf"
            pdfs.extend(sorted(path.glob(pattern)))
    return sorted({pdf.resolve() for pdf in pdfs})


def run_parser(
    table_parser: TableParser,
    parser_name: str,
    regions: list[TableRegion],
) -> tuple[list[LayoutChunk], int, str]:
    start = time.perf_counter()
    if not regions:
        return [], 0, ""
    try:
        if parser_name == "region_guided":
            chunks = table_parser._keyword_extractor.parse_regions(regions)
        elif parser_name == "pdfplumber":
            chunks = table_parser._pdfplumber_extractor.parse_regions(regions, feedback=None)
        else:
            raise ValueError(f"unknown parser: {parser_name}")
        return chunks, int((time.perf_counter() - start) * 1000), ""
    except Exception as exc:
        return [], int((time.perf_counter() - start) * 1000), f"{type(exc).__name__}: {exc}"


def run_parser_on_single_region(
    table_parser: TableParser,
    parser_name: str,
    region: TableRegion,
    *,
    feedback: str | None = None,
) -> tuple[LayoutChunk | None, int, str, float]:
    chunks, duration_ms, error = run_parser(
        table_parser,
        parser_name,
        [region],
    )
    if feedback and parser_name == "pdfplumber":
        start = time.perf_counter()
        try:
            chunks = table_parser._pdfplumber_extractor.parse_regions([region], feedback=feedback)
            duration_ms = int((time.perf_counter() - start) * 1000)
            error = ""
        except Exception as exc:
            chunks = []
            duration_ms = int((time.perf_counter() - start) * 1000)
            error = f"{type(exc).__name__}: {exc}"
    chunk, overlap = best_chunk_for_region(region, chunks)
    return chunk, duration_ms, error, overlap


def region_caption(region: TableRegion) -> str:
    for evidence in region.evidence:
        if evidence.source == "caption":
            return str(evidence.details.get("text", "")).strip()
    return ""


def best_chunk_for_region(region: TableRegion, chunks: list[LayoutChunk]) -> tuple[LayoutChunk | None, float]:
    best_chunk: LayoutChunk | None = None
    best_score = 0.0
    for chunk in chunks:
        source_region_id = chunk.metadata.get("source_region_id")
        if source_region_id and source_region_id == region.region_id:
            return chunk, 1.0
        if chunk.page != region.page:
            continue
        overlap = max(
            bbox_iou(chunk.bbox, region.bbox),
            bbox_overlap_ratio(chunk.bbox, region.bbox),
            bbox_overlap_ratio(region.bbox, chunk.bbox),
        )
        if overlap > best_score:
            best_chunk = chunk
            best_score = overlap
    if best_score < 0.10:
        return None, best_score
    return best_chunk, best_score


def result_for_region(
    *,
    pdf: Path,
    table_parser: TableParser,
    region: TableRegion,
    parser_name: str,
    chunks: list[LayoutChunk],
    duration_ms: int,
    parser_error: str,
    text_context_by_region: dict[str, dict],
    retry_enabled: bool,
    bbox_candidates_enabled: bool,
) -> ParserResult:
    caption = region_caption(region)
    table_label = table_parser._table_label_from_caption(caption)
    chunk, overlap = best_chunk_for_region(region, chunks)
    if parser_error:
        failure_reason = parser_error
    elif chunk is None:
        failure_reason = "no overlapping table chunk produced for this region"
    else:
        failure_reason = ""

    metadata = chunk.metadata if chunk else {}
    text_context = text_context_by_region.get(region.region_id, {}).get("context")
    feedback = classify_table_result(
        region=region,
        chunk_metadata=metadata,
        parser_succeeded=chunk is not None and not parser_error,
        parser_error=parser_error,
        overlap=overlap,
        text_context=text_context,
    )
    candidate_score = score_table_candidate(
        metadata=metadata,
        bbox=chunk.bbox if chunk else region.bbox,
        region_bbox=region.bbox,
        text_context=text_context,
        parser_succeeded=chunk is not None and not parser_error,
    )
    scored_metadata = {
        **metadata,
        "table_candidate_score": candidate_score.score,
        "table_candidate_tier": candidate_score.tier,
        "table_candidate_reasons": candidate_score.reasons,
        "table_region_unparsed": bool(chunk is None or parser_error or metadata.get("table_region_unparsed")),
        "rule_failure_category": feedback.failure_category,
        "rule_recommended_action": feedback.recommended_action,
    }
    storage_decision = build_storage_decision(
        content_type="table",
        metadata=scored_metadata,
    )
    retry = retry_result_for_action(
        table_parser=table_parser,
        parser_name=parser_name,
        region=region,
        original_metadata=metadata,
        original_bbox=chunk.bbox if chunk else region.bbox,
        original_score=candidate_score.to_dict(),
        feedback=feedback,
        text_context_by_region=text_context_by_region,
        retry_enabled=retry_enabled,
        bbox_candidates_enabled=bbox_candidates_enabled,
    )
    return ParserResult(
        pdf=pdf.name,
        page=region.page,
        table_label=table_label,
        caption=caption,
        region_id=region.region_id,
        region_bbox=[float(value) for value in region.bbox],
        region_confidence=round(float(region.confidence), 3),
        region_band=region.band,
        parser=parser_name,
        succeeded=chunk is not None and not parser_error,
        duration_ms=duration_ms,
        rows=int(metadata.get("rows") or 0),
        cols=int(metadata.get("cols") or 0),
        chunk_bbox=[float(value) for value in chunk.bbox] if chunk else None,
        overlap=round(float(overlap), 3),
        rule_category=feedback.failure_category,
        recommended_action=feedback.recommended_action,
        needs_llm_judge=feedback.needs_llm_judge,
        rule_confidence=round(float(feedback.confidence), 3),
        rule_reasons=" | ".join(feedback.reasons),
        retry_attempted=retry["attempted"],
        retry_succeeded=retry["succeeded"],
        retry_action=retry["action"],
        retry_rows=retry["rows"],
        retry_cols=retry["cols"],
        retry_overlap=retry["overlap"],
        retry_failure_reason=retry["failure_reason"],
        candidate_score=candidate_score.score,
        candidate_tier=candidate_score.tier,
        candidate_reasons=" | ".join(candidate_score.reasons),
        table_repair_actions=" | ".join(str(item) for item in (metadata.get("table_repair_actions") or [])),
        retry_candidate_score=retry["candidate_score"],
        retry_score_delta=retry["score_delta"],
        retry_score_verdict=retry["score_verdict"],
        retry_adoptable=retry["adoptable"],
        excluded_from_storage=bool(storage_decision.get("excluded_from_storage")),
        storage_exclusion_reason=str(storage_decision.get("storage_exclusion_reason") or ""),
        storage_decision_source=str(storage_decision.get("storage_decision_source") or ""),
        failure_reason=failure_reason,
        preview=(chunk.raw_content[:240].replace("\n", " ") if chunk else ""),
    )


def judge_result_row_with_llm(
    *,
    row: dict,
    text_context: dict[str, Any],
    model: str,
) -> dict:
    empty = {
        "attempted": False,
        "failure_category": "",
        "recommended_action": "",
        "confidence": 0.0,
        "reason": "",
        "error": "",
        "agree": False,
    }
    feedback = TableJudgeFeedback(
        failure_category=row.get("rule_category") or "unknown",
        recommended_action=row.get("recommended_action") or "needs_llm_judge",
        needs_llm_judge=bool(row.get("needs_llm_judge")),
        confidence=float(row.get("rule_confidence") or 0.0),
        reasons=str(row.get("rule_reasons") or "").split(" | ") if row.get("rule_reasons") else [],
    )
    try:
        judged = judge_table_failure_with_llm(
            pdf_name=str(row.get("pdf") or ""),
            page=int(row.get("page") or 0),
            table_label=str(row.get("table_label") or ""),
            caption=str(row.get("caption") or ""),
            region_bbox=[float(value) for value in row.get("region_bbox") or []],
            parser_name=str(row.get("parser") or ""),
            parser_succeeded=bool(row.get("succeeded")),
            rows=int(row.get("rows") or 0),
            cols=int(row.get("cols") or 0),
            failure_reason=str(row.get("failure_reason") or ""),
            parser_preview=str(row.get("preview") or ""),
            rule_feedback=feedback,
            text_context=text_context,
            model=model,
        )
        data = judged.to_dict()
        agree = (
            data["failure_category"] == row.get("rule_category")
            and data["recommended_action"] == row.get("recommended_action")
        )
        return {
            "attempted": True,
            "failure_category": data["failure_category"],
            "recommended_action": data["recommended_action"],
            "confidence": data["confidence"],
            "reason": data["reason"],
            "error": "",
            "agree": agree,
        }
    except Exception as exc:
        return dict(empty, attempted=True, error=f"{type(exc).__name__}: {exc}")


def select_llm_judge_rows(
    report: dict,
    max_regions: int,
    *,
    only_failures: bool = False,
) -> list[dict]:
    rows = [
        row
        for pdf_report in report.get("pdf_reports", [])
        for row in pdf_report.get("results", [])
    ]

    def _key(row: dict) -> tuple[str, str]:
        return str(row.get("pdf", "")), str(row.get("region_id", ""))

    selected: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def _add(candidates: list[dict]) -> None:
        for row in candidates:
            if len(selected) >= max_regions:
                return
            key = _key(row)
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)

    primary = [
        row for row in rows
        if row.get("needs_llm_judge") and not row.get("retry_succeeded")
    ]
    _add(primary)
    fallback_categories = {
        "body_text_contamination",
        "no_parser_output",
        "header_or_span_complexity",
    }
    fallback = [
        row for row in rows
        if row.get("rule_category") in fallback_categories
    ]
    _add(fallback)
    if only_failures:
        return selected
    return selected


def build_llm_judge_summary(report: dict) -> dict:
    rows = [
        row
        for pdf_report in report.get("pdf_reports", [])
        for row in pdf_report.get("results", [])
        if row.get("llm_judge_attempted")
    ]
    agree_count = sum(1 for row in rows if row.get("rule_llm_agree"))
    errors = Counter(str(row.get("llm_error") or "") for row in rows)
    errors.pop("", None)
    return {
        "attempted_rows": len(rows),
        "agree_count": agree_count,
        "agreement_rate": round(agree_count / len(rows), 3) if rows else 0.0,
        "errors": dict(errors),
        "rule_categories": dict(Counter(str(row.get("rule_category") or "") for row in rows)),
        "rule_actions": dict(Counter(str(row.get("recommended_action") or "") for row in rows)),
        "llm_categories": dict(Counter(str(row.get("llm_failure_category") or "") for row in rows)),
        "llm_actions": dict(Counter(str(row.get("llm_recommended_action") or "") for row in rows)),
        "safe_auto_actions": sum(1 for row in rows if row.get("safe_auto_action")),
        "human_review_required": sum(1 for row in rows if row.get("human_review_required")),
        "would_retry": sum(1 for row in rows if row.get("would_retry")),
        "would_adjust_bbox": sum(1 for row in rows if row.get("would_adjust_bbox")),
        "would_drop_candidate": sum(1 for row in rows if row.get("would_drop_candidate")),
    }


def build_self_check_summary(report: dict) -> dict:
    rows = [
        row
        for pdf_report in report.get("pdf_reports", [])
        for row in pdf_report.get("results", [])
    ]
    pdf_reports = report.get("pdf_reports", [])
    zero_region_with_caption = _zero_region_with_caption(pdf_reports)
    if not rows:
        return {
            "rows": 0,
            "success_rate": 0.0,
            "avg_candidate_score": 0.0,
            "candidate_tiers": {},
            "retry_verdicts": {},
            "risk_flags": {},
            "repair_actions": {},
            "storage": {
                "candidate_rows": 0,
                "storable_rows": 0,
                "excluded_from_storage": 0,
                "excluded_reasons": {},
                "decision_sources": {},
            },
            "zero_region_pdfs": sum(1 for pdf_report in pdf_reports if not pdf_report.get("regions")),
            "zero_region_with_caption": zero_region_with_caption[:20],
            "slow_pdfs": [],
            "slow_parsers": [],
        }

    succeeded = sum(1 for row in rows if row.get("succeeded"))
    avg_score = sum(float(row.get("candidate_score") or 0.0) for row in rows) / len(rows)
    risk_flags = Counter()
    for row in rows:
        reasons = str(row.get("candidate_reasons") or "")
        if "prose" in reasons or "body" in reasons:
            risk_flags["body_text_contamination"] += 1
        if "low_consistency" in reasons:
            risk_flags["low_consistency"] += 1
        if row.get("rule_category") == "candidate_false_positive":
            risk_flags["candidate_false_positive"] += 1
        if row.get("needs_llm_judge"):
            risk_flags["needs_llm_judge"] += 1
        if row.get("retry_score_verdict") == "regressed":
            risk_flags["retry_regressed"] += 1
        if row.get("excluded_from_storage"):
            risk_flags["excluded_from_storage"] += 1

    slow_pdfs = sorted(
        [
            {
                "pdf": Path(str(pdf_report.get("pdf") or "")).name,
                "detect_ms": int(pdf_report.get("detect_ms") or 0),
                "regions": len(pdf_report.get("regions") or []),
            }
            for pdf_report in pdf_reports
        ],
        key=lambda item: item["detect_ms"],
        reverse=True,
    )[:10]
    slow_parsers = sorted(
        [
            {
                "pdf": Path(str(pdf_report.get("pdf") or "")).name,
                "parser": parser_name,
                "duration_ms": int(summary.get("duration_ms") or 0),
                "chunks": int(summary.get("chunks") or 0),
            }
            for pdf_report in pdf_reports
            for parser_name, summary in (pdf_report.get("parser_summaries") or {}).items()
        ],
        key=lambda item: item["duration_ms"],
        reverse=True,
    )[:10]
    return {
        "rows": len(rows),
        "success_rate": round(succeeded / len(rows), 3),
        "avg_candidate_score": round(avg_score, 3),
        "candidate_tiers": dict(Counter(str(row.get("candidate_tier") or "") for row in rows)),
        "rule_categories": dict(Counter(str(row.get("rule_category") or "") for row in rows)),
        "recommended_actions": dict(Counter(str(row.get("recommended_action") or "") for row in rows)),
        "retry_attempted": sum(1 for row in rows if row.get("retry_attempted")),
        "retry_succeeded": sum(1 for row in rows if row.get("retry_succeeded")),
        "retry_adoptable": sum(1 for row in rows if row.get("retry_adoptable")),
        "retry_verdicts": dict(Counter(str(row.get("retry_score_verdict") or "") for row in rows if row.get("retry_attempted"))),
        "repair_actions": dict(Counter(
            action
            for row in rows
            for action in str(row.get("table_repair_actions") or "").split(" | ")
            if action
        )),
        "risk_flags": dict(risk_flags),
        "storage": {
            "candidate_rows": len(rows),
            "storable_rows": sum(1 for row in rows if not row.get("excluded_from_storage")),
            "excluded_from_storage": sum(1 for row in rows if row.get("excluded_from_storage")),
            "excluded_reasons": dict(Counter(
                str(row.get("storage_exclusion_reason") or "")
                for row in rows
                if row.get("excluded_from_storage")
            )),
            "decision_sources": dict(Counter(
                str(row.get("storage_decision_source") or "")
                for row in rows
                if row.get("storage_decision_source")
            )),
        },
        "zero_region_pdfs": sum(1 for pdf_report in pdf_reports if not pdf_report.get("regions")),
        "zero_region_with_caption": zero_region_with_caption[:20],
        "slow_pdfs": slow_pdfs,
        "slow_parsers": slow_parsers,
    }


def _zero_region_with_caption(pdf_reports: list[dict]) -> list[dict]:
    return [
        {
            "pdf": Path(str(pdf_report.get("pdf") or "")).name,
            "anchored_count": int((pdf_report.get("caption_scan") or {}).get("anchored_count") or 0),
            "broad_count": int((pdf_report.get("caption_scan") or {}).get("broad_count") or 0),
        }
        for pdf_report in pdf_reports
        if not pdf_report.get("regions")
        and (
            int((pdf_report.get("caption_scan") or {}).get("anchored_count") or 0) > 0
            or int((pdf_report.get("caption_scan") or {}).get("broad_count") or 0) > 0
        )
    ]


def apply_llm_judge_sample(
    report: dict,
    max_regions: int,
    model: str,
    *,
    only_failures: bool,
    decision_mode: str = "shadow",
) -> dict:
    selected = select_llm_judge_rows(report, max_regions, only_failures=only_failures)
    selected_keys = {
        (str(row.get("pdf", "")), str(row.get("region_id", "")), str(row.get("parser", "")))
        for row in selected
    }
    contexts_by_pdf_region: dict[tuple[str, str], dict] = {}
    for pdf_report in report.get("pdf_reports", []):
        contexts = pdf_report.get("region_text_contexts", {})
        for region_id, context in contexts.items():
            for row in pdf_report.get("results", []):
                if row.get("region_id") == region_id:
                    contexts_by_pdf_region[(str(row.get("pdf", "")), str(region_id))] = context
                    break

    attempted = 0
    for pdf_report in report.get("pdf_reports", []):
        for row in pdf_report.get("results", []):
            key = (str(row.get("pdf", "")), str(row.get("region_id", "")), str(row.get("parser", "")))
            if key not in selected_keys:
                continue
            context = contexts_by_pdf_region.get((key[0], key[1]), {})
            llm_payload = judge_result_row_with_llm(row=row, text_context=context, model=model)
            row["llm_judge_attempted"] = llm_payload["attempted"]
            row["llm_failure_category"] = llm_payload["failure_category"]
            row["llm_recommended_action"] = llm_payload["recommended_action"]
            row["llm_confidence"] = llm_payload["confidence"]
            row["llm_reason"] = llm_payload["reason"]
            row["llm_error"] = llm_payload["error"]
            row["rule_llm_agree"] = llm_payload["agree"]
            decision = build_auto_decision(
                mode=decision_mode,
                llm_recommended_action=row["llm_recommended_action"],
                llm_confidence=float(row["llm_confidence"] or 0.0),
                llm_error=row["llm_error"],
            )
            row.update(decision)
            attempted += 1
    report["llm_judge_sample"] = {
        "requested_max_regions": max_regions,
        "selected_regions": len(selected),
        "attempted": attempted,
        "model": model,
        "decision_mode": decision_mode,
    }
    report["llm_judge_summary"] = build_llm_judge_summary(report)
    return report


def retry_result_for_action(
    *,
    table_parser: TableParser,
    parser_name: str,
    region: TableRegion,
    original_metadata: dict,
    original_bbox,
    original_score: dict,
    feedback: TableJudgeFeedback,
    text_context_by_region: dict[str, dict],
    retry_enabled: bool,
    bbox_candidates_enabled: bool = True,
) -> dict:
    empty = {
        "attempted": False,
        "succeeded": False,
        "action": "",
        "rows": 0,
        "cols": 0,
        "overlap": 0.0,
        "candidate_score": 0.0,
        "score_delta": 0.0,
        "score_verdict": "",
        "adoptable": False,
        "failure_reason": "",
    }
    if not retry_enabled:
        return empty
    if feedback.recommended_action not in {"expand_bbox", "shrink_bbox", "retry_text_strategy"}:
        return empty
    if feedback.recommended_action in {"expand_bbox", "shrink_bbox"} and not bbox_candidates_enabled:
        return empty

    page_info = text_context_by_region.get(region.region_id, {})
    page_width = float(page_info.get("page_width") or 0)
    page_height = float(page_info.get("page_height") or 0)

    retry_region = region
    if feedback.recommended_action in {"expand_bbox", "shrink_bbox"}:
        return bbox_candidate_retry_result(
            table_parser=table_parser,
            parser_name=parser_name,
            region=region,
            action=feedback.recommended_action,
            original_bbox=original_bbox,
            original_score=original_score,
            text_context_by_region=text_context_by_region,
            page_width=page_width,
            page_height=page_height,
        )

    retry_feedback = "retry_text_strategy" if feedback.recommended_action == "retry_text_strategy" else None
    if retry_feedback:
        start = time.perf_counter()
        try:
            chunks = table_parser._pdfplumber_extractor.parse_regions([retry_region], feedback=retry_feedback)
            error = ""
        except Exception as exc:
            chunks = []
            error = f"{type(exc).__name__}: {exc}"
        chunk, overlap = best_chunk_for_region(retry_region, chunks)
        _duration_ms = int((time.perf_counter() - start) * 1000)
    else:
        chunk, _duration_ms, error, overlap = run_parser_on_single_region(
            table_parser,
            parser_name,
            retry_region,
            feedback=retry_feedback,
        )
    metadata = chunk.metadata if chunk else {}
    candidate_score = score_table_candidate(
        metadata=metadata,
        bbox=chunk.bbox if chunk else retry_region.bbox,
        region_bbox=original_bbox,
        text_context=text_context_by_region.get(region.region_id, {}).get("context"),
        parser_succeeded=bool(chunk and not error),
    )
    comparison = compare_table_candidate_scores(original_score, candidate_score)
    failure_reason = error or ("" if chunk else "retry produced no overlapping table chunk")
    return {
        "attempted": True,
        "succeeded": bool(chunk and not error),
        "action": feedback.recommended_action,
        "rows": int(metadata.get("rows") or 0),
        "cols": int(metadata.get("cols") or 0),
        "overlap": round(float(overlap), 3),
        "candidate_score": candidate_score.score,
        "score_delta": comparison["score_delta"],
        "score_verdict": comparison["verdict"],
        "adoptable": comparison["verdict"] == "improved",
        "failure_reason": failure_reason,
    }


def bbox_candidate_retry_result(
    *,
    table_parser: TableParser,
    parser_name: str,
    region: TableRegion,
    action: str,
    original_bbox,
    original_score: dict,
    text_context_by_region: dict[str, dict],
    page_width: float,
    page_height: float,
) -> dict:
    empty = {
        "attempted": False,
        "succeeded": False,
        "action": "",
        "rows": 0,
        "cols": 0,
        "overlap": 0.0,
        "candidate_score": 0.0,
        "score_delta": 0.0,
        "score_verdict": "",
        "adoptable": False,
        "failure_reason": "",
    }
    candidates = build_bbox_candidates(
        region,
        action,
        page_width=page_width,
        page_height=page_height,
    )
    if not candidates:
        return empty

    best = dict(empty, attempted=True, action=action, failure_reason="candidate produced no table")
    best_score = -1.0
    for candidate in candidates:
        retry_region = TableRegion(
            page=region.page,
            bbox=candidate.bbox,
            confidence=region.confidence,
            evidence=region.evidence,
            source=region.source,
        )
        chunk, _duration_ms, error, overlap = run_parser_on_single_region(
            table_parser,
            parser_name,
            retry_region,
            feedback=None,
        )
        metadata = chunk.metadata if chunk else {}
        candidate_score = score_table_candidate(
            metadata=metadata,
            bbox=chunk.bbox if chunk else retry_region.bbox,
            region_bbox=original_bbox,
            text_context=text_context_by_region.get(region.region_id, {}).get("context"),
            parser_succeeded=bool(chunk and not error),
        )
        comparison = compare_table_candidate_scores(original_score, candidate_score)
        if candidate_score.score > best_score:
            best_score = candidate_score.score
            best = {
                "attempted": True,
                "succeeded": bool(chunk and not error),
                "action": action,
                "rows": int(metadata.get("rows") or 0),
                "cols": int(metadata.get("cols") or 0),
                "overlap": round(float(overlap), 3),
                "candidate_score": candidate_score.score,
                "score_delta": comparison["score_delta"],
                "score_verdict": comparison["verdict"],
                "adoptable": comparison["verdict"] == "improved",
                "failure_reason": error or ("" if chunk else "candidate produced no overlapping table chunk"),
            }
    return best


def collect_region_text_contexts(pdf: Path, regions: list[TableRegion]) -> dict[str, dict]:
    import pdfplumber

    if not regions:
        return {}
    by_page: dict[int, list[TableRegion]] = {}
    for region in regions:
        by_page.setdefault(region.page, []).append(region)

    result: dict[str, dict] = {}
    with pdfplumber.open(str(pdf)) as doc:
        for page_index, page_regions in by_page.items():
            if page_index < 1 or page_index > len(doc.pages):
                continue
            page = doc.pages[page_index - 1]
            page_width = float(page.width or 0)
            page_height = float(page.height or 0)
            for region in page_regions:
                try:
                    words = page.within_bbox(region.bbox).extract_words() or []
                except Exception:
                    words = []
                result[region.region_id] = {
                    "context": build_text_context(words),
                    "page_width": page_width,
                    "page_height": page_height,
                }
    return result


def scan_caption_pages(pdf: Path) -> dict[str, Any]:
    try:
        import fitz
    except Exception:
        return {"error": "fitz unavailable", "anchored_pages": [], "broad_pages": []}

    anchored_pages: list[int] = []
    broad_pages: list[int] = []
    try:
        with fitz.open(str(pdf)) as doc:
            for index, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                anchored, broad = TableRegionDetector._caption_page_matches(text)
                if anchored:
                    anchored_pages.append(index)
                if broad:
                    broad_pages.append(index)
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "anchored_pages": [],
            "broad_pages": [],
        }
    return {
        "error": "",
        "anchored_pages": anchored_pages,
        "broad_pages": broad_pages,
        "anchored_count": len(anchored_pages),
        "broad_count": len(broad_pages),
    }


def benchmark_pdf(
    pdf: Path,
    parsers: list[str],
    *,
    retry_enabled: bool,
    llm_judge_enabled: bool,
    llm_model: str,
    bbox_candidates_enabled: bool,
) -> dict:
    table_parser = TableParser(str(pdf), enable_vision_fallback=False)
    caption_scan = scan_caption_pages(pdf)
    detect_start = time.perf_counter()
    regions = [region for region in table_parser._detect_regions() if region.band != "weak"]
    detect_ms = int((time.perf_counter() - detect_start) * 1000)
    text_context_by_region = collect_region_text_contexts(pdf, regions)

    results: list[ParserResult] = []
    parser_summaries: dict[str, dict] = {}
    for parser_name in parsers:
        chunks, duration_ms, error = run_parser(table_parser, parser_name, regions)
        parser_summaries[parser_name] = {
            "duration_ms": duration_ms,
            "chunks": len(chunks),
            "error": error,
        }
        for region in regions:
            results.append(result_for_region(
                pdf=pdf,
                table_parser=table_parser,
                region=region,
                parser_name=parser_name,
                chunks=chunks,
                duration_ms=duration_ms,
                parser_error=error,
                text_context_by_region=text_context_by_region,
                retry_enabled=retry_enabled,
                bbox_candidates_enabled=bbox_candidates_enabled,
            ))

    return {
        "pdf": str(pdf),
        "detect_ms": detect_ms,
        "caption_scan": caption_scan,
        "regions": [region.to_metadata() for region in regions],
        "region_text_contexts": {
            region_id: payload["context"].to_dict()
            for region_id, payload in text_context_by_region.items()
        },
        "parser_summaries": parser_summaries,
        "results": [asdict(result) for result in results],
    }


def write_json(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for pdf_report in report["pdf_reports"]
        for row in pdf_report["results"]
    ]
    fieldnames = list(ParserResult.__dataclass_fields__.keys())
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run table-region-only parser benchmarks for one PDF or a PDF directory.",
    )
    parser.add_argument("paths", nargs="+", help="PDF file(s) or directory/directories")
    parser.add_argument(
        "--parsers",
        default=",".join(DEFAULT_PARSERS),
        help="Comma-separated parsers: region_guided,pdfplumber",
    )
    parser.add_argument("--recursive", action="store_true", help="Scan directories recursively")
    parser.add_argument("--max-pdfs", type=int, default=0, help="Limit number of PDFs, 0 means no limit")
    parser.add_argument("--sample-pdfs", type=int, default=0, help="Randomly sample N PDFs before --max-pdfs")
    parser.add_argument("--seed", type=int, default=20260520, help="Random seed for --sample-pdfs")
    parser.add_argument("--json", dest="json_output", default="", help="Write JSON report path")
    parser.add_argument("--csv", dest="csv_output", default="", help="Write CSV report path")
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="Only classify failures; do not apply recommended retry actions",
    )
    parser.add_argument("--llm-judge", action="store_true", help="Call DeepSeek for rows marked needs_llm_judge")
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Debug switch: keep LLM Judge sampling disabled",
    )
    parser.add_argument("--llm-model", default="deepseek-chat", help="DeepSeek model for --llm-judge")
    parser.add_argument("--llm-max-regions", type=int, default=100, help="Maximum unique failed/suspicious regions to judge")
    parser.add_argument("--llm-only-failures", action="store_true", help="Only judge failed/suspicious sampled regions")
    parser.add_argument(
        "--llm-shadow-report",
        action="store_true",
        help="Add shadow-mode what-would-happen automation fields to the report",
    )
    parser.add_argument(
        "--no-bbox-candidates",
        action="store_true",
        help="Disable bbox retry/candidate probes in benchmark retry simulation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_project_env()
    args = build_arg_parser().parse_args(argv)
    parsers = [item.strip() for item in args.parsers.split(",") if item.strip()]
    invalid = sorted(set(parsers) - set(DEFAULT_PARSERS))
    if invalid:
        raise SystemExit(f"Unknown parser(s): {', '.join(invalid)}")
    if args.llm_judge and args.no_llm_judge:
        raise SystemExit("--llm-judge and --no-llm-judge cannot be used together.")
    if args.llm_judge and not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY is required when --llm-judge is enabled.")
    if args.llm_judge and args.llm_max_regions <= 0:
        raise SystemExit("--llm-max-regions must be greater than 0.")

    pdfs = iter_pdfs(args.paths, recursive=args.recursive)
    if args.sample_pdfs > 0 and len(pdfs) > args.sample_pdfs:
        rng = random.Random(args.seed)
        pdfs = sorted(rng.sample(pdfs, args.sample_pdfs))
    if args.max_pdfs > 0:
        pdfs = pdfs[: args.max_pdfs]
    if not pdfs:
        raise SystemExit("No PDF files found.")

    pdf_reports = []
    for pdf in pdfs:
        safe_print(f"[table-benchmark] {pdf}")
        pdf_report = benchmark_pdf(
            pdf,
            parsers,
            retry_enabled=not args.no_retry,
            llm_judge_enabled=args.llm_judge and not args.no_llm_judge,
            llm_model=args.llm_model,
            bbox_candidates_enabled=not args.no_bbox_candidates,
        )
        pdf_reports.append(pdf_report)
        region_count = len(pdf_report["regions"])
        safe_print(f"  regions={region_count} detect={pdf_report['detect_ms']}ms")
        for parser_name, summary in pdf_report["parser_summaries"].items():
            ok = sum(
                1 for row in pdf_report["results"]
                if row["parser"] == parser_name and row["succeeded"]
            )
            safe_print(
                f"  {parser_name}: ok={ok}/{region_count} "
                f"chunks={summary['chunks']} time={summary['duration_ms']}ms"
                + (f" error={summary['error']}" if summary["error"] else "")
            )
        retry_ok = sum(1 for row in pdf_report["results"] if row["retry_succeeded"])
        retry_attempted = sum(1 for row in pdf_report["results"] if row["retry_attempted"])
        if retry_attempted:
            safe_print(f"  retries: ok={retry_ok}/{retry_attempted}")
        llm_attempted = sum(1 for row in pdf_report["results"] if row["llm_judge_attempted"])
        if llm_attempted:
            safe_print(f"  llm_judge: attempted={llm_attempted}")

    report = {
        "parsers": parsers,
        "pdf_count": len(pdf_reports),
        "llm_judge_enabled": bool(args.llm_judge and not args.no_llm_judge),
        "bbox_candidates_enabled": not args.no_bbox_candidates,
        "pdf_reports": pdf_reports,
    }
    report["self_check_summary"] = build_self_check_summary(report)
    if args.llm_judge:
        report = apply_llm_judge_sample(
            report,
            max_regions=args.llm_max_regions,
            model=args.llm_model,
            only_failures=args.llm_only_failures,
            decision_mode="shadow" if args.llm_shadow_report else "off",
        )

    if args.json_output:
        write_json(report, Path(args.json_output))
        safe_print(f"JSON report: {args.json_output}")
    if args.csv_output:
        write_csv(report, Path(args.csv_output))
        safe_print(f"CSV report: {args.csv_output}")
    if args.llm_judge:
        summary = report.get("llm_judge_summary", {})
        safe_print(
            "LLM Judge summary: "
            f"attempted={summary.get('attempted_rows', 0)} "
            f"agree={summary.get('agree_count', 0)} "
            f"agreement_rate={summary.get('agreement_rate', 0.0)}"
        )
    self_check = report.get("self_check_summary", {})
    storage = self_check.get("storage", {})
    safe_print(
        "Self-check summary: "
        f"rows={self_check.get('rows', 0)} "
        f"success_rate={self_check.get('success_rate', 0.0)} "
        f"avg_score={self_check.get('avg_candidate_score', 0.0)} "
        f"tiers={self_check.get('candidate_tiers', {})} "
        f"retry={self_check.get('retry_verdicts', {})} "
        f"storage={storage.get('storable_rows', 0)}/{storage.get('candidate_rows', 0)} "
        f"excluded={storage.get('excluded_from_storage', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
