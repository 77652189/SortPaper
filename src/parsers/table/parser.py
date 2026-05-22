"""
表格解析器：协调关键词/pdfplumber 等表格候选提取，并交给统一后处理器打包。
"""

from __future__ import annotations

import logging
from pathlib import Path
import re

from src.judge.table_judge import (
    build_auto_decision,
    BBoxCandidate,
    build_bbox_candidates,
    build_storage_decision,
    build_text_context,
    classify_table_result,
    compare_table_candidate_scores,
    judge_table_failure_with_llm,
    score_bbox_candidate,
    score_table_candidate,
)
from src.parsers.layout_chunk import LayoutChunk
from src.parsers.config import TABLE_PARSER as CFG
from src.parsers.table import cleanup
from src.parsers.table import geometry
from src.parsers.table.diagnostics import (
    attach_region_attempts,
    attach_region_diagnostics,
    build_region_attempts,
    bbox_iou,
    bbox_overlap_ratio,
    find_best_region,
)
from src.parsers.table.keyword_extractor import KeywordGuidedTableExtractor
from src.parsers.table.models import TableRegion
from src.parsers.table.parse_router import TableParseRouter
from src.parsers.table.pdfplumber_extractor import PdfplumberTableExtractor
from src.parsers.table.post_processor import TablePostProcessor
from src.parsers.table.region_detector import TableRegionDetector
from src.parsers.table.vision_fallback import parse_table_with_vision

logger = logging.getLogger(__name__)

class TableParser:
    def __init__(self, pdf_path: str | Path, enable_vision_fallback: bool = False) -> None:
        self.pdf_path = str(pdf_path)
        self.enable_vision_fallback = enable_vision_fallback
        self._region_detector = TableRegionDetector(self.pdf_path)
        self._post_processor = TablePostProcessor(
            pdf_path=self.pdf_path,
            normalize_table=cleanup.normalize_table,
            unsplit_twin_columns=cleanup.unsplit_twin_columns,
            truncate_at_body_text=cleanup.truncate_at_body_text,
            repair_table_structure=cleanup.repair_table_structure,
            is_valid_table=cleanup.is_valid_table,
            structural_quality=cleanup.structural_quality,
            refine_table_bbox=geometry.refine_table_bbox,
            parse_table_with_vision=parse_table_with_vision,
            to_markdown=cleanup.to_markdown,
            allow_vision_fallback=enable_vision_fallback,
        )
        self._keyword_extractor = KeywordGuidedTableExtractor(
            pdf_path=self.pdf_path,
            validate_and_fallback=self._validate_and_fallback,
        )
        self._pdfplumber_extractor = PdfplumberTableExtractor(
            pdf_path=self.pdf_path,
            post_process_table=self._post_process_table,
            validate_and_fallback=self._validate_and_fallback,
        )
        self._router = TableParseRouter(
            keyword_guided=self._keyword_extractor.parse,
            keyword_regions=self._keyword_extractor.parse_regions,
            pdfplumber=self._pdfplumber_extractor.parse,
            pdfplumber_regions=self._pdfplumber_extractor.parse_regions,
            deduplicate=TableParser._deduplicate_chunks,
        )

    def parse(self, feedback: str | None = None, strategy: str = "all") -> list[LayoutChunk]:
        """
        提取 PDF 所有表格，返回 LayoutChunk 列表。

        strategy 可选值：
          "all"             : 关键词引导 + pdfplumber（默认）
          "pdfplumber_only" : 单引擎模式
        """
        regions = self._detect_regions() if strategy == "all" else []
        if strategy == "all" and CFG.DISCOVERY_ONLY:
            chunks = self._region_discovery_chunks(regions)
            return self._attach_judge_feedback(chunks, regions)
        try:
            chunks = self._router.parse(
                strategy=strategy,
                feedback=feedback,
                regions=regions,
            )
        except Exception as exc:
            logger.exception("table structure extraction failed")
            if not regions:
                raise
            chunks = []
        if regions:
            chunks = attach_region_diagnostics(chunks, regions)
            attempts_by_region = build_region_attempts(chunks, regions)
            chunks = attach_region_attempts(chunks, attempts_by_region)
            chunks.extend(self._unparsed_region_chunks(chunks, regions, attempts_by_region))
            chunks = self._deduplicate_chunks(chunks)
            chunks = self._expand_vision_clips_by_region(chunks)
            chunks = self._attach_judge_feedback(chunks, regions)
        return chunks

    def _detect_regions(self) -> list[TableRegion]:
        try:
            return self._region_detector.detect()
        except Exception:
            logger.exception("table region detection failed; continuing with broad parser fallback")
            return []

    def _region_discovery_chunks(self, regions: list[TableRegion]) -> list[LayoutChunk]:
        chunks: list[LayoutChunk] = []
        for index, region in enumerate(regions):
            if region.band == "weak":
                continue
            caption = self._region_caption_text(region)
            table_label = self._table_label_from_caption(caption)
            metadata = {
                "parser": "table_region_detector",
                "rows": 0,
                "cols": 0,
                "table_region": region.to_metadata(),
                "table_label": table_label,
                "table_caption": caption,
                "table_region_match": "region_only",
                "table_region_unparsed": True,
                "table_region_discovery_only": True,
                "manual_review_needed": True,
                "structure_extraction_pending": True,
                "structure_reparse_needed": True,
                "vision_fallback_disabled": True,
                "vision_fallback_attempted": False,
                "vision_fallback_succeeded": False,
                "caption": caption,
                "unparsed_region_reason": (
                    "table region discovery mode: structure extraction is intentionally disabled"
                ),
                "region_extraction_attempts": [],
            }
            raw_content = self._region_discovery_content(region, caption)
            chunks.append(LayoutChunk(
                content_type="table",
                raw_content=raw_content,
                page=region.page,
                bbox=region.bbox,
                column=2,
                order_in_page=index,
                metadata=metadata,
            ))
        return chunks

    @staticmethod
    def _region_caption_text(region: TableRegion) -> str:
        for evidence in region.evidence:
            if evidence.source == "caption":
                return str(evidence.details.get("text", "")).strip()
        return ""

    @staticmethod
    def _table_label_from_caption(caption: str) -> str:
        match = re.search(
            r"\b(supplementary\s+table|table|tab\.)\s*(s?\d+)\b",
            caption,
            re.IGNORECASE,
        )
        if not match:
            return ""
        prefix = "Supplementary Table" if "supplementary" in match.group(1).lower() else "Table"
        return f"{prefix} {match.group(2).upper()}"

    @staticmethod
    def _region_discovery_content(region: TableRegion, caption: str) -> str:
        bbox = ", ".join(f"{value:.1f}" for value in region.bbox)
        lines = [
            "[Table region detected]",
            f"page: {region.page}",
            f"bbox: [{bbox}]",
            f"confidence: {region.confidence:.2f}",
            f"band: {region.band}",
        ]
        if caption:
            lines.insert(1, f"caption: {caption}")
        return "\n".join(lines)

    def _unparsed_region_chunks(
        self,
        chunks: list[LayoutChunk],
        regions: list[TableRegion],
        attempts_by_region: dict[str, list[dict]] | None = None,
    ) -> list[LayoutChunk]:
        attempts_by_region = attempts_by_region or {}
        placeholders: list[LayoutChunk] = []
        for region in regions:
            if region.band == "weak":
                continue
            if not self._should_emit_unparsed_placeholder(region):
                continue
            probe = LayoutChunk(
                content_type="table",
                raw_content="",
                page=region.page,
                bbox=region.bbox,
                column=0,
                order_in_page=0,
            )
            if find_best_region(probe, [
                TableRegion(
                    page=chunk.page,
                    bbox=chunk.bbox,
                    confidence=1.0,
                    evidence=[],
                    source="extracted_table",
                )
                for chunk in chunks
                if chunk.content_type == "table"
            ], min_overlap=0.20):
                continue

            metadata = {
                "parser": "table_region_detector",
                "rows": 0,
                "cols": 0,
                "table_region": region.to_metadata(),
                "table_label": self._table_label_from_caption(self._region_caption_text(region)),
                "table_caption": self._region_caption_text(region),
                "table_region_match": "unparsed",
                "table_region_unparsed": True,
                "manual_review_needed": True,
                "structure_extraction_pending": True,
                "vision_fallback_attempted": False,
                "vision_fallback_succeeded": False,
                "unparsed_region_reason": (
                    "table-like region detected but no structure extractor produced a usable table"
                ),
                "region_extraction_attempts": attempts_by_region.get(region.region_id, []),
            }
            placeholders.append(LayoutChunk(
                content_type="table",
                raw_content="[Table region detected]\nstructure extraction pending",
                page=region.page,
                bbox=region.bbox,
                column=2,
                order_in_page=0,
                metadata=metadata,
            ))
        return placeholders

    def _attach_judge_feedback(
        self,
        chunks: list[LayoutChunk],
        regions: list[TableRegion],
    ) -> list[LayoutChunk]:
        if not chunks or not regions:
            return chunks
        regions_by_id = {region.region_id: region for region in regions}
        context_by_region = self._collect_region_text_contexts(regions)
        llm_enabled = bool(getattr(CFG, "LLM_JUDGE_ENABLED", True))

        for index, chunk in enumerate(list(chunks)):
            if chunk.content_type != "table":
                continue
            region = self._region_for_chunk(chunk, regions_by_id)
            if region is None:
                continue
            self._attach_region_identity(chunk, region)
            text_context = context_by_region.get(region.region_id)
            parser_succeeded = (
                not chunk.metadata.get("table_region_unparsed")
                and int(chunk.metadata.get("rows") or 0) >= 1
                and int(chunk.metadata.get("cols") or 0) >= 1
            )
            overlap = max(
                bbox_iou(chunk.bbox, region.bbox),
                bbox_overlap_ratio(chunk.bbox, region.bbox),
                bbox_overlap_ratio(region.bbox, chunk.bbox),
            )
            feedback = classify_table_result(
                region=region,
                chunk_metadata=chunk.metadata,
                parser_succeeded=parser_succeeded,
                overlap=overlap,
                text_context=text_context,
            )
            self._write_candidate_score(
                chunk=chunk,
                region=region,
                text_context=text_context,
                parser_succeeded=parser_succeeded,
            )
            self._write_rule_feedback(chunk, feedback)
            if llm_enabled and feedback.needs_llm_judge:
                self._write_llm_feedback(
                    chunk=chunk,
                    region=region,
                    feedback=feedback,
                    text_context=text_context,
                    parser_succeeded=parser_succeeded,
                )
                if chunk.metadata.get("auto_action_allowed"):
                    replacement = self._assistive_retry_chunk(chunk, region)
                    if replacement is not None:
                        chunks[index] = replacement
                        self._write_storage_decision(replacement)
                    else:
                        self._write_storage_decision(chunk)
                else:
                    self._write_storage_decision(chunk)
            else:
                chunk.metadata["llm_decision_mode"] = "disabled" if not llm_enabled else "not_needed"
                self._write_storage_decision(chunk)
        return chunks

    def _attach_region_identity(self, chunk: LayoutChunk, region: TableRegion) -> None:
        chunk.metadata.setdefault("source_region_id", region.region_id)
        chunk.metadata.setdefault("table_region", region.to_metadata())
        chunk.metadata.setdefault("table_region_match", "matched")
        caption = self._region_caption_text(region)
        if caption:
            chunk.metadata.setdefault("table_caption", caption)
            chunk.metadata.setdefault("caption", caption)
            chunk.metadata.setdefault("table_label", self._table_label_from_caption(caption))

    def _region_for_chunk(
        self,
        chunk: LayoutChunk,
        regions_by_id: dict[str, TableRegion],
    ) -> TableRegion | None:
        region_id = chunk.metadata.get("source_region_id")
        if not region_id:
            region_meta = chunk.metadata.get("table_region") or {}
            region_id = region_meta.get("region_id")
        return regions_by_id.get(str(region_id)) if region_id else None

    def _collect_region_text_contexts(self, regions: list[TableRegion]) -> dict[str, object]:
        try:
            import pdfplumber
        except Exception:
            return {}

        by_page: dict[int, list[TableRegion]] = {}
        for region in regions:
            by_page.setdefault(region.page, []).append(region)

        contexts: dict[str, object] = {}
        try:
            with pdfplumber.open(self.pdf_path) as doc:
                for page_index, page_regions in by_page.items():
                    if page_index < 1 or page_index > len(doc.pages):
                        continue
                    page = doc.pages[page_index - 1]
                    for region in page_regions:
                        try:
                            words = page.within_bbox(region.bbox).extract_words() or []
                        except Exception:
                            words = []
                        contexts[region.region_id] = build_text_context(words)
        except Exception:
            return {}
        return contexts

    @staticmethod
    def _write_rule_feedback(chunk: LayoutChunk, feedback) -> None:
        chunk.metadata["rule_failure_category"] = feedback.failure_category
        chunk.metadata["rule_recommended_action"] = feedback.recommended_action
        chunk.metadata["rule_needs_llm_judge"] = feedback.needs_llm_judge
        chunk.metadata["rule_confidence"] = round(float(feedback.confidence), 3)
        chunk.metadata["rule_reasons"] = list(feedback.reasons)

    @staticmethod
    def _write_candidate_score(
        *,
        chunk: LayoutChunk,
        region: TableRegion,
        text_context,
        parser_succeeded: bool,
    ) -> None:
        score = score_table_candidate(
            metadata=chunk.metadata,
            bbox=chunk.bbox,
            region_bbox=region.bbox,
            text_context=text_context,
            parser_succeeded=parser_succeeded,
        ).to_dict()
        chunk.metadata["table_candidate_score"] = score["score"]
        chunk.metadata["table_candidate_tier"] = score["tier"]
        chunk.metadata["table_candidate_reasons"] = score["reasons"]
        chunk.metadata["table_candidate_metrics"] = score["metrics"]

    def _write_llm_feedback(
        self,
        *,
        chunk: LayoutChunk,
        region: TableRegion,
        feedback,
        text_context,
        parser_succeeded: bool,
    ) -> None:
        chunk.metadata["llm_decision_mode"] = "default"
        try:
            judged = judge_table_failure_with_llm(
                pdf_name=Path(self.pdf_path).name,
                page=region.page,
                table_label=str(chunk.metadata.get("table_label") or ""),
                caption=str(chunk.metadata.get("table_caption") or chunk.metadata.get("caption") or ""),
                region_bbox=[float(value) for value in region.bbox],
                parser_name=str(chunk.metadata.get("parser") or ""),
                parser_succeeded=parser_succeeded,
                rows=int(chunk.metadata.get("rows") or 0),
                cols=int(chunk.metadata.get("cols") or 0),
                failure_reason=str(chunk.metadata.get("unparsed_region_reason") or ""),
                parser_preview=chunk.raw_content[:1200],
                rule_feedback=feedback,
                text_context=text_context.to_dict() if hasattr(text_context, "to_dict") else {},
                model=str(getattr(CFG, "LLM_JUDGE_MODEL", "deepseek-chat")),
            )
            data = judged.to_dict()
            chunk.metadata["llm_failure_category"] = data["failure_category"]
            chunk.metadata["llm_recommended_action"] = data["recommended_action"]
            chunk.metadata["llm_confidence"] = data["confidence"]
            chunk.metadata["llm_reason"] = data["reason"]
            chunk.metadata["llm_error"] = ""
            chunk.metadata["rule_llm_agree"] = (
                data["failure_category"] == feedback.failure_category
                and data["recommended_action"] == feedback.recommended_action
            )
            self._write_auto_decision(chunk)
        except Exception as exc:
            self._write_llm_error(chunk, f"{type(exc).__name__}: {exc}")

    def _write_llm_error(self, chunk: LayoutChunk, error: str) -> None:
        chunk.metadata["llm_decision_mode"] = "degraded"
        chunk.metadata["llm_failure_category"] = ""
        chunk.metadata["llm_recommended_action"] = ""
        chunk.metadata["llm_confidence"] = 0.0
        chunk.metadata["llm_reason"] = ""
        chunk.metadata["llm_error"] = error
        chunk.metadata["rule_llm_agree"] = False
        self._write_auto_decision(chunk)

    def _write_auto_decision(self, chunk: LayoutChunk) -> None:
        decision = build_auto_decision(
            llm_recommended_action=str(chunk.metadata.get("llm_recommended_action") or ""),
            llm_confidence=float(chunk.metadata.get("llm_confidence") or 0.0),
            llm_error=str(chunk.metadata.get("llm_error") or ""),
            confidence_threshold=float(getattr(CFG, "LLM_JUDGE_SAFE_CONFIDENCE", 0.65)),
            enabled=True,
        )
        chunk.metadata.update(decision)

    @staticmethod
    def _write_storage_decision(chunk: LayoutChunk) -> None:
        decision = build_storage_decision(
            content_type=chunk.content_type,
            metadata=chunk.metadata,
        )
        unparseable_reason = decision.pop("unparseable_reason", "")
        chunk.metadata.update(decision)
        if unparseable_reason and not chunk.metadata.get("unparseable_reason"):
            chunk.metadata["unparseable_reason"] = unparseable_reason

    def _assistive_retry_chunk(
        self,
        original: LayoutChunk,
        region: TableRegion,
    ) -> LayoutChunk | None:
        action = str(original.metadata.get("llm_recommended_action") or "")
        original.metadata["auto_action_attempted"] = True
        original.metadata["auto_action_succeeded"] = False
        if action == "mark_unparseable":
            original.metadata["unparseable_reason"] = original.metadata.get("llm_reason") or "marked unparseable by table judge"
            original.metadata["excluded_from_storage"] = True
            original.metadata["storage_exclusion_reason"] = original.metadata["unparseable_reason"]
            original.metadata["auto_action_succeeded"] = True
            return None
        if action == "drop_candidate":
            original.metadata["excluded_from_storage"] = True
            original.metadata["storage_exclusion_reason"] = original.metadata.get("llm_reason") or "candidate marked as false positive by table judge"
            original.metadata["auto_action_succeeded"] = True
            return None

        retry_region = region
        feedback = "retry_text_strategy" if action == "retry_text_strategy" else None
        if action in {"expand_bbox", "shrink_bbox"}:
            return self._run_bbox_candidates(original, region, action)

        original.metadata["llm_assistive_attempted"] = True
        try:
            chunks = self._pdfplumber_extractor.parse_regions([retry_region], feedback=feedback)
        except Exception as exc:
            original.metadata["llm_assistive_succeeded"] = False
            original.metadata["llm_assistive_reason"] = f"{type(exc).__name__}: {exc}"
            return None

        replacement = self._best_retry_chunk(retry_region, chunks)
        if replacement is None:
            original.metadata["llm_assistive_succeeded"] = False
            original.metadata["llm_assistive_reason"] = "assistive retry produced no valid table"
            return None

        original_score = score_table_candidate(
            metadata=original.metadata,
            bbox=original.bbox,
            region_bbox=region.bbox,
            parser_succeeded=not original.metadata.get("table_region_unparsed"),
        )
        replacement_score = score_table_candidate(
            metadata=replacement.metadata,
            bbox=replacement.bbox,
            region_bbox=region.bbox,
            parser_succeeded=True,
        )
        comparison = compare_table_candidate_scores(original_score, replacement_score)
        original.metadata["llm_assistive_score"] = replacement_score.to_dict()
        original.metadata["llm_assistive_score_delta"] = comparison["score_delta"]
        original.metadata["llm_assistive_score_verdict"] = comparison["verdict"]
        if comparison["verdict"] != "improved":
            original.metadata["llm_assistive_succeeded"] = False
            original.metadata["llm_assistive_reason"] = "assistive retry did not improve self-check score"
            return None

        replacement.metadata.update({
            key: value
            for key, value in original.metadata.items()
            if key.startswith("rule_")
            or key.startswith("llm_")
            or key in {
                "table_region",
                "table_label",
                "table_caption",
                "caption",
                "table_region_match",
                "region_extraction_attempts",
                "would_retry",
                "would_adjust_bbox",
                "would_drop_candidate",
                "safe_auto_action",
                "human_review_required",
                "auto_action_allowed",
                "auto_decision_reason",
                "table_candidate_score",
                "table_candidate_tier",
                "table_candidate_reasons",
                "table_candidate_metrics",
            }
        })
        replacement.metadata["table_candidate_score"] = replacement_score.score
        replacement.metadata["table_candidate_tier"] = replacement_score.tier
        replacement.metadata["table_candidate_reasons"] = replacement_score.reasons
        replacement.metadata["table_candidate_metrics"] = replacement_score.metrics
        replacement.metadata["table_region_match"] = "assistive_retry"
        replacement.metadata["llm_assistive_attempted"] = True
        replacement.metadata["llm_assistive_succeeded"] = True
        replacement.metadata["llm_assistive_original_chunk_id"] = original.chunk_id
        replacement.metadata["llm_assistive_action"] = action
        replacement.metadata["llm_assistive_score_delta"] = comparison["score_delta"]
        replacement.metadata["llm_assistive_score_verdict"] = comparison["verdict"]
        replacement.metadata["auto_action_attempted"] = True
        replacement.metadata["auto_action_succeeded"] = True
        return replacement

    def _run_bbox_candidates(
        self,
        original: LayoutChunk,
        region: TableRegion,
        action: str,
    ) -> LayoutChunk | None:
        if not getattr(CFG, "BBOX_CANDIDATE_ENABLED", True):
            original.metadata["bbox_candidate_adopted"] = False
            original.metadata["bbox_candidate_reason"] = "bbox candidate runner disabled"
            return None
        page_bbox = self._page_bbox(region.page)
        page_width, page_height = page_bbox[2], page_bbox[3]
        candidates = build_bbox_candidates(
            region,
            action,
            page_width=page_width,
            page_height=page_height,
        )
        candidates = self._clip_bbox_candidates(candidates, page_bbox)
        original.metadata["bbox_candidates_tried"] = [candidate.to_dict() for candidate in candidates]
        best_chunk: LayoutChunk | None = None
        best_candidate: dict | None = None
        best_score = -1.0
        best_reason = "no_candidate"
        for candidate in candidates:
            candidate_region = TableRegion(
                page=region.page,
                bbox=candidate.bbox,
                confidence=region.confidence,
                evidence=region.evidence,
                source=region.source,
            )
            try:
                chunks = self._pdfplumber_extractor.parse_regions([candidate_region], feedback=None)
            except Exception as exc:
                best_reason = f"{type(exc).__name__}: {exc}"
                continue
            candidate_chunk = self._best_retry_chunk(candidate_region, chunks)
            if candidate_chunk is None:
                best_reason = "candidate produced no valid table"
                continue
            score = score_bbox_candidate(
                original_metadata=original.metadata,
                candidate_metadata=candidate_chunk.metadata,
                original_bbox=region.bbox,
                candidate_bbox=candidate.bbox,
            )
            candidate_meta = {
                **candidate.to_dict(),
                "score": score["score"],
                "score_delta": score.get("score_delta", 0.0),
                "tier": score.get("candidate_tier", ""),
                "adoptable": score["adoptable"],
                "reason": score["reason"],
            }
            if float(score["score"]) > best_score:
                best_score = float(score["score"])
                best_reason = str(score["reason"])
                best_candidate = candidate_meta
                best_chunk = candidate_chunk if score["adoptable"] else None

        original.metadata["bbox_candidate_best"] = best_candidate or {}
        original.metadata["bbox_candidate_reason"] = best_reason
        if best_chunk is None:
            original.metadata["bbox_candidate_adopted"] = False
            original.metadata["auto_action_succeeded"] = False
            return None

        best_chunk.metadata.update({
            key: value
            for key, value in original.metadata.items()
            if key.startswith("rule_")
            or key.startswith("llm_")
            or key in {
                "table_region",
                "table_label",
                "table_caption",
                "caption",
                "region_extraction_attempts",
                "auto_action",
                "would_retry",
                "would_adjust_bbox",
                "would_drop_candidate",
                "safe_auto_action",
                "human_review_required",
                "auto_action_allowed",
                "auto_decision_reason",
                "bbox_candidates_tried",
            }
        })
        best_chunk.metadata["bbox_candidate_best"] = best_candidate or {}
        best_chunk.metadata["bbox_candidate_adopted"] = True
        best_chunk.metadata["bbox_candidate_reason"] = best_reason
        best_chunk.metadata["table_region_match"] = "bbox_candidate_adopted"
        adopted_score = score_table_candidate(
            metadata=best_chunk.metadata,
            bbox=best_chunk.bbox,
            region_bbox=region.bbox,
            parser_succeeded=True,
        )
        best_chunk.metadata["table_candidate_score"] = adopted_score.score
        best_chunk.metadata["table_candidate_tier"] = adopted_score.tier
        best_chunk.metadata["table_candidate_reasons"] = adopted_score.reasons
        best_chunk.metadata["table_candidate_metrics"] = adopted_score.metrics
        best_chunk.metadata["llm_assistive_attempted"] = True
        best_chunk.metadata["llm_assistive_succeeded"] = True
        best_chunk.metadata["llm_assistive_original_chunk_id"] = original.chunk_id
        best_chunk.metadata["llm_assistive_action"] = action
        best_chunk.metadata["auto_action_attempted"] = True
        best_chunk.metadata["auto_action_succeeded"] = True
        return best_chunk

    @staticmethod
    def _best_retry_chunk(region: TableRegion, chunks: list[LayoutChunk]) -> LayoutChunk | None:
        best: LayoutChunk | None = None
        best_score = 0.0
        for chunk in chunks:
            if chunk.content_type != "table":
                continue
            rows = int(chunk.metadata.get("rows") or 0)
            cols = int(chunk.metadata.get("cols") or 0)
            if rows < 2 or cols < 2:
                continue
            score = max(
                bbox_iou(chunk.bbox, region.bbox),
                bbox_overlap_ratio(chunk.bbox, region.bbox),
                bbox_overlap_ratio(region.bbox, chunk.bbox),
            )
            if score >= best_score:
                best = chunk
                best_score = score
        return best

    def _page_size(self, page_number: int) -> tuple[float, float]:
        page_bbox = self._page_bbox(page_number)
        return page_bbox[2], page_bbox[3]

    def _page_bbox(self, page_number: int) -> tuple[float, float, float, float]:
        try:
            import pdfplumber
            with pdfplumber.open(self.pdf_path) as doc:
                if 1 <= page_number <= len(doc.pages):
                    page = doc.pages[page_number - 1]
                    return tuple(float(value) for value in getattr(
                        page,
                        "bbox",
                        (0.0, 0.0, float(page.width or 0), float(page.height or 0)),
                    ))
        except Exception:
            pass
        return 0.0, 0.0, 0.0, 0.0

    @staticmethod
    def _clip_bbox_candidates(
        candidates: list[BBoxCandidate],
        page_bbox: tuple[float, float, float, float],
    ) -> list[BBoxCandidate]:
        px0, py0, px1, py1 = page_bbox
        clipped: list[BBoxCandidate] = []
        seen: set[tuple[float, float, float, float]] = set()
        for candidate in candidates:
            x0, y0, x1, y1 = candidate.bbox
            bbox = (
                max(px0, float(x0)),
                max(py0, float(y0)),
                min(px1, float(x1)),
                min(py1, float(y1)),
            )
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            key = tuple(round(value, 2) for value in bbox)
            if key in seen:
                continue
            seen.add(key)
            clipped.append(BBoxCandidate(candidate.name, bbox))
        return clipped

    @staticmethod
    def _should_emit_unparsed_placeholder(region: TableRegion) -> bool:
        """Keep preview placeholders useful: show caption-backed or very strong misses only."""
        evidence_sources = {item.source for item in region.evidence}
        if "caption" in evidence_sources:
            return True
        return region.confidence >= 0.85

    @staticmethod
    def _deduplicate_chunks(chunks: list[LayoutChunk]) -> list[LayoutChunk]:
        """去重：同一区域或包含关系保留结构质量更好的表格候选。"""
        if not chunks:
            return chunks

        def compute_iou(b1, b2) -> float:
            """计算两个 bbox 的 IoU (Intersection over Union)"""
            if not b1 or not b2:
                return 0.0
            x0_1, y0_1, x1_1, y1_1 = b1
            x0_2, y0_2, x1_2, y1_2 = b2

            # 计算交集
            x0_inter = max(x0_1, x0_2)
            y0_inter = max(y0_1, y0_2)
            x1_inter = min(x1_1, x1_2)
            y1_inter = min(y1_1, y1_2)

            if x0_inter >= x1_inter or y0_inter >= y1_inter:
                return 0.0

            inter_area = (x1_inter - x0_inter) * (y1_inter - y0_inter)
            area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
            area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
            union_area = area1 + area2 - inter_area

            if union_area <= 0:
                return 0.0
            return inter_area / union_area

        unique: list[LayoutChunk] = []
        for chunk in chunks:
            duplicate_index: int | None = None
            duplicate_reason = ""
            duplicate_score = 0.0
            for index, existing in enumerate(unique):
                if chunk.page != existing.page:
                    continue
                iou = compute_iou(chunk.bbox, existing.bbox)
                contained = max(
                    bbox_overlap_ratio(chunk.bbox, existing.bbox),
                    bbox_overlap_ratio(existing.bbox, chunk.bbox),
                )
                same_region_exact = TableParser._same_region_exact_duplicate(
                    chunk, existing, iou
                )
                same_source_region = TableParser._same_source_region(chunk, existing)
                if same_source_region and not same_region_exact:
                    priority_delta = abs(
                        TableParser._parser_source_priority(str(chunk.metadata.get("parser", "")))
                        - TableParser._parser_source_priority(str(existing.metadata.get("parser", "")))
                    )
                    if (
                        iou <= CFG.DEDUP_IOU_THRESHOLD
                        and contained < CFG.SAME_REGION_DEDUP_CONTAINED_RATIO
                    ) or priority_delta < CFG.SAME_REGION_DEDUP_MIN_PRIORITY_DELTA:
                        continue
                subset = TableParser._is_contained_subset(chunk.bbox, existing.bbox)
                overflow_subset = TableParser._is_overflowing_subset(chunk.bbox, existing.bbox)
                if (
                    same_region_exact
                    or iou > CFG.DEDUP_IOU_THRESHOLD
                    or contained >= CFG.DEDUP_CONTAINED_RATIO
                    or subset
                    or overflow_subset
                ):
                    duplicate_index = index
                    duplicate_score = max(iou, contained)
                    duplicate_reason = (
                        "same_region_exact" if same_region_exact
                        else "iou" if iou > CFG.DEDUP_IOU_THRESHOLD
                        else "contained_subset" if subset
                        else "overlap_subset" if overflow_subset
                        else "contained"
                    )
                    break

            if duplicate_index is None:
                unique.append(chunk)
                continue

            existing = unique[duplicate_index]
            if TableParser._chunk_quality_score(chunk) > TableParser._chunk_quality_score(existing):
                TableParser._record_dedup_replacement(
                    winner=chunk,
                    loser=existing,
                    reason=duplicate_reason,
                    score=duplicate_score,
                )
                unique[duplicate_index] = chunk
            else:
                TableParser._record_dedup_replacement(
                    winner=existing,
                    loser=chunk,
                    reason=duplicate_reason,
                    score=duplicate_score,
                )
        return unique

    @staticmethod
    def _expand_vision_clips_by_region(chunks: list[LayoutChunk]) -> list[LayoutChunk]:
        by_region: dict[tuple[int, str], list[LayoutChunk]] = {}
        for chunk in chunks:
            region_id = chunk.metadata.get("source_region_id") or (
                chunk.metadata.get("table_region") or {}
            ).get("region_id")
            if region_id:
                by_region.setdefault((chunk.page, region_id), []).append(chunk)

        for group in by_region.values():
            if len(group) < 2:
                continue
            union_bbox = TableParser._bbox_union(*(chunk.bbox for chunk in group))
            for chunk in group:
                if chunk.metadata.get("vision_fallback_needed"):
                    TableParser._expand_vision_clip_bbox(chunk, union_bbox)
                    chunk.metadata["region_fragment_count_for_vision_clip"] = len(group)
        return chunks

    @staticmethod
    def _is_contained_subset(a: tuple, b: tuple) -> bool:
        a_area = TableParser._bbox_area(a)
        b_area = TableParser._bbox_area(b)
        if a_area <= 0 or b_area <= 0:
            return False

        small, large = (a, b) if a_area <= b_area else (b, a)
        small_area = min(a_area, b_area)
        large_area = max(a_area, b_area)
        area_ratio = small_area / large_area
        if area_ratio > CFG.DEDUP_SUBSET_MAX_AREA_RATIO:
            return False

        return bbox_overlap_ratio(small, large) >= CFG.DEDUP_SUBSET_CONTAINED_RATIO

    @staticmethod
    def _same_source_region(a: LayoutChunk, b: LayoutChunk) -> bool:
        a_region = a.metadata.get("source_region_id") or (
            a.metadata.get("table_region") or {}
        ).get("region_id")
        b_region = b.metadata.get("source_region_id") or (
            b.metadata.get("table_region") or {}
        ).get("region_id")
        return bool(a_region and b_region and a_region == b_region and a.page == b.page)

    @staticmethod
    def _same_region_exact_duplicate(
        a: LayoutChunk,
        b: LayoutChunk,
        iou: float,
    ) -> bool:
        if not TableParser._same_source_region(a, b):
            return False
        if iou >= 0.95:
            return True

        diffs = [abs(float(av) - float(bv)) for av, bv in zip(a.bbox, b.bbox)]
        return max(diffs, default=float("inf")) <= CFG.SAME_REGION_EXACT_BBOX_TOLERANCE

    @staticmethod
    def _is_overflowing_subset(a: tuple, b: tuple) -> bool:
        """Detect a table fragment whose bbox spills into nearby prose.

        Some PDF text extractors merge a left-column table row with same-y right-column
        body text. The fragment is not geometrically contained, but its vertical span
        and left edge still place it inside the parent table region.
        """
        a_area = TableParser._bbox_area(a)
        b_area = TableParser._bbox_area(b)
        if a_area <= 0 or b_area <= 0:
            return False

        small, large = (a, b) if a_area <= b_area else (b, a)
        sx0, sy0, sx1, sy1 = small
        lx0, ly0, lx1, ly1 = large
        small_h = max(0.0, sy1 - sy0)
        vertical_overlap = max(0.0, min(sy1, ly1) - max(sy0, ly0))
        if small_h <= 0 or vertical_overlap / small_h < 0.85:
            return False

        left_inside = lx0 - 12 <= sx0 <= lx1 + 12
        starts_inside_vertically = ly0 - 12 <= sy0 <= ly1 + 12
        horizontal_overlap = max(0.0, min(sx1, lx1) - max(sx0, lx0))
        small_w = max(0.0, sx1 - sx0)
        return (
            left_inside
            and starts_inside_vertically
            and small_w > 0
            and horizontal_overlap / small_w >= 0.35
        )

    @staticmethod
    def _bbox_area(bbox: tuple) -> float:
        x0, y0, x1, y1 = bbox
        return max(0.0, float(x1) - float(x0)) * max(0.0, float(y1) - float(y0))

    @staticmethod
    def _record_dedup_replacement(
        *,
        winner: LayoutChunk,
        loser: LayoutChunk,
        reason: str,
        score: float,
    ) -> None:
        if TableParser._should_expand_vision_clip_from_dedup(winner, loser):
            TableParser._expand_vision_clip_bbox(winner, loser.bbox)

        winner.metadata.setdefault("dedup_replaced", []).append({
            "chunk_id": loser.chunk_id,
            "parser": loser.metadata.get("parser", ""),
            "bbox": [float(value) for value in loser.bbox],
            "reason": reason,
            "score": round(float(score), 3),
        })

    @staticmethod
    def _should_expand_vision_clip_from_dedup(winner: LayoutChunk, loser: LayoutChunk) -> bool:
        if not winner.metadata.get("vision_fallback_needed"):
            return False
        if winner.page != loser.page:
            return False

        winner_region = winner.metadata.get("source_region_id") or (
            winner.metadata.get("table_region") or {}
        ).get("region_id")
        loser_region = loser.metadata.get("source_region_id") or (
            loser.metadata.get("table_region") or {}
        ).get("region_id")
        if winner_region and loser_region and winner_region == loser_region:
            return True

        return TableParser._is_vertically_related_fragment(winner.bbox, loser.bbox)

    @staticmethod
    def _expand_vision_clip_bbox(chunk: LayoutChunk, extra_bbox: tuple) -> None:
        current_bbox = tuple(chunk.metadata.get("vision_clip_bbox") or chunk.bbox)
        expanded = TableParser._bbox_union(current_bbox, chunk.bbox, extra_bbox)
        chunk.metadata["vision_clip_bbox"] = [float(value) for value in expanded]
        chunk.metadata["refined_bbox"] = [float(value) for value in expanded]
        chunk.metadata["dedup_expanded_vision_clip"] = True

    @staticmethod
    def _bbox_union(*bboxes: tuple) -> tuple[float, float, float, float]:
        valid = [bbox for bbox in bboxes if bbox and len(bbox) == 4]
        return (
            min(float(bbox[0]) for bbox in valid),
            min(float(bbox[1]) for bbox in valid),
            max(float(bbox[2]) for bbox in valid),
            max(float(bbox[3]) for bbox in valid),
        )

    @staticmethod
    def _is_vertically_related_fragment(a: tuple, b: tuple) -> bool:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        x_overlap = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        min_width = min(max(0.0, ax1 - ax0), max(0.0, bx1 - bx0))
        if min_width <= 0 or x_overlap / min_width < 0.60:
            return False
        vertical_gap = max(by0 - ay1, ay0 - by1, 0.0)
        return vertical_gap <= 80.0

    @staticmethod
    def _chunk_quality_score(chunk: LayoutChunk) -> float:
        metadata = chunk.metadata or {}
        parser_name = str(metadata.get("parser", ""))
        source_priority = TableParser._parser_source_priority(parser_name)
        quality = metadata.get("structural_quality") or {}
        consistency = float(quality.get("consistency_score", 0.0) or 0.0)
        fill_rate = float(quality.get("fill_rate", 0.0) or 0.0)
        rows = int(metadata.get("rows", 0) or 0)
        cols = int(metadata.get("cols", 0) or 0)

        score = source_priority
        score += consistency * CFG.CHUNK_SCORE_CONSISTENCY_WEIGHT
        score += fill_rate * CFG.CHUNK_SCORE_FILL_WEIGHT
        score += min(rows, CFG.CHUNK_SCORE_MAX_ROWS) * CFG.CHUNK_SCORE_ROW_WEIGHT
        score += min(cols, CFG.CHUNK_SCORE_MAX_COLS) * CFG.CHUNK_SCORE_COL_WEIGHT
        if metadata.get("vision_fallback_needed"):
            score -= CFG.CHUNK_SCORE_VISION_PENALTY
        if metadata.get("table_region_unparsed"):
            score -= CFG.CHUNK_SCORE_UNPARSED_PENALTY
        return score

    @staticmethod
    def _parser_source_priority(parser_name: str) -> float:
        if parser_name.startswith("region_guided_geometry"):
            return 40.0
        if parser_name.startswith("pdfplumber_region"):
            return 38.0
        if parser_name.startswith("region_guided_text"):
            return 36.0
        if parser_name.startswith("keyword_guided"):
            return 34.0
        if parser_name.startswith("pdfplumber"):
            return 30.0
        if parser_name == "table_region_detector":
            return 0.0
        return 10.0

    def _post_process_table(
        self, raw_rows: list[list[str | None]], bbox: tuple, page_index: int,
        page_width: float, page_height: float, *,
        parser_name: str = "unknown", pdf_page=None, table_index: int = 0,
        vision_clip_bbox: tuple[float, float, float, float] | None = None,
    ) -> LayoutChunk | None:
        """统一后处理（原始行入口）：标准化 → 截断 → 验证 → 质量 → Vision → 打包。"""
        return self._post_processor.post_process(
            raw_rows, bbox, page_index, page_width, page_height,
            parser_name=parser_name, pdf_page=pdf_page, table_index=table_index,
            vision_clip_bbox=vision_clip_bbox,
        )

    def _validate_and_fallback(
        self, normalized: list[list[str]], bbox: tuple, page_index: int,
        page_width: float, page_height: float, *,
        parser_name: str = "unknown", pdf_page=None, table_index: int = 0,
        vision_clip_bbox: tuple[float, float, float, float] | None = None,
    ) -> LayoutChunk | None:
        """统一后处理（已规范化行入口）：验证 → 质量评估 → Vision fallback → 打包。"""
        return self._post_processor.validate_and_fallback(
            normalized, bbox, page_index, page_width, page_height,
            parser_name=parser_name, pdf_page=pdf_page, table_index=table_index,
            vision_clip_bbox=vision_clip_bbox,
        )

