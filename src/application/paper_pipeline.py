from __future__ import annotations

import logging
import threading
import time
import types
from pathlib import Path
from typing import Any

from src.domain.pipeline_contracts import BatchPaperResult, ParsedPaperResult
from src.ports.repositories import ParsedChunkStore
from src.ports.quality import QualityEnrichmentStore, QualityEvaluator

logger = logging.getLogger(__name__)

_STORE_LOCKS: dict[str, threading.Lock] = {}
_STORE_LOCKS_GUARD = threading.Lock()


def paper_store_lock(paper_id: str) -> threading.Lock:
    with _STORE_LOCKS_GUARD:
        lock = _STORE_LOCKS.get(paper_id)
        if lock is None:
            lock = threading.Lock()
            _STORE_LOCKS[paper_id] = lock
        return lock


def embedding_content_for_chunk(chunk: dict) -> str:
    raw_content = str(chunk.get("raw_content", "") or "")
    metadata = chunk.get("metadata", {}) or {}
    if metadata.get("parser") == "mineru_vlm" and chunk.get("content_type") == "image":
        parts = [
            str(metadata.get("vision_group_description") or "").strip(),
            str(metadata.get("figure_caption") or "").strip(),
            str(metadata.get("mineru_visual_content") or "").strip(),
            raw_content,
        ]
        return "\n\n".join(part for part in parts if part)
    if chunk.get("content_type") != "table":
        return raw_content
    try:
        from src.domain.table_embedding_text import build_table_embedding_text

        return build_table_embedding_text(
            raw_content=raw_content,
            metadata=metadata,
        ) or raw_content
    except Exception:
        return raw_content


def run_mineru_preview(
    pdf_bytes: bytes,
    paper_id: str,
    filename: str = "",
    *,
    model_version: str | None = None,
    describe_figure_groups: bool = False,
    vision_model: str | None = None,
    force_refresh: bool = False,
    poll_interval_seconds: float = 10.0,
    poll_timeout_seconds: float = 600.0,
    cache_root: Path,
    figure_vision_max_workers: int,
) -> ParsedPaperResult:
    from src.adapters.parsers.mineru import run_preview

    return run_preview(
        pdf_bytes,
        paper_id,
        filename,
        model_version=model_version,
        describe_figure_groups=describe_figure_groups,
        vision_model=vision_model,
        force_refresh=force_refresh,
        poll_interval_seconds=poll_interval_seconds,
        poll_timeout_seconds=poll_timeout_seconds,
        cache_root=cache_root,
        figure_vision_max_workers=figure_vision_max_workers,
    )


def run_mineru_ingest(
    pdf_bytes: bytes,
    paper_id: str,
    filename: str = "",
    *,
    model_version: str | None = None,
    describe_figure_groups: bool = False,
    vision_model: str | None = None,
    force_refresh: bool = False,
    poll_interval_seconds: float = 10.0,
    poll_timeout_seconds: float = 600.0,
    preview_runner,
    store_runner,
) -> ParsedPaperResult:
    started = time.time()
    result = preview_runner(
        pdf_bytes,
        paper_id,
        filename,
        model_version=model_version,
        describe_figure_groups=describe_figure_groups,
        vision_model=vision_model,
        force_refresh=force_refresh,
        poll_interval_seconds=poll_interval_seconds,
        poll_timeout_seconds=poll_timeout_seconds,
    )
    store_start = time.time()
    store_result = store_runner(result)
    result["mode"] = "mineru_auto_store"
    result["status"] = "mineru_stored" if not store_result.get("error") else "mineru_store_failed"
    result["store_result"] = store_result
    result.setdefault("mineru", {})["store_result"] = store_result
    result.setdefault("worker_timing", {})["store"] = round(time.time() - store_start, 3)
    result["worker_timing"]["total"] = round(time.time() - started, 3)
    return result


def store_parsed_chunks(
    result: dict,
    *,
    embedding_api_key_env: str,
    parsed_chunk_store: ParsedChunkStore | None = None,
) -> dict:
    if parsed_chunk_store is None:
        from src.adapters.store.parsed_chunks import LegacyParsedChunkStore

        parsed_chunk_store = LegacyParsedChunkStore()

    return parsed_chunk_store.store(
        result,
        embedding_api_key_env=embedding_api_key_env,
        embed_content_for_chunk=embedding_content_for_chunk,
    )


def evaluate_parsed_chunks(result: dict, *, enrich_runner) -> dict:
    paper_id = result.get("paper_id", "")
    if not paper_id:
        return {"skip": True, "error": "missing paper_id"}
    return enrich_runner(paper_id)


def point_to_quality_chunk(point) -> types.SimpleNamespace:
    payload = point.payload or {}
    raw_content = str(payload.get("raw_content") or payload.get("content") or "")
    chunk_id = str(payload.get("chunk_id") or point.id)
    return types.SimpleNamespace(
        chunk_id=chunk_id,
        raw_content=raw_content,
        content_type=payload.get("content_type") or payload.get("worker_type") or "text",
        page=payload.get("page"),
        bbox=payload.get("bbox") or [0, 0, 0, 0],
        column=payload.get("column"),
        order_in_page=payload.get("order_in_page"),
        global_order=payload.get("global_order") or 0,
        metadata=payload,
    )


def paper_payload_from_quality(quality: dict, *, title: str, paper_id: str) -> dict:
    return {
        "category": quality.get("category"),
        "credibility": quality.get("credibility"),
        "fermentation_relevance": quality.get("fermentation_relevance"),
        "is_actionable": quality.get("is_actionable"),
        "paper_title": quality.get("paper_title") or title or paper_id,
        "paper_summary": quality.get("paper_summary"),
        "target_products": quality.get("target_products", []),
        "organisms": quality.get("organisms", []),
        "classify_reason": quality.get("classify_reason", ""),
        "classify_status": quality.get("classify_status", ""),
        "enrichment_status": "done",
    }


def chunk_payload_update_for_quality(
    payload: dict,
    *,
    paper_payload: dict,
    context: str,
    search_text_builder,
) -> dict:
    raw_content = str(payload.get("raw_content") or payload.get("content") or "")
    updated_payload = {**payload, **paper_payload}
    point_payload: dict[str, object] = {}
    if context:
        updated_payload.update({
            "context": context,
            "content": f"{context}\n\n{raw_content}",
            "raw_content": raw_content,
        })
        point_payload.update({
            "context": context,
            "content": updated_payload["content"],
            "raw_content": raw_content,
        })
    point_payload["search_text"] = search_text_builder(updated_payload)
    return point_payload


def evaluate_and_enrich_paper(
    paper_id: str,
    *,
    store: QualityEnrichmentStore,
    evaluator: QualityEvaluator,
    search_text_builder,
) -> dict:
    points = store.scroll_by_paper_id(paper_id)

    if not points:
        return {"skip": True, "error": f"paper_id {paper_id} not found in Qdrant"}

    chunks = []
    title = ""
    for point in points:
        payload = point.payload or {}
        title = title or str(payload.get("paper_title") or "")
        chunks.append(point_to_quality_chunk(point))

    chunks.sort(key=lambda c: (c.global_order if c.global_order is not None else 0))
    quality = evaluator.evaluate(chunks, title=title)
    if quality.get("error"):
        return quality

    paper_payload = paper_payload_from_quality(quality, title=title, paper_id=paper_id)
    point_ids = [point.id for point in points]
    quality["updated_points"] = store.update_paper_payload(
        paper_id,
        paper_payload,
        point_ids=point_ids,
    )

    chunk_contexts = quality.get("chunk_contexts", {}) or {}
    for point in points:
        payload = point.payload or {}
        chunk_id = str(payload.get("chunk_id") or point.id)
        context = chunk_contexts.get(chunk_id, "")
        point_payload = chunk_payload_update_for_quality(
            payload,
            paper_payload=paper_payload,
            context=context,
            search_text_builder=search_text_builder,
        )
        store.set_chunk_payload(point.id, point_payload)
    store.invalidate_lexical_cache()

    return quality


def evaluate_and_enrich_from_qdrant(paper_id: str) -> dict:
    from src.adapters.quality import default_quality_enrichment_dependencies

    dependencies = default_quality_enrichment_dependencies()

    return evaluate_and_enrich_paper(
        paper_id,
        store=dependencies.store,
        evaluator=dependencies.evaluator,
        search_text_builder=dependencies.search_text_builder,
    )


def process_one_pdf(
    file_bytes: bytes,
    paper_id: str,
    filename: str,
    *,
    pipeline_runner,
    store_runner,
) -> BatchPaperResult:
    t0 = time.time()
    stage_start = t0
    logger.info("[batch] %s | pipeline start", filename)
    pipe_result = pipeline_runner(file_bytes, paper_id, filename)
    parse_seconds = time.time() - stage_start
    logger.info("[batch] %s | pipeline done (%.0fs) | store start", filename, parse_seconds)

    stage_start = time.time()
    with paper_store_lock(paper_id):
        store_result = store_runner(pipe_result)
    store_seconds = time.time() - stage_start
    logger.info("[batch] %s | store done (%.0fs)", filename, time.time() - t0)

    stored = int(store_result.get("stored", 0) or 0)
    degraded = int(store_result.get("degraded", 0) or 0)
    store_failed = int(store_result.get("failed", 0) or 0)
    excluded = int(store_result.get("excluded", 0) or 0)
    attempted = int(store_result.get("attempted", 0) or 0)
    duplicate = bool(store_result.get("duplicate", False))
    if duplicate:
        status = "skip"
        reason = f"already stored {store_result.get('existing_count', 0)} records"
    elif store_failed:
        status = "partial" if stored or degraded else "fail"
        reason = str(store_result.get("error") or f"store failed {store_failed}/{attempted}")
    elif attempted == 0 and not excluded:
        status = "no_store"
        reason = f"no storable chunks; missing verdicts {store_result.get('missing_verdicts', 0)}"
    else:
        status = "ok"
        reason = ""

    return {
        "status": status,
        "reason": reason,
        "category": "pending",
        "credibility": 0,
        "chunks": len(pipe_result.get("merged_chunks", [])),
        "stored": stored,
        "degraded": degraded,
        "store_failed": store_failed,
        "excluded": excluded,
        "store_attempted": attempted,
        "parse_seconds": round(parse_seconds, 1),
        "eval_seconds": 0,
        "store_seconds": round(store_seconds, 1),
        "total_seconds": round(time.time() - t0, 1),
    }
