"""Compatibility facade for PaperSort parsing, storage, and enrichment use cases."""
from __future__ import annotations

from src.runtime_env import load_project_env

load_project_env()

from app_config import EMBEDDING_API_KEY_ENV as _CONFIG_EMBEDDING_API_KEY_ENV
from app_config import MINERU_CACHE_DIR, MINERU_FIGURE_VISION_MAX_WORKERS
from src.adapters.parsers.legacy import run_pipeline as _run_legacy_pipeline
from src.adapters.parsers.legacy import run_preview as _run_legacy_preview
from src.adapters.parsers.mineru import default_mineru_verdicts as _default_mineru_verdicts
from src.adapters.parsers.mineru import extract_mineru_zip_url as _extract_mineru_zip_url
from src.application.paper_pipeline import (
    embedding_content_for_chunk as _embedding_content_for_chunk,
    evaluate_and_enrich_from_qdrant as _evaluate_and_enrich_from_qdrant,
    evaluate_parsed_chunks as _evaluate_parsed_chunks,
    paper_store_lock as _paper_store_lock,
    process_one_pdf as _process_one_pdf_use_case,
    run_mineru_ingest as _run_mineru_ingest_use_case,
    run_mineru_preview as _run_mineru_preview_use_case,
    store_parsed_chunks as _store_parsed_chunks_use_case,
)

EMBEDDING_API_KEY_ENV = _CONFIG_EMBEDDING_API_KEY_ENV or "DASHSCOPE_API_KEY"


def run_preview(pdf_bytes: bytes) -> dict:
    """Compatibility wrapper for the legacy quick-preview parser."""
    return _run_legacy_preview(pdf_bytes)


def run_pipeline(
    pdf_bytes: bytes,
    paper_id: str,
    filename: str = "",
) -> dict:
    """Compatibility wrapper for the legacy LangGraph parser pipeline."""
    return _run_legacy_pipeline(pdf_bytes, paper_id, filename)


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
) -> dict:
    """Run MinerU extraction and convert the cached result zip to LayoutChunk objects."""
    return _run_mineru_preview_use_case(
        pdf_bytes,
        paper_id,
        filename,
        model_version=model_version,
        describe_figure_groups=describe_figure_groups,
        vision_model=vision_model,
        force_refresh=force_refresh,
        poll_interval_seconds=poll_interval_seconds,
        poll_timeout_seconds=poll_timeout_seconds,
        cache_root=MINERU_CACHE_DIR,
        figure_vision_max_workers=MINERU_FIGURE_VISION_MAX_WORKERS,
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
) -> dict:
    """Run MinerU extraction and immediately store converted chunks in Qdrant."""
    return _run_mineru_ingest_use_case(
        pdf_bytes,
        paper_id,
        filename,
        model_version=model_version,
        describe_figure_groups=describe_figure_groups,
        vision_model=vision_model,
        force_refresh=force_refresh,
        poll_interval_seconds=poll_interval_seconds,
        poll_timeout_seconds=poll_timeout_seconds,
        preview_runner=run_mineru_preview,
        store_runner=store_parsed_chunks,
    )


def store_parsed_chunks(result: dict) -> dict:
    """Store an already parsed pipeline result in Qdrant."""
    return _store_parsed_chunks_use_case(
        result,
        embedding_api_key_env=_CONFIG_EMBEDDING_API_KEY_ENV,
    )


def evaluate_parsed_chunks(result: dict) -> dict:
    """Compatibility entrypoint for paper-level quality enrichment."""
    return _evaluate_parsed_chunks(
        result,
        enrich_runner=evaluate_and_enrich_from_qdrant,
    )


def evaluate_and_enrich_from_qdrant(paper_id: str) -> dict:
    """Run Map-Reduce from stored Qdrant chunks and enrich their payloads."""
    return _evaluate_and_enrich_from_qdrant(paper_id)


def _process_one_pdf(file_bytes: bytes, paper_id: str, filename: str) -> dict:
    """Batch helper kept for the Streamlit compatibility boundary."""
    return _process_one_pdf_use_case(
        file_bytes,
        paper_id,
        filename,
        pipeline_runner=run_pipeline,
        store_runner=store_parsed_chunks,
    )
