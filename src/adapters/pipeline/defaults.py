from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app_config import (
    EMBEDDING_API_KEY_ENV,
    MINERU_CACHE_DIR,
    MINERU_FIGURE_VISION_MAX_WORKERS,
)
from src.application.paper_pipeline import (
    run_mineru_ingest as run_mineru_ingest_use_case,
    run_mineru_preview as run_mineru_preview_use_case,
    store_parsed_chunks,
)
from src.application.pipeline_utils import build_paper_id
from src.application.vector_library import paper_count


@dataclass(frozen=True)
class ExternalCandidatePipelineDependencies:
    paper_id_builder: Callable[[bytes], str]
    ingest_runner: Callable[..., dict]
    paper_count_lookup: Callable[[str], int]


def run_mineru_preview_with_defaults(
    pdf_bytes: bytes,
    paper_id: str,
    filename: str = "",
    **kwargs,
) -> dict:
    return run_mineru_preview_use_case(
        pdf_bytes,
        paper_id,
        filename,
        cache_root=MINERU_CACHE_DIR,
        figure_vision_max_workers=MINERU_FIGURE_VISION_MAX_WORKERS,
        **kwargs,
    )


def store_parsed_chunks_with_defaults(result: dict) -> dict:
    return store_parsed_chunks(
        result,
        embedding_api_key_env=EMBEDDING_API_KEY_ENV,
    )


def run_mineru_ingest_with_defaults(
    pdf_bytes: bytes,
    paper_id: str,
    filename: str = "",
    **kwargs,
) -> dict:
    return run_mineru_ingest_use_case(
        pdf_bytes,
        paper_id,
        filename,
        preview_runner=run_mineru_preview_with_defaults,
        store_runner=store_parsed_chunks_with_defaults,
        **kwargs,
    )


def default_external_candidate_pipeline_dependencies() -> ExternalCandidatePipelineDependencies:
    return ExternalCandidatePipelineDependencies(
        paper_id_builder=build_paper_id,
        ingest_runner=run_mineru_ingest_with_defaults,
        paper_count_lookup=paper_count,
    )
