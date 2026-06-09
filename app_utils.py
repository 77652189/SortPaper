from __future__ import annotations

from app_config import RESULTS_DIR
from src.application.pipeline_utils import (
    build_paper_id as _build_paper_id_impl,
    chunk_to_dict,
    generate_chunk_description,
)


def build_paper_id(pdf_bytes: bytes) -> str:
    return _build_paper_id_impl(pdf_bytes)


def _chunk_to_dict(chunk) -> dict:
    return chunk_to_dict(chunk)


def _generate_chunk_description(content_type: str, raw_content: str) -> str:
    return generate_chunk_description(content_type, raw_content)


def build_snapshot(result: dict, filename: str) -> dict:
    from src.application.saved_results import build_snapshot as build_saved_snapshot

    return build_saved_snapshot(result, filename)


def save_result(snapshot: dict, filename: str) -> str | None:
    from src.application.saved_results import save_result as save_saved_result

    return save_saved_result(snapshot, filename, results_dir=RESULTS_DIR)


def overwrite_result(snapshot: dict, filename: str) -> str:
    from src.application.saved_results import overwrite_result as overwrite_saved_result

    return overwrite_saved_result(snapshot, filename, results_dir=RESULTS_DIR)


def qdrant_search(
    query: str,
    paper_id: str = None,
    top_k: int = 10,
    retrieval_profile: str | None = None,
    rerank: bool = False,
    selective_rerank: bool = False,
    rerank_top_n: int | None = None,
    filter_kwargs: dict | None = None,
    lexical_backfill: bool = True,
    neighbor_backfill: bool = False,
    query_rewrite: bool = False,
    multi_query: bool = False,
) -> list[dict]:
    from src.application.search import search_chunks

    return search_chunks(
        query,
        paper_id=paper_id,
        top_k=top_k,
        retrieval_profile=retrieval_profile,
        rerank=rerank,
        selective_rerank=selective_rerank,
        rerank_top_n=rerank_top_n,
        filter_kwargs=filter_kwargs,
        lexical_backfill=lexical_backfill,
        neighbor_backfill=neighbor_backfill,
        query_rewrite=query_rewrite,
        multi_query=multi_query,
    )
