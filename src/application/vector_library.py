from __future__ import annotations

import os
from typing import Callable

from src.application.settings import EMBEDDING_API_KEY_ENV, EMBEDDING_PROVIDER
from src.ports.repositories import VectorLibraryRepository


ProgressCallback = Callable[[int, int, dict, int, int], None]
QualityEnrichRunner = Callable[[str], dict]


def default_repository() -> VectorLibraryRepository:
    from src.adapters.store.defaults import default_vector_library_repository

    return default_vector_library_repository()


def paper_count(paper_id: str, *, repository: VectorLibraryRepository | None = None) -> int:
    return (repository or default_repository()).paper_count(paper_id)


def total_count(*, repository: VectorLibraryRepository | None = None) -> int:
    return (repository or default_repository()).count


def list_papers(*, repository: VectorLibraryRepository | None = None) -> list[dict]:
    return (repository or default_repository()).list_papers()


def delete_paper(paper_id: str, *, repository: VectorLibraryRepository | None = None) -> int:
    return (repository or default_repository()).delete_by_paper_id(paper_id)


def clear_library(*, repository: VectorLibraryRepository | None = None) -> int:
    repo = repository or default_repository()
    before = repo.count
    repo.clear()
    return before


def default_quality_enricher(paper_id: str) -> dict:
    from src.application.paper_pipeline import evaluate_and_enrich_from_qdrant

    return evaluate_and_enrich_from_qdrant(paper_id)


def enrich_paper(
    paper_id: str,
    *,
    enrich_runner: QualityEnrichRunner | None = None,
) -> dict:
    runner = enrich_runner or default_quality_enricher
    return runner(paper_id)


def enrich_pending_papers(
    papers: list[dict],
    *,
    enrich_runner: QualityEnrichRunner | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    pending_papers = [
        paper for paper in papers
        if paper.get("enrichment_status", "done") == "pending"
    ]
    success_n = 0
    fail_n = 0
    failures: list[str] = []
    total_pending = len(pending_papers)
    runner = enrich_runner or default_quality_enricher

    for index, paper in enumerate(pending_papers, start=1):
        paper_id = paper["paper_id"]
        title = paper.get("paper_title", paper_id)
        try:
            result = runner(paper_id)
            if result.get("error"):
                fail_n += 1
                failures.append(f"{title}: {result.get('error')}")
            else:
                success_n += 1
        except Exception as exc:
            fail_n += 1
            failures.append(f"{title}: {exc}")
        if progress_callback is not None:
            progress_callback(index, total_pending, paper, success_n, fail_n)

    return {
        "total": total_pending,
        "success_n": success_n,
        "fail_n": fail_n,
        "failures": failures,
    }


def embedding_preflight_error(*, repository: VectorLibraryRepository | None = None) -> str:
    if EMBEDDING_PROVIDER != "qwen3_local" and not os.getenv(EMBEDDING_API_KEY_ENV):
        return f"缺少 {EMBEDDING_API_KEY_ENV}，无法生成向量入库"
    try:
        (repository or default_repository()).embed("SortPaper embedding preflight")
    except Exception as exc:
        return f"Embedding 预检失败：{type(exc).__name__}: {str(exc)[:300]}"
    return ""
