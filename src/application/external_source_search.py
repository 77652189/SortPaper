from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from src.domain.external_papers import ExternalPaperCandidate, ExternalSearchQuery, utc_now_iso
from src.ports.external_papers import ExternalCandidateRepository, ExternalPaperSource


ExternalRefreshProgress = Callable[[dict[str, Any]], None]
RepositoryFactory = Callable[[], ExternalCandidateRepository]
SourceFactory = Callable[[list[str] | None], list[ExternalPaperSource]]


class ExternalSourceSearchService:
    """Application service for querying external paper sources and merging candidates."""

    def __init__(
        self,
        *,
        repository_factory: RepositoryFactory,
        source_factory: SourceFactory,
        default_source_names: list[str],
    ) -> None:
        self.repository_factory = repository_factory
        self.source_factory = source_factory
        self.default_source_names = default_source_names

    def refresh_candidates(
        self,
        query: str,
        *,
        source_names: list[str] | None = None,
        from_date: str = "",
        until_date: str = "",
        max_results_per_source: int = 20,
        repository: ExternalCandidateRepository | None = None,
        sources: list[ExternalPaperSource] | None = None,
        progress_callback: ExternalRefreshProgress | None = None,
    ) -> dict[str, Any]:
        selected_sources = source_names or self.default_source_names
        search_query = ExternalSearchQuery(
            query=query.strip(),
            sources=selected_sources,
            from_date=from_date,
            until_date=until_date,
            max_results_per_source=max_results_per_source,
        )
        if not search_query.query:
            return {"error": "query 不能为空", "errors": [], "candidates": []}

        source_adapters = sources or self.source_factory(selected_sources)
        candidates, errors = self._search_sources(
            search_query,
            source_adapters=source_adapters,
            progress_callback=progress_callback,
        )

        repo = repository or self.repository_factory()
        emit_progress(
            progress_callback,
            event="merge_start",
            fetched_count=len(candidates),
            error_count=len(errors),
        )
        upsert = repo.upsert_candidates(candidates) if candidates else {
            "new_count": 0,
            "updated_count": 0,
            "candidate_ids": [],
            "total_count": len(repo.list_candidates()),
        }
        summary = {
            "query": search_query.query,
            "sources": selected_sources,
            "from_date": from_date,
            "until_date": until_date,
            "fetched_count": len(candidates),
            "errors": errors,
            **upsert,
        }
        if hasattr(repo, "append_run_summary"):
            repo.append_run_summary({"kind": "manual_refresh", "run_at": utc_now_iso(), **summary})
        emit_progress(progress_callback, event="merge_done", **summary)
        return summary

    def _search_sources(
        self,
        search_query: ExternalSearchQuery,
        *,
        source_adapters: list[ExternalPaperSource],
        progress_callback: ExternalRefreshProgress | None,
    ) -> tuple[list[ExternalPaperCandidate], list[dict[str, str]]]:
        candidates: list[ExternalPaperCandidate] = []
        errors: list[dict[str, str]] = []
        if not source_adapters:
            return candidates, errors

        max_workers = min(len(source_adapters), 6)
        for source in source_adapters:
            emit_progress(
                progress_callback,
                event="source_start",
                source=getattr(source, "name", "?"),
                total_sources=len(source_adapters),
            )
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_sources = {
                pool.submit(source.search, search_query): source
                for source in source_adapters
            }
            completed_sources = 0
            for future in as_completed(future_sources):
                source = future_sources[future]
                source_name = getattr(source, "name", "?")
                completed_sources += 1
                try:
                    source_candidates = future.result()
                    candidates.extend(source_candidates)
                    emit_progress(
                        progress_callback,
                        event="source_done",
                        source=source_name,
                        count=len(source_candidates),
                        completed_sources=completed_sources,
                        total_sources=len(source_adapters),
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {str(exc)[:300]}"
                    errors.append({"source": source_name, "error": error})
                    emit_progress(
                        progress_callback,
                        event="source_error",
                        source=source_name,
                        error=error,
                        completed_sources=completed_sources,
                        total_sources=len(source_adapters),
                    )
        return candidates, errors


def emit_progress(
    callback: ExternalRefreshProgress | None,
    **event: Any,
) -> None:
    if callback is None:
        return
    try:
        callback(event)
    except Exception:
        return
