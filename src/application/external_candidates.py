from __future__ import annotations

from typing import Any, Callable

from src.domain.external_papers import ExternalPaperCandidate
from src.ports.external_papers import ExternalCandidateRepository


RepositoryFactory = Callable[[], ExternalCandidateRepository]


class ExternalCandidateQueryService:
    """Application service for candidate-library filtering and sorting."""

    def __init__(
        self,
        *,
        repository_factory: RepositoryFactory,
        source_priority: dict[str, int],
    ) -> None:
        self.repository_factory = repository_factory
        self.source_priority = source_priority

    def list_candidates(
        self,
        *,
        repository: ExternalCandidateRepository | None = None,
        source_names: list[str] | None = None,
        status: str = "",
        only_with_pdf: bool = False,
        query: str = "",
    ) -> list[ExternalPaperCandidate]:
        candidates = (repository or self.repository_factory()).list_candidates()
        source_set = set(source_names or [])
        text_query = query.strip().lower()
        result: list[ExternalPaperCandidate] = []
        for candidate in candidates:
            if source_set and not source_set.intersection(candidate.sources):
                continue
            if status:
                if candidate.import_status != status:
                    continue
            elif candidate.import_status == "skipped":
                continue
            if only_with_pdf and not candidate.pdf_url:
                continue
            if text_query:
                haystack = " ".join([
                    candidate.title,
                    candidate.abstract,
                    candidate.doi,
                    " ".join(candidate.authors),
                ]).lower()
                if text_query not in haystack:
                    continue
            result.append(candidate)
        return sorted(result, key=self._candidate_sort_key)

    def _candidate_sort_key(self, item: ExternalPaperCandidate) -> tuple[Any, ...]:
        return (
            item.import_status != "candidate",
            -(item.year or 0),
            self.source_priority.get(item.sources[0], 999) if item.sources else 999,
            item.title,
        )
