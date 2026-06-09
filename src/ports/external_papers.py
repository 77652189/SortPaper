from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.domain.external_papers import ExternalPaperCandidate, ExternalSearchQuery, MonitorProfile


class ExternalPaperSource(Protocol):
    name: str

    def search(self, query: ExternalSearchQuery) -> list[ExternalPaperCandidate]:
        ...


class ExternalCandidateRepository(Protocol):
    def list_candidates(self) -> list[ExternalPaperCandidate]:
        ...

    def get_candidate(self, candidate_id: str) -> ExternalPaperCandidate | None:
        ...

    def upsert_candidates(self, candidates: list[ExternalPaperCandidate]) -> dict:
        ...

    def update_candidate(self, candidate: ExternalPaperCandidate) -> None:
        ...


class ExternalPdfDownloader(Protocol):
    def download(
        self,
        candidate: ExternalPaperCandidate,
        *,
        root_dir: str | Path,
    ) -> tuple[bytes, Path]:
        ...


class MonitorRepository(Protocol):
    def list_monitors(self) -> list[MonitorProfile]:
        ...

    def save_monitors(self, monitors: list[MonitorProfile]) -> None:
        ...
