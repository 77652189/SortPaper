from __future__ import annotations

from pathlib import Path

from src.adapters.external_papers.repository import JsonExternalImportRepository
from src.application import external_filters
from src.domain.external_papers import make_candidate


def test_y103_candidate_filter_marks_other_candidates_skipped(tmp_path: Path) -> None:
    repo = JsonExternalImportRepository(tmp_path)
    relevant = make_candidate(
        source="pubmed",
        title="Recombinant lactoferrin expression in Pichia pastoris",
        abstract="Pichia pastoris secretes recombinant lactoferrin.",
        doi="10.1/relevant-filter",
    )
    irrelevant = make_candidate(
        source="pubmed",
        title="Human development disparities",
        abstract="Persistent inequality constrains human development.",
        doi="10.1/irrelevant-filter",
    )
    repo.upsert_candidates([relevant, irrelevant])

    summary = external_filters.apply_candidate_filter(
        repo,
        [relevant.candidate_id, irrelevant.candidate_id],
        {"enabled": True, "filter_type": "y103", "include_other": False},
    )

    stored_relevant = repo.get_candidate(relevant.candidate_id)
    stored_irrelevant = repo.get_candidate(irrelevant.candidate_id)
    assert summary["filter_type"] == "y103"
    assert summary["kept_count"] == 1
    assert summary["skipped_count"] == 1
    assert stored_relevant.classification_id == "03"
    assert stored_relevant.import_status == "candidate"
    assert stored_irrelevant.classification_id == "other"
    assert stored_irrelevant.import_status == "skipped"
    assert stored_irrelevant.import_error.startswith("候选过滤")


def test_disabled_candidate_filter_does_not_touch_candidates(tmp_path: Path) -> None:
    repo = JsonExternalImportRepository(tmp_path)
    candidate = make_candidate(
        source="pubmed",
        title="Human development disparities",
        abstract="Persistent inequality constrains human development.",
        doi="10.1/disabled-filter",
    )
    repo.upsert_candidates([candidate])

    summary = external_filters.apply_candidate_filter(
        repo,
        [candidate.candidate_id],
        {"enabled": False, "filter_type": "none", "include_other": True},
    )

    stored = repo.get_candidate(candidate.candidate_id)
    assert summary["enabled"] is False
    assert summary["checked_count"] == 0
    assert stored.import_status == "candidate"
    assert stored.classification_id == ""
