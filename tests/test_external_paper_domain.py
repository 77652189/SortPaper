from __future__ import annotations

from src.adapters.external_papers.repository import JsonExternalImportRepository
from src.domain.external_papers import MonitorProfile, make_candidate, normalize_doi


def test_normalize_doi_accepts_url_and_case() -> None:
    assert normalize_doi("https://doi.org/10.1101/ABC.123.") == "10.1101/abc.123"
    assert normalize_doi("doi: 10.1038/NATURE") == "10.1038/nature"


def test_repository_upsert_merges_by_doi_and_preserves_sources(tmp_path) -> None:
    repo = JsonExternalImportRepository(tmp_path)
    pubmed = make_candidate(
        source="pubmed",
        title="Metabolic engineering for LNT II",
        doi="10.1000/ABC",
        pmid="123",
        published_date="2026-01-02",
    )
    crossref = make_candidate(
        source="crossref",
        title="Metabolic engineering for LNT II",
        doi="https://doi.org/10.1000/abc",
        abstract="Long abstract",
        pdf_url="https://example.test/paper.pdf",
        published_date="2026-01-02",
    )

    first = repo.upsert_candidates([pubmed])
    second = repo.upsert_candidates([crossref])
    candidates = repo.list_candidates()

    assert first["new_count"] == 1
    assert second["updated_count"] == 1
    assert len(candidates) == 1
    assert candidates[0].sources == ["pubmed", "crossref"]
    assert candidates[0].pmid == "123"
    assert candidates[0].pdf_url == "https://example.test/paper.pdf"


def test_repository_upsert_merges_title_year_fallback(tmp_path) -> None:
    repo = JsonExternalImportRepository(tmp_path)
    a = make_candidate(
        source="openalex",
        title="A paper without DOI",
        published_date="2026-02-01",
    )
    b = make_candidate(
        source="crossref",
        title="A Paper Without DOI!",
        published_date="2026",
        doi="10.1000/fill-in",
    )

    repo.upsert_candidates([a, b])
    candidates = repo.list_candidates()

    assert len(candidates) == 1
    assert candidates[0].doi == "10.1000/fill-in"


def test_monitor_profile_preserves_filter_config_and_allows_legacy_payload() -> None:
    legacy = MonitorProfile.from_dict({
        "monitor_id": "mon-legacy",
        "name": "Legacy",
        "query": "lactoferrin",
    })
    configured = MonitorProfile.from_dict({
        "monitor_id": "mon-filtered",
        "name": "Y103",
        "query": "Pichia lactoferrin",
        "filter_config": {"enabled": True, "filter_type": "y103", "include_other": False},
    })

    assert legacy.filter_config == {}
    assert configured.to_dict()["filter_config"] == {
        "enabled": True,
        "filter_type": "y103",
        "include_other": False,
    }
