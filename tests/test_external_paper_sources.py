from __future__ import annotations

from typing import Any

from src.adapters.external_papers.sources import (
    BioRxivSource,
    CrossrefSource,
    EuropePmcSource,
    OpenAlexSource,
    PubMedSource,
)
from src.domain.external_papers import ExternalSearchQuery


class FakeTransport:
    def __init__(self, *, json_responses: list[dict[str, Any]] | None = None, text_responses: list[str] | None = None) -> None:
        self.json_responses = json_responses or []
        self.text_responses = text_responses or []
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def get_json(self, url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append((url, params))
        return self.json_responses.pop(0)

    def get_text(self, url: str, *, params: dict[str, Any] | None = None) -> str:
        self.calls.append((url, params))
        return self.text_responses.pop(0)


def test_pubmed_source_esearches_then_efetches_xml() -> None:
    xml = """
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>123</PMID>
          <Article>
            <ArticleTitle>LNT II biosynthesis</ArticleTitle>
            <Abstract><AbstractText>Abstract text</AbstractText></Abstract>
            <Journal><Title>Journal</Title><JournalIssue><PubDate><Year>2026</Year><Month>Jan</Month><Day>02</Day></PubDate></JournalIssue></Journal>
            <AuthorList><Author><ForeName>Ada</ForeName><LastName>Lovelace</LastName></Author></AuthorList>
          </Article>
        </MedlineCitation>
        <PubmedData><ArticleIdList>
          <ArticleId IdType="doi">10.1000/ABC</ArticleId>
          <ArticleId IdType="pmc">PMC12345</ArticleId>
        </ArticleIdList></PubmedData>
      </PubmedArticle>
    </PubmedArticleSet>
    """
    transport = FakeTransport(
        json_responses=[{"esearchresult": {"idlist": ["123"]}}],
        text_responses=[xml],
    )

    results = PubMedSource(transport=transport, contact_email="me@example.test").search(
        ExternalSearchQuery(query="LNT", max_results_per_source=5)
    )

    assert transport.calls[0][0].endswith("esearch.fcgi")
    assert transport.calls[0][1]["email"] == "me@example.test"
    assert transport.calls[1][0].endswith("efetch.fcgi")
    assert results[0].doi == "10.1000/abc"
    assert results[0].pmcid == "PMC12345"
    assert results[0].pdf_url.endswith("/PMC12345/pdf/")


def test_europe_pmc_source_maps_open_pdf() -> None:
    transport = FakeTransport(json_responses=[{
        "resultList": {
            "result": [{
                "title": "Europe paper",
                "abstractText": "abstract",
                "authorString": "Ada; Bob",
                "journalTitle": "Journal",
                "firstPublicationDate": "2026-01-01",
                "doi": "10.1/eu",
                "pmid": "1",
                "isOpenAccess": "Y",
                "fullTextUrlList": {"fullTextUrl": [{"documentStyle": "pdf", "availability": "Open access", "url": "https://pdf.test/a.pdf"}]},
            }]
        }
    }])

    results = EuropePmcSource(transport=transport).search(ExternalSearchQuery(query="x"))

    assert results[0].sources == ["europe_pmc"]
    assert results[0].pdf_url == "https://pdf.test/a.pdf"
    assert results[0].is_open_access is True


def test_openalex_source_reconstructs_abstract_and_uses_mailto() -> None:
    transport = FakeTransport(json_responses=[{
        "results": [{
            "id": "https://openalex.org/W1",
            "doi": "https://doi.org/10.1/oa",
            "title": "OpenAlex paper",
            "publication_date": "2026-01-01",
            "abstract_inverted_index": {"hello": [0], "world": [1]},
            "open_access": {"is_oa": True, "oa_url": "https://oa.test/paper.pdf"},
            "authorships": [{"author": {"display_name": "Ada"}}],
            "primary_location": {"source": {"display_name": "Journal"}, "pdf_url": "https://pdf.test/open.pdf"},
        }]
    }])

    results = OpenAlexSource(transport=transport, contact_email="me@example.test").search(ExternalSearchQuery(query="x"))

    assert transport.calls[0][1]["mailto"] == "me@example.test"
    assert results[0].abstract == "hello world"
    assert results[0].openalex_id == "W1"


def test_openalex_source_does_not_treat_oa_url_as_pdf_url() -> None:
    transport = FakeTransport(json_responses=[{
        "results": [{
            "id": "https://openalex.org/W2",
            "doi": "https://doi.org/10.1/landing",
            "title": "OpenAlex landing only",
            "publication_date": "2026-01-01",
            "open_access": {"is_oa": True, "oa_url": "https://doi.org/10.1/landing"},
            "primary_location": {"source": {"display_name": "Journal"}},
        }]
    }])

    results = OpenAlexSource(transport=transport).search(ExternalSearchQuery(query="x"))

    assert results[0].is_open_access is True
    assert results[0].landing_url == "https://doi.org/10.1/landing"
    assert results[0].pdf_url == ""


def test_openalex_source_filters_broad_boolean_results_locally() -> None:
    transport = FakeTransport(json_responses=[{
        "results": [
            {
                "id": "https://openalex.org/W1",
                "title": "Human development disparities",
                "publication_date": "2026-01-01",
            },
            {
                "id": "https://openalex.org/W2",
                "title": "Pichia pastoris lactoferrin secretion",
                "publication_date": "2026-01-01",
            },
        ]
    }])

    results = OpenAlexSource(transport=transport).search(
        ExternalSearchQuery(query="(Pichia OR Komagataella) AND (lactoferrin OR osteopontin)")
    )

    assert [item.openalex_id for item in results] == ["W2"]


def test_biorxiv_source_filters_details_locally() -> None:
    transport = FakeTransport(json_responses=[{
        "messages": [{"total": 2}],
        "collection": [
            {"title": "Other", "abstract": "nope", "doi": "10.1101/no", "date": "2026-01-01", "version": "1"},
            {"title": "LNT fermentation", "abstract": "Escherichia coli", "doi": "10.1101/yes", "date": "2026-01-02", "version": "2"},
        ],
    }])

    results = BioRxivSource(server="biorxiv", transport=transport).search(
        ExternalSearchQuery(query="LNT fermentation", from_date="2026-01-01", until_date="2026-01-03")
    )

    assert len(results) == 1
    assert results[0].doi == "10.1101/yes"
    assert "v2.full.pdf" in results[0].pdf_url


def test_biorxiv_source_matches_simple_boolean_or_groups() -> None:
    transport = FakeTransport(json_responses=[{
        "messages": [{"total": 2}],
        "collection": [
            {"title": "Other", "abstract": "Komagataella enzyme", "doi": "10.1101/no", "date": "2026-01-01", "version": "1"},
            {"title": "Pichia lactoferrin expression", "abstract": "secreted recombinant protein", "doi": "10.1101/yes", "date": "2026-01-02", "version": "1"},
        ],
    }])

    results = BioRxivSource(server="biorxiv", transport=transport).search(
        ExternalSearchQuery(
            query="(Pichia OR Komagataella) AND (lactoferrin OR osteopontin) AND (expression OR secretion)",
            from_date="2026-01-01",
            until_date="2026-01-03",
        )
    )

    assert len(results) == 1
    assert results[0].doi == "10.1101/yes"


def test_biorxiv_source_stops_after_scan_page_limit() -> None:
    transport = FakeTransport(json_responses=[
        {
            "messages": [{"total": 300}],
            "collection": [
                {"title": "Other", "abstract": "nope", "doi": "10.1101/no1", "date": "2026-01-01", "version": "1"},
            ],
        },
        {
            "messages": [{"total": 300}],
            "collection": [
                {"title": "Other again", "abstract": "nope", "doi": "10.1101/no2", "date": "2026-01-02", "version": "1"},
            ],
        },
        {
            "messages": [{"total": 300}],
            "collection": [
                {"title": "Should not be requested", "abstract": "LNT fermentation", "doi": "10.1101/yes", "date": "2026-01-03", "version": "1"},
            ],
        },
    ])

    results = BioRxivSource(server="biorxiv", transport=transport, max_scan_pages=2).search(
        ExternalSearchQuery(query="LNT fermentation", from_date="2026-01-01", until_date="2026-01-03")
    )

    assert results == []
    assert len(transport.calls) == 2


def test_crossref_source_maps_pdf_link() -> None:
    transport = FakeTransport(json_responses=[{
        "message": {
            "items": [{
                "title": ["Crossref paper"],
                "abstract": "<p>abstract</p>",
                "container-title": ["Journal"],
                "DOI": "10.1/CROSS",
                "published": {"date-parts": [[2026, 1, 3]]},
                "author": [{"given": "Ada", "family": "Lovelace"}],
                "link": [{"content-type": "application/pdf", "URL": "https://pdf.test/cross.pdf"}],
                "license": [{"URL": "https://license.test"}],
            }]
        }
    }])

    results = CrossrefSource(transport=transport).search(ExternalSearchQuery(query="x"))

    assert results[0].title == "Crossref paper"
    assert results[0].doi == "10.1/cross"
    assert results[0].pdf_url == "https://pdf.test/cross.pdf"


def test_crossref_source_filters_broad_boolean_results_locally() -> None:
    transport = FakeTransport(json_responses=[{
        "message": {
            "items": [
                {
                    "title": ["Human development disparities"],
                    "DOI": "10.1/no",
                    "published": {"date-parts": [[2026, 1, 1]]},
                },
                {
                    "title": ["Pichia pastoris lactoferrin secretion"],
                    "DOI": "10.1/yes",
                    "published": {"date-parts": [[2026, 1, 2]]},
                },
            ]
        }
    }])

    results = CrossrefSource(transport=transport).search(
        ExternalSearchQuery(query="(Pichia OR Komagataella) AND (lactoferrin OR osteopontin)")
    )

    assert [item.doi for item in results] == ["10.1/yes"]
