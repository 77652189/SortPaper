from __future__ import annotations

import os
import re
from html import unescape
from typing import Any
from xml.etree import ElementTree

import requests

from src.domain.external_papers import (
    ExternalPaperCandidate,
    ExternalSearchQuery,
    make_candidate,
    normalize_doi,
)


DEFAULT_TIMEOUT_SECONDS = 20.0


class ExternalSourceError(RuntimeError):
    pass


class RequestsExternalTransport:
    def __init__(self, *, timeout: float = DEFAULT_TIMEOUT_SECONDS, contact_email: str = "") -> None:
        self.timeout = timeout
        self.contact_email = contact_email
        self.session = requests.Session()

    def get_json(self, url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self.session.get(
            url,
            params=params,
            timeout=self.timeout,
            headers=_headers(self.contact_email),
        )
        response.raise_for_status()
        value = response.json()
        if not isinstance(value, dict):
            raise ExternalSourceError(f"Unexpected JSON from {url}")
        return value

    def get_text(self, url: str, *, params: dict[str, Any] | None = None) -> str:
        response = self.session.get(
            url,
            params=params,
            timeout=self.timeout,
            headers=_headers(self.contact_email),
        )
        response.raise_for_status()
        return response.text


class PubMedSource:
    name = "pubmed"

    def __init__(
        self,
        *,
        transport: RequestsExternalTransport | None = None,
        contact_email: str = "",
        api_key: str = "",
    ) -> None:
        self.transport = transport or RequestsExternalTransport(contact_email=contact_email)
        self.contact_email = contact_email
        self.api_key = api_key or _env_value("NCBI_API_KEY")

    def search(self, query: ExternalSearchQuery) -> list[ExternalPaperCandidate]:
        params: dict[str, Any] = {
            "db": "pubmed",
            "term": query.query,
            "retmode": "json",
            "retmax": query.max_results_per_source,
            "sort": "pub date",
        }
        if query.from_date or query.until_date:
            params["datetype"] = "pdat"
            if query.from_date:
                params["mindate"] = query.from_date
            if query.until_date:
                params["maxdate"] = query.until_date
        if self.contact_email:
            params["email"] = self.contact_email
        if self.api_key:
            params["api_key"] = self.api_key
        search_payload = self.transport.get_json(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params=params,
        )
        ids = (search_payload.get("esearchresult") or {}).get("idlist") or []
        ids = [str(item) for item in ids if str(item).strip()]
        if not ids:
            return []

        fetch_params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml",
        }
        if self.contact_email:
            fetch_params["email"] = self.contact_email
        if self.api_key:
            fetch_params["api_key"] = self.api_key
        xml_text = self.transport.get_text(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params=fetch_params,
        )
        return _parse_pubmed_xml(xml_text)


class EuropePmcSource:
    name = "europe_pmc"

    def __init__(self, *, transport: RequestsExternalTransport | None = None, contact_email: str = "") -> None:
        self.transport = transport or RequestsExternalTransport(contact_email=contact_email)

    def search(self, query: ExternalSearchQuery) -> list[ExternalPaperCandidate]:
        term = query.query
        if query.from_date or query.until_date:
            start = query.from_date or "1900-01-01"
            end = query.until_date or "2999-12-31"
            term = f"({term}) AND FIRST_PDATE:[{start} TO {end}]"
        payload = self.transport.get_json(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={
                "query": term,
                "format": "json",
                "resultType": "core",
                "pageSize": query.max_results_per_source,
            },
        )
        results = ((payload.get("resultList") or {}).get("result") or [])
        return [_candidate_from_europe_pmc(item) for item in results if isinstance(item, dict)]


class OpenAlexSource:
    name = "openalex"

    def __init__(self, *, transport: RequestsExternalTransport | None = None, contact_email: str = "") -> None:
        self.transport = transport or RequestsExternalTransport(contact_email=contact_email)
        self.contact_email = contact_email

    def search(self, query: ExternalSearchQuery) -> list[ExternalPaperCandidate]:
        filters = []
        if query.from_date:
            filters.append(f"from_publication_date:{query.from_date}")
        if query.until_date:
            filters.append(f"to_publication_date:{query.until_date}")
        params: dict[str, Any] = {
            "search": query.query,
            "per-page": query.max_results_per_source,
            "sort": "publication_date:desc",
        }
        if filters:
            params["filter"] = ",".join(filters)
        if self.contact_email:
            params["mailto"] = self.contact_email
        payload = self.transport.get_json("https://api.openalex.org/works", params=params)
        results = payload.get("results") or []
        candidates = [_candidate_from_openalex(item) for item in results if isinstance(item, dict)]
        return _filter_candidates_by_query(candidates, query.query)


class BioRxivSource:
    name = "biorxiv"

    def __init__(
        self,
        *,
        server: str = "biorxiv",
        transport: RequestsExternalTransport | None = None,
        contact_email: str = "",
        max_scan_pages: int = 2,
    ) -> None:
        self.server = server
        self.name = server
        self.transport = transport or RequestsExternalTransport(contact_email=contact_email)
        self.max_scan_pages = max(max_scan_pages, 1)

    def search(self, query: ExternalSearchQuery) -> list[ExternalPaperCandidate]:
        start = query.from_date or "2020-01-01"
        end = query.until_date or start
        cursor = 0
        scanned_pages = 0
        candidates: list[ExternalPaperCandidate] = []
        max_results = max(query.max_results_per_source, 1)
        while len(candidates) < max_results and scanned_pages < self.max_scan_pages:
            payload = self.transport.get_json(
                f"https://api.biorxiv.org/details/{self.server}/{start}/{end}/{cursor}",
            )
            scanned_pages += 1
            collection = payload.get("collection") or []
            if not collection:
                break
            for item in collection:
                if not isinstance(item, dict):
                    continue
                if _matches_query(item, query.query):
                    candidates.append(_candidate_from_biorxiv(self.server, item))
                    if len(candidates) >= max_results:
                        break
            messages = payload.get("messages") or []
            total = _safe_int(messages[0].get("total")) if messages and isinstance(messages[0], dict) else len(collection)
            cursor += len(collection)
            if cursor >= total:
                break
        return candidates


class CrossrefSource:
    name = "crossref"

    def __init__(self, *, transport: RequestsExternalTransport | None = None, contact_email: str = "") -> None:
        self.transport = transport or RequestsExternalTransport(contact_email=contact_email)
        self.contact_email = contact_email

    def search(self, query: ExternalSearchQuery) -> list[ExternalPaperCandidate]:
        filters = []
        if query.from_date:
            filters.append(f"from-pub-date:{query.from_date}")
        if query.until_date:
            filters.append(f"until-pub-date:{query.until_date}")
        params: dict[str, Any] = {
            "query.bibliographic": query.query,
            "rows": query.max_results_per_source,
            "sort": "published",
            "order": "desc",
        }
        if filters:
            params["filter"] = ",".join(filters)
        if self.contact_email:
            params["mailto"] = self.contact_email
        payload = self.transport.get_json("https://api.crossref.org/works", params=params)
        items = (payload.get("message") or {}).get("items") or []
        candidates = [_candidate_from_crossref(item) for item in items if isinstance(item, dict)]
        return _filter_candidates_by_query(candidates, query.query)


def build_default_sources(
    *,
    source_names: list[str],
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    contact_email: str = "",
) -> list[Any]:
    sources: list[Any] = []
    for name in source_names:
        transport = RequestsExternalTransport(timeout=timeout, contact_email=contact_email)
        if name == "pubmed":
            sources.append(PubMedSource(transport=transport, contact_email=contact_email))
        elif name == "europe_pmc":
            sources.append(EuropePmcSource(transport=transport, contact_email=contact_email))
        elif name == "openalex":
            sources.append(OpenAlexSource(transport=transport, contact_email=contact_email))
        elif name == "biorxiv":
            sources.append(BioRxivSource(server="biorxiv", transport=transport, contact_email=contact_email))
        elif name == "medrxiv":
            sources.append(BioRxivSource(server="medrxiv", transport=transport, contact_email=contact_email))
        elif name == "crossref":
            sources.append(CrossrefSource(transport=transport, contact_email=contact_email))
    return sources


def _headers(contact_email: str = "") -> dict[str, str]:
    suffix = f" mailto:{contact_email}" if contact_email else ""
    return {
        "User-Agent": f"SortPaper/1.0 external-paper-importer{suffix}",
        "Accept": "application/json, application/xml, text/xml, text/plain",
    }


def _env_value(name: str, default: str = "") -> str:
    return (os.getenv(name) or os.getenv(f"\ufeff{name}") or default).strip()


def _parse_pubmed_xml(xml_text: str) -> list[ExternalPaperCandidate]:
    root = ElementTree.fromstring(xml_text)
    candidates: list[ExternalPaperCandidate] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = _node_text(article.find(".//PMID"))
        title = _iter_text(article.find(".//ArticleTitle"))
        abstract = " ".join(_iter_text(node) for node in article.findall(".//Abstract/AbstractText"))
        journal = _iter_text(article.find(".//Journal/Title")) or _iter_text(article.find(".//Journal/ISOAbbreviation"))
        published_date = _pubmed_date(article)
        ids = {
            str(node.attrib.get("IdType", "")).lower(): _node_text(node)
            for node in article.findall(".//PubmedData/ArticleIdList/ArticleId")
        }
        doi = ids.get("doi", "")
        pmcid = ids.get("pmc", "")
        authors = []
        for author in article.findall(".//AuthorList/Author"):
            name = " ".join(part for part in [
                _node_text(author.find("ForeName")),
                _node_text(author.find("LastName")),
            ] if part)
            if name:
                authors.append(name)
        pdf_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/" if pmcid else ""
        landing_url = (
            f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            if pmid
            else (f"https://doi.org/{normalize_doi(doi)}" if doi else "")
        )
        if title:
            candidates.append(make_candidate(
                source="pubmed",
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                published_date=published_date,
                doi=doi,
                pmid=pmid,
                pmcid=pmcid,
                pdf_url=pdf_url,
                landing_url=landing_url,
                is_open_access=bool(pmcid),
                external_id=pmid,
            ))
    return candidates


def _candidate_from_europe_pmc(item: dict[str, Any]) -> ExternalPaperCandidate:
    pdf_url = ""
    for url_item in (((item.get("fullTextUrlList") or {}).get("fullTextUrl")) or []):
        if not isinstance(url_item, dict):
            continue
        style = str(url_item.get("documentStyle") or "").lower()
        availability = str(url_item.get("availability") or "").lower()
        if style == "pdf" and ("open" in availability or item.get("isOpenAccess") == "Y"):
            pdf_url = str(url_item.get("url") or "")
            break
    pmcid = item.get("pmcid") or ""
    if not pdf_url and pmcid:
        pdf_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/"
    return make_candidate(
        source="europe_pmc",
        title=str(item.get("title") or ""),
        abstract=str(item.get("abstractText") or ""),
        authors=_split_authors(str(item.get("authorString") or "")),
        journal=str(item.get("journalTitle") or ""),
        published_date=str(item.get("firstPublicationDate") or item.get("firstIndexDate") or ""),
        doi=str(item.get("doi") or ""),
        pmid=str(item.get("pmid") or ""),
        pmcid=str(pmcid),
        pdf_url=pdf_url,
        landing_url=str(item.get("journalUrl") or item.get("url") or ""),
        is_open_access=str(item.get("isOpenAccess") or "").upper() == "Y" or bool(pdf_url),
        external_id=str(item.get("id") or item.get("pmid") or ""),
        raw={"source": item.get("source"), "citedByCount": item.get("citedByCount")},
    )


def _candidate_from_openalex(item: dict[str, Any]) -> ExternalPaperCandidate:
    authors = []
    for authorship in item.get("authorships") or []:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author") or {}
        name = author.get("display_name")
        if name:
            authors.append(str(name))
    location = item.get("primary_location") or {}
    source = location.get("source") or {}
    open_access = item.get("open_access") or {}
    pdf_url = str(location.get("pdf_url") or "")
    landing_url = str(location.get("landing_page_url") or open_access.get("oa_url") or item.get("id") or "")
    return make_candidate(
        source="openalex",
        title=str(item.get("title") or item.get("display_name") or ""),
        abstract=_openalex_abstract(item.get("abstract_inverted_index") or {}),
        authors=authors,
        journal=str(source.get("display_name") or ""),
        published_date=str(item.get("publication_date") or ""),
        doi=str(item.get("doi") or ""),
        openalex_id=str(item.get("id") or ""),
        pdf_url=pdf_url,
        landing_url=landing_url,
        is_open_access=bool(open_access.get("is_oa")) or bool(pdf_url),
        external_id=str(item.get("id") or ""),
        raw={"cited_by_count": item.get("cited_by_count")},
    )


def _candidate_from_biorxiv(server: str, item: dict[str, Any]) -> ExternalPaperCandidate:
    doi = str(item.get("doi") or "")
    version = str(item.get("version") or "").strip()
    version_suffix = f"v{version}" if version and not version.startswith("v") else version
    pdf_url = f"https://www.{server}.org/content/{doi}{version_suffix}.full.pdf" if doi else ""
    return make_candidate(
        source=server,
        title=str(item.get("title") or ""),
        abstract=str(item.get("abstract") or ""),
        authors=_split_authors(str(item.get("authors") or "")),
        journal=server,
        published_date=str(item.get("date") or ""),
        doi=doi,
        pdf_url=pdf_url,
        landing_url=f"https://www.{server}.org/content/{doi}{version_suffix}" if doi else "",
        is_open_access=bool(pdf_url),
        external_id=doi,
        raw={"category": item.get("category"), "version": version, "license": item.get("license")},
    )


def _candidate_from_crossref(item: dict[str, Any]) -> ExternalPaperCandidate:
    authors = []
    for author in item.get("author") or []:
        if not isinstance(author, dict):
            continue
        name = " ".join(str(author.get(part) or "").strip() for part in ("given", "family")).strip()
        if name:
            authors.append(name)
    published_date = _crossref_date(item)
    pdf_url = ""
    for link in item.get("link") or []:
        if not isinstance(link, dict):
            continue
        content_type = str(link.get("content-type") or "").lower()
        if "pdf" in content_type:
            pdf_url = str(link.get("URL") or "")
            break
    title = _first_text(item.get("title"))
    abstract = unescape(str(item.get("abstract") or ""))
    container = _first_text(item.get("container-title"))
    return make_candidate(
        source="crossref",
        title=title,
        abstract=abstract,
        authors=authors,
        journal=container,
        published_date=published_date,
        doi=str(item.get("DOI") or ""),
        pdf_url=pdf_url,
        landing_url=str(item.get("URL") or ""),
        is_open_access=bool(pdf_url and item.get("license")),
        external_id=str(item.get("DOI") or ""),
        raw={"type": item.get("type"), "publisher": item.get("publisher")},
    )


def _openalex_abstract(index: dict[str, Any]) -> str:
    if not isinstance(index, dict) or not index:
        return ""
    words: list[tuple[int, str]] = []
    for word, positions in index.items():
        if isinstance(positions, list):
            words.extend((int(pos), str(word)) for pos in positions if isinstance(pos, int))
    return " ".join(word for _, word in sorted(words))


def _pubmed_date(article: Any) -> str:
    article_date = article.find(".//ArticleDate")
    if article_date is not None:
        parts = [_node_text(article_date.find(part)) for part in ("Year", "Month", "Day")]
        if parts[0]:
            return "-".join(part.zfill(2) if index else part for index, part in enumerate(parts) if part)
    pub_date = article.find(".//JournalIssue/PubDate")
    if pub_date is not None:
        year = _node_text(pub_date.find("Year")) or _node_text(pub_date.find("MedlineDate"))[:4]
        month = _month_number(_node_text(pub_date.find("Month")))
        day = _node_text(pub_date.find("Day"))
        return "-".join(part for part in [year, month, day.zfill(2) if day else ""] if part)
    return ""


def _crossref_date(item: dict[str, Any]) -> str:
    for key in ("published-print", "published-online", "published", "created", "issued"):
        date_parts = ((item.get(key) or {}).get("date-parts") or [])
        if date_parts and isinstance(date_parts[0], list):
            parts = [str(part).zfill(2) if index else str(part) for index, part in enumerate(date_parts[0])]
            return "-".join(parts)
    return ""


def _matches_query(item: dict[str, Any], query: str) -> bool:
    clauses = _boolean_query_clauses(query)
    if not clauses:
        return True
    haystack = " ".join(str(item.get(key) or "") for key in ("title", "abstract", "authors", "category")).lower()
    return all(any(term in haystack for term in clause) for clause in clauses)


def _boolean_query_clauses(query: str) -> list[list[str]]:
    text = str(query or "").strip()
    if not text:
        return []
    if not re.search(r"\bAND\b|\bOR\b", text, flags=re.I):
        return [
            [term.lower()]
            for term in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9'-]{2,}", text)
            if len(term) >= 3
        ][:6]
    clauses: list[list[str]] = []
    for raw_clause in re.split(r"\bAND\b", text, flags=re.I):
        terms = [
            term.lower()
            for term in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9'-]{2,}", raw_clause)
            if len(term) >= 3 and term.lower() not in {"and", "or", "not"}
        ]
        if terms:
            clauses.append(terms)
    return clauses[:6]


def _filter_candidates_by_query(candidates: list[ExternalPaperCandidate], query: str) -> list[ExternalPaperCandidate]:
    return [
        candidate
        for candidate in candidates
        if _matches_query(
            {
                "title": candidate.title,
                "abstract": candidate.abstract,
                "authors": " ".join(candidate.authors),
                "category": candidate.journal,
            },
            query,
        )
    ]


def _iter_text(node: Any) -> str:
    if node is None:
        return ""
    return re.sub(r"\s+", " ", "".join(node.itertext())).strip()


def _node_text(node: Any) -> str:
    return str(node.text or "").strip() if node is not None else ""


def _month_number(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    if text.isdigit():
        return text.zfill(2)
    months = {
        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
        "may": "05", "jun": "06", "jul": "07", "aug": "08",
        "sep": "09", "oct": "10", "nov": "11", "dec": "12",
    }
    return months.get(text[:3].lower(), "")


def _split_authors(value: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r";|\band\b", value)
        if item.strip()
    ]


def _first_text(value: Any) -> str:
    if isinstance(value, list) and value:
        return str(value[0] or "")
    return str(value or "")


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
