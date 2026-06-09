from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


ExternalSourceName = Literal[
    "pubmed",
    "europe_pmc",
    "openalex",
    "biorxiv",
    "medrxiv",
    "crossref",
]
ImportStatus = Literal["candidate", "importing", "imported", "duplicate", "failed", "skipped"]
PdfAccessStatus = Literal["unknown", "checking", "verified", "unavailable"]
BriefPriority = Literal["high", "medium", "low"]


SOURCE_PRIORITY: dict[str, int] = {
    "pubmed": 10,
    "europe_pmc": 20,
    "openalex": 30,
    "biorxiv": 40,
    "medrxiv": 41,
    "crossref": 50,
}


@dataclass(frozen=True)
class ExternalSearchQuery:
    query: str
    sources: list[str] = field(default_factory=list)
    from_date: str = ""
    until_date: str = ""
    max_results_per_source: int = 20


@dataclass
class ExternalPaperCandidate:
    candidate_id: str
    title: str
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    published_date: str = ""
    year: int | None = None
    doi: str = ""
    pmid: str = ""
    pmcid: str = ""
    openalex_id: str = ""
    source_priority: int = 999
    sources: list[str] = field(default_factory=list)
    source_records: list[dict[str, Any]] = field(default_factory=list)
    pdf_url: str = ""
    landing_url: str = ""
    is_open_access: bool = False
    pdf_access_status: PdfAccessStatus = "unknown"
    pdf_checked_at: str = ""
    pdf_access_error: str = ""
    import_status: ImportStatus = "candidate"
    imported_paper_id: str = ""
    downloaded_pdf_path: str = ""
    import_error: str = ""
    classification_id: str = ""
    classification_name: str = ""
    classification_confidence: float = 0.0
    classification_status: str = ""
    classification_reason: str = ""
    judge_reason: str = ""
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExternalPaperCandidate":
        data = {**ExternalPaperCandidate(candidate_id="", title="").to_dict(), **dict(value)}
        data["authors"] = list(data.get("authors") or [])
        data["sources"] = list(data.get("sources") or [])
        data["source_records"] = list(data.get("source_records") or [])
        return cls(**{key: data.get(key) for key in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MonitorProfile:
    monitor_id: str
    name: str
    query: str
    sources: list[str] = field(default_factory=list)
    filter_config: dict[str, Any] = field(default_factory=dict)
    lookback_days: int = 14
    interval_hours: int = 24
    max_results_per_source: int = 20
    enabled: bool = True
    created_at: str = ""
    last_run_at: str = ""
    next_run_at: str = ""
    last_summary: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MonitorProfile":
        data = {**MonitorProfile(monitor_id="", name="", query="").to_dict(), **dict(value)}
        data["sources"] = list(data.get("sources") or [])
        data["filter_config"] = dict(data.get("filter_config") or {})
        data["last_summary"] = dict(data.get("last_summary") or {})
        return cls(**{key: data.get(key) for key in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DailyBriefItem:
    candidate_id: str
    title: str
    journal: str = ""
    published_date: str = ""
    year: int | None = None
    sources: list[str] = field(default_factory=list)
    doi: str = ""
    pmid: str = ""
    pmcid: str = ""
    landing_url: str = ""
    pdf_url: str = ""
    pdf_access_status: PdfAccessStatus = "unknown"
    is_open_access: bool = False
    topics: list[str] = field(default_factory=list)
    classification_id: str = ""
    classification_name: str = ""
    classification_confidence: float = 0.0
    classification_status: str = ""
    classification_reason: str = ""
    judge_reason: str = ""
    brief: str = ""
    priority: BriefPriority = "medium"
    priority_reason: str = ""
    recommended_action: str = ""
    evidence_level: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DailyBriefItem":
        data = {**DailyBriefItem(candidate_id="", title="").to_dict(), **dict(value)}
        data["sources"] = list(data.get("sources") or [])
        data["topics"] = list(data.get("topics") or [])
        return cls(**{key: data.get(key) for key in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DailyBrief:
    brief_id: str
    brief_date: str
    generated_at: str
    title: str
    source_filter: list[str] = field(default_factory=list)
    lookback_days: int = 14
    max_items: int = 30
    total_candidates: int = 0
    included_count: int = 0
    topic_counts: dict[str, int] = field(default_factory=dict)
    source_counts: dict[str, int] = field(default_factory=dict)
    priority_counts: dict[str, int] = field(default_factory=dict)
    pdf_counts: dict[str, int] = field(default_factory=dict)
    highlights: list[str] = field(default_factory=list)
    items: list[DailyBriefItem] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DailyBrief":
        data = {**DailyBrief(brief_id="", brief_date="", generated_at="", title="").to_dict(), **dict(value)}
        data["source_filter"] = list(data.get("source_filter") or [])
        data["topic_counts"] = dict(data.get("topic_counts") or {})
        data["source_counts"] = dict(data.get("source_counts") or {})
        data["priority_counts"] = dict(data.get("priority_counts") or {})
        data["pdf_counts"] = dict(data.get("pdf_counts") or {})
        data["highlights"] = list(data.get("highlights") or [])
        data["items"] = [
            DailyBriefItem.from_dict(item)
            for item in (data.get("items") or [])
            if isinstance(item, dict)
        ]
        return cls(**{key: data.get(key) for key in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_doi(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.I)
    text = re.sub(r"^doi:\s*", "", text, flags=re.I)
    text = text.strip().strip(".").lower()
    match = re.search(r"(10\.\d{4,9}/\S+)", text, flags=re.I)
    return match.group(1).rstrip(").,;").lower() if match else text


def normalize_pmid(value: str | None) -> str:
    text = re.sub(r"\D+", "", str(value or ""))
    return text


def normalize_pmcid(value: str | None) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    digits = re.sub(r"\D+", "", text)
    return f"PMC{digits}" if digits else ""


def normalize_openalex_id(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.rstrip("/")
    return text.rsplit("/", 1)[-1].upper()


def normalize_title(value: str | None) -> str:
    text = str(value or "").lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def infer_year(published_date: str | None) -> int | None:
    match = re.search(r"\b(19|20)\d{2}\b", str(published_date or ""))
    return int(match.group(0)) if match else None


def title_year_key(title: str, year: int | None) -> str:
    normalized = normalize_title(title)
    if not normalized:
        return ""
    return f"title:{normalized}|{year or ''}"


def candidate_identity_keys(candidate: ExternalPaperCandidate) -> list[str]:
    keys: list[str] = []
    if candidate.doi:
        keys.append(f"doi:{normalize_doi(candidate.doi)}")
    if candidate.pmid:
        keys.append(f"pmid:{normalize_pmid(candidate.pmid)}")
    if candidate.pmcid:
        keys.append(f"pmcid:{normalize_pmcid(candidate.pmcid)}")
    if candidate.openalex_id:
        keys.append(f"openalex:{normalize_openalex_id(candidate.openalex_id)}")
    title_key = title_year_key(candidate.title, candidate.year)
    if title_key:
        keys.append(title_key)
    return [key for key in keys if key and not key.endswith(":")]


def canonical_candidate_id(candidate: ExternalPaperCandidate) -> str:
    keys = candidate_identity_keys(candidate)
    primary = keys[0] if keys else f"external:{candidate.title}"
    digest = hashlib.sha1(primary.encode("utf-8")).hexdigest()[:16]
    return f"ext_{digest}"


def make_candidate(
    *,
    source: str,
    title: str,
    abstract: str = "",
    authors: list[str] | None = None,
    journal: str = "",
    published_date: str = "",
    doi: str = "",
    pmid: str = "",
    pmcid: str = "",
    openalex_id: str = "",
    pdf_url: str = "",
    landing_url: str = "",
    is_open_access: bool = False,
    external_id: str = "",
    raw: dict[str, Any] | None = None,
) -> ExternalPaperCandidate:
    now = utc_now_iso()
    candidate = ExternalPaperCandidate(
        candidate_id="",
        title=_clean_text(title),
        abstract=_clean_text(abstract),
        authors=[_clean_text(author) for author in (authors or []) if _clean_text(author)],
        journal=_clean_text(journal),
        published_date=str(published_date or "").strip(),
        year=infer_year(published_date),
        doi=normalize_doi(doi),
        pmid=normalize_pmid(pmid),
        pmcid=normalize_pmcid(pmcid),
        openalex_id=normalize_openalex_id(openalex_id),
        source_priority=SOURCE_PRIORITY.get(source, 999),
        sources=[source],
        source_records=[{
            "source": source,
            "external_id": str(external_id or "").strip(),
            "landing_url": str(landing_url or "").strip(),
            "pdf_url": str(pdf_url or "").strip(),
            "raw": raw or {},
        }],
        pdf_url=str(pdf_url or "").strip(),
        landing_url=str(landing_url or "").strip(),
        is_open_access=bool(is_open_access),
        created_at=now,
        updated_at=now,
    )
    candidate.candidate_id = canonical_candidate_id(candidate)
    return candidate


def merge_candidate(
    target: ExternalPaperCandidate,
    incoming: ExternalPaperCandidate,
) -> tuple[ExternalPaperCandidate, bool]:
    changed = False
    incoming_is_primary = incoming.source_priority < target.source_priority

    def assign(field_name: str, value: Any) -> None:
        nonlocal changed
        if getattr(target, field_name) != value:
            setattr(target, field_name, value)
            changed = True

    for field_name in ("doi", "pmid", "pmcid", "openalex_id"):
        if not getattr(target, field_name) and getattr(incoming, field_name):
            assign(field_name, getattr(incoming, field_name))

    primary_fields = ("title", "journal", "published_date", "year", "landing_url")
    for field_name in primary_fields:
        current = getattr(target, field_name)
        value = getattr(incoming, field_name)
        if value and (not current or incoming_is_primary):
            assign(field_name, value)

    if incoming.abstract and (
        not target.abstract
        or (incoming_is_primary and len(incoming.abstract) >= len(target.abstract) * 0.7)
        or (incoming.source_priority == target.source_priority and len(incoming.abstract) > len(target.abstract))
    ):
        assign("abstract", incoming.abstract)

    if incoming.authors and (not target.authors or incoming_is_primary):
        assign("authors", incoming.authors)

    if incoming.pdf_url and (not target.pdf_url or incoming_is_primary):
        assign("pdf_url", incoming.pdf_url)
    if incoming.is_open_access and not target.is_open_access:
        assign("is_open_access", True)

    if incoming.source_priority < target.source_priority:
        assign("source_priority", incoming.source_priority)

    for source in incoming.sources:
        if source not in target.sources:
            target.sources.append(source)
            changed = True
    target.sources.sort(key=lambda item: SOURCE_PRIORITY.get(item, 999))

    existing_records = {
        (record.get("source"), record.get("external_id"), record.get("landing_url"))
        for record in target.source_records
    }
    for record in incoming.source_records:
        key = (record.get("source"), record.get("external_id"), record.get("landing_url"))
        if key not in existing_records:
            target.source_records.append(record)
            changed = True

    if changed:
        target.updated_at = utc_now_iso()
    return target, changed


def _clean_text(value: str | None) -> str:
    text = str(value or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()
