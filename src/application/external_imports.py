from __future__ import annotations

import hashlib
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from src.application.settings import (
    EXTERNAL_CONTACT_EMAIL,
    EXTERNAL_HTTP_TIMEOUT_SECONDS,
    EXTERNAL_IMPORTS_DIR,
)
from src.application import external_filters
from src.application.external_briefing import (
    BriefClassificationDecision,
    Y103_CATEGORIES,
    classify_candidates_for_brief,
)
from src.application import external_brief_items
from src.application.external_candidate_import import (
    ExternalCandidateImportService,
    candidate_filename as candidate_import_filename,
    read_validated_cached_pdf,
    repository_root_dir,
)
from src.application.external_candidates import ExternalCandidateQueryService
from src.application.external_daily_briefs import ExternalDailyBriefService
from src.application.external_monitoring import (
    ExternalMonitorService,
    monitor_due,
    monitor_id,
    utc_now as monitor_utc_now,
)
from src.application.external_pdf_access import (
    ExternalPdfAccessService,
    emit_pdf_check_progress,
)
from src.application.external_source_search import (
    ExternalSourceSearchService,
    emit_progress,
)
from src.domain.external_papers import (
    DailyBrief,
    DailyBriefItem,
    ExternalPaperCandidate,
    MonitorProfile,
    SOURCE_PRIORITY,
    utc_now_iso,
)
from src.ports.external_papers import ExternalCandidateRepository, ExternalPaperSource, ExternalPdfDownloader


DEFAULT_SOURCE_NAMES = ["pubmed", "europe_pmc", "openalex", "biorxiv", "medrxiv", "crossref"]
FIRST_PRIORITY_SOURCE_NAMES = ["pubmed", "europe_pmc", "openalex"]
SECOND_PRIORITY_SOURCE_NAMES = ["biorxiv", "medrxiv", "crossref"]
LATEST_PAPER_FETCH_KEY = "latest_paper_fetch_at"
PaperIdBuilder = Callable[[bytes], str]
IngestRunner = Callable[..., dict]
PaperCountLookup = Callable[[str], int]
ExternalRefreshProgress = Callable[[dict[str, Any]], None]
ExternalPdfCheckProgress = Callable[[dict[str, Any]], None]


def default_repository() -> Any:
    from src.adapters.external_papers.repository import JsonExternalImportRepository

    return JsonExternalImportRepository(EXTERNAL_IMPORTS_DIR)


def default_sources(source_names: list[str] | None = None) -> list[ExternalPaperSource]:
    from src.adapters.external_papers.sources import build_default_sources

    return build_default_sources(
        source_names=source_names or DEFAULT_SOURCE_NAMES,
        timeout=EXTERNAL_HTTP_TIMEOUT_SECONDS,
        contact_email=EXTERNAL_CONTACT_EMAIL,
    )


def default_pdf_downloader() -> ExternalPdfDownloader:
    from src.adapters.external_papers.downloader import default_pdf_downloader as build_downloader

    return build_downloader()


def _monitor_service() -> ExternalMonitorService:
    return ExternalMonitorService(
        repository_factory=default_repository,
        refresh_candidates=refresh_candidates,
        default_source_names=DEFAULT_SOURCE_NAMES,
        latest_fetch_key=LATEST_PAPER_FETCH_KEY,
    )


def _candidate_import_service() -> ExternalCandidateImportService:
    return ExternalCandidateImportService(
        repository_factory=default_repository,
        pdf_downloader_factory=default_pdf_downloader,
    )


def _candidate_query_service() -> ExternalCandidateQueryService:
    return ExternalCandidateQueryService(
        repository_factory=default_repository,
        source_priority=SOURCE_PRIORITY,
    )


def _pdf_access_service() -> ExternalPdfAccessService:
    return ExternalPdfAccessService(
        repository_factory=default_repository,
        pdf_downloader_factory=default_pdf_downloader,
    )


def _source_search_service() -> ExternalSourceSearchService:
    return ExternalSourceSearchService(
        repository_factory=default_repository,
        source_factory=default_sources,
        default_source_names=DEFAULT_SOURCE_NAMES,
    )


def _daily_brief_service() -> ExternalDailyBriefService:
    return ExternalDailyBriefService(
        repository_factory=default_repository,
        list_candidates=list_candidates,
        candidate_in_window=_candidate_in_brief_window,
        candidate_priority_score=external_brief_items.candidate_priority_score,
        daily_brief_decisions=_daily_brief_decisions,
        daily_brief_item=external_brief_items.daily_brief_item,
        count_values=external_brief_items.count_values,
        daily_brief_highlights=external_brief_items.daily_brief_highlights,
        daily_brief_id=_daily_brief_id,
    )


def refresh_candidates(
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
    return _source_search_service().refresh_candidates(
        query,
        source_names=source_names,
        from_date=from_date,
        until_date=until_date,
        max_results_per_source=max_results_per_source,
        repository=repository,
        sources=sources,
        progress_callback=progress_callback,
    )


def list_candidates(
    *,
    repository: ExternalCandidateRepository | None = None,
    source_names: list[str] | None = None,
    status: str = "",
    only_with_pdf: bool = False,
    query: str = "",
) -> list[ExternalPaperCandidate]:
    return _candidate_query_service().list_candidates(
        repository=repository,
        source_names=source_names,
        status=status,
        only_with_pdf=only_with_pdf,
        query=query,
    )


def generate_daily_brief(
    *,
    brief_date: date | None = None,
    lookback_days: int = 14,
    max_items: int = 30,
    source_names: list[str] | None = None,
    query: str = "",
    repository: Any | None = None,
    use_llm: bool = False,
    small_llm_client: Any | None = None,
    judge_llm_client: Any | None = None,
    progress_callback: ExternalRefreshProgress | None = None,
) -> DailyBrief:
    return _daily_brief_service().generate_daily_brief(
        brief_date=brief_date,
        lookback_days=lookback_days,
        max_items=max_items,
        source_names=source_names,
        query=query,
        repository=repository,
        use_llm=use_llm,
        small_llm_client=small_llm_client,
        judge_llm_client=judge_llm_client,
        progress_callback=progress_callback,
    )


def list_daily_briefs(*, repository: Any | None = None) -> list[DailyBrief]:
    repo = repository or default_repository()
    if not hasattr(repo, "list_daily_briefs"):
        return []
    return sorted(repo.list_daily_briefs(), key=lambda item: item.generated_at, reverse=True)


def _daily_brief_decisions(
    candidates: list[ExternalPaperCandidate],
    *,
    use_llm: bool,
    small_llm_client: Any | None,
    judge_llm_client: Any | None,
    progress_callback: ExternalRefreshProgress | None,
) -> dict[str, BriefClassificationDecision]:
    decisions: dict[str, BriefClassificationDecision] = {}
    missing: list[ExternalPaperCandidate] = []
    for candidate in candidates:
        stored = _stored_classification_decision(candidate)
        if stored is None:
            missing.append(candidate)
        else:
            decisions[candidate.candidate_id] = stored
    if missing:
        decisions.update(classify_candidates_for_brief(
            missing,
            use_llm=use_llm,
            small_llm_client=small_llm_client,
            judge_llm_client=judge_llm_client,
            progress_callback=progress_callback,
        ))
    return decisions


def _stored_classification_decision(candidate: ExternalPaperCandidate) -> BriefClassificationDecision | None:
    category_id = str(getattr(candidate, "classification_id", "") or "").strip()
    if not category_id:
        return None
    category_name = str(getattr(candidate, "classification_name", "") or "").strip()
    if not category_name:
        category_name = "其他/不纳入Y103" if category_id == "other" else category_id
    return BriefClassificationDecision(
        category_id=category_id,
        category_name=category_name,
        confidence=float(getattr(candidate, "classification_confidence", 0.0) or 0.0),
        status=str(getattr(candidate, "classification_status", "") or "stored"),
        reason=str(getattr(candidate, "classification_reason", "") or ""),
        judge_reason=str(getattr(candidate, "judge_reason", "") or ""),
    )


def latest_paper_fetch_time(*, repository: Any | None = None) -> str:
    repo = repository or default_repository()
    if not hasattr(repo, "get_metadata"):
        return ""
    metadata = repo.get_metadata()
    return str(metadata.get(LATEST_PAPER_FETCH_KEY) or "")


def latest_paper_fetch_status(
    *,
    repository: Any | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    latest_fetch_at = latest_paper_fetch_time(repository=repository)
    return {
        "latest_fetch_at": latest_fetch_at,
        "ran_today": _same_beijing_date(latest_fetch_at, now or _utc_now()),
    }


def background_monitor_task_status() -> dict[str, Any]:
    from src.application.external_monitor_task_status import background_monitor_task_status as get_status

    return get_status()


def y103_categories() -> list[dict[str, Any]]:
    return [
        {
            "category_id": category.category_id,
            "name": category.name,
            "description": category.description,
            "keywords": list(category.keywords),
            "retrieval_query": getattr(category, "retrieval_query", ""),
            "smoke_profile": getattr(category, "smoke_profile", "evidence"),
        }
        for category in Y103_CATEGORIES
    ]


def candidate_filter_type_labels() -> dict[str, str]:
    return external_filters.filter_type_labels()


def default_filter_config_for_monitor(name: str, query: str) -> dict[str, Any]:
    return external_filters.default_filter_config_for_monitor(name, query)


def effective_monitor_filter_config(monitor: MonitorProfile) -> dict[str, Any]:
    return external_filters.effective_filter_config(
        monitor.filter_config,
        monitor_name=monitor.name,
        monitor_query=monitor.query,
    )


def normalize_candidate_filter_config(config: dict[str, Any] | None) -> dict[str, Any]:
    return external_filters.normalize_filter_config(config)


def candidate_filter_label(config: dict[str, Any] | None) -> str:
    return external_filters.filter_label(config)


def candidate_filter_behavior_label(config: dict[str, Any] | None) -> str:
    return external_filters.filter_behavior_label(config)


def candidate_filter_criteria_text(config: dict[str, Any] | None) -> str:
    return external_filters.filter_criteria_text(config)


def create_monitor(
    *,
    name: str,
    query: str,
    sources: list[str] | None = None,
    lookback_days: int = 14,
    interval_hours: int = 24,
    max_results_per_source: int = 20,
    filter_config: dict[str, Any] | None = None,
    repository: Any | None = None,
) -> MonitorProfile:
    return _monitor_service().create_monitor(
        name=name,
        query=query,
        sources=sources,
        lookback_days=lookback_days,
        interval_hours=interval_hours,
        max_results_per_source=max_results_per_source,
        filter_config=filter_config,
        repository=repository,
    )


def list_monitors(*, repository: Any | None = None) -> list[MonitorProfile]:
    return _monitor_service().list_monitors(repository=repository)


def update_monitor(
    monitor: MonitorProfile,
    *,
    repository: Any | None = None,
) -> None:
    _monitor_service().update_monitor(monitor, repository=repository)


def delete_monitor(monitor_id: str, *, repository: Any | None = None) -> bool:
    return _monitor_service().delete_monitor(monitor_id, repository=repository)


def run_due_monitors(
    *,
    repository: Any | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    return _monitor_service().run_due_monitors(repository=repository, now=now)


def run_enabled_monitors_now(
    *,
    repository: Any | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    return _monitor_service().run_enabled_monitors_now(repository=repository, now=now)


def run_monitor(
    monitor_id: str,
    *,
    repository: Any | None = None,
    now: datetime | None = None,
    record_latest_fetch: bool = True,
) -> dict[str, Any]:
    return _monitor_service().run_monitor(
        monitor_id,
        repository=repository,
        now=now,
        record_latest_fetch=record_latest_fetch,
    )


def import_candidates(
    candidate_ids: list[str],
    *,
    repository: Any | None = None,
    paper_id_builder: PaperIdBuilder,
    ingest_runner: IngestRunner,
    paper_count_lookup: PaperCountLookup | None = None,
    pdf_downloader: ExternalPdfDownloader | None = None,
) -> dict[str, Any]:
    return _candidate_import_service().import_candidates(
        candidate_ids,
        repository=repository,
        paper_id_builder=paper_id_builder,
        ingest_runner=ingest_runner,
        paper_count_lookup=paper_count_lookup,
        pdf_downloader=pdf_downloader,
    )


def import_candidates_with_default_pipeline(
    candidate_ids: list[str],
    *,
    repository: Any | None = None,
    pdf_downloader: ExternalPdfDownloader | None = None,
) -> dict[str, Any]:
    from src.adapters.pipeline.defaults import default_external_candidate_pipeline_dependencies

    dependencies = default_external_candidate_pipeline_dependencies()

    return import_candidates(
        candidate_ids,
        repository=repository,
        paper_id_builder=dependencies.paper_id_builder,
        ingest_runner=dependencies.ingest_runner,
        paper_count_lookup=lambda paper_id: _safe_paper_count(paper_id, dependencies.paper_count_lookup),
        pdf_downloader=pdf_downloader,
    )


def check_candidate_pdf_access(
    candidate_ids: list[str],
    *,
    repository: Any | None = None,
    pdf_downloader: ExternalPdfDownloader | None = None,
    progress_callback: ExternalPdfCheckProgress | None = None,
) -> dict[str, Any]:
    return _pdf_access_service().check_candidate_pdf_access(
        candidate_ids,
        repository=repository,
        pdf_downloader=pdf_downloader,
        progress_callback=progress_callback,
    )


def download_candidate_pdf(
    candidate: ExternalPaperCandidate,
    *,
    root_dir: str | Path = EXTERNAL_IMPORTS_DIR,
    timeout: float = EXTERNAL_HTTP_TIMEOUT_SECONDS,
    max_pdf_mb: int | None = None,
) -> tuple[bytes, Path]:
    from src.adapters.external_papers.downloader import RequestsExternalPdfDownloader

    kwargs: dict[str, Any] = {"timeout": timeout}
    if max_pdf_mb is not None:
        kwargs["max_pdf_mb"] = max_pdf_mb
    return RequestsExternalPdfDownloader(**kwargs).download(candidate, root_dir=root_dir)



def candidate_filename(candidate: ExternalPaperCandidate) -> str:
    return candidate_import_filename(candidate)


def _candidate_filename(candidate: ExternalPaperCandidate) -> str:
    return candidate_filename(candidate)


def _monitor_due(monitor: MonitorProfile, now: datetime) -> bool:
    return monitor_due(monitor, now)


def _monitor_id(*, name: str, query: str) -> str:
    return monitor_id(name=name, query=query)


def _utc_now() -> datetime:
    return monitor_utc_now()


def _record_latest_paper_fetch(repo: Any, when: datetime) -> str:
    return _monitor_service().record_latest_paper_fetch(repo, when)


def _same_beijing_date(value: str, now: datetime) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    beijing_tz = timezone(timedelta(hours=8))
    return parsed.astimezone(beijing_tz).date() == now.astimezone(beijing_tz).date()


def _repository_root_dir(repo: Any) -> Path:
    return repository_root_dir(repo)


def _candidate_in_brief_window(
    candidate: ExternalPaperCandidate,
    *,
    target_date: date,
    lookback_days: int,
) -> bool:
    candidate_date = _candidate_date(candidate)
    if candidate_date is None:
        return True
    since = target_date - timedelta(days=max(int(lookback_days), 1))
    return since <= candidate_date <= target_date


def _candidate_date(candidate: ExternalPaperCandidate) -> date | None:
    for value in (candidate.published_date, candidate.updated_at, candidate.created_at):
        parsed = _parse_date(value)
        if parsed is not None:
            return parsed
    return None


def _parse_date(value: str) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    match = re.search(r"\b(19|20)\d{2}(?:-\d{1,2})?(?:-\d{1,2})?\b", text)
    if not match:
        return None
    parts = match.group(0).split("-")
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
        day = int(parts[2]) if len(parts) > 2 else 1
        return date(year, month, day)
    except ValueError:
        return None


def _candidate_priority_score(candidate: ExternalPaperCandidate) -> int:
    return external_brief_items.candidate_priority_score(candidate)


def _daily_brief_item(
    candidate: ExternalPaperCandidate,
    decision: BriefClassificationDecision | None = None,
) -> DailyBriefItem:
    return external_brief_items.daily_brief_item(candidate, decision)


def _candidate_topics(candidate: ExternalPaperCandidate) -> list[str]:
    return external_brief_items.candidate_topics(candidate)


def _candidate_priority(
    candidate: ExternalPaperCandidate,
    decision: BriefClassificationDecision,
) -> tuple[str, str]:
    return external_brief_items.candidate_priority(candidate, decision)


def _candidate_brief(candidate: ExternalPaperCandidate, decision: BriefClassificationDecision) -> str:
    return external_brief_items.candidate_brief(candidate, decision)


def _first_sentence(text: str) -> str:
    return external_brief_items.first_sentence(text)


def _candidate_recommended_action(candidate: ExternalPaperCandidate, priority: str) -> str:
    return external_brief_items.candidate_recommended_action(candidate, priority)


def _candidate_evidence_level(candidate: ExternalPaperCandidate) -> str:
    return external_brief_items.candidate_evidence_level(candidate)


def _pdf_bucket(pdf_url: str, status: str) -> str:
    return external_brief_items.pdf_bucket(pdf_url, status)


def _count_values(values) -> dict[str, int]:
    return external_brief_items.count_values(values)


def _daily_brief_highlights(items: list[DailyBriefItem], *, total_candidates: int) -> list[str]:
    return external_brief_items.daily_brief_highlights(items, total_candidates=total_candidates)


def _daily_brief_id(*, target_date: date) -> str:
    digest = hashlib.sha1(f"daily-brief|{target_date.isoformat()}|{utc_now_iso()}".encode("utf-8")).hexdigest()[:12]
    return f"brief_{target_date.isoformat()}_{digest}"


def _read_validated_cached_pdf(candidate: ExternalPaperCandidate) -> tuple[bytes, Path] | None:
    return read_validated_cached_pdf(candidate)


def _safe_paper_count(paper_id: str, paper_count: PaperCountLookup) -> int:
    try:
        return int(paper_count(paper_id))
    except Exception:
        return 0


def _emit_progress(
    callback: ExternalRefreshProgress | None,
    **event: Any,
) -> None:
    emit_progress(callback, **event)


def _emit_pdf_check_progress(
    callback: ExternalPdfCheckProgress | None,
    item: dict[str, Any],
    *,
    title: str,
    total: int,
    completed: int,
    verified: int,
    failed: int,
    missing_pdf_url: int,
) -> None:
    emit_pdf_check_progress(
        callback,
        total=total,
        completed=completed,
        verified=verified,
        failed=failed,
        missing_pdf_url=missing_pdf_url,
        item=item,
        title=title,
    )


def _user_agent() -> str:
    suffix = f" mailto:{EXTERNAL_CONTACT_EMAIL}" if EXTERNAL_CONTACT_EMAIL else ""
    return f"SortPaper/1.0 external-paper-importer{suffix}"
