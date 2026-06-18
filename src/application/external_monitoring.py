from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from src.application import external_filters
from src.domain.external_papers import MonitorProfile, utc_now_iso


RepositoryFactory = Callable[[], Any]
RefreshCandidates = Callable[..., dict[str, Any]]
PostFilterCandidates = Callable[[Any, dict[str, Any]], dict[str, Any]]


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def monitor_due(monitor: MonitorProfile, now: datetime) -> bool:
    if not monitor.next_run_at:
        return True
    try:
        next_run = datetime.fromisoformat(monitor.next_run_at)
    except ValueError:
        return True
    if next_run.tzinfo is None:
        next_run = next_run.replace(tzinfo=timezone.utc)
    return next_run <= now


def monitor_id(*, name: str, query: str) -> str:
    digest = hashlib.sha1(f"{name}|{query}|{utc_now_iso()}".encode("utf-8")).hexdigest()[:12]
    return f"mon_{digest}"


class ExternalMonitorService:
    """Application service for scheduled external-paper monitor workflows."""

    def __init__(
        self,
        *,
        repository_factory: RepositoryFactory,
        refresh_candidates: RefreshCandidates,
        default_source_names: list[str],
        latest_fetch_key: str,
        post_filter_candidates: PostFilterCandidates | None = None,
    ) -> None:
        self.repository_factory = repository_factory
        self.refresh_candidates = refresh_candidates
        self.default_source_names = default_source_names
        self.latest_fetch_key = latest_fetch_key
        self.post_filter_candidates = post_filter_candidates

    def create_monitor(
        self,
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
        repo = repository or self.repository_factory()
        now = utc_now()
        effective_filter = (
            external_filters.normalize_filter_config(filter_config)
            if filter_config is not None
            else external_filters.default_filter_config_for_monitor(name, query)
        )
        monitor = MonitorProfile(
            monitor_id=monitor_id(name=name, query=query),
            name=name.strip() or query.strip()[:40] or "external monitor",
            query=query.strip(),
            sources=sources or self.default_source_names,
            filter_config=effective_filter,
            lookback_days=max(int(lookback_days), 1),
            interval_hours=max(int(interval_hours), 1),
            max_results_per_source=max(int(max_results_per_source), 1),
            enabled=True,
            created_at=now.isoformat(),
            next_run_at=now.isoformat(),
        )
        monitors = [item for item in repo.list_monitors() if item.monitor_id != monitor.monitor_id]
        monitors.append(monitor)
        repo.save_monitors(monitors)
        return monitor

    def list_monitors(self, *, repository: Any | None = None) -> list[MonitorProfile]:
        return sorted(
            (repository or self.repository_factory()).list_monitors(),
            key=lambda item: item.created_at,
            reverse=True,
        )

    def update_monitor(self, monitor: MonitorProfile, *, repository: Any | None = None) -> None:
        repo = repository or self.repository_factory()
        monitors = repo.list_monitors()
        for index, item in enumerate(monitors):
            if item.monitor_id == monitor.monitor_id:
                monitors[index] = monitor
                break
        else:
            monitors.append(monitor)
        repo.save_monitors(monitors)

    def delete_monitor(self, monitor_id: str, *, repository: Any | None = None) -> bool:
        repo = repository or self.repository_factory()
        monitors = repo.list_monitors()
        kept = [item for item in monitors if item.monitor_id != monitor_id]
        repo.save_monitors(kept)
        return len(kept) != len(monitors)

    def run_due_monitors(
        self,
        *,
        repository: Any | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        repo = repository or self.repository_factory()
        current = now or utc_now()
        summaries = []
        for monitor in repo.list_monitors():
            if monitor.enabled and monitor_due(monitor, current):
                summaries.append(self.run_monitor(
                    monitor.monitor_id,
                    repository=repo,
                    now=current,
                    record_latest_fetch=False,
                ))
        latest_fetch_at = self.record_latest_paper_fetch(repo, current)
        return {"run_count": len(summaries), "summaries": summaries, "latest_fetch_at": latest_fetch_at}

    def run_enabled_monitors_now(
        self,
        *,
        repository: Any | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        repo = repository or self.repository_factory()
        current = now or utc_now()
        summaries = []
        for monitor in repo.list_monitors():
            if monitor.enabled:
                summaries.append(self.run_monitor(
                    monitor.monitor_id,
                    repository=repo,
                    now=current,
                    record_latest_fetch=False,
                ))
        latest_fetch_at = self.record_latest_paper_fetch(repo, current)
        return {"run_count": len(summaries), "summaries": summaries, "latest_fetch_at": latest_fetch_at}

    def run_monitor(
        self,
        monitor_id: str,
        *,
        repository: Any | None = None,
        now: datetime | None = None,
        record_latest_fetch: bool = True,
    ) -> dict[str, Any]:
        repo = repository or self.repository_factory()
        monitors = repo.list_monitors()
        monitor = next((item for item in monitors if item.monitor_id == monitor_id), None)
        if monitor is None:
            return {"error": f"monitor {monitor_id} not found"}

        current = now or utc_now()
        until = current.date()
        since = until - timedelta(days=max(monitor.lookback_days, 1))
        summary = self.refresh_candidates(
            monitor.query,
            source_names=monitor.sources,
            from_date=since.isoformat(),
            until_date=until.isoformat(),
            max_results_per_source=monitor.max_results_per_source,
            repository=repo,
        )
        monitor.filter_config = external_filters.effective_filter_config(
            monitor.filter_config,
            monitor_name=monitor.name,
            monitor_query=monitor.query,
        )
        summary["candidate_filter"] = external_filters.apply_candidate_filter(
            repo,
            [str(candidate_id) for candidate_id in summary.get("candidate_ids", [])],
            monitor.filter_config,
        )
        if self.post_filter_candidates is not None:
            summary["abstract_translation"] = self.post_filter_candidates(repo, summary)
        monitor.last_run_at = current.isoformat()
        monitor.next_run_at = (current + timedelta(hours=max(monitor.interval_hours, 1))).isoformat()
        monitor.last_summary = summary
        self.update_monitor(monitor, repository=repo)
        if hasattr(repo, "append_run_summary"):
            repo.append_run_summary({
                "kind": "monitor_run",
                "monitor_id": monitor.monitor_id,
                "run_at": monitor.last_run_at,
                **summary,
            })
        if record_latest_fetch:
            summary["latest_fetch_at"] = self.record_latest_paper_fetch(repo, current)
        return summary

    def record_latest_paper_fetch(self, repo: Any, when: datetime) -> str:
        value = when.isoformat()
        if hasattr(repo, "update_metadata"):
            repo.update_metadata({self.latest_fetch_key: value})
        return value
