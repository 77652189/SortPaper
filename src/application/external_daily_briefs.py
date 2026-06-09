from __future__ import annotations

from datetime import date
from typing import Any, Callable

from src.domain.external_papers import DailyBrief, DailyBriefItem, ExternalPaperCandidate, utc_now_iso


CandidateList = Callable[..., list[ExternalPaperCandidate]]
CandidatePredicate = Callable[..., bool]
CandidateScore = Callable[[ExternalPaperCandidate], int]
DecisionBuilder = Callable[..., dict[str, Any]]
BriefItemBuilder = Callable[[ExternalPaperCandidate, Any], DailyBriefItem]
CountValues = Callable[[Any], dict[str, int]]
HighlightsBuilder = Callable[[list[DailyBriefItem]], list[str]]
BriefIdBuilder = Callable[[date], str]


class ExternalDailyBriefService:
    """Application service for turning filtered candidates into a daily brief."""

    def __init__(
        self,
        *,
        repository_factory: Callable[[], Any],
        list_candidates: CandidateList,
        candidate_in_window: CandidatePredicate,
        candidate_priority_score: CandidateScore,
        daily_brief_decisions: DecisionBuilder,
        daily_brief_item: BriefItemBuilder,
        count_values: CountValues,
        daily_brief_highlights: Callable[..., list[str]],
        daily_brief_id: Callable[..., str],
    ) -> None:
        self.repository_factory = repository_factory
        self.list_candidates = list_candidates
        self.candidate_in_window = candidate_in_window
        self.candidate_priority_score = candidate_priority_score
        self.daily_brief_decisions = daily_brief_decisions
        self.daily_brief_item = daily_brief_item
        self.count_values = count_values
        self.daily_brief_highlights = daily_brief_highlights
        self.daily_brief_id = daily_brief_id

    def generate_daily_brief(
        self,
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
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> DailyBrief:
        repo = repository or self.repository_factory()
        target_date = brief_date or date.today()
        candidates = self.list_candidates(
            repository=repo,
            source_names=source_names or [],
            query=query,
        )
        in_window = [
            candidate for candidate in candidates
            if self.candidate_in_window(
                candidate,
                target_date=target_date,
                lookback_days=lookback_days,
            )
        ]
        ranked = sorted(
            in_window,
            key=lambda candidate: (
                self.candidate_priority_score(candidate),
                candidate.year or 0,
                bool(candidate.abstract),
                candidate.title,
            ),
            reverse=True,
        )
        pool_size = max(max_items, 1) * (4 if use_llm else 8)
        selected = ranked[:max(pool_size, max_items, 1)]
        decisions = self.daily_brief_decisions(
            selected,
            use_llm=use_llm,
            small_llm_client=small_llm_client,
            judge_llm_client=judge_llm_client,
            progress_callback=progress_callback,
        )
        classified_items = [
            self.daily_brief_item(candidate, decisions.get(candidate.candidate_id))
            for candidate in selected
        ]
        items = classified_items[:max(max_items, 1)]
        brief = DailyBrief(
            brief_id=self.daily_brief_id(target_date=target_date),
            brief_date=target_date.isoformat(),
            generated_at=utc_now_iso(),
            title=f"{target_date.isoformat()} 每日简报",
            source_filter=list(source_names or []),
            lookback_days=max(int(lookback_days), 1),
            max_items=max(int(max_items), 1),
            total_candidates=len(in_window),
            included_count=len(items),
            topic_counts=self.count_values(topic for item in items for topic in item.topics),
            source_counts=self.count_values(source for item in items for source in item.sources),
            priority_counts=self.count_values(item.priority for item in items),
            pdf_counts=self.count_values(_pdf_bucket(item.pdf_url, item.pdf_access_status) for item in items),
            highlights=self.daily_brief_highlights(items, total_candidates=len(in_window)),
            items=items,
        )
        if hasattr(repo, "save_daily_brief"):
            repo.save_daily_brief(brief)
        return brief


def _pdf_bucket(pdf_url: str, status: str) -> str:
    if status == "verified":
        return "PDF可访问"
    if status == "unavailable":
        return "PDF不可访问"
    if pdf_url:
        return "PDF待预检"
    return "无PDF链接"
