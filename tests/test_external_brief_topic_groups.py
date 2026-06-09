from __future__ import annotations

from src.application.external_brief_topic_groups import (
    daily_brief_topic_groups,
    daily_brief_topic_summary,
)
from src.domain.external_papers import DailyBrief, DailyBriefItem


def test_daily_brief_topic_groups_allow_duplicate_display_across_y103_topics() -> None:
    item = DailyBriefItem(
        candidate_id="c1",
        title="Pichia pastoris recombinant lactoferrin protein expression",
        classification_id="03",
        classification_name="毕赤酵母（乳铁蛋白/骨桥蛋白）",
        brief=(
            "Pichia pastoris was engineered for recombinant lactoferrin protein expression "
            "and production."
        ),
        priority="high",
    )
    brief = DailyBrief(
        brief_id="brief-1",
        brief_date="2026-06-09",
        generated_at="2026-06-09T00:00:00+00:00",
        title="2026-06-09 每日简报",
        items=[item],
    )

    groups = daily_brief_topic_groups(brief)
    summary = daily_brief_topic_summary(groups)
    hit_ids = {
        group["category_id"]
        for group in groups
        if any(entry.candidate_id == "c1" for entry in group["items"])
    }

    assert len(groups) == 16
    assert "03" in hit_ids
    assert "13" in hit_ids
    assert summary["unique_count"] == 1
    assert summary["display_count"] >= 2
    assert summary["duplicate_count"] >= 1


def test_daily_brief_topic_groups_do_not_show_other_candidates() -> None:
    item = DailyBriefItem(
        candidate_id="c1",
        title="Unrelated urban transport paper",
        classification_id="other",
        classification_name="其他/不纳入Y103",
        priority="low",
    )
    brief = DailyBrief(
        brief_id="brief-1",
        brief_date="2026-06-09",
        generated_at="2026-06-09T00:00:00+00:00",
        title="2026-06-09 每日简报",
        items=[item],
    )

    groups = daily_brief_topic_groups(brief)
    summary = daily_brief_topic_summary(groups)

    assert summary["display_count"] == 0
    assert all(not group["items"] for group in groups)
