from __future__ import annotations

from src.application.external_briefing import Y103_CATEGORIES
from src.application.y103_retrieval_cases import y103_rag_smoke_cases
from src.domain.y103_topics import Y103_TOPICS


def test_y103_topics_are_shared_by_briefing_and_rag_cases() -> None:
    assert Y103_CATEGORIES is Y103_TOPICS

    cases = y103_rag_smoke_cases()

    assert len(cases) == 16
    assert cases[0]["topic"] == "Y103-01"
    assert cases[0]["query"] == Y103_TOPICS[0].retrieval_query
    assert cases[0]["profile"] == Y103_TOPICS[0].smoke_profile
