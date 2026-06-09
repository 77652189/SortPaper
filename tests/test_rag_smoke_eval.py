from __future__ import annotations

from evals.rag_smoke_eval import load_cases, run_smoke_cases


def test_rag_smoke_runner_uses_case_profile_and_summarizes_hits() -> None:
    captured: list[dict] = []

    def fake_search_runner(query, **kwargs):
        captured.append({"query": query, **kwargs})
        return [
            {
                "score": 0.9,
                "payload": {
                    "paper_title": "Paper A",
                    "paper_id": "paper-a",
                    "chunk_id": "chunk-a",
                    "page": 1,
                    "content_type": "text",
                },
            }
        ]

    report = run_smoke_cases(
        [{"id": "case-1", "topic": "Y103-03", "query": "Pichia lactoferrin", "profile": "evidence"}],
        top_k=3,
        default_profile="quick",
        search_runner=fake_search_runner,
    )

    assert captured[0]["query"] == "Pichia lactoferrin"
    assert captured[0]["top_k"] == 3
    assert captured[0]["retrieval_profile"] == "evidence"
    assert report["case_count"] == 1
    assert report["results"][0]["top_hits"][0]["paper_title"] == "Paper A"


def test_rag_smoke_default_cases_come_from_y103_topics() -> None:
    cases = load_cases()

    assert len(cases) == 16
    assert cases[0]["topic"] == "Y103-01"
    assert cases[0]["query"]
    assert cases[0]["profile"] in {"quick", "evidence", "agent"}


def test_rag_smoke_report_marks_placeholder_only_hits() -> None:
    def fake_search_runner(_query, **_kwargs):
        return [
            {
                "score": 0.9,
                "payload": {
                    "paper_title": "default-reset-smoke.pdf",
                    "paper_id": "codex-default-reset-smoke-1",
                    "chunk_id": "chunk-a",
                    "page": 1,
                    "content_type": "text",
                },
            }
        ]

    report = run_smoke_cases(
        [{"id": "case-1", "topic": "Y103-03", "query": "Pichia lactoferrin"}],
        top_k=3,
        default_profile="quick",
        search_runner=fake_search_runner,
    )

    assert report["library_hint"]["only_placeholder_hits"] is True
    assert "只能证明链路可跑" in report["library_hint"]["note"]
