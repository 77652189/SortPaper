from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from src.adapters.external_papers.repository import JsonExternalImportRepository
from src.application import external_imports
from src.domain.external_papers import ExternalSearchQuery, make_candidate


class FakeSource:
    name = "pubmed"

    def __init__(self, candidate):
        self.candidate = candidate

    def search(self, query: ExternalSearchQuery):
        assert query.query == "LNT"
        return [self.candidate]


class FailingSource:
    name = "crossref"

    def search(self, query: ExternalSearchQuery):
        raise RuntimeError("boom")


class FakeDownloader:
    def download(self, candidate, *, root_dir):
        path = Path(root_dir) / "pdfs" / "paper.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-1.7")
        return b"%PDF-1.7", path


class FailingDownloader:
    def download(self, candidate, *, root_dir):
        raise ValueError("not a pdf")


def test_refresh_candidates_keeps_source_errors_non_fatal(tmp_path: Path) -> None:
    repo = JsonExternalImportRepository(tmp_path)
    candidate = make_candidate(source="pubmed", title="LNT paper", doi="10.1/a")

    summary = external_imports.refresh_candidates(
        "LNT",
        repository=repo,
        sources=[FakeSource(candidate), FailingSource()],
    )

    assert summary["fetched_count"] == 1
    assert summary["new_count"] == 1
    assert summary["errors"] == [{"source": "crossref", "error": "RuntimeError: boom"}]
    assert repo.list_candidates()[0].title == "LNT paper"


def test_refresh_candidates_reports_progress_events(tmp_path: Path) -> None:
    repo = JsonExternalImportRepository(tmp_path)
    candidate = make_candidate(source="pubmed", title="LNT paper", doi="10.1/progress")
    events = []

    summary = external_imports.refresh_candidates(
        "LNT",
        repository=repo,
        sources=[FakeSource(candidate), FailingSource()],
        progress_callback=lambda event: events.append(event),
    )

    assert summary["fetched_count"] == 1
    assert {event["event"] for event in events} >= {"source_start", "source_done", "source_error", "merge_done"}
    assert any(event.get("source") == "pubmed" and event.get("count") == 1 for event in events)
    assert any(event.get("source") == "crossref" for event in events)


def test_monitor_due_run_updates_next_run_and_candidates(tmp_path: Path) -> None:
    repo = JsonExternalImportRepository(tmp_path)
    monitor = external_imports.create_monitor(
        name="LNT",
        query="LNT",
        sources=["pubmed"],
        repository=repo,
    )
    monitor.next_run_at = "2026-01-01T00:00:00+00:00"
    external_imports.update_monitor(monitor, repository=repo)

    class SourceFactory:
        def __call__(self, source_names=None):
            return [FakeSource(make_candidate(source="pubmed", title="LNT monitor", doi="10.1/mon"))]

    old_default_sources = external_imports.default_sources
    try:
        external_imports.default_sources = SourceFactory()
        summary = external_imports.run_due_monitors(
            repository=repo,
            now=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )
    finally:
        external_imports.default_sources = old_default_sources

    monitors = repo.list_monitors()
    assert summary["run_count"] == 1
    assert summary["latest_fetch_at"].startswith("2026-01-02")
    assert external_imports.latest_paper_fetch_time(repository=repo).startswith("2026-01-02")
    assert monitors[0].last_run_at.startswith("2026-01-02")
    assert monitors[0].next_run_at.startswith("2026-01-03")
    assert repo.list_candidates()[0].doi == "10.1/mon"


def test_direct_monitor_run_updates_same_latest_fetch_time(tmp_path: Path) -> None:
    repo = JsonExternalImportRepository(tmp_path)
    monitor = external_imports.create_monitor(
        name="LNT",
        query="LNT",
        sources=["pubmed"],
        repository=repo,
    )

    class SourceFactory:
        def __call__(self, source_names=None):
            return [FakeSource(make_candidate(source="pubmed", title="Direct monitor", doi="10.1/direct"))]

    old_default_sources = external_imports.default_sources
    try:
        external_imports.default_sources = SourceFactory()
        summary = external_imports.run_monitor(
            monitor.monitor_id,
            repository=repo,
            now=datetime(2026, 1, 3, 8, 30, tzinfo=timezone.utc),
        )
    finally:
        external_imports.default_sources = old_default_sources

    assert summary["latest_fetch_at"].startswith("2026-01-03T08:30")
    assert external_imports.latest_paper_fetch_time(repository=repo).startswith("2026-01-03T08:30")


def test_monitor_run_filters_candidates_before_candidate_library(tmp_path: Path) -> None:
    repo = JsonExternalImportRepository(tmp_path)
    monitor = external_imports.create_monitor(
        name="Y103",
        query="Pichia lactoferrin",
        sources=["pubmed"],
        repository=repo,
    )
    relevant = make_candidate(
        source="pubmed",
        title="Recombinant lactoferrin expression in Pichia pastoris",
        abstract="Pichia pastoris secretes recombinant lactoferrin.",
        doi="10.1/relevant",
        published_date="2026-01-01",
    )
    irrelevant = make_candidate(
        source="pubmed",
        title="Human development disparities",
        abstract="Persistent inequality constrains human development.",
        doi="10.1/irrelevant",
        published_date="2026-01-01",
    )

    class LocalSource:
        name = "pubmed"

        def __init__(self, candidates):
            self.candidates = candidates

        def search(self, query: ExternalSearchQuery):
            return self.candidates

    class SourceFactory:
        def __call__(self, source_names=None):
            return [LocalSource([relevant, irrelevant])]

    old_default_sources = external_imports.default_sources
    try:
        external_imports.default_sources = SourceFactory()
        summary = external_imports.run_monitor(
            monitor.monitor_id,
            repository=repo,
            now=datetime(2026, 1, 3, 8, 30, tzinfo=timezone.utc),
        )
    finally:
        external_imports.default_sources = old_default_sources

    visible = external_imports.list_candidates(repository=repo)
    skipped = external_imports.list_candidates(repository=repo, status="skipped")

    assert summary["candidate_filter"]["filter_type"] == "y103"
    assert summary["candidate_filter"]["checked_count"] == 2
    assert summary["candidate_filter"]["kept_count"] == 1
    assert summary["candidate_filter"]["skipped_count"] == 1
    assert [candidate.doi for candidate in visible] == ["10.1/relevant"]
    assert [candidate.doi for candidate in skipped] == ["10.1/irrelevant"]
    assert visible[0].classification_id == "03"
    assert skipped[0].import_error.startswith("候选过滤")


def test_monitor_filter_can_be_disabled(tmp_path: Path) -> None:
    repo = JsonExternalImportRepository(tmp_path)
    monitor = external_imports.create_monitor(
        name="Y103",
        query="Pichia lactoferrin",
        sources=["pubmed"],
        filter_config={"enabled": False, "filter_type": "none", "include_other": True},
        repository=repo,
    )
    irrelevant = make_candidate(
        source="pubmed",
        title="Human development disparities",
        abstract="Persistent inequality constrains human development.",
        doi="10.1/not-filtered",
        published_date="2026-01-01",
    )

    class LocalSource:
        name = "pubmed"

        def search(self, query: ExternalSearchQuery):
            return [irrelevant]

    class SourceFactory:
        def __call__(self, source_names=None):
            return [LocalSource()]

    old_default_sources = external_imports.default_sources
    try:
        external_imports.default_sources = SourceFactory()
        summary = external_imports.run_monitor(
            monitor.monitor_id,
            repository=repo,
            now=datetime(2026, 1, 3, 8, 30, tzinfo=timezone.utc),
        )
    finally:
        external_imports.default_sources = old_default_sources

    visible = external_imports.list_candidates(repository=repo)
    skipped = external_imports.list_candidates(repository=repo, status="skipped")

    assert summary["candidate_filter"]["enabled"] is False
    assert summary["candidate_filter"]["checked_count"] == 0
    assert [candidate.doi for candidate in visible] == ["10.1/not-filtered"]
    assert skipped == []


def test_due_monitor_check_records_click_time_even_without_due_topics(tmp_path: Path) -> None:
    repo = JsonExternalImportRepository(tmp_path)

    summary = external_imports.run_due_monitors(
        repository=repo,
        now=datetime(2026, 1, 4, 9, 15, tzinfo=timezone.utc),
    )

    assert summary["run_count"] == 0
    assert summary["latest_fetch_at"].startswith("2026-01-04T09:15")
    assert external_imports.latest_paper_fetch_time(repository=repo).startswith("2026-01-04T09:15")


def test_latest_paper_fetch_status_uses_beijing_day(tmp_path: Path) -> None:
    repo = JsonExternalImportRepository(tmp_path)
    repo.update_metadata({"latest_paper_fetch_at": "2026-01-03T16:30:00+00:00"})

    status = external_imports.latest_paper_fetch_status(
        repository=repo,
        now=datetime(2026, 1, 4, 5, 0, tzinfo=timezone.utc),
    )

    assert status["latest_fetch_at"] == "2026-01-03T16:30:00+00:00"
    assert status["ran_today"] is True


def test_run_enabled_monitors_now_ignores_next_run_time(tmp_path: Path) -> None:
    repo = JsonExternalImportRepository(tmp_path)
    monitor = external_imports.create_monitor(
        name="LNT",
        query="LNT",
        sources=["pubmed"],
        repository=repo,
    )
    monitor.next_run_at = "2999-01-01T00:00:00+00:00"
    external_imports.update_monitor(monitor, repository=repo)

    class SourceFactory:
        def __call__(self, source_names=None):
            return [FakeSource(make_candidate(source="pubmed", title="Immediate monitor", doi="10.1/immediate"))]

    old_default_sources = external_imports.default_sources
    try:
        external_imports.default_sources = SourceFactory()
        summary = external_imports.run_enabled_monitors_now(
            repository=repo,
            now=datetime(2026, 1, 4, 10, 0, tzinfo=timezone.utc),
        )
    finally:
        external_imports.default_sources = old_default_sources

    assert summary["run_count"] == 1
    assert summary["latest_fetch_at"].startswith("2026-01-04T10:00")
    assert repo.list_candidates()[0].doi == "10.1/immediate"
    assert repo.list_monitors()[0].next_run_at.startswith("2026-01-05")


def test_generate_daily_brief_processes_candidate_metadata(tmp_path: Path) -> None:
    repo = JsonExternalImportRepository(tmp_path)
    high = make_candidate(
        source="pubmed",
        title="Recombinant lactoferrin expression in Pichia pastoris",
        abstract="Pichia pastoris was engineered for heterologous recombinant lactoferrin expression and secretion.",
        doi="10.1/high",
        pmid="123",
        pmcid="PMC123",
        pdf_url="https://pdf.test/high.pdf",
        published_date="2026-06-01",
        is_open_access=True,
    )
    high.pdf_access_status = "verified"
    low = make_candidate(
        source="crossref",
        title="Older unrelated note",
        doi="10.1/low",
        published_date="2020-01-01",
    )
    repo.upsert_candidates([high, low])

    brief = external_imports.generate_daily_brief(
        brief_date=date(2026, 6, 5),
        lookback_days=14,
        max_items=10,
        repository=repo,
    )

    assert brief.title == "2026-06-05 每日简报"
    assert brief.total_candidates == 1
    assert brief.included_count == 1
    assert brief.items[0].candidate_id == high.candidate_id
    assert brief.items[0].priority == "high"
    assert brief.items[0].classification_id == "03"
    assert brief.items[0].classification_name == "毕赤酵母（乳铁蛋白/骨桥蛋白）"
    assert brief.items[0].classification_status == "rule_only"
    assert brief.items[0].recommended_action == "优先导入 PDF"
    assert repo.list_daily_briefs()[0].brief_id == brief.brief_id


def test_generate_daily_brief_reuses_stored_classification_without_llm(tmp_path: Path) -> None:
    repo = JsonExternalImportRepository(tmp_path)
    candidate = make_candidate(
        source="pubmed",
        title="Stored Y103 classification",
        abstract="This candidate was already classified by the monitor.",
        doi="10.1/stored-classification",
        published_date="2026-06-01",
    )
    candidate.classification_id = "03"
    candidate.classification_name = "毕赤酵母（乳铁蛋白/骨桥蛋白）"
    candidate.classification_confidence = 0.88
    candidate.classification_status = "rule_only"
    candidate.classification_reason = "自动定时搜索已分类"
    repo.upsert_candidates([candidate])

    class FailingClient:
        def complete(self, **kwargs):
            raise AssertionError("daily brief should reuse stored classification")

    brief = external_imports.generate_daily_brief(
        brief_date=date(2026, 6, 5),
        lookback_days=14,
        max_items=5,
        repository=repo,
        use_llm=True,
        small_llm_client=FailingClient(),
        judge_llm_client=FailingClient(),
    )

    item = brief.items[0]
    assert item.classification_id == "03"
    assert item.classification_name == "毕赤酵母（乳铁蛋白/骨桥蛋白）"
    assert item.classification_confidence == 0.88
    assert item.classification_reason == "自动定时搜索已分类"


def test_generate_daily_brief_uses_small_model_and_judge(tmp_path: Path) -> None:
    repo = JsonExternalImportRepository(tmp_path)
    candidate = make_candidate(
        source="pubmed",
        title="Signal peptide screening in Pichia pastoris",
        abstract="Signal peptide variants improve secretion in Pichia pastoris.",
        doi="10.1/signal",
        published_date="2026-06-01",
    )
    repo.upsert_candidates([candidate])
    calls = []

    class FakeClient:
        def complete(self, **kwargs):
            calls.append(kwargs)
            if kwargs["label"] == "daily-brief-small-classifier":
                return '{"category_id":"13","category_name":"毕赤酵母（蛋白表达）","confidence":0.61,"reason":"小模型按蛋白表达初判","key_evidence":["Pichia pastoris"]}'
            return '{"accepted":false,"category_id":"10","category_name":"毕赤酵母（信号肽）","confidence":0.88,"reason":"核心证据是signal peptide筛选","key_evidence":["Signal peptide"]}'

    brief = external_imports.generate_daily_brief(
        brief_date=date(2026, 6, 5),
        lookback_days=14,
        max_items=5,
        repository=repo,
        use_llm=True,
        small_llm_client=FakeClient(),
        judge_llm_client=FakeClient(),
    )

    item = brief.items[0]
    assert item.classification_id == "10"
    assert item.classification_name == "毕赤酵母（信号肽）"
    assert item.classification_status == "judge_revised"
    assert item.judge_reason == "核心证据是signal peptide筛选"
    assert {call["label"] for call in calls} == {"daily-brief-small-classifier", "daily-brief-judge"}


def test_list_daily_briefs_returns_latest_first(tmp_path: Path) -> None:
    repo = JsonExternalImportRepository(tmp_path)
    candidate = make_candidate(
        source="pubmed",
        title="Brief candidate",
        abstract="Fermentation paper.",
        doi="10.1/brief",
        published_date="2026-06-01",
    )
    repo.upsert_candidates([candidate])

    first = external_imports.generate_daily_brief(
        brief_date=date(2026, 6, 4),
        repository=repo,
    )
    second = external_imports.generate_daily_brief(
        brief_date=date(2026, 6, 5),
        repository=repo,
    )

    briefs = external_imports.list_daily_briefs(repository=repo)

    assert {brief.brief_id for brief in briefs[:2]} == {first.brief_id, second.brief_id}
    assert briefs[0].generated_at >= briefs[1].generated_at


def test_import_candidates_downloads_pdf_and_marks_imported(tmp_path: Path) -> None:
    repo = JsonExternalImportRepository(tmp_path)
    candidate = make_candidate(
        source="europe_pmc",
        title="Import me",
        doi="10.1/import",
        pdf_url="https://pdf.test/import.pdf",
    )
    repo.upsert_candidates([candidate])

    ingest_calls = []

    def fake_ingest(pdf_bytes, paper_id, filename, **kwargs):
        ingest_calls.append((pdf_bytes, paper_id, filename, kwargs))
        return {"store_result": {"stored": 2, "degraded": 0, "error": ""}}

    result = external_imports.import_candidates(
        [candidate.candidate_id],
        repository=repo,
        paper_id_builder=lambda data: "paper-1",
        ingest_runner=fake_ingest,
        paper_count_lookup=lambda paper_id: 0,
        pdf_downloader=FakeDownloader(),
    )

    stored = repo.get_candidate(candidate.candidate_id)
    assert result["success"] == 1
    assert stored.import_status == "imported"
    assert stored.imported_paper_id == "paper-1"
    assert stored.pdf_access_status == "verified"
    assert ingest_calls[0][3]["describe_figure_groups"] is True


def test_check_candidate_pdf_access_downloads_and_marks_verified(tmp_path: Path) -> None:
    repo = JsonExternalImportRepository(tmp_path)
    candidate = make_candidate(
        source="europe_pmc",
        title="Check me",
        doi="10.1/check",
        pdf_url="https://pdf.test/check.pdf",
    )
    repo.upsert_candidates([candidate])

    result = external_imports.check_candidate_pdf_access(
        [candidate.candidate_id],
        repository=repo,
        pdf_downloader=FakeDownloader(),
    )

    stored = repo.get_candidate(candidate.candidate_id)
    assert result["verified"] == 1
    assert stored.pdf_access_status == "verified"
    assert stored.pdf_access_error == ""
    assert stored.downloaded_pdf_path.endswith("paper.pdf")
    assert Path(stored.downloaded_pdf_path).exists()


def test_check_candidate_pdf_access_records_failed_validation(tmp_path: Path) -> None:
    repo = JsonExternalImportRepository(tmp_path)
    candidate = make_candidate(
        source="crossref",
        title="Broken pdf",
        doi="10.1/broken",
        pdf_url="https://pdf.test/broken.pdf",
    )
    repo.upsert_candidates([candidate])

    result = external_imports.check_candidate_pdf_access(
        [candidate.candidate_id],
        repository=repo,
        pdf_downloader=FailingDownloader(),
    )

    stored = repo.get_candidate(candidate.candidate_id)
    assert result["failed"] == 1
    assert stored.pdf_access_status == "unavailable"
    assert "not a pdf" in stored.pdf_access_error


def test_check_candidate_pdf_access_reports_progress(tmp_path: Path) -> None:
    repo = JsonExternalImportRepository(tmp_path)
    first = make_candidate(
        source="europe_pmc",
        title="Progress ok",
        doi="10.1/progress-ok",
        pdf_url="https://pdf.test/ok.pdf",
    )
    second = make_candidate(
        source="pubmed",
        title="Progress missing",
        doi="10.1/progress-missing",
    )
    repo.upsert_candidates([first, second])
    events = []

    result = external_imports.check_candidate_pdf_access(
        [first.candidate_id, second.candidate_id],
        repository=repo,
        pdf_downloader=FakeDownloader(),
        progress_callback=lambda event: events.append(event),
    )

    assert result["verified"] == 1
    assert result["missing_pdf_url"] == 1
    assert events[0]["event"] == "start"
    assert [event["event"] for event in events].count("candidate_done") == 2
    assert events[-1]["event"] == "done"
    assert events[-1]["completed"] == 2
    assert events[-1]["verified"] == 1
    assert events[-1]["missing_pdf_url"] == 1


def test_import_candidates_reuses_verified_local_pdf(tmp_path: Path) -> None:
    repo = JsonExternalImportRepository(tmp_path)
    local_pdf = tmp_path / "pdfs" / "cached.pdf"
    local_pdf.parent.mkdir(parents=True, exist_ok=True)
    local_pdf.write_bytes(b"%PDF-cached")
    candidate = make_candidate(
        source="europe_pmc",
        title="Cached import",
        doi="10.1/cached",
        pdf_url="https://pdf.test/cached.pdf",
    )
    candidate.pdf_access_status = "verified"
    candidate.downloaded_pdf_path = str(local_pdf)
    repo.upsert_candidates([candidate])
    ingest_calls = []

    def fake_ingest(pdf_bytes, paper_id, filename, **kwargs):
        ingest_calls.append(pdf_bytes)
        return {"store_result": {"stored": 1, "error": ""}}

    result = external_imports.import_candidates(
        [candidate.candidate_id],
        repository=repo,
        paper_id_builder=lambda data: "paper-cached",
        ingest_runner=fake_ingest,
        paper_count_lookup=lambda paper_id: 0,
        pdf_downloader=FailingDownloader(),
    )

    assert result["success"] == 1
    assert ingest_calls == [b"%PDF-cached"]


def test_import_candidates_skips_existing_pdf(tmp_path: Path) -> None:
    repo = JsonExternalImportRepository(tmp_path)
    candidate = make_candidate(
        source="europe_pmc",
        title="Duplicate",
        doi="10.1/dup",
        pdf_url="https://pdf.test/dup.pdf",
    )
    repo.upsert_candidates([candidate])

    result = external_imports.import_candidates(
        [candidate.candidate_id],
        repository=repo,
        paper_id_builder=lambda data: "paper-dup",
        ingest_runner=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ingest should not run")),
        paper_count_lookup=lambda paper_id: 5,
        pdf_downloader=FakeDownloader(),
    )

    stored = repo.get_candidate(candidate.candidate_id)
    assert result["duplicate"] == 1
    assert stored.import_status == "duplicate"
    assert stored.import_error == "该 PDF 已在向量库中"
