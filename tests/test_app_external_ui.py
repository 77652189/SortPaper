from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from src.domain.external_papers import DailyBrief, DailyBriefItem, MonitorProfile
from src.domain.external_papers import make_candidate


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


class _Progress:
    def __init__(self, owner) -> None:
        self.owner = owner

    def progress(self, value, text=None):
        self.owner.messages.append(("progress_update", {"value": value, "text": text}))


class _Placeholder:
    def __init__(self, owner) -> None:
        self.owner = owner

    def info(self, text):
        self.owner.messages.append(("placeholder_info", text))

    def success(self, text):
        self.owner.messages.append(("placeholder_success", text))

    def warning(self, text):
        self.owner.messages.append(("placeholder_warning", text))


class FakeStreamlit:
    def __init__(self) -> None:
        self.messages: list[tuple[str, object]] = []
        self.button_values: dict[str, bool] = {}
        self.button_inputs: list[dict] = []
        self.date_inputs: list[dict] = []
        self.text_input_values: dict[str, str] = {}
        self.text_inputs: list[dict] = []
        self.tabs_labels: list[str] = []
        self.selectbox_inputs: list[dict] = []
        self.selectbox_values: dict[str, object] = {}
        self.checkbox_values: dict[str, bool] = {}
        self.multiselect_values: dict[str, list] = {}
        self.session_state = {}

    def subheader(self, text):
        self.messages.append(("subheader", text))

    def text_input(self, *args, **kwargs):
        self.text_inputs.append({"args": args, "kwargs": kwargs})
        key = kwargs.get("key", "")
        if key in self.text_input_values:
            return self.text_input_values[key]
        label = args[0] if args else ""
        if label in self.text_input_values:
            return self.text_input_values[label]
        return ""

    def date_input(self, *args, **kwargs):
        self.date_inputs.append({"args": args, "kwargs": kwargs})
        return kwargs.get("value")

    def selectbox(self, *args, **kwargs):
        self.selectbox_inputs.append({"args": args, "kwargs": kwargs})
        self.messages.append(("selectbox", args[0] if args else kwargs.get("label", "")))
        key = kwargs.get("key", "")
        if key in self.selectbox_values:
            return self.selectbox_values[key]
        options = args[1] if len(args) > 1 else kwargs.get("options", [])
        return options[0] if options else ""

    def checkbox(self, *args, **kwargs):
        key = kwargs.get("key", "")
        if key in self.checkbox_values:
            return self.checkbox_values[key]
        return kwargs.get("value", False)

    def slider(self, *args, **kwargs):
        return args[3] if len(args) > 3 else kwargs.get("value")

    def multiselect(self, *args, **kwargs):
        key = kwargs.get("key", "")
        if key in self.multiselect_values:
            return self.multiselect_values[key]
        return kwargs.get("default", [])

    def caption(self, text):
        self.messages.append(("caption", text))

    def info(self, text):
        self.messages.append(("info", text))

    def dataframe(self, *args, **kwargs):
        self.messages.append(("dataframe", args[0]))

    def button(self, label, *args, **kwargs):
        self.button_inputs.append({"label": label, "args": args, "kwargs": kwargs})
        self.messages.append(("button", label))
        if kwargs.get("disabled"):
            return False
        key = kwargs.get("key", label)
        return self.button_values.get(key, False)

    def success(self, text):
        self.messages.append(("success", text))

    def warning(self, text):
        self.messages.append(("warning", text))

    def json(self, value):
        self.messages.append(("json", value))

    def spinner(self, text):
        self.messages.append(("spinner", text))
        return _Context()

    def progress(self, value, text=None):
        self.messages.append(("progress", {"value": value, "text": text}))
        return _Progress(self)

    def empty(self):
        self.messages.append(("empty", True))
        return _Placeholder(self)

    def expander(self, *args, **kwargs):
        self.messages.append(("expander", args[0] if args else ""))
        return _Context()

    def form(self, *args, **kwargs):
        self.messages.append(("form", args[0] if args else ""))
        return _Context()

    def markdown(self, text):
        self.messages.append(("markdown", text))

    def code(self, text):
        self.messages.append(("code", text))

    def number_input(self, *args, **kwargs):
        return kwargs.get("value", 0)

    def form_submit_button(self, *args, **kwargs):
        key = kwargs.get("key", args[0] if args else "submit")
        return self.button_values.get(key, False)

    def divider(self):
        self.messages.append(("divider", True))

    def rerun(self):
        self.messages.append(("rerun", True))

    def columns(self, spec):
        count = spec if isinstance(spec, int) else len(spec)
        return [_Context() for _ in range(count)]

    def tabs(self, labels):
        self.tabs_labels = list(labels)
        return [_Context() for _ in labels]


def test_candidate_library_renders_empty_state(monkeypatch) -> None:
    import app_external_ui

    fake_st = FakeStreamlit()
    monkeypatch.setattr(app_external_ui, "st", fake_st)
    monkeypatch.setattr(app_external_ui.external_imports, "list_candidates", lambda **kwargs: [])

    app_external_ui._render_candidate_library()

    assert ("info", "暂无候选论文。") in fake_st.messages


def test_candidate_row_exposes_article_and_pdf_links() -> None:
    import app_external_ui

    candidate = make_candidate(
        source="openalex",
        title="Linked paper",
        doi="10.1/linked",
        pmcid="PMC123",
        openalex_id="https://openalex.org/W123",
        landing_url="https://article.test/paper",
        pdf_url="https://article.test/paper.pdf",
        is_open_access=True,
    )
    candidate.pdf_access_status = "verified"
    candidate.downloaded_pdf_path = "data/external_imports/pdfs/paper.pdf"

    row = app_external_ui._candidate_row(candidate)

    assert row["原文链接"] == "https://article.test/paper"
    assert row["PDF链接"] == "https://article.test/paper.pdf"
    assert row["PDF状态"] == "已校验可下载"
    assert row["可直接导入"] == "是"
    assert row["本地PDF"] == "data/external_imports/pdfs/paper.pdf"
    assert row["PMCID"] == "PMC123"
    assert row["OpenAlex ID"] == "W123"


def test_candidate_library_imports_selected_pdf(monkeypatch) -> None:
    import app_external_ui

    candidate = make_candidate(
        source="europe_pmc",
        title="Importable paper",
        doi="10.1/ui",
        pdf_url="https://pdf.test/paper.pdf",
    )
    fake_st = FakeStreamlit()
    fake_st.multiselect_values["external_import_selected"] = [candidate.candidate_id]
    fake_st.button_values["external_import_button"] = True
    calls: dict[str, object] = {}

    monkeypatch.setattr(app_external_ui, "st", fake_st)
    monkeypatch.setattr(
        app_external_ui.external_imports,
        "list_candidates",
        lambda **kwargs: [candidate],
    )
    monkeypatch.setattr(
        app_external_ui.external_imports,
        "import_candidates_with_default_pipeline",
        lambda selected: (
            calls.update({"selected": selected}) or {
            "success": 1,
            "duplicate": 0,
            "failed": 0,
            "results": [{"candidate_id": selected[0], "status": "imported"}],
            }
        ),
    )

    app_external_ui._render_candidate_library()

    assert calls["selected"] == [candidate.candidate_id]
    assert any(kind == "success" and "成功 1" in str(text) for kind, text in fake_st.messages)


def test_candidate_library_pdf_precheck_stores_result_before_rerun(monkeypatch) -> None:
    import app_external_ui

    candidate = make_candidate(
        source="europe_pmc",
        title="Precheck paper",
        doi="10.1/precheck",
        pdf_url="https://pdf.test/precheck.pdf",
    )
    fake_st = FakeStreamlit()
    fake_st.multiselect_values["external_pdf_check_selected"] = [candidate.candidate_id]
    fake_st.button_values["external_pdf_check_button"] = True

    monkeypatch.setattr(app_external_ui, "st", fake_st)
    monkeypatch.setattr(
        app_external_ui.external_imports,
        "list_candidates",
        lambda **kwargs: [candidate],
    )
    monkeypatch.setattr(
        app_external_ui.external_imports,
        "check_candidate_pdf_access",
        lambda selected, progress_callback=None: (
            progress_callback and progress_callback({
                "event": "candidate_done",
                "completed": 1,
                "total": 1,
                "verified": 1,
                "failed": 0,
                "missing_pdf_url": 0,
                "title": "Precheck paper",
            })
        ) or {
            "verified": 1,
            "failed": 0,
            "missing_pdf_url": 0,
            "results": [{"candidate_id": selected[0], "status": "verified", "path": "pdfs/paper.pdf"}],
        },
    )

    app_external_ui._render_candidate_library()

    assert fake_st.session_state["external_pdf_check_result"]["verified"] == 1
    assert any(kind == "progress_update" for kind, _ in fake_st.messages)
    assert any(kind == "placeholder_info" and "已完成 1/1" in str(text) for kind, text in fake_st.messages)
    assert ("rerun", True) in fake_st.messages


def test_candidate_library_pdf_precheck_all_uses_current_downloadable_candidates(monkeypatch) -> None:
    import app_external_ui

    first = make_candidate(
        source="europe_pmc",
        title="First PDF",
        doi="10.1/first",
        pdf_url="https://pdf.test/first.pdf",
    )
    second = make_candidate(
        source="crossref",
        title="Second PDF",
        doi="10.1/second",
        pdf_url="https://pdf.test/second.pdf",
    )
    no_pdf = make_candidate(
        source="pubmed",
        title="No PDF",
        doi="10.1/no-pdf",
    )
    fake_st = FakeStreamlit()
    fake_st.button_values["external_pdf_check_all_button"] = True
    calls: dict[str, object] = {}

    monkeypatch.setattr(app_external_ui, "st", fake_st)
    monkeypatch.setattr(
        app_external_ui.external_imports,
        "list_candidates",
        lambda **kwargs: [first, second, no_pdf],
    )
    monkeypatch.setattr(
        app_external_ui.external_imports,
        "check_candidate_pdf_access",
        lambda selected, progress_callback=None: (
            calls.update({"selected": selected}) or {
                "verified": len(selected),
                "failed": 0,
                "missing_pdf_url": 0,
                "results": [],
            }
        ),
    )

    app_external_ui._render_candidate_library()

    assert calls["selected"] == [first.candidate_id, second.candidate_id]
    assert fake_st.session_state["external_pdf_check_result"]["requested_count"] == 2
    assert ("rerun", True) in fake_st.messages


def test_candidate_library_shows_persisted_pdf_precheck_result(monkeypatch) -> None:
    import app_external_ui

    candidate = make_candidate(
        source="crossref",
        title="Failed precheck paper",
        doi="10.1/failed",
        pdf_url="https://pdf.test/failed.pdf",
    )
    candidate.pdf_access_status = "unavailable"
    candidate.pdf_access_error = "HTTPError: 403 Client Error"
    fake_st = FakeStreamlit()
    fake_st.session_state["external_pdf_check_result"] = {
        "verified": 0,
        "failed": 1,
        "missing_pdf_url": 0,
        "results": [{"candidate_id": candidate.candidate_id, "status": "failed", "error": "HTTPError: 403 Client Error"}],
    }

    monkeypatch.setattr(app_external_ui, "st", fake_st)
    monkeypatch.setattr(
        app_external_ui.external_imports,
        "list_candidates",
        lambda **kwargs: [candidate],
    )

    app_external_ui._render_candidate_library()

    assert any(kind == "warning" and "失败 1" in str(text) for kind, text in fake_st.messages)
    summary_tables = [
        value for kind, value in fake_st.messages
        if kind == "dataframe"
        and isinstance(value, list)
        and value
        and "结果" in value[0]
    ]
    assert summary_tables
    assert summary_tables[0][0]["结果"] == "失败"
    assert "错误" not in summary_tables[0][0]
    assert ("code", "HTTPError: 403 Client Error") in fake_st.messages


def test_pdf_precheck_detail_rows_include_full_links_and_paths(monkeypatch) -> None:
    import app_external_ui

    fake_st = FakeStreamlit()
    monkeypatch.setattr(app_external_ui, "st", fake_st)

    app_external_ui._render_pdf_check_result_items([
        {
            "标题": "Downloaded paper",
            "结果": "可访问",
            "本地PDF": r"data\external_imports\pdfs\downloaded.pdf",
            "大小": "2.28 MB",
            "错误": "",
            "PDF链接": "https://pdf.test/downloaded.pdf",
            "原文链接": "https://article.test/downloaded",
        }
    ])

    assert ("code", r"data\external_imports\pdfs\downloaded.pdf") in fake_st.messages
    assert any(
        kind == "markdown" and "PDF链接：[打开 PDF](https://pdf.test/downloaded.pdf)" in str(text)
        for kind, text in fake_st.messages
    )


def test_daily_brief_generates_and_persists_selection(monkeypatch) -> None:
    import app_external_ui

    fake_st = FakeStreamlit()
    fake_st.button_values["external_daily_brief_generate"] = True
    brief = DailyBrief(
        brief_id="brief-1",
        brief_date="2026-06-05",
        generated_at="2026-06-05T00:00:00+00:00",
        title="2026-06-05 每日简报",
        total_candidates=2,
        included_count=1,
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr(app_external_ui, "st", fake_st)
    monkeypatch.setattr(
        app_external_ui.external_imports,
        "generate_daily_brief",
        lambda **kwargs: calls.update(kwargs) or brief,
    )
    monkeypatch.setattr(app_external_ui.external_imports, "list_daily_briefs", lambda: [brief])
    monkeypatch.setattr(
        app_external_ui.external_imports,
        "latest_paper_fetch_status",
        lambda: {"latest_fetch_at": "", "ran_today": False},
    )

    app_external_ui._render_daily_brief()

    assert calls["brief_date"] == date.today()
    assert calls["query"] == ""
    assert calls["use_llm"] is False
    assert fake_st.date_inputs == []
    assert not any(item["args"] and item["args"][0] == "主题筛选" for item in fake_st.text_inputs)
    assert fake_st.session_state["external_daily_brief_id"] == "brief-1"
    assert any(kind == "success" and "已生成" in str(text) for kind, text in fake_st.messages)
    assert ("rerun", True) in fake_st.messages


def test_daily_brief_shows_latest_paper_fetch_time(monkeypatch) -> None:
    import app_external_ui

    fake_st = FakeStreamlit()
    monkeypatch.setattr(app_external_ui, "st", fake_st)
    monkeypatch.setattr(app_external_ui.external_imports, "list_daily_briefs", lambda: [])
    monkeypatch.setattr(
        app_external_ui.external_imports,
        "latest_paper_fetch_status",
        lambda: {"latest_fetch_at": "2026-06-05T00:00:00+00:00", "ran_today": True},
    )

    app_external_ui._render_daily_brief()

    assert any(
        kind == "info" and "2026-06-05 08:00:00" in str(text) and "|" in str(text)
        for kind, text in fake_st.messages
    )


def test_daily_brief_can_auto_fetch_latest_papers(monkeypatch) -> None:
    import app_external_ui

    fake_st = FakeStreamlit()
    fake_st.button_values["external_daily_brief_run_due_monitors"] = True
    calls = {"run_enabled_monitors_now": 0}

    monkeypatch.setattr(app_external_ui, "st", fake_st)
    monkeypatch.setattr(app_external_ui.external_imports, "list_daily_briefs", lambda: [])
    monkeypatch.setattr(
        app_external_ui.external_imports,
        "latest_paper_fetch_status",
        lambda: {"latest_fetch_at": "", "ran_today": False},
    )

    def run_enabled_monitors_now():
        calls["run_enabled_monitors_now"] += 1
        return {"run_count": 2, "summaries": [{"new_count": 3}]}

    monkeypatch.setattr(app_external_ui.external_imports, "run_enabled_monitors_now", run_enabled_monitors_now)

    app_external_ui._render_daily_brief()

    assert calls["run_enabled_monitors_now"] == 1
    assert ("success", "已立即运行 2 个启用主题") in fake_st.messages
    assert ("json", [{"new_count": 3}]) in fake_st.messages


def test_daily_brief_fetch_button_shows_today_status_and_avoids_repeat(monkeypatch) -> None:
    import app_external_ui

    fake_st = FakeStreamlit()
    fake_st.button_values["external_daily_brief_run_due_monitors"] = True
    calls = {"run_enabled_monitors_now": 0}

    monkeypatch.setattr(app_external_ui, "st", fake_st)
    monkeypatch.setattr(app_external_ui.external_imports, "list_daily_briefs", lambda: [])
    monkeypatch.setattr(
        app_external_ui.external_imports,
        "latest_paper_fetch_status",
        lambda: {"latest_fetch_at": "2026-06-05T00:00:00+00:00", "ran_today": True},
    )
    monkeypatch.setattr(
        app_external_ui.external_imports,
        "run_enabled_monitors_now",
        lambda: calls.update({"run_enabled_monitors_now": calls["run_enabled_monitors_now"] + 1}) or {"run_count": 1, "summaries": []},
    )

    app_external_ui._render_daily_brief()

    assert calls["run_enabled_monitors_now"] == 0
    assert ("success", "✅ 今日已运行") in fake_st.messages
    assert any(
        item["label"] == "自动获取最新论文" and item["kwargs"].get("disabled") is True
        for item in fake_st.button_inputs
    )


def test_external_import_tab_defaults_to_daily_brief_first(monkeypatch) -> None:
    import app_external_ui

    fake_st = FakeStreamlit()
    render_order: list[str] = []

    monkeypatch.setattr(app_external_ui, "st", fake_st)
    monkeypatch.setattr(app_external_ui, "_auto_run_due_monitors_once", lambda: None)
    monkeypatch.setattr(app_external_ui, "_render_daily_brief", lambda: render_order.append("daily"))
    monkeypatch.setattr(app_external_ui, "_render_candidate_library", lambda: render_order.append("candidates"))
    monkeypatch.setattr(app_external_ui, "_render_manual_refresh", lambda: render_order.append("manual"))
    monkeypatch.setattr(app_external_ui, "_render_monitors", lambda: render_order.append("topics"))

    app_external_ui.render_external_import_tab()

    assert fake_st.tabs_labels == ["🗞️ 每日简报", "📥 候选库", "🔎 手动获取", "⚙️ 设置每日简报主题"]
    assert render_order == ["daily", "candidates", "manual", "topics"]


def test_auto_run_due_monitors_checks_once_per_beijing_day(monkeypatch) -> None:
    import app_external_ui

    fake_st = FakeStreamlit()
    fake_st.session_state["external_due_monitors_checked_date"] = "2026-06-07"
    calls = {"run_due_monitors": 0}

    monkeypatch.setattr(app_external_ui, "st", fake_st)
    monkeypatch.setattr(app_external_ui, "EXTERNAL_MONITOR_AUTO_RUN", True)
    monkeypatch.setattr(app_external_ui, "_beijing_today_key", lambda: "2026-06-08")

    def run_due_monitors():
        calls["run_due_monitors"] += 1
        return {"run_count": 1, "summaries": []}

    monkeypatch.setattr(app_external_ui.external_imports, "run_due_monitors", run_due_monitors)

    app_external_ui._auto_run_due_monitors_once()
    app_external_ui._auto_run_due_monitors_once()

    assert calls["run_due_monitors"] == 1
    assert fake_st.session_state["external_due_monitors_checked_date"] == "2026-06-08"
    assert ("info", "已自动运行 1 个到期每日简报主题。") in fake_st.messages


def test_auto_run_due_monitors_skips_after_today_check(monkeypatch) -> None:
    import app_external_ui

    fake_st = FakeStreamlit()
    fake_st.session_state["external_due_monitors_checked_date"] = "2026-06-08"
    calls = {"run_due_monitors": 0}

    monkeypatch.setattr(app_external_ui, "st", fake_st)
    monkeypatch.setattr(app_external_ui, "EXTERNAL_MONITOR_AUTO_RUN", True)
    monkeypatch.setattr(app_external_ui, "_beijing_today_key", lambda: "2026-06-08")
    monkeypatch.setattr(
        app_external_ui.external_imports,
        "run_due_monitors",
        lambda: calls.update({"run_due_monitors": calls["run_due_monitors"] + 1}) or {"run_count": 1, "summaries": []},
    )

    app_external_ui._auto_run_due_monitors_once()

    assert calls["run_due_monitors"] == 0


def test_daily_brief_item_row_labels_priority_and_pdf_status() -> None:
    import app_external_ui

    item = DailyBriefItem(
        candidate_id="c1",
        title="Lactoferrin trial",
        published_date="2026-06-01",
        sources=["pubmed", "openalex"],
        doi="10.1/x",
        pdf_access_status="verified",
        topics=["毕赤酵母（乳铁蛋白/骨桥蛋白）"],
        classification_id="03",
        classification_name="毕赤酵母（乳铁蛋白/骨桥蛋白）",
        classification_confidence=0.86,
        classification_status="judge_confirmed",
        priority="high",
        recommended_action="优先导入 PDF",
        evidence_level="研究论文/预印本",
    )

    row = app_external_ui._daily_brief_item_row(item)

    assert row["重点"] == "🔥 高"
    assert row["Y103分类"] == "毕赤酵母（乳铁蛋白/骨桥蛋白）"
    assert row["PDF"] == "已校验可下载"
    assert row["来源"] == "PubMed, OpenAlex"
    assert "DOI" not in row
    assert "置信度" not in row


def test_daily_brief_content_lists_items_with_visible_detail_buttons(monkeypatch) -> None:
    import app_external_ui

    fake_st = FakeStreamlit()
    fake_st.button_values["external_daily_brief_detail_button_brief-1_03_1_c1"] = True
    monkeypatch.setattr(app_external_ui, "st", fake_st)
    item = DailyBriefItem(
        candidate_id="c1",
        title="Pichia lactoferrin expression",
        sources=["pubmed"],
        published_date="2026-06-01",
        pdf_access_status="verified",
        classification_id="03",
        classification_name="毕赤酵母（乳铁蛋白/骨桥蛋白）",
        classification_confidence=0.88,
        classification_status="rule_only",
        classification_reason="自动定时搜索已分类",
        priority="high",
        brief="候选摘要。",
        recommended_action="优先导入 PDF",
    )
    brief = DailyBrief(
        brief_id="brief-1",
        brief_date="2026-06-05",
        generated_at="2026-06-05T00:00:00+00:00",
        title="2026-06-05 每日简报",
        total_candidates=1,
        included_count=1,
        topic_counts={"毕赤酵母（乳铁蛋白/骨桥蛋白）": 1},
        source_counts={"pubmed": 1},
        pdf_counts={"PDF可访问": 1},
        items=[item],
    )

    app_external_ui._render_daily_brief_content(brief)

    list_index = next(
        i for i, message in enumerate(fake_st.messages)
        if message[0] == "markdown" and message[1] == "**📚 按 Y103 16 主题展示**"
    )
    button_index = next(i for i, message in enumerate(fake_st.messages) if message == ("button", "🔎 查看详情"))
    detail_index = next(
        i for i, message in enumerate(fake_st.messages)
        if message[0] == "markdown" and str(message[1]).startswith("**🔎 1/1")
    )
    assert list_index < button_index < detail_index
    assert not any(item["args"] and item["args"][0] == "选择条目" for item in fake_st.selectbox_inputs)
    assert any(item["kwargs"].get("type") == "primary" for item in fake_st.button_inputs if item["label"] == "🔎 查看详情")
    reason_tables = [
        value for kind, value in fake_st.messages
        if kind == "dataframe"
        and isinstance(value, list)
        and value
        and value[0].get("类型") == "摘要信息"
    ]
    assert reason_tables
    assert reason_tables[0][0]["内容"] == "候选摘要。"
    assert any(kind == "success" and "📌 今日需看 1 条" in str(text) for kind, text in fake_st.messages)


def test_daily_brief_content_groups_items_by_y103_topics_with_duplicate_display(monkeypatch) -> None:
    import app_external_ui

    fake_st = FakeStreamlit()
    monkeypatch.setattr(app_external_ui, "st", fake_st)
    item = DailyBriefItem(
        candidate_id="c1",
        title="Pichia pastoris recombinant lactoferrin protein expression",
        sources=["pubmed"],
        published_date="2026-06-01",
        pdf_access_status="verified",
        classification_id="03",
        classification_name="毕赤酵母（乳铁蛋白/骨桥蛋白）",
        classification_confidence=0.88,
        classification_status="rule_only",
        classification_reason="规则命中主分类",
        priority="high",
        brief="Pichia pastoris was engineered for recombinant lactoferrin protein expression and production.",
        recommended_action="优先导入 PDF",
    )
    brief = DailyBrief(
        brief_id="brief-1",
        brief_date="2026-06-05",
        generated_at="2026-06-05T00:00:00+00:00",
        title="2026-06-05 每日简报",
        total_candidates=1,
        included_count=1,
        source_counts={"pubmed": 1},
        pdf_counts={"PDF可访问": 1},
        items=[item],
    )

    app_external_ui._render_daily_brief_content(brief)

    expander_labels = [text for kind, text in fake_st.messages if kind == "expander"]
    assert any("03 毕赤酵母（乳铁蛋白/骨桥蛋白） | 1 条" in str(label) for label in expander_labels)
    assert any("13 毕赤酵母（蛋白表达） | 1 条" in str(label) for label in expander_labels)
    assert any(kind == "caption" and "重复展示" in str(text) for kind, text in fake_st.messages)
    assert sum(1 for kind, text in fake_st.messages if kind == "button" and text == "🔎 查看详情") >= 2


def test_y103_category_rows_always_show_all_16_categories() -> None:
    import app_external_ui

    item = DailyBriefItem(
        candidate_id="c1",
        title="Pichia lactoferrin expression",
        classification_id="03",
        classification_name="毕赤酵母（乳铁蛋白/骨桥蛋白）",
        priority="high",
    )
    brief = DailyBrief(
        brief_id="brief-1",
        brief_date="2026-06-05",
        generated_at="2026-06-05T00:00:00+00:00",
        title="2026-06-05 每日简报",
        items=[item],
    )

    rows = app_external_ui._y103_category_rows(brief)

    assert len(rows) == 16
    hit = next(row for row in rows if row["编号"] == "03")
    empty = next(row for row in rows if row["编号"] == "01")
    assert hit["今日状态"] == "需看"
    assert hit["命中条数"] == 1
    assert hit["高优先级"] == 1
    assert empty["今日状态"] == "无"


def test_daily_brief_attention_makes_need_to_read_obvious(monkeypatch) -> None:
    import app_external_ui

    fake_st = FakeStreamlit()
    monkeypatch.setattr(app_external_ui, "st", fake_st)
    brief = DailyBrief(
        brief_id="brief-1",
        brief_date="2026-06-05",
        generated_at="2026-06-05T00:00:00+00:00",
        title="2026-06-05 每日简报",
        items=[
            DailyBriefItem(
                candidate_id="c1",
                title="Pichia lactoferrin expression",
                classification_id="03",
                classification_name="毕赤酵母（乳铁蛋白/骨桥蛋白）",
                priority="high",
            )
        ],
    )

    app_external_ui._render_daily_brief_attention(brief)

    assert any(kind == "success" and "今日有 1 条 Y103 相关候选需要看" in str(text) for kind, text in fake_st.messages)


def test_daily_brief_attention_reports_no_y103_hits(monkeypatch) -> None:
    import app_external_ui

    fake_st = FakeStreamlit()
    monkeypatch.setattr(app_external_ui, "st", fake_st)
    brief = DailyBrief(
        brief_id="brief-1",
        brief_date="2026-06-05",
        generated_at="2026-06-05T00:00:00+00:00",
        title="2026-06-05 每日简报",
        items=[DailyBriefItem(candidate_id="c1", title="Other paper", classification_id="other")],
    )

    app_external_ui._render_daily_brief_attention(brief)

    assert ("info", "今日暂无命中 Y103 16 分类的候选。") in fake_st.messages


def test_monitor_time_formats_as_beijing_time() -> None:
    import app_external_ui

    formatted = app_external_ui._format_monitor_time("2999-01-01T00:00:00+00:00")

    assert formatted == "2999-01-01 08:00:00 北京时间"


def test_manual_date_bounds_allow_pre_2016_and_cap_to_current_year(monkeypatch) -> None:
    import app_external_ui

    fake_st = FakeStreamlit()
    monkeypatch.setattr(app_external_ui, "st", fake_st)

    app_external_ui._render_manual_refresh()

    assert len(fake_st.date_inputs) == 2
    for item in fake_st.date_inputs:
        assert item["kwargs"]["min_value"] == date(1900, 1, 1)
        assert item["kwargs"]["max_value"] == date(date.today().year, 12, 31)


def test_manual_date_error_rejects_future_year() -> None:
    import app_external_ui

    error = app_external_ui._manual_date_error(
        date(2026, 1, 1),
        date(2027, 1, 1),
        today=date(2026, 6, 5),
    )

    assert error == "日期不能晚于 2026 年"


def test_manual_date_error_rejects_reversed_range() -> None:
    import app_external_ui

    error = app_external_ui._manual_date_error(
        date(2026, 6, 1),
        date(2026, 5, 1),
        today=date(2026, 6, 5),
    )

    assert error == "开始日期不能晚于结束日期"


def test_monitor_row_includes_explicit_next_run_time() -> None:
    import app_external_ui

    monitor = MonitorProfile(
        monitor_id="mon-1",
        name="HMO weekly",
        query="HMO fermentation",
        sources=["pubmed", "crossref"],
        next_run_at="2999-01-01T00:00:00+00:00",
        last_run_at="",
    )

    row = app_external_ui._monitor_row(monitor)

    assert row["下次运行时间"] == "2999-01-01 08:00:00 北京时间"
    assert row["上次运行时间"] == "尚未运行"
    assert row["来源"] == "PubMed, Crossref"
    assert row["过滤规则"] == "不过滤"


def test_monitors_render_current_state_before_collapsed_create_form(monkeypatch) -> None:
    import app_external_ui

    fake_st = FakeStreamlit()
    monitor = MonitorProfile(
        monitor_id="mon-1",
        name="hLF",
        query="hLF",
        sources=["pubmed"],
        next_run_at="2999-01-01T00:00:00+00:00",
    )

    monkeypatch.setattr(app_external_ui, "st", fake_st)
    monkeypatch.setattr(app_external_ui.external_imports, "list_monitors", lambda: [monitor])

    app_external_ui._render_monitors()

    dataframe_index = next(i for i, item in enumerate(fake_st.messages) if item[0] == "dataframe")
    create_index = next(i for i, item in enumerate(fake_st.messages) if item == ("expander", "新建每日简报主题"))
    assert dataframe_index < create_index
    assert ("subheader", "设置每日简报主题") in fake_st.messages


def test_monitor_row_shows_y103_filter_for_existing_y103_topic() -> None:
    import app_external_ui

    monitor = MonitorProfile(
        monitor_id="mon-y103",
        name="Y103-03 毕赤酵母（乳铁蛋白/骨桥蛋白）",
        query="Pichia pastoris AND lactoferrin",
        sources=["pubmed"],
    )

    row = app_external_ui._monitor_row(monitor)

    assert row["过滤规则"] == "Y103 16分类过滤"
    assert "候选库默认隐藏" in row["过滤行为"]


def test_monitors_can_save_filter_config(monkeypatch) -> None:
    import app_external_ui

    fake_st = FakeStreamlit()
    fake_st.selectbox_values["external_monitor_filter_type_mon-1"] = "y103"
    fake_st.checkbox_values["external_monitor_filter_include_other_mon-1"] = False
    fake_st.button_values["external_monitor_filter_save_mon-1"] = True
    monitor = MonitorProfile(
        monitor_id="mon-1",
        name="General topic",
        query="lactoferrin",
        sources=["pubmed"],
        filter_config={"enabled": False, "filter_type": "none", "include_other": True},
    )
    updated = []

    monkeypatch.setattr(app_external_ui, "st", fake_st)
    monkeypatch.setattr(app_external_ui.external_imports, "list_monitors", lambda: [monitor])
    monkeypatch.setattr(app_external_ui.external_imports, "update_monitor", lambda item: updated.append(item))

    app_external_ui._render_monitors()

    assert updated
    assert updated[-1].filter_config == {
        "enabled": True,
        "filter_type": "y103",
        "include_other": False,
    }
    assert ("success", "已保存筛选标准") in fake_st.messages


def test_new_monitor_auto_filter_uses_submitted_y103_name(monkeypatch) -> None:
    import app_external_ui

    fake_st = FakeStreamlit()
    fake_st.text_input_values["名称"] = "Y103 new topic"
    fake_st.text_input_values["检索式 / 主题"] = "Pichia pastoris AND lactoferrin"
    fake_st.button_values["保存主题"] = True
    calls: dict[str, object] = {}

    monkeypatch.setattr(app_external_ui, "st", fake_st)
    monkeypatch.setattr(app_external_ui.external_imports, "list_monitors", lambda: [])

    def create_monitor(**kwargs):
        calls.update(kwargs)
        return MonitorProfile(
            monitor_id="mon-new",
            name=kwargs["name"],
            query=kwargs["query"],
            filter_config=kwargs["filter_config"],
        )

    monkeypatch.setattr(app_external_ui.external_imports, "create_monitor", create_monitor)

    app_external_ui._render_monitors()

    assert calls["filter_config"] == {
        "enabled": True,
        "filter_type": "y103",
        "include_other": False,
    }
    assert ("rerun", True) in fake_st.messages
