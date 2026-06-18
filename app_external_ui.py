from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import streamlit as st

from app_config import EXTERNAL_MONITOR_AUTO_RUN
from src.application import external_imports
from src.application.external_brief_topic_groups import (
    daily_brief_topic_groups,
    daily_brief_topic_summary,
)


SOURCE_LABELS = {
    "pubmed": "PubMed",
    "europe_pmc": "Europe PMC",
    "openalex": "OpenAlex",
    "biorxiv": "bioRxiv",
    "medrxiv": "medRxiv",
    "crossref": "Crossref",
}
STATUS_LABELS = {
    "candidate": "候选",
    "importing": "导入中",
    "imported": "已入库",
    "duplicate": "已存在",
    "failed": "失败",
    "skipped": "跳过",
}
PDF_ACCESS_LABELS = {
    "unknown": "待校验链接",
    "checking": "校验中",
    "verified": "已校验可下载",
    "unavailable": "不可访问",
}
BEIJING_TZ = timezone(timedelta(hours=8), "Asia/Shanghai")


def render_external_import_tab() -> None:
    st.caption("从 PubMed / Europe PMC / OpenAlex / bioRxiv / medRxiv / Crossref 获取候选论文；仅开放 PDF 可进入 MinerU 入库。")
    _auto_run_due_monitors_once()

    tabs = st.tabs(["🗞️ 每日简报", "📥 候选库", "🔎 手动获取", "⚙️ 设置每日简报主题"])
    with tabs[0]:
        _render_daily_brief()
    with tabs[1]:
        _render_candidate_library()
    with tabs[2]:
        _render_manual_refresh()
    with tabs[3]:
        _render_monitors()


def render_daily_brief_view() -> None:
    _auto_run_due_monitors_once()
    _render_daily_brief()


def render_candidate_library_view() -> None:
    _auto_run_due_monitors_once()
    _render_candidate_library()


def render_manual_discovery_view() -> None:
    _auto_run_due_monitors_once()
    _render_manual_refresh()


def render_topics_view() -> None:
    _auto_run_due_monitors_once()
    _render_monitors()


def _render_manual_refresh() -> None:
    st.subheader("手动获取候选论文")
    query = st.text_input(
        "检索式 / 主题",
        placeholder="e.g. lacto-N-tetraose biosynthesis Escherichia coli fermentation",
        key="external_manual_query",
    )
    today = date.today()
    min_external_date = _manual_min_date()
    max_external_date = _manual_max_date(today)
    default_start = max(today - timedelta(days=14), min_external_date)
    col_a, col_b = st.columns(2)
    with col_a:
        from_date = st.date_input(
            "开始日期",
            value=default_start,
            min_value=min_external_date,
            max_value=max_external_date,
            key="external_manual_from",
        )
    with col_b:
        until_date = st.date_input(
            "结束日期",
            value=min(today, max_external_date),
            min_value=min_external_date,
            max_value=max_external_date,
            key="external_manual_until",
        )
    sources = st.multiselect(
        "来源",
        options=external_imports.DEFAULT_SOURCE_NAMES,
        default=external_imports.DEFAULT_SOURCE_NAMES,
        format_func=lambda value: SOURCE_LABELS.get(value, value),
        key="external_manual_sources",
    )
    max_results = st.slider("每个来源最多返回", 1, 100, 20, key="external_manual_max")
    if st.button("刷新候选", type="primary", key="external_manual_refresh"):
        if not query.strip():
            st.warning("请输入检索式或主题")
            return
        validation_error = _manual_date_error(from_date, until_date, today=today)
        if validation_error:
            st.warning(validation_error)
            return
        progress = st.progress(0.0, "准备查询外部来源...")
        status = st.empty()
        source_lines: dict[str, str] = {
            source: f"等待 {SOURCE_LABELS.get(source, source)}"
            for source in sources
        }

        def update_progress(event: dict) -> None:
            source = event.get("source", "")
            source_label = SOURCE_LABELS.get(source, source)
            total = int(event.get("total_sources") or max(len(sources), 1))
            completed = int(event.get("completed_sources") or 0)
            if event.get("event") == "source_start" and source:
                source_lines[source] = f"查询中 {source_label}"
            elif event.get("event") == "source_done":
                source_lines[source] = f"完成 {source_label}: {event.get('count', 0)} 条"
            elif event.get("event") == "source_error":
                source_lines[source] = f"失败 {source_label}: {event.get('error', '')}"
            elif event.get("event") == "merge_start":
                progress.progress(0.95, "正在合并候选...")
                status.info("正在合并候选与本地候选库...")
                return
            elif event.get("event") == "merge_done":
                progress.progress(1.0, "外源查询完成")
                status.success(
                    f"外源查询完成：获取 {event.get('fetched_count', 0)} 条，"
                    f"新增 {event.get('new_count', 0)} 条，更新 {event.get('updated_count', 0)} 条"
                )
                return
            progress.progress(
                min(0.9, completed / max(total, 1)),
                f"外源查询 {completed}/{total}",
            )
            status.info("\n".join(source_lines.values()))

        summary = external_imports.refresh_candidates(
            query,
            source_names=sources,
            from_date=from_date.isoformat(),
            until_date=until_date.isoformat(),
            max_results_per_source=max_results,
            progress_callback=update_progress,
        )
        _render_refresh_summary(summary)


def _render_candidate_library() -> None:
    st.subheader("候选库")
    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        query = st.text_input("库内筛选", placeholder="标题 / 摘要 / DOI / 作者", key="external_candidate_query")
    with col_b:
        status = st.selectbox(
            "状态",
            [""] + list(STATUS_LABELS.keys()),
            format_func=lambda value: "全部" if not value else STATUS_LABELS.get(value, value),
            key="external_candidate_status",
        )
    with col_c:
        only_pdf = st.checkbox("仅有 PDF", value=False, key="external_candidate_pdf")
    sources = st.multiselect(
        "来源筛选",
        options=external_imports.DEFAULT_SOURCE_NAMES,
        default=[],
        format_func=lambda value: SOURCE_LABELS.get(value, value),
        key="external_candidate_sources",
    )
    candidates = external_imports.list_candidates(
        source_names=sources,
        status=status,
        only_with_pdf=only_pdf,
        query=query,
    )
    if not candidates:
        st.info("暂无候选论文。")
        return

    _render_pdf_check_result(candidates)
    st.caption(f"共 {len(candidates)} 条候选")
    _render_candidate_translation_maintenance(candidates)
    rows = [_candidate_row(candidate) for candidate in candidates]
    st.dataframe(rows, width="stretch", hide_index=True)
    _render_candidate_link_details(candidates)

    downloadable = [candidate for candidate in candidates if candidate.pdf_url]
    selected_check_ids = st.multiselect(
        "选择要预检下载到本地的 PDF",
        options=[candidate.candidate_id for candidate in downloadable],
        format_func=lambda candidate_id: _candidate_label(downloadable, candidate_id),
        key="external_pdf_check_selected",
    )
    all_check_ids = [candidate.candidate_id for candidate in downloadable]
    col_selected_check, col_all_check = st.columns(2)
    with col_selected_check:
        if st.button("预检选中 PDF 到本地", disabled=not selected_check_ids, key="external_pdf_check_button"):
            _run_pdf_precheck(selected_check_ids, label="正在请求选中 PDF 链接并校验内容...")
    with col_all_check:
        if st.button(
            f"全部预检并下载当前列表 PDF（{len(all_check_ids)}）",
            disabled=not all_check_ids,
            key="external_pdf_check_all_button",
        ):
            _run_pdf_precheck(all_check_ids, label="正在请求当前列表全部 PDF 链接并校验内容...")

    importable = [
        candidate
        for candidate in candidates
        if candidate.pdf_url and candidate.import_status in {"candidate", "failed", "duplicate"}
    ]
    selected_ids = st.multiselect(
        "选择要导入的开放 PDF",
        options=[candidate.candidate_id for candidate in importable],
        format_func=lambda candidate_id: _candidate_label(importable, candidate_id),
        key="external_import_selected",
    )
    if st.button("导入选中 PDF 到 Qdrant", disabled=not selected_ids, type="primary", key="external_import_button"):
        with st.spinner("正在下载 PDF 并调用 MinerU 入库..."):
            result = external_imports.import_candidates_with_default_pipeline(selected_ids)
        st.success(
            f"导入完成：成功 {result['success']}，已存在 {result['duplicate']}，失败 {result['failed']}"
        )
        with st.expander("查看导入详情", expanded=bool(result["failed"])):
            st.json(result["results"])
        st.rerun()


def _render_candidate_translation_maintenance(candidates: list) -> None:
    candidate_ids = _candidate_translation_candidate_ids(candidates)
    config_status = external_imports.abstract_translation_config_status()
    _render_candidate_translation_result(config_status=config_status)
    translated_count = sum(
        1
        for candidate in candidates
        if str(getattr(candidate, "abstract_translation_zh", "") or "").strip()
    )
    pending_count = len(candidate_ids)
    col_text, col_action = st.columns([3, 1])
    with col_text:
        st.markdown("**摘要翻译维护**")
        st.caption(
            f"当前列表已翻译 {translated_count} 条，待补翻译 {pending_count} 条。"
            " 已翻译会自动跳过，失败会记录错误，不会重复消耗。"
        )
        if not config_status.get("ready"):
            st.warning(str(config_status.get("error") or "摘要翻译配置未就绪。"))
    with col_action:
        if st.button(
            f"补翻译当前候选摘要（{pending_count}）",
            disabled=not candidate_ids or not bool(config_status.get("ready")),
            type="primary",
            key="external_candidate_translate_missing",
        ):
            with st.spinner("正在补翻译当前候选摘要..."):
                result = external_imports.ensure_candidate_abstract_translations(candidate_ids)
            st.session_state["external_candidate_translation_result"] = result
            st.rerun()


def _render_candidate_translation_result(config_status: dict | None = None) -> None:
    result = st.session_state.get("external_candidate_translation_result")
    if not result:
        return
    if config_status and not config_status.get("ready") and not result.get("configuration_error"):
        return
    config_error = str(result.get("configuration_error") or "")
    if config_error:
        st.warning(f"摘要补翻译未执行：{config_error}")
        return
    translated = int(result.get("translated_count") or 0)
    skipped = int(result.get("skipped_count") or 0)
    failed = int(result.get("failed_count") or 0)
    message = f"摘要补翻译完成：新增 {translated}，已跳过 {skipped}，失败 {failed}"
    if failed:
        st.warning(message)
    else:
        st.success(message)
    errors = result.get("errors") or []
    if errors:
        with st.expander("查看翻译错误", expanded=True):
            st.json(errors)


def _candidate_translation_candidate_ids(candidates: list) -> list[str]:
    ids: list[str] = []
    for candidate in candidates:
        if getattr(candidate, "import_status", "") == "skipped":
            continue
        if not str(getattr(candidate, "abstract", "") or "").strip():
            continue
        if str(getattr(candidate, "abstract_translation_zh", "") or "").strip():
            continue
        candidate_id = str(getattr(candidate, "candidate_id", "") or "").strip()
        if candidate_id:
            ids.append(candidate_id)
    return ids


def _run_pdf_precheck(candidate_ids: list[str], *, label: str) -> None:
    progress = st.progress(0.0, "准备预检 PDF...")
    status = st.empty()

    def update_progress(event: dict) -> None:
        total = max(int(event.get("total") or len(candidate_ids) or 1), 1)
        completed = int(event.get("completed") or 0)
        if event.get("event") == "candidate_start":
            title = str(event.get("title") or event.get("candidate_id") or "")[:80]
            progress.progress(
                min(completed / total, 0.99),
                f"正在预检 {completed + 1}/{total}: {title}",
            )
            status.info(_pdf_check_progress_text(event, total=total, completed=completed))
            return
        if event.get("event") == "candidate_done":
            progress.progress(
                min(completed / total, 1.0),
                f"PDF 预检 {completed}/{total}",
            )
            status.info(_pdf_check_progress_text(event, total=total, completed=completed))
            return
        if event.get("event") == "done":
            progress.progress(1.0, "PDF 预检完成")
            status.success(_pdf_check_progress_text(event, total=total, completed=completed))

    with st.spinner(label):
        result = external_imports.check_candidate_pdf_access(
            candidate_ids,
            progress_callback=update_progress,
        )
    result["requested_count"] = len(candidate_ids)
    st.session_state["external_pdf_check_result"] = result
    st.rerun()


def _pdf_check_progress_text(event: dict, *, total: int, completed: int) -> str:
    verified = int(event.get("verified") or 0)
    failed = int(event.get("failed") or 0)
    missing = int(event.get("missing_pdf_url") or 0)
    parts = [
        f"已完成 {completed}/{total}",
        f"可访问 {verified}",
        f"失败 {failed}",
        f"无链接 {missing}",
    ]
    title = str(event.get("title") or "")
    if event.get("event") == "candidate_start" and title:
        parts.append(f"当前：{title[:80]}")
    error = str(event.get("error") or "")
    if error:
        parts.append(f"最近错误：{error[:120]}")
    return " | ".join(parts)


def _render_daily_brief() -> None:
    st.subheader("每日简报")
    st.caption("基于候选库元数据进行初步加工，不依赖 PDF 下载成功，不写入 Qdrant。")
    today = date.today()
    fetch_status = external_imports.latest_paper_fetch_status()
    checked_today = bool(fetch_status.get("checked_today", fetch_status.get("ran_today")))
    st.info(f"今日简报时间：{_format_candidate_time(datetime.now(timezone.utc).isoformat())}")
    _render_daily_fetch_status(fetch_status)
    col_lookback, col_max = st.columns(2)
    with col_lookback:
        lookback_days = st.number_input("回看天数", min_value=1, max_value=3650, value=14, step=1, key="external_daily_brief_lookback")
    with col_max:
        max_items = st.number_input("最多条目", min_value=1, max_value=200, value=30, step=1, key="external_daily_brief_max")
    sources = st.multiselect(
        "来源筛选",
        options=external_imports.DEFAULT_SOURCE_NAMES,
        default=[],
        format_func=lambda value: SOURCE_LABELS.get(value, value),
        key="external_daily_brief_sources",
    )
    st.caption("每日简报复用候选库已有分类；缺少分类的旧候选只做规则兜底，不调用 LLM。")

    col_generate, col_fetch, col_fetch_status = st.columns([1, 1, 1])
    with col_generate:
        if st.button("生成每日简报", type="primary", key="external_daily_brief_generate"):
            progress = st.progress(0.0, "准备生成每日简报...")
            status = st.empty()

            def update_progress(event: dict) -> None:
                total = max(int(event.get("total") or 1), 1)
                completed = int(event.get("completed") or 0)
                if event.get("event") == "candidate_done":
                    progress.progress(min(completed / total, 1.0), f"Y103 分类 {completed}/{total}")
                    status.info(_daily_brief_progress_text(event, total=total, completed=completed))
                elif event.get("event") == "done":
                    progress.progress(1.0, "每日简报分类完成")
                    status.success(f"已完成 {completed}/{total} 条候选分类")

            with st.spinner("正在加工候选库元数据..."):
                brief = external_imports.generate_daily_brief(
                    brief_date=today,
                    lookback_days=int(lookback_days),
                    max_items=int(max_items),
                    source_names=sources,
                    query="",
                    use_llm=False,
                    progress_callback=update_progress,
                )
            st.session_state["external_daily_brief_id"] = brief.brief_id
            st.success(f"已生成：{brief.title}，纳入 {brief.included_count}/{brief.total_candidates} 条")
            st.rerun()
    with col_fetch:
        if st.button(
            "自动获取最新论文",
            type="primary",
            disabled=checked_today,
            help="今天已经检查过时不会重复触发较慢的外部检索；需要强制重跑可到自动定时搜索中运行具体主题。",
            key="external_daily_brief_run_due_monitors",
        ):
            with st.spinner("正在立即运行所有启用主题..."):
                summary = external_imports.run_enabled_monitors_now()
            if summary["run_count"]:
                st.success(f"已立即运行 {summary['run_count']} 个启用主题")
            else:
                st.warning("没有启用的自动定时搜索主题。")
            if summary["summaries"]:
                st.json(summary["summaries"])
    with col_fetch_status:
        check_at = str(fetch_status.get("latest_check_at") or fetch_status.get("latest_fetch_at") or "")
        check_label = _format_candidate_time(check_at) if check_at else "尚未检查"
        if checked_today:
            st.success("✅ 今天已检查")
            st.caption(f"最近检查：{check_label}")
        else:
            st.warning("⚠️ 今天未检查")
            st.caption("可点击左侧按钮获取最新论文")

    briefs = external_imports.list_daily_briefs()
    if not briefs:
        st.info("暂无每日简报。")
        return

    selected_id = st.selectbox(
        "查看简报",
        options=[brief.brief_id for brief in briefs],
        index=_brief_select_index(briefs, st.session_state.get("external_daily_brief_id", "")),
        format_func=lambda brief_id: _brief_label(briefs, brief_id),
        key="external_daily_brief_select",
    )
    brief = next((item for item in briefs if item.brief_id == selected_id), briefs[0])
    _render_daily_brief_content(brief)


def _render_daily_fetch_status(fetch_status: dict) -> None:
    check_at = str(fetch_status.get("latest_check_at") or fetch_status.get("latest_fetch_at") or "")
    monitor_run_at = str(fetch_status.get("latest_monitor_run_at") or "")
    candidate_activity_at = str(fetch_status.get("latest_candidate_activity_at") or "")
    candidate_activity = fetch_status.get("latest_candidate_activity") or {}

    col_check, col_monitor, col_candidates = st.columns(3)
    with col_check:
        _render_status_line(
            "后台任务检查",
            check_at,
            bool(fetch_status.get("checked_today", fetch_status.get("ran_today"))),
            empty="尚未检查",
        )
    with col_monitor:
        _render_status_line(
            "主题实际运行",
            monitor_run_at,
            bool(fetch_status.get("monitor_ran_today")),
            empty="尚未运行主题",
        )
    with col_candidates:
        _render_status_line(
            "候选获取/更新",
            candidate_activity_at,
            bool(fetch_status.get("candidate_activity_today")),
            empty="尚无候选活动",
        )
        if candidate_activity:
            st.caption(
                "获取 {fetched} | 新增 {new} | 更新 {updated}".format(
                    fetched=int(candidate_activity.get("fetched_count") or 0),
                    new=int(candidate_activity.get("new_count") or 0),
                    updated=int(candidate_activity.get("updated_count") or 0),
                )
            )


def _render_status_line(title: str, value: str, is_today: bool, *, empty: str) -> None:
    label = _format_candidate_time(value) if value else empty
    if value and is_today:
        st.success(f"✅ {title}")
    elif value:
        st.warning(f"⚠️ {title}")
    else:
        st.info(f"ℹ️ {title}")
    st.caption(label)


def _render_daily_brief_content(brief) -> None:
    st.markdown(f"**🗞️ {brief.title}**")
    st.caption(
        f"生成时间：{_format_candidate_time(brief.generated_at)} | "
        f"回看 {brief.lookback_days} 天 | 纳入 {brief.included_count}/{brief.total_candidates} 条"
    )
    groups = _daily_brief_topic_groups(brief)
    _render_daily_brief_overview(brief, groups)

    if not brief.items:
        st.info("该简报没有纳入条目。")
        return

    detail_active_key = f"external_daily_brief_detail_active_{brief.brief_id}"
    _render_daily_brief_topic_sections(brief, groups, active_key=detail_active_key)


def _render_daily_brief_overview(brief, groups: list[dict] | None = None) -> None:
    topic_groups = groups if groups is not None else _daily_brief_topic_groups(brief)
    summary = daily_brief_topic_summary(topic_groups)
    high_count = sum(int(group.get("high_count") or 0) for group in topic_groups)
    if summary["unique_count"]:
        st.success(
            f"📌 今日需看 {summary['unique_count']} 条 | "
            f"按主题展示 {summary['display_count']} 次 | "
            f"高优先级 {high_count} 次 | 覆盖 {summary['hit_category_count']} 个 Y103 分类"
        )
        if summary["duplicate_count"]:
            st.caption(f"同一候选可重复出现在多个主题下；本次重复展示 {summary['duplicate_count']} 次。")
    else:
        st.info("📌 今日暂无命中 Y103 16 分类的候选。")

    st.dataframe(_daily_brief_overview_rows(brief, topic_groups), width="stretch", hide_index=True)
    hit_rows = _brief_hit_category_rows(brief, topic_groups)
    if hit_rows:
        with st.expander("📊 命中分类", expanded=False):
            st.dataframe(hit_rows, width="stretch", hide_index=True)


def _daily_brief_overview_rows(brief, groups: list[dict] | None = None) -> list[dict]:
    verified_count = sum(1 for item in brief.items if item.pdf_access_status == "verified")
    topic_counts = _daily_brief_topic_count_map(groups if groups is not None else _daily_brief_topic_groups(brief))
    return [
        {"项目": "主题分布", "摘要": _top_count_text(topic_counts)},
        {"项目": "来源分布", "摘要": _top_count_text(brief.source_counts)},
        {"项目": "PDF状态", "摘要": _top_count_text(brief.pdf_counts)},
        {"项目": "可直接导入", "摘要": f"{verified_count} 条 PDF 已校验可下载"},
    ]


def _top_count_text(counts: dict, *, limit: int = 4) -> str:
    items = list((counts or {}).items())
    if not items:
        return "暂无"
    visible = "；".join(f"{key} {value}" for key, value in items[:limit])
    hidden_count = len(items) - limit
    return f"{visible}；另 {hidden_count} 类" if hidden_count > 0 else visible


def _daily_brief_topic_count_map(groups: list[dict]) -> dict[str, int]:
    return {
        str(group.get("name") or ""): len(group.get("items") or [])
        for group in groups
        if group.get("items")
    }


def _brief_hit_category_rows(brief, groups: list[dict] | None = None) -> list[dict]:
    return [row for row in _y103_category_rows(brief, groups) if row["命中条数"] > 0]


def _render_daily_brief_topic_sections(brief, groups: list[dict], *, active_key: str) -> None:
    st.markdown("**📚 按 Y103 16 主题展示**")
    st.caption("同一篇候选如果同时符合多个主题，会在对应主题中重复展示。")
    summary = daily_brief_topic_summary(groups)
    if not summary["display_count"]:
        st.info("当前简报没有可按 Y103 主题展示的条目。")
        return

    for group in groups:
        items = list(group.get("items") or [])
        category_id = str(group.get("category_id") or "")
        name = str(group.get("name") or "")
        title = f"{category_id} {name} | {len(items)} 条"
        if group.get("high_count"):
            title += f" | 高优先级 {group['high_count']}"
        with st.expander(title, expanded=bool(items)):
            st.caption(str(group.get("description") or ""))
            retrieval_query = str(group.get("retrieval_query") or "")
            if retrieval_query:
                st.caption(f"默认检索式：`{retrieval_query}`")
            if not items:
                st.caption("暂无条目")
                continue
            for index, item in enumerate(items, start=1):
                detail_id = f"{category_id}:{item.candidate_id}"
                _render_daily_brief_item_summary(
                    item,
                    index=index,
                    total=len(items),
                    active_key=active_key,
                    detail_id=detail_id,
                    key_suffix=f"{brief.brief_id}_{category_id}_{index}",
                    display_topic=name,
                )
                if st.session_state.get(active_key, "") == detail_id:
                    _render_daily_brief_item_detail(
                        item,
                        index=index,
                        total=len(items),
                        display_topic=name,
                    )
                if index != len(items):
                    st.divider()


def _render_daily_brief_item_summary(
    item,
    *,
    index: int,
    total: int,
    active_key: str,
    detail_id: str | None = None,
    key_suffix: str = "",
    display_topic: str = "",
) -> None:
    col_info, col_button = st.columns([5, 1])
    with col_info:
        st.markdown(f"**{index}/{total} {_priority_badge(item.priority)} {item.title}**")
        _render_daily_brief_abstract_translation(item)
        topic_text = f"展示主题：{display_topic} | " if display_topic else ""
        st.caption(
            f"{topic_text}主分类：{item.classification_name or '其他/不纳入Y103'} | "
            f"{item.recommended_action} | "
            f"{', '.join(SOURCE_LABELS.get(source, source) for source in item.sources)} | "
            f"{PDF_ACCESS_LABELS.get(item.pdf_access_status, item.pdf_access_status)}"
        )
    with col_button:
        detail_value = detail_id or item.candidate_id
        suffix = key_suffix or item.candidate_id
        button_key = (
            f"external_daily_brief_detail_button_{suffix}_{item.candidate_id}"
            if key_suffix
            else f"external_daily_brief_detail_button_{item.candidate_id}"
        )
        if st.button(
            "🔎 查看详情",
            type="primary",
            use_container_width=True,
            key=button_key,
        ):
            st.session_state[active_key] = detail_value


def _render_daily_brief_item_detail(item, *, index: int, total: int, display_topic: str = "") -> None:
    st.markdown(f"**🔎 {index}/{total} {item.title}**")
    st.caption(
        f"{_priority_badge(item.priority)} | "
        f"{'展示主题：' + display_topic + ' | ' if display_topic else ''}"
        f"分类：{item.classification_name or '其他/不纳入Y103'} | "
        f"状态：{_classification_status_label(item.classification_status)} | "
        f"置信度：{float(item.classification_confidence or 0.0):.2f} | "
        f"证据层级：{item.evidence_level}"
    )
    _render_daily_brief_abstract_translation(item)
    st.markdown(item.brief)
    st.info(f"✅ 建议：{item.recommended_action}")
    reasons = [
        ("摘要信息", item.brief),
        ("分类依据", item.classification_reason),
        ("Judge意见", item.judge_reason),
        ("判断依据", item.priority_reason),
    ]
    reason_rows = [{"类型": label, "内容": value} for label, value in reasons if value]
    if reason_rows:
        with st.expander("🧾 判断依据", expanded=False):
            st.dataframe(reason_rows, width="stretch", hide_index=True)
    link_parts = []
    if item.landing_url:
        link_parts.append(f"原文：{_markdown_link(item.landing_url, '打开原文')}")
    if item.pdf_url:
        link_parts.append(f"PDF：{_markdown_link(item.pdf_url, '打开 PDF')}")
    if item.doi:
        link_parts.append(f"DOI：`{item.doi}`")
    if link_parts:
        st.markdown(" | ".join(link_parts))


def _render_daily_brief_abstract_translation(item) -> None:
    translation = str(getattr(item, "abstract_translation_zh", "") or "").strip()
    if translation:
        st.info(f"中文摘要：{translation}")
        return
    error = str(getattr(item, "abstract_translation_error", "") or "").strip()
    if error and getattr(item, "abstract", ""):
        st.caption(f"中文摘要暂未生成：{error}")


def _render_daily_brief_attention(brief) -> None:
    groups = _daily_brief_topic_groups(brief)
    summary = daily_brief_topic_summary(groups)
    high_count = sum(int(group.get("high_count") or 0) for group in groups)
    if summary["unique_count"]:
        st.success(
            f"今日有 {summary['unique_count']} 条 Y103 相关候选需要看，"
            f"按主题展示 {summary['display_count']} 次，覆盖 {summary['hit_category_count']} 个分类；"
            f"其中高优先级 {high_count} 次。"
        )
    else:
        st.info("今日暂无命中 Y103 16 分类的候选。")


def _render_y103_category_overview(brief) -> None:
    rows = _y103_category_rows(brief)
    hit_rows = [row for row in rows if row["命中条数"] > 0]
    if hit_rows:
        st.caption("今日命中的 Y103 分类")
        st.dataframe(hit_rows, width="stretch", hide_index=True)
    st.caption("Y103 16 分类总览")
    st.dataframe(rows, width="stretch", hide_index=True)


def _brief_y103_items(brief) -> list:
    return [
        item for item in brief.items
        if str(getattr(item, "classification_id", "") or "") not in {"", "other"}
    ]


def _daily_brief_topic_groups(brief) -> list[dict]:
    return daily_brief_topic_groups(brief)


def _y103_category_rows(brief, groups: list[dict] | None = None) -> list[dict]:
    categories = external_imports.y103_categories()
    topic_groups = groups if groups is not None else _daily_brief_topic_groups(brief)
    items_by_category: dict[str, list] = {
        str(group.get("category_id") or ""): list(group.get("items") or [])
        for group in topic_groups
    }
    rows = []
    for category in categories:
        items = items_by_category.get(category["category_id"], [])
        high_count = sum(1 for item in items if item.priority == "high")
        rows.append({
            "编号": category["category_id"],
            "分类": category["name"],
            "默认检索式": category.get("retrieval_query", ""),
            "今日状态": "需看" if items else "无",
            "命中条数": len(items),
            "高优先级": high_count,
            "代表论文": items[0].title if items else "",
        })
    return rows


def _daily_brief_item_row(item) -> dict:
    return {
        "重点": _priority_badge(item.priority),
        "标题": item.title,
        "Y103分类": item.classification_name or "其他/不纳入Y103",
        "建议动作": item.recommended_action,
        "来源": ", ".join(SOURCE_LABELS.get(source, source) for source in item.sources),
        "PDF": PDF_ACCESS_LABELS.get(item.pdf_access_status, item.pdf_access_status),
        "日期": item.published_date,
    }


def _count_rows(counts: dict, *, label: str = "类别") -> list[dict]:
    return [{label: key, "数量": value} for key, value in (counts or {}).items()]


def _priority_label(priority: str) -> str:
    return {"high": "高", "medium": "中", "low": "低"}.get(priority, priority)


def _priority_badge(priority: str) -> str:
    return {"high": "🔥 高", "medium": "🟡 中", "low": "⚪ 低"}.get(priority, priority)


def _classification_status_label(status: str) -> str:
    return {
        "rule_only": "规则初判",
        "small_only": "小模型初判",
        "judge_confirmed": "Judge确认",
        "judge_revised": "Judge修正",
        "small_failed_fallback_rule": "小模型失败-规则兜底",
        "judge_failed_small_only": "Judge失败-小模型结果",
        "failed_fallback_rule": "失败-规则兜底",
    }.get(status, status or "未分类")


def _daily_brief_progress_text(event: dict, *, total: int, completed: int) -> str:
    title = str(event.get("title") or "")[:80]
    category = str(event.get("category_name") or "")
    status = _classification_status_label(str(event.get("status") or ""))
    return f"已完成 {completed}/{total} | {status} | {category} | {title}"


def _brief_select_index(briefs: list, brief_id: str) -> int:
    for index, brief in enumerate(briefs):
        if brief.brief_id == brief_id:
            return index
    return 0


def _brief_label(briefs: list, brief_id: str) -> str:
    brief = next((item for item in briefs if item.brief_id == brief_id), None)
    if brief is None:
        return brief_id
    return f"{brief.title} | {brief.included_count} 条 | {_format_candidate_time(brief.generated_at)}"


def _render_monitors() -> None:
    st.subheader("设置每日简报主题")
    if st.button("运行到期主题", key="external_run_due_monitors"):
        with st.spinner("正在运行到期的每日简报主题..."):
            summary = external_imports.run_due_monitors()
        st.success(f"已运行 {summary['run_count']} 个到期主题")
        if summary["summaries"]:
            st.json(summary["summaries"])

    _render_background_monitor_task_status()
    monitors = external_imports.list_monitors()
    if not monitors:
        st.info("暂无每日简报主题。可以在下方新建一个主题。")
    else:
        st.caption("下次运行时间按北京时间显示；启用的主题会在打开外源导入页时自动检查是否到期，并按各自的结果过滤条件处理候选库。")
        st.dataframe([_monitor_row(monitor) for monitor in monitors], width="stretch", hide_index=True)

        for monitor in monitors:
            filter_config = external_imports.effective_monitor_filter_config(monitor)
            title = (
                f"{'✅' if monitor.enabled else '⏸️'} {monitor.name} | "
                f"下次运行时间：{_format_monitor_time(monitor.next_run_at)}"
            )
            with st.expander(title, expanded=False):
                st.caption(f"query: `{monitor.query}`")
                st.caption(f"下次运行时间：{_format_monitor_time(monitor.next_run_at)}")
                st.caption(f"上次运行时间：{_format_monitor_time(monitor.last_run_at, empty='尚未运行')}")
                st.caption("来源: " + ", ".join(SOURCE_LABELS.get(item, item) for item in monitor.sources))
                st.caption(f"回看 {monitor.lookback_days} 天 | 间隔 {monitor.interval_hours} 小时 | 每源 {monitor.max_results_per_source} 条")
                st.caption(f"过滤规则：{external_imports.candidate_filter_label(filter_config)}")
                st.caption(f"过滤行为：{external_imports.candidate_filter_behavior_label(filter_config)}")
                st.caption(f"筛选标准：{external_imports.candidate_filter_criteria_text(filter_config)}")
                if monitor.last_summary:
                    st.caption(
                        f"上次：新增 {monitor.last_summary.get('new_count', 0)}，"
                        f"更新 {monitor.last_summary.get('updated_count', 0)}，"
                        f"错误 {len(monitor.last_summary.get('errors', []))}"
                    )
                    _render_filter_summary(monitor.last_summary.get("candidate_filter"))
                _render_monitor_filter_editor(monitor, filter_config)
                col_run, col_toggle, col_delete = st.columns(3)
                with col_run:
                    if st.button("立即运行", key=f"external_monitor_run_{monitor.monitor_id}"):
                        with st.spinner("正在运行主题..."):
                            summary = external_imports.run_monitor(monitor.monitor_id)
                        _render_refresh_summary(summary)
                        st.rerun()
                with col_toggle:
                    label = "停用" if monitor.enabled else "启用"
                    if st.button(label, key=f"external_monitor_toggle_{monitor.monitor_id}"):
                        monitor.enabled = not monitor.enabled
                        external_imports.update_monitor(monitor)
                        st.rerun()
                with col_delete:
                    if st.button("删除", key=f"external_monitor_delete_{monitor.monitor_id}"):
                        external_imports.delete_monitor(monitor.monitor_id)
                        st.rerun()

    st.divider()
    with st.expander("新建每日简报主题", expanded=False):
        with st.form("external_monitor_form", clear_on_submit=True):
            name = st.text_input("名称", placeholder="HMO fermentation weekly")
            query = st.text_input("检索式 / 主题", placeholder="2'-fucosyllactose fermentation Escherichia coli")
            sources = st.multiselect(
                "来源",
                options=external_imports.DEFAULT_SOURCE_NAMES,
                default=external_imports.DEFAULT_SOURCE_NAMES,
                format_func=lambda value: SOURCE_LABELS.get(value, value),
                key="external_monitor_sources",
            )
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                lookback = st.number_input("回看天数", min_value=1, max_value=365, value=14, step=1)
            with col_b:
                interval = st.number_input("间隔小时", min_value=1, max_value=720, value=24, step=1)
            with col_c:
                max_results = st.number_input("每源上限", min_value=1, max_value=100, value=20, step=1)
            filter_type = st.selectbox(
                "结果过滤规则",
                options=_new_filter_type_options(),
                index=0,
                format_func=_new_filter_type_label,
                key="external_monitor_new_filter_type",
            )
            default_filter = external_imports.default_filter_config_for_monitor(name, query)
            include_other = st.checkbox(
                "保留未命中筛选标准的候选",
                value=bool(default_filter.get("include_other", filter_type == "none")),
                disabled=filter_type in {"auto", "none"},
                key="external_monitor_new_filter_include_other",
            )
            new_filter_config = (
                default_filter
                if filter_type == "auto"
                else _build_filter_config(filter_type, include_other)
            )
            st.caption(f"筛选标准：{external_imports.candidate_filter_criteria_text(new_filter_config)}")
            submitted = st.form_submit_button("保存主题", type="primary")
            if submitted:
                if not query.strip():
                    st.warning("请输入主题检索式")
                else:
                    monitor = external_imports.create_monitor(
                        name=name,
                        query=query,
                        sources=sources,
                        lookback_days=int(lookback),
                        interval_hours=int(interval),
                        max_results_per_source=int(max_results),
                        filter_config=new_filter_config,
                    )
                    st.success(f"已创建每日简报主题：{monitor.name}")
                    st.rerun()


def _render_background_monitor_task_status() -> None:
    try:
        status = external_imports.background_monitor_task_status()
    except Exception as exc:
        st.warning(f"后台计划任务状态读取失败：{type(exc).__name__}: {str(exc)[:200]}")
        return

    if not status.get("installed"):
        st.warning("后台计划任务未安装；只有打开应用时才会检查到期主题。")
        if status.get("error"):
            st.caption(f"状态详情：{status.get('error')}")
        return

    if status.get("hidden"):
        st.success("后台计划任务已安装，并以静默方式运行。")
    else:
        st.warning("后台计划任务已安装，但当前启动方式可能会弹出窗口。")

    cols = st.columns(3)
    with cols[0]:
        st.caption(f"上次运行：{status.get('last_run_time') or '未知'}")
    with cols[1]:
        st.caption(f"下次运行：{status.get('next_run_time') or '未知'}")
    with cols[2]:
        st.caption(f"退出码：{status.get('last_result') or '未知'}")
    if status.get("latest_log_marker"):
        st.caption(f"最近日志：{status.get('latest_log_marker')}")
    st.caption(f"日志文件：{status.get('log_path')}")
    _render_background_monitor_task_commands(status.get("commands") or {})


def _render_background_monitor_task_commands(commands: dict) -> None:
    if not commands:
        return
    with st.expander("后台任务修复与手动命令", expanded=False):
        st.caption("如果后台任务未安装、不是静默运行，或需要手动触发，可在命令行执行下面的命令。")
        labels = {
            "install_or_repair": "安装/修复静默后台任务",
            "run_now": "手动触发一次",
            "query_status": "查看计划任务状态",
            "tail_log": "查看最近日志",
        }
        for key, label in labels.items():
            command = commands.get(key)
            if not command:
                continue
            st.caption(label)
            st.code(command)


def _render_monitor_filter_editor(monitor, filter_config: dict) -> None:
    st.markdown("**结果过滤条件**")
    filter_type = st.selectbox(
        "过滤规则",
        options=_filter_type_options(),
        index=_filter_type_index(filter_config.get("filter_type", "none")),
        format_func=_filter_type_label,
        key=f"external_monitor_filter_type_{monitor.monitor_id}",
    )
    include_other = st.checkbox(
        "保留未命中筛选标准的候选",
        value=bool(filter_config.get("include_other", filter_type == "none")),
        disabled=filter_type == "none",
        key=f"external_monitor_filter_include_other_{monitor.monitor_id}",
    )
    edited_config = _build_filter_config(filter_type, include_other)
    st.caption(f"修改后行为：{external_imports.candidate_filter_behavior_label(edited_config)}")
    if st.button("保存筛选标准", key=f"external_monitor_filter_save_{monitor.monitor_id}"):
        monitor.filter_config = edited_config
        external_imports.update_monitor(monitor)
        st.success("已保存筛选标准")
        st.rerun()


def _render_filter_summary(filter_summary: dict | None) -> None:
    if not filter_summary:
        return
    if not filter_summary.get("enabled"):
        st.caption("上次过滤：未启用")
        return
    st.caption(
        f"上次过滤：{filter_summary.get('filter_name', '')} | "
        f"检查 {filter_summary.get('checked_count', 0)} 条 | "
        f"保留 {filter_summary.get('kept_count', 0)} 条 | "
        f"隐藏 {filter_summary.get('skipped_count', 0)} 条"
    )


def _filter_type_options() -> list[str]:
    return list(external_imports.candidate_filter_type_labels().keys())


def _new_filter_type_options() -> list[str]:
    return ["auto", *_filter_type_options()]


def _filter_type_label(value: str) -> str:
    return external_imports.candidate_filter_type_labels().get(value, value)


def _new_filter_type_label(value: str) -> str:
    if value == "auto":
        return "自动推荐"
    return _filter_type_label(value)


def _filter_type_index(value: str) -> int:
    options = _filter_type_options()
    return options.index(value) if value in options else 0


def _build_filter_config(filter_type: str, include_other: bool) -> dict:
    return external_imports.normalize_candidate_filter_config({
        "enabled": filter_type != "none",
        "filter_type": filter_type,
        "include_other": include_other if filter_type != "none" else True,
    })


def _auto_run_due_monitors_once() -> None:
    if not EXTERNAL_MONITOR_AUTO_RUN:
        return
    today_key = _beijing_today_key()
    checked_key = "external_due_monitors_checked_date"
    if st.session_state.get(checked_key) == today_key:
        return
    st.session_state[checked_key] = today_key
    summary = external_imports.run_due_monitors()
    if summary["run_count"]:
        st.info(f"已自动运行 {summary['run_count']} 个到期每日简报主题。")


def _beijing_today_key() -> str:
    return datetime.now(timezone.utc).astimezone(BEIJING_TZ).date().isoformat()


def _render_refresh_summary(summary: dict) -> None:
    if summary.get("error"):
        st.error(summary["error"])
        return
    st.success(
        f"获取 {summary.get('fetched_count', 0)} 条，"
        f"新增 {summary.get('new_count', 0)} 条，"
        f"更新 {summary.get('updated_count', 0)} 条"
    )
    _render_filter_summary(summary.get("candidate_filter"))
    errors = summary.get("errors") or []
    if errors:
        with st.expander(f"来源错误（{len(errors)}）", expanded=True):
            for item in errors:
                st.warning(f"{SOURCE_LABELS.get(item.get('source'), item.get('source'))}: {item.get('error')}")


def _candidate_row(candidate) -> dict:
    return {
        "标题": candidate.title,
        "日期": candidate.published_date,
        "来源": ", ".join(SOURCE_LABELS.get(source, source) for source in candidate.sources),
        "状态": STATUS_LABELS.get(candidate.import_status, candidate.import_status),
        "Y103分类": getattr(candidate, "classification_name", "") or "",
        "分类依据": getattr(candidate, "classification_reason", "") or "",
        "摘要翻译": _translation_status_label(candidate),
        "PDF状态": _pdf_status_label(candidate),
        "可直接导入": _importable_pdf_label(candidate),
        "OA": "是" if candidate.is_open_access else "否",
        "原文链接": candidate.landing_url,
        "PDF链接": candidate.pdf_url,
        "本地PDF": candidate.downloaded_pdf_path,
        "PDF校验时间": _format_candidate_time(getattr(candidate, "pdf_checked_at", "")),
        "PDF校验错误": getattr(candidate, "pdf_access_error", ""),
        "DOI": candidate.doi,
        "PMID": candidate.pmid,
        "PMCID": candidate.pmcid,
        "OpenAlex ID": candidate.openalex_id,
        "期刊": candidate.journal,
        "错误": candidate.import_error,
    }


def _render_candidate_link_details(candidates: list) -> None:
    with st.expander("候选链接详情", expanded=False):
        candidate = st.selectbox(
            "选择候选论文",
            options=candidates,
            format_func=lambda item: _candidate_label(candidates, item.candidate_id),
            key="external_candidate_link_detail",
        )
        if candidate is None:
            return
        st.markdown(f"**标题**：{candidate.title}")
        st.markdown(f"**PDF状态**：{_pdf_status_label(candidate)}")
        st.markdown(f"**原文链接**：{_markdown_link(candidate.landing_url, '打开原文')}")
        st.markdown(f"**PDF链接**：{_markdown_link(candidate.pdf_url, '打开 PDF')}")
        st.markdown(f"**本地PDF**：{candidate.downloaded_pdf_path or '无'}")
        if candidate.pdf_access_error:
            st.warning(candidate.pdf_access_error)
        st.caption("PDF链接只代表外部来源返回了可尝试下载的地址；导入时仍会校验响应内容必须是真实 PDF。")

        source_rows = [_source_record_row(record) for record in candidate.source_records]
        if source_rows:
            st.dataframe(source_rows, width="stretch", hide_index=True)


def _render_pdf_check_result(candidates: list) -> None:
    result = st.session_state.get("external_pdf_check_result")
    if not result:
        return
    verified = int(result.get("verified") or 0)
    failed = int(result.get("failed") or 0)
    missing_pdf_url = int(result.get("missing_pdf_url") or 0)
    message = f"PDF 预检完成：可访问 {verified}，失败 {failed}，无 PDF 链接 {missing_pdf_url}"
    if failed or missing_pdf_url:
        st.warning(message)
    else:
        st.success(message)

    rows = _pdf_check_result_rows(result, candidates)
    with st.expander("查看 PDF 预检详情", expanded=bool(failed or missing_pdf_url)):
        if rows:
            st.dataframe(_pdf_check_summary_rows(rows), width="stretch", hide_index=True)
            _render_pdf_check_result_items(rows)
        else:
            st.json(result.get("results", []))
        if st.button("清除预检结果提示", key="external_pdf_check_clear"):
            st.session_state.pop("external_pdf_check_result", None)
            st.rerun()


def _render_pdf_check_result_items(rows: list[dict]) -> None:
    for index, row in enumerate(rows, start=1):
        st.markdown(f"**{index}. {row['结果']}｜{row['标题']}**")
        if row.get("大小"):
            st.caption(f"大小：{row['大小']}")
        if row.get("本地PDF"):
            st.markdown("本地PDF")
            st.code(row["本地PDF"])
        if row.get("PDF链接"):
            st.markdown(f"PDF链接：{_markdown_link(row['PDF链接'], '打开 PDF')}")
        if row.get("原文链接"):
            st.markdown(f"原文链接：{_markdown_link(row['原文链接'], '打开原文')}")
        if row.get("错误"):
            st.markdown("错误")
            st.code(row["错误"])
        if index != len(rows):
            st.divider()


def _pdf_check_summary_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "标题": row.get("标题", ""),
            "结果": row.get("结果", ""),
            "大小": row.get("大小", ""),
        }
        for row in rows
    ]


def _pdf_check_result_rows(result: dict, candidates: list) -> list[dict]:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    rows = []
    for item in result.get("results") or []:
        candidate = by_id.get(item.get("candidate_id", ""))
        rows.append({
            "标题": candidate.title if candidate else item.get("candidate_id", ""),
            "结果": _pdf_check_result_label(str(item.get("status") or "")),
            "本地PDF": item.get("path") or (candidate.downloaded_pdf_path if candidate else ""),
            "大小": _format_bytes(item.get("size_bytes")),
            "错误": item.get("error") or (getattr(candidate, "pdf_access_error", "") if candidate else ""),
            "PDF链接": candidate.pdf_url if candidate else "",
            "原文链接": candidate.landing_url if candidate else "",
        })
    return rows


def _pdf_check_result_label(status: str) -> str:
    return {
        "verified": "可访问",
        "failed": "失败",
        "missing_pdf_url": "无PDF链接",
    }.get(status, status)


def _source_record_row(record: dict) -> dict:
    source = str(record.get("source") or "")
    return {
        "来源": SOURCE_LABELS.get(source, source),
        "外部ID": record.get("external_id") or "",
        "原文链接": record.get("landing_url") or "",
        "PDF链接": record.get("pdf_url") or "",
    }


def _markdown_link(url: str, label: str) -> str:
    value = str(url or "").strip()
    return f"[{label}]({value})" if value else "无"


def _pdf_status_label(candidate) -> str:
    if not candidate.pdf_url:
        return "无PDF链接"
    status = getattr(candidate, "pdf_access_status", "unknown") or "unknown"
    return PDF_ACCESS_LABELS.get(status, status)


def _translation_status_label(candidate) -> str:
    if getattr(candidate, "abstract_translation_error", ""):
        return "失败"
    if getattr(candidate, "abstract_translation_zh", ""):
        return "已翻译"
    if getattr(candidate, "abstract", ""):
        return "待翻译"
    return "无摘要"


def _importable_pdf_label(candidate) -> str:
    if not candidate.pdf_url:
        return "否"
    if getattr(candidate, "pdf_access_status", "unknown") == "verified":
        return "是"
    if getattr(candidate, "pdf_access_status", "unknown") == "unavailable":
        return "否"
    return "可尝试"


def _format_candidate_time(value: str) -> str:
    if not value:
        return ""
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return str(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S 北京时间")


def _format_bytes(value) -> str:
    try:
        size = int(value)
    except (TypeError, ValueError):
        return ""
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.2f} MB"


def _candidate_label(candidates: list, candidate_id: str) -> str:
    candidate = next((item for item in candidates if item.candidate_id == candidate_id), None)
    if candidate is None:
        return candidate_id
    return f"{candidate.title[:90]} ({', '.join(candidate.sources)})"


def _manual_min_date() -> date:
    return date(1900, 1, 1)


def _manual_max_date(today: date | None = None) -> date:
    current = today or date.today()
    return date(current.year, 12, 31)


def _manual_date_error(from_date: date, until_date: date, *, today: date | None = None) -> str:
    if from_date > until_date:
        return "开始日期不能晚于结束日期"
    max_date = _manual_max_date(today)
    if from_date > max_date or until_date > max_date:
        return f"日期不能晚于 {max_date.year} 年"
    return ""


def _monitor_row(monitor) -> dict:
    filter_config = external_imports.effective_monitor_filter_config(monitor)
    return {
        "名称": monitor.name,
        "状态": "启用" if monitor.enabled else "停用",
        "过滤规则": external_imports.candidate_filter_label(filter_config),
        "过滤行为": external_imports.candidate_filter_behavior_label(filter_config),
        "下次运行时间": _format_monitor_time(monitor.next_run_at),
        "上次运行时间": _format_monitor_time(monitor.last_run_at, empty="尚未运行"),
        "间隔": f"{monitor.interval_hours} 小时",
        "回看": f"{monitor.lookback_days} 天",
        "每源上限": monitor.max_results_per_source,
        "来源": ", ".join(SOURCE_LABELS.get(item, item) for item in monitor.sources),
    }


def _format_monitor_time(value: str, *, empty: str = "立即运行") -> str:
    if not value:
        return empty
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return str(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    local_time = parsed.astimezone(BEIJING_TZ)
    label = local_time.strftime("%Y-%m-%d %H:%M:%S 北京时间")
    now_utc = datetime.now(timezone.utc)
    if parsed <= now_utc:
        label += "（已到期）"
    return label


def _safe_paper_count(paper_id: str, paper_count) -> int:
    try:
        return int(paper_count(paper_id))
    except Exception:
        return 0
