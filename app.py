"""
SortPaper — Streamlit 前端
功能：导入 PDF → 解析（预览 / 完整流水线）→ 查看结果
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import streamlit as st
from src.runtime_env import load_project_env

load_project_env()

from app_utils import (
    build_paper_id, build_snapshot, save_result, overwrite_result,
    qdrant_search,
)
from app_config import (
    BATCH_MAX_PAPER_WORKERS,
    EMBEDDING_API_KEY_ENV,
    PIPELINE_TIMEOUT_SECONDS,
    RESULTS_DIR,
    SAMPLE_DIR,
)
from app_modes import (
    is_auto_store_mode,
    is_legacy_auto_store_mode,
    is_mineru_auto_store_mode,
    normalize_mode,
)
from app_pipeline import (
    run_preview, run_pipeline, run_mineru_preview, run_mineru_ingest, store_parsed_chunks,
    evaluate_and_enrich_from_qdrant, _process_one_pdf,
)
from app_ui import (
    type_badge, verdict_badge, category_label,
    render_overview, render_chunk_card,
    render_reconstruction_tab, render_text_tab, render_table_tab, render_image_tab,
    render_search_tab, render_standalone_search_tab,
    render_manual_search_view, render_agent_search_view,
)
from app_sidebar import sidebar, render_vector_library_workspace
from app_external_ui import (
    render_candidate_library_view,
    render_daily_brief_view,
    render_external_import_tab,
    render_manual_discovery_view,
    render_topics_view,
)
from app_workbench import ensure_workspace_state

logger = logging.getLogger(__name__)

# ─── 页面配置 ──────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SortPaper",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    div[data-testid="stSegmentedControl"] {
        background: rgba(148, 163, 184, 0.10);
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 10px;
        padding: 4px;
        margin: 10px 0 18px;
    }
    div[data-testid="stSegmentedControl"] button {
        min-height: 38px;
        border-radius: 8px;
        border: 1px solid transparent;
        font-weight: 700;
        letter-spacing: 0;
        transition: background-color 140ms ease, border-color 140ms ease, color 140ms ease;
    }
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
    div[data-testid="stSegmentedControl"] button[aria-selected="true"] {
        color: #f8fafc;
        background: linear-gradient(135deg, #2563eb, #0891b2);
        border-color: rgba(125, 211, 252, 0.70);
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.22);
    }
    div[data-testid="stSegmentedControl"] button:not([aria-pressed="true"]):not([aria-selected="true"]):hover {
        background: rgba(37, 99, 235, 0.12);
        border-color: rgba(96, 165, 250, 0.55);
        color: #93c5fd;
    }
    div[data-testid="stButton"] > button[kind="primary"] {
        border: 1px solid rgba(125, 211, 252, 0.70);
        background: linear-gradient(135deg, #2563eb, #0891b2);
        color: #f8fafc;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.22);
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        border-color: rgba(186, 230, 253, 0.90);
        background: linear-gradient(135deg, #1d4ed8, #0e7490);
        color: #ffffff;
    }
    div[data-testid="stButton"] > button[kind="secondary"] {
        border: 1px solid rgba(148, 163, 184, 0.40);
        background: rgba(15, 23, 42, 0.48);
        color: #e2e8f0;
        font-weight: 700;
    }
    div[data-testid="stButton"] > button[kind="secondary"]:hover {
        border-color: rgba(96, 165, 250, 0.72);
        background: rgba(37, 99, 235, 0.16);
        color: #bfdbfe;
    }
    .nav-label-primary {
        margin: 18px 0 6px;
        color: #93c5fd;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0;
    }
    .nav-label-secondary {
        margin: 14px 0 4px;
        color: #cbd5e1;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0;
    }
    .st-key-nav_workspace {
        padding: 8px;
        border: 1px solid rgba(96, 165, 250, 0.34);
        border-radius: 14px;
        background:
            linear-gradient(180deg, rgba(37, 99, 235, 0.16), rgba(8, 145, 178, 0.06)),
            rgba(15, 23, 42, 0.52);
        margin-bottom: 24px;
    }
    .st-key-nav_workspace div[data-testid="stButton"] > button {
        min-height: 50px;
        border-radius: 11px;
        font-size: 1rem;
    }
    .st-key-nav_workspace div[data-testid="stButton"] > button[kind="primary"] {
        border-color: rgba(191, 219, 254, 0.95);
        background: linear-gradient(135deg, #1d4ed8, #0891b2);
        box-shadow: 0 10px 28px rgba(37, 99, 235, 0.36);
    }
    .st-key-nav_online_view,
    .st-key-nav_import_view,
    .st-key-nav_search_view {
        padding: 5px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 10px;
        background: rgba(15, 23, 42, 0.30);
        margin: 4px 0 18px;
    }
    .st-key-nav_online_view div[data-testid="stButton"] > button,
    .st-key-nav_import_view div[data-testid="stButton"] > button,
    .st-key-nav_search_view div[data-testid="stButton"] > button {
        min-height: 36px;
        border-radius: 8px;
        font-size: 0.9rem;
        box-shadow: none;
    }
    .st-key-nav_online_view div[data-testid="stButton"] > button[kind="primary"],
    .st-key-nav_import_view div[data-testid="stButton"] > button[kind="primary"],
    .st-key-nav_search_view div[data-testid="stButton"] > button[kind="primary"] {
        background: rgba(37, 99, 235, 0.88);
        border-color: rgba(147, 197, 253, 0.86);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.18);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── 常量 ──────────────────────────────────────────────────────────────────────

LOGO = "📄"


WORKSPACE_LABELS = {
    "online": "在线获取论文",
    "import": "导入论文",
    "search": "检索召回",
}
ONLINE_VIEW_LABELS = {
    "daily_brief": "每日简报",
    "manual_discovery": "手动搜索",
    "topics": "自动定时搜索",
    "candidates": "候选库",
}
IMPORT_VIEW_LABELS = {
    "local_pdf": "本地 PDF",
    "library_quality": "论文库与质量",
}
SEARCH_VIEW_LABELS = {
    "manual": "手动召回",
    "agent": "Agent 召回",
}


def _qdrant_paper_count(paper_id: str) -> int:
    from src.application.vector_library import paper_count

    return paper_count(paper_id)


def _existing_paper_reason(paper_id: str) -> str:
    qdrant_count = _qdrant_paper_count(paper_id)
    if qdrant_count > 0:
        return f"已入库 {qdrant_count} 条记录"
    return ""


def _embedding_preflight_error() -> str:
    from src.application.vector_library import embedding_preflight_error

    return embedding_preflight_error()


def _mineru_batch_timing_fields(result: dict, fallback_elapsed: float) -> dict:
    from src.application.batch_processing import mineru_batch_timing_fields

    return mineru_batch_timing_fields(result, fallback_elapsed)


def _format_duration(value: object, *, skipped: bool = False) -> str:
    from src.application.batch_processing import format_duration

    return format_duration(value, skipped=skipped)


def _process_batch_job(file_bytes: bytes, paper_id: str, filename: str, mode: str) -> dict:
    from src.application.batch_processing import process_batch_job

    return process_batch_job(
        file_bytes,
        paper_id,
        filename,
        mode,
        run_preview=run_preview,
        run_pipeline=run_pipeline,
        run_mineru_preview=run_mineru_preview,
        run_mineru_ingest=run_mineru_ingest,
        fallback_runner=_process_one_pdf,
        snapshot_builder=build_snapshot,
        result_saver=save_result,
    )


def _ensure_workspace_state() -> None:
    ensure_workspace_state(
        st.session_state,
        workspace_labels=WORKSPACE_LABELS,
        online_view_labels=ONLINE_VIEW_LABELS,
        import_view_labels=IMPORT_VIEW_LABELS,
        search_view_labels=SEARCH_VIEW_LABELS,
    )


def _segmented_state(key: str, labels: dict[str, str], *, label: str) -> str:
    existing = st.session_state.get(key)
    current = existing if existing in labels else next(iter(labels))
    label_class = "nav-label-primary" if key == "workspace" else "nav-label-secondary"
    st.markdown(f'<div class="{label_class}">{label}</div>', unsafe_allow_html=True)
    with st.container(key=f"nav_{key}"):
        columns = st.columns(len(labels), gap="small")
        for column, (value, text) in zip(columns, labels.items()):
            with column:
                clicked = st.button(
                    text,
                    key=f"{key}_{value}_nav",
                    type="primary" if value == current else "secondary",
                    use_container_width=True,
                )
                if clicked and value != current:
                    st.session_state[key] = value
                    st.rerun()
    return current


def _render_workspace_nav() -> str:
    st.markdown("## SortPaper")
    return _segmented_state("workspace", WORKSPACE_LABELS, label="工作区")


def _empty_import_input(mode: str) -> dict:
    return {
        "pdf_bytes": None,
        "paper_id": "",
        "mode": mode,
        "parse_clicked": False,
        "filename": "",
        "batch": None,
    }


def _render_local_pdf_import(mode: str) -> dict:
    st.subheader("本地 PDF")
    st.caption("上传一篇或多篇 PDF；解析模式在左侧上下文栏中选择。")
    source = st.segmented_control(
        "PDF 来源",
        ["upload", "sample"],
        default=st.session_state.get("local_pdf_source", "upload"),
        format_func=lambda value: "上传文件" if value == "upload" else "示例论文",
        key="local_pdf_source",
        label_visibility="collapsed",
    ) or "upload"

    pdf_bytes: bytes | None = None
    filename = ""
    uploaded_files: list = []

    if source == "upload":
        uploaded = st.file_uploader(
            "选择 PDF 文件",
            type="pdf",
            accept_multiple_files=True,
            key="local_pdf_uploader",
        )
        uploaded_files = list(uploaded) if uploaded else []
        if uploaded_files:
            st.caption(f"已选择 {len(uploaded_files)} 个 PDF 文件")
    else:
        samples = sorted(SAMPLE_DIR.glob("*.pdf")) if SAMPLE_DIR.exists() else []
        if not samples:
            st.warning("data/sample_papers/ 目录为空")
        else:
            choice = st.selectbox(
                "选择示例论文",
                options=samples,
                format_func=lambda path: path.name,
                key="local_pdf_sample",
            )
            if choice:
                pdf_bytes = choice.read_bytes()
                filename = choice.name

    is_batch = len(uploaded_files) > 1
    if len(uploaded_files) == 1:
        pdf_bytes = uploaded_files[0].getvalue()
        filename = uploaded_files[0].name

    if pdf_bytes:
        paper_id = build_paper_id(pdf_bytes)
        st.caption(f"文件：{filename}")
        st.caption(f"paper_id: `{paper_id}`")
    elif is_batch:
        paper_id = "batch"
        filename = f"批量 {len(uploaded_files)} 篇"
        effective_batches = (len(uploaded_files) + BATCH_MAX_PAPER_WORKERS - 1) // BATCH_MAX_PAPER_WORKERS
        est_min = effective_batches * 2
        st.caption(f"批量 {len(uploaded_files)} 篇；预计约 {est_min} 分钟")
    else:
        paper_id = ""

    selected_count = len(uploaded_files) if source == "upload" else int(pdf_bytes is not None)
    can_parse = selected_count > 0
    button_label = "开始批量处理" if selected_count > 1 else "开始解析"
    parse_clicked = st.button(
        button_label,
        disabled=not can_parse,
        type="primary",
        key="local_pdf_parse_button",
    )
    if not can_parse:
        st.caption("请先选择 PDF 文件")

    return {
        "pdf_bytes": pdf_bytes,
        "paper_id": paper_id,
        "mode": mode,
        "parse_clicked": parse_clicked,
        "filename": filename,
        "batch": {"files": uploaded_files, "pending": parse_clicked, "mode": mode} if is_batch else None,
    }


def _render_online_workspace(mode: str) -> dict:
    st.markdown("### 在线获取论文")
    view = _segmented_state("online_view", ONLINE_VIEW_LABELS, label="在线获取视图")
    if view == "daily_brief":
        render_daily_brief_view()
        return _empty_import_input(mode)
    if view == "manual_discovery":
        render_manual_discovery_view()
        return _empty_import_input(mode)
    if view == "topics":
        render_topics_view()
        return _empty_import_input(mode)
    render_candidate_library_view()
    return _empty_import_input(mode)


def _render_import_workspace(mode: str) -> dict:
    st.markdown("### 导入论文")
    view = _segmented_state("import_view", IMPORT_VIEW_LABELS, label="导入视图")
    if view == "local_pdf":
        return _render_local_pdf_import(mode)
    _render_library_workspace()
    return _empty_import_input(mode)


def _render_search_workspace() -> None:
    st.markdown("### 检索召回")
    view = _segmented_state("search_view", SEARCH_VIEW_LABELS, label="检索召回视图")
    paper_id = st.session_state.get("current_paper") or None
    if view == "manual":
        render_manual_search_view(paper_id)
    else:
        render_agent_search_view(paper_id)


def _render_library_workspace() -> None:
    st.markdown("### 论文库与质量分析")
    render_vector_library_workspace()


def _render_analysis_workspace() -> None:
    _render_library_workspace()


def _render_result_details(result: dict) -> None:
    verdicts = result.get("verdicts", {})
    tabs = st.tabs(["概览", "文本块", "图片", "表格", "PDF 重建"])
    with tabs[0]:
        render_overview(result)
        _render_result_actions(result)
    with tabs[1]:
        text_chunks = [c for c in result.get("merged_chunks", []) if c.get("content_type") == "text"]
        render_text_tab(text_chunks, verdicts)
    with tabs[2]:
        render_image_tab(result)
    with tabs[3]:
        table_chunks = [c for c in result.get("merged_chunks", []) if c.get("content_type") == "table"]
        render_table_tab(table_chunks, verdicts)
    with tabs[4]:
        render_reconstruction_tab(result)


def _render_result_actions(result: dict) -> None:
    from src.application.result_actions import (
        enrich_result_quality,
        has_quality as result_has_quality,
        is_pipeline_result,
        overwrite_result_snapshot,
        resolve_storage_state,
        save_result_snapshot,
        store_result_chunks,
    )

    if not is_pipeline_result(result):
        return

    quality = result.get("quality", {})
    has_quality = result_has_quality(result)
    storage_state = resolve_storage_state(
        result,
        session_stored_state=st.session_state.get(f"stored_{result.get('paper_id', '')}"),
        paper_count_lookup=_qdrant_paper_count,
    )
    stored_key = storage_state["stored_key"]
    stored_state = st.session_state.get(stored_key)
    is_stored = storage_state["is_stored"]

    st.divider()
    if not is_stored:
        st.info("入库到 Qdrant 后可检索；质量分析可稍后补充。")
        if st.button("入库到 Qdrant", type="primary", key=f"store_{result.get('paper_id', '')}"):
            with st.spinner("正在检查重复并写入 Qdrant..."):
                store_r = store_result_chunks(result, store_runner=store_parsed_chunks)
            st.session_state[stored_key] = store_r
            st.rerun()
        return

    store_r = stored_state if isinstance(stored_state, dict) else {}
    if store_r.get("duplicate"):
        st.warning(f"该论文已入库（{store_r.get('existing_count', '?')} 条记录）。")
    elif store_r:
        st.success(f"已入库 {store_r.get('stored', 0)} chunks + {store_r.get('degraded', 0)} degraded。")
    else:
        st.success("已入库，可立即检索。")

    if has_quality:
        st.success(f"质量分析已完成：{quality.get('category', '?')} / credibility={quality.get('credibility', '?')}")
        eval_label = "重新质量评估"
    else:
        st.info("质量分析待完成，可从 Qdrant 读取 chunks 执行 Map-Reduce。")
        eval_label = "一键质量评估"

    if st.button(eval_label, type="primary", key=f"enrich_{result.get('paper_id', '')}"):
        with st.spinner("正在执行质量分析..."):
            enrich_state = enrich_result_quality(result, enrich_runner=evaluate_and_enrich_from_qdrant)
        st.session_state.result = enrich_state["result"]
        st.rerun()

    if quality.get("category") and st.button("保存解析结果", key=f"save_{result.get('paper_id', '')}"):
        fname = st.session_state.orig_filename or result.get("paper_id", "result")
        save_state = save_result_snapshot(result, fname, snapshot_builder=build_snapshot, result_saver=save_result)
        st.session_state.pending_save_filename = save_state["pending_save_filename"]
        st.session_state.save_success_msg = save_state["success_message"]
        st.rerun()

    if st.session_state.pending_save_filename:
        pfname = st.session_state.pending_save_filename
        st.warning(f"文件 `{pfname}.json` 已存在，是否覆盖？")
        col_ov, col_ca, _ = st.columns([1, 1, 3])
        with col_ov:
            if st.button("覆盖", key=f"overwrite_{pfname}"):
                fname = pfname or "result"
                save_state = overwrite_result_snapshot(
                    result,
                    fname,
                    snapshot_builder=build_snapshot,
                    result_overwriter=overwrite_result,
                )
                st.session_state.pending_save_filename = save_state["pending_save_filename"]
                st.session_state.save_success_msg = save_state["success_message"]
                st.rerun()
        with col_ca:
            if st.button("取消", key=f"cancel_save_{pfname}"):
                st.session_state.pending_save_filename = ""
                st.rerun()

    if st.session_state.save_success_msg:
        st.success(st.session_state.save_success_msg)
        if st.button("清除消息", key="clear_save_msg"):
            st.session_state.save_success_msg = ""
            st.rerun()


def _handle_local_pdf_submission(import_input: dict) -> bool:
    pdf_bytes: bytes | None = import_input["pdf_bytes"]
    paper_id: str = import_input["paper_id"]
    mode: str = normalize_mode(import_input["mode"])
    parse_clicked: bool = import_input["parse_clicked"]
    filename: str = import_input["filename"]
    batch_info: dict | None = import_input.get("batch")

    if filename:
        st.session_state.orig_filename = filename

    if paper_id and paper_id != st.session_state.current_paper:
        st.session_state.result = None
        st.session_state.current_paper = paper_id

    if batch_info and batch_info.get("pending"):
        _run_batch_import(batch_info, mode)
        return True

    if parse_clicked and pdf_bytes:
        _run_single_pdf_parse(pdf_bytes, paper_id, filename, mode)
    return False


def _run_batch_import(batch_info: dict, mode: str) -> None:
    batch_files = batch_info.get("files", [])
    batch_mode = normalize_mode(str(batch_info.get("mode") or mode))
    if not batch_files:
        return

    progress = st.progress(0, f"批量{batch_mode}中...")
    status = st.empty()
    results_view = st.empty()
    st.session_state.batch_results = []
    st.session_state.batch_preview_results = {}
    success_n = skip_n = fail_n = 0
    skip_existing_n = 0
    skip_batch_duplicate_n = 0

    prepared_jobs: list[dict] = []
    seen_paper_ids: dict[str, dict] = {}
    from src.application.batch_processing import precheck_batch_file, precheck_failure_row

    for i, pdf_file in enumerate(batch_files):
        status.info(f"({i + 1}/{len(batch_files)}) 预检查 {pdf_file.name}")
        try:
            file_bytes = pdf_file.getvalue() if hasattr(pdf_file, "getvalue") else pdf_file.read()
        except Exception as exc:
            fail_n += 1
            st.session_state.batch_results.append(precheck_failure_row(i + 1, pdf_file.name, exc))
            progress.progress((i + 1) / len(batch_files), f"预检查 {i + 1}/{len(batch_files)}")
            continue
        precheck = precheck_batch_file(
            no=i + 1,
            filename=pdf_file.name,
            file_bytes=file_bytes,
            mode=batch_mode,
            seen_paper_ids=seen_paper_ids,
            paper_id_builder=build_paper_id,
            existing_reason_lookup=_existing_paper_reason,
        )
        if precheck["kind"] == "prepared":
            prepared_jobs.append(precheck["job"])
        elif precheck["kind"] == "skip_batch_duplicate":
            skip_n += 1
            skip_batch_duplicate_n += 1
            st.session_state.batch_results.append(precheck["row"])
        elif precheck["kind"] == "skip_existing":
            skip_n += 1
            skip_existing_n += 1
            st.session_state.batch_results.append(precheck["row"])
        else:
            fail_n += 1
            st.session_state.batch_results.append(precheck["row"])
        progress.progress((i + 1) / len(batch_files), f"预检查 {i + 1}/{len(batch_files)}")

    if st.session_state.batch_results:
        import pandas as pd

        results_view.dataframe(
            pd.DataFrame(sorted(st.session_state.batch_results, key=lambda row: row.get("no", 0))),
            width="stretch",
            hide_index=True,
        )

    embedding_error = _embedding_preflight_error() if prepared_jobs and is_auto_store_mode(batch_mode) else ""
    if prepared_jobs and embedding_error:
        from src.application.batch_processing import embedding_preflight_failure_rows

        fail_n += len(prepared_jobs)
        st.session_state.batch_results.extend(embedding_preflight_failure_rows(prepared_jobs, embedding_error))
        prepared_jobs = []
        st.error(embedding_error)

    if prepared_jobs:
        max_workers = min(BATCH_MAX_PAPER_WORKERS, len(prepared_jobs))
        status.info(f"开始并行处理 {len(prepared_jobs)} 篇；论文并发 {max_workers}")
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_jobs = {
                pool.submit(_process_batch_job, job["bytes"], job["paper_id"], job["file"], batch_mode): {
                    **job,
                    "started_at": time.time(),
                }
                for job in prepared_jobs
            }
            pending = set(future_jobs)
            while pending:
                done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for fut in done:
                    job = future_jobs[fut]
                    try:
                        from src.application.batch_processing import completed_batch_result

                        q_result = fut.result()
                        completed_result = completed_batch_result(job, q_result)
                        if completed_result["counter"] == "success":
                            success_n += 1
                        elif completed_result["counter"] == "skip_existing":
                            skip_n += 1
                            skip_existing_n += 1
                        elif completed_result["counter"] == "skip":
                            skip_n += 1
                        else:
                            fail_n += 1
                        if completed_result["preview_entry"]:
                            st.session_state.batch_preview_results[job["paper_id"]] = completed_result["preview_entry"]
                        st.session_state.batch_results.append(completed_result["row"])
                    except Exception as exc:
                        fail_n += 1
                        st.session_state.batch_results.append({
                            "no": job["no"],
                            "file": job["file"],
                            "paper_id": job["paper_id"],
                            "status": "fail",
                            "reason": str(exc)[:200],
                        })

                completed = success_n + skip_n + fail_n
                partial = sum(
                    min(0.9, (time.time() - future_jobs[fut]["started_at"]) / max(PIPELINE_TIMEOUT_SECONDS, 1))
                    for fut in pending
                )
                progress.progress(
                    min(1.0, (completed + partial) / len(batch_files)),
                    f"完成 {completed}/{len(batch_files)}；待完成 {len(pending)}",
                )
                if st.session_state.batch_results:
                    import pandas as pd

                    results_view.dataframe(
                        pd.DataFrame(sorted(st.session_state.batch_results, key=lambda row: row.get("no", 0))),
                        width="stretch",
                        hide_index=True,
                    )

    progress.empty()
    status.empty()
    st.markdown(f"## 批量{batch_mode}完成")
    st.markdown(
        f"成功 **{success_n}** | "
        f"已入库跳过 **{skip_existing_n}** | "
        f"本批重复 **{skip_batch_duplicate_n}** | "
        f"其他跳过 **{max(skip_n - skip_existing_n - skip_batch_duplicate_n, 0)}** | "
        f"失败 **{fail_n}**"
    )
    if st.session_state.batch_results:
        import pandas as pd

        st.dataframe(pd.DataFrame(st.session_state.batch_results), width="stretch", hide_index=True)
    _render_batch_preview_results()


def _run_single_pdf_parse(pdf_bytes: bytes, paper_id: str, filename: str, mode: str) -> None:
    from src.application.single_paper_processing import duplicate_reparse_decision

    dup_decision = duplicate_reparse_decision(
        paper_id=paper_id,
        mode=mode,
        existing_count=_qdrant_paper_count(paper_id),
        force_reparse_enabled=bool(st.session_state.get(f"force_reparse_{paper_id}")),
    )
    if dup_decision["should_prompt"]:
        force_key = dup_decision["force_key"]
        st.warning(dup_decision["message"])
        col_f1, col_f2, _ = st.columns([1, 1, 3])
        with col_f1:
            if st.button("强制重新解析", key=f"force_{paper_id}"):
                st.session_state[force_key] = True
                st.rerun()
        with col_f2:
            if st.button("取消", key=f"cancel_{paper_id}"):
                return
        return

    st.session_state._pdf_bytes = pdf_bytes
    start_ts = time.time()
    try:
        from src.application.single_paper_processing import execute_single_parse, single_parse_execution_plan

        parse_plan = single_parse_execution_plan(
            mode,
            timeout_seconds=PIPELINE_TIMEOUT_SECONDS,
            embedding_api_key_env=EMBEDDING_API_KEY_ENV,
        )
        timeout_banner = None
        if parse_plan["kind"] != "direct":
            timeout_banner = st.info(parse_plan["timeout_intro"])

        with st.spinner(parse_plan["spinner"]):
            parse_state = execute_single_parse(
                pdf_bytes,
                paper_id,
                filename,
                mode,
                timeout_seconds=PIPELINE_TIMEOUT_SECONDS,
                embedding_api_key_env=EMBEDDING_API_KEY_ENV,
                run_mineru_preview=run_mineru_preview,
                run_mineru_ingest=run_mineru_ingest,
                run_preview=run_preview,
                run_pipeline=run_pipeline,
            )

        if parse_state["status"] == "timeout":
            if timeout_banner is not None:
                timeout_banner.error(parse_plan["timeout_error"])
            return
        if timeout_banner is not None:
            timeout_banner.empty()

        result = parse_state["result"]
        elapsed = parse_state["elapsed"]
        st.caption(f"解析完成，耗时 {elapsed:.1f} 秒")

        if is_legacy_auto_store_mode(mode):
            steps = st.empty()
            steps.info("阶段 2/2: 入库到 Qdrant...")
            from src.application.single_paper_processing import auto_store_feedback, store_legacy_auto_result

            store_r = store_legacy_auto_result(
                result,
                filename,
                store_runner=store_parsed_chunks,
                snapshot_builder=build_snapshot,
                result_saver=save_result,
                before_save=lambda: steps.info("保存解析结果..."),
            )
            steps.empty()
            result["_elapsed"] = time.time() - start_ts
            feedback = auto_store_feedback(mode=mode, paper_id=paper_id, store_result=store_r)
            st.session_state[feedback["stored_key"]] = feedback["store_result"]
            st.success(feedback["message"])
        elif is_mineru_auto_store_mode(mode):
            from src.application.single_paper_processing import auto_store_feedback

            feedback = auto_store_feedback(mode=mode, paper_id=paper_id, result=result)
            st.session_state[feedback["stored_key"]] = feedback["store_result"]
            if feedback["level"] == "warning":
                st.warning(feedback["message"])
            elif feedback["level"] == "error":
                st.error(feedback["message"])
            else:
                st.success(feedback["message"])

        st.session_state.result = result
    except Exception as exc:
        st.error(f"解析失败：{exc}")
        logging.exception("parse error")


def _render_batch_preview_results() -> None:
    previews = st.session_state.get("batch_preview_results", {})
    if not previews:
        return

    st.divider()
    st.markdown("## 预览内容")
    options = sorted(previews.keys(), key=lambda pid: previews[pid].get("no", 0))
    selected = st.selectbox(
        "选择要查看的论文",
        options=options,
        format_func=lambda pid: f"{previews[pid].get('no', '?')}. {previews[pid].get('file', pid)}",
        key="batch_preview_selected",
    )
    item = previews.get(selected)
    if not item:
        return

    result = item["result"]
    verdicts = result.get("verdicts", {})
    st.caption(f"paper_id: `{result.get('paper_id', selected)}`")

    tabs = st.tabs(["📊 概览", "📝 文本块", "🖼️ 图片", "📋 表格", "📐 PDF 重建"])
    with tabs[0]:
        render_overview(result)
    with tabs[1]:
        text_chunks = [c for c in result.get("merged_chunks", []) if c.get("content_type") == "text"]
        render_text_tab(text_chunks, verdicts)
    with tabs[2]:
        render_image_tab(result)
    with tabs[3]:
        table_chunks = [c for c in result.get("merged_chunks", []) if c.get("content_type") == "table"]
        render_table_tab(table_chunks, verdicts)
    with tabs[4]:
        render_reconstruction_tab(result)

# ─── 主界面 ────────────────────────────────────────────────────────────────────

def main() -> None:
    _ensure_workspace_state()
    workspace = _render_workspace_nav()
    sb = sidebar(workspace)
    mode = normalize_mode(sb.get("mode", st.session_state.get("sidebar_parse_mode", "")))

    if workspace == "online":
        _render_online_workspace(mode)
        return

    if workspace == "import":
        import_input = _render_import_workspace(mode)
        handled_batch = _handle_local_pdf_submission(import_input)
        if handled_batch:
            return
        if st.session_state.get("result") is not None and st.session_state.get("import_view") == "local_pdf":
            st.divider()
            _render_result_details(st.session_state.result)
        return

    if workspace == "search":
        _render_search_workspace()
        return

    _render_import_workspace(mode)


def _legacy_tabbed_main() -> None:
    from app_sidebar import legacy_sidebar

    sb = legacy_sidebar()
    pdf_bytes: bytes | None = sb["pdf_bytes"]
    paper_id: str = sb["paper_id"]
    mode: str = normalize_mode(sb["mode"])
    parse_clicked: bool = sb["parse_clicked"]
    filename: str = sb["filename"]
    batch_info: dict | None = sb.get("batch")

    if "result" not in st.session_state:
        st.session_state.result = None
    if "current_paper" not in st.session_state:
        st.session_state.current_paper = None
    if "orig_filename" not in st.session_state:
        st.session_state.orig_filename = ""
    if "pending_save_filename" not in st.session_state:
        st.session_state.pending_save_filename = ""
    if "save_success_msg" not in st.session_state:
        st.session_state.save_success_msg = ""
    if "batch_results" not in st.session_state:
        st.session_state.batch_results: list[dict] = []
    if "batch_preview_results" not in st.session_state:
        st.session_state.batch_preview_results = {}

    if filename:
        st.session_state.orig_filename = filename

    if paper_id and paper_id != st.session_state.current_paper:
        st.session_state.result = None
        st.session_state.current_paper = paper_id

    # ── 批量模式 ──
    if batch_info and batch_info.get("pending"):
        batch_files = batch_info.get("files", [])
        batch_mode = normalize_mode(str(batch_info.get("mode") or mode))
        if batch_files:
            progress = st.progress(0, f"批量{batch_mode}中…")
            status = st.empty()
            results_view = st.empty()
            st.session_state.batch_results = []
            st.session_state.batch_preview_results = {}
            success_n = skip_n = fail_n = 0
            skip_existing_n = 0
            skip_batch_duplicate_n = 0

            prepared_jobs: list[dict] = []
            seen_paper_ids: dict[str, dict] = {}
            from src.application.batch_processing import precheck_batch_file, precheck_failure_row

            for i, pdf_file in enumerate(batch_files):
                status.info(f"({i+1}/{len(batch_files)}) 预检查: {pdf_file.name}")
                try:
                    file_bytes = pdf_file.getvalue() if hasattr(pdf_file, "getvalue") else pdf_file.read()
                except Exception as exc:
                    fail_n += 1
                    st.session_state.batch_results.append(precheck_failure_row(i + 1, pdf_file.name, exc))
                    progress.progress((i + 1) / len(batch_files), f"预检查 {i+1}/{len(batch_files)}")
                    continue
                precheck = precheck_batch_file(
                    no=i + 1,
                    filename=pdf_file.name,
                    file_bytes=file_bytes,
                    mode=batch_mode,
                    seen_paper_ids=seen_paper_ids,
                    paper_id_builder=build_paper_id,
                    existing_reason_lookup=_existing_paper_reason,
                )
                if precheck["kind"] == "prepared":
                    prepared_jobs.append(precheck["job"])
                elif precheck["kind"] == "skip_batch_duplicate":
                    skip_n += 1
                    skip_batch_duplicate_n += 1
                    st.session_state.batch_results.append(precheck["row"])
                elif precheck["kind"] == "skip_existing":
                    skip_n += 1
                    skip_existing_n += 1
                    st.session_state.batch_results.append(precheck["row"])
                else:
                    fail_n += 1
                    st.session_state.batch_results.append(precheck["row"])
                progress.progress((i + 1) / len(batch_files), f"预检查 {i+1}/{len(batch_files)}")

            if st.session_state.batch_results:
                import pandas as pd
                results_view.dataframe(
                    pd.DataFrame(sorted(st.session_state.batch_results, key=lambda row: row.get("no", 0))),
                    width="stretch",
                    hide_index=True,
                )

            embedding_error = _embedding_preflight_error() if prepared_jobs and is_auto_store_mode(batch_mode) else ""
            if prepared_jobs and embedding_error:
                from src.application.batch_processing import embedding_preflight_failure_rows

                fail_n += len(prepared_jobs)
                reason = embedding_error
                st.session_state.batch_results.extend(
                    embedding_preflight_failure_rows(prepared_jobs, reason)
                )
                prepared_jobs = []
                st.error(reason)

            if prepared_jobs:
                max_workers = min(BATCH_MAX_PAPER_WORKERS, len(prepared_jobs))
                status.info(
                    f"开始并行处理 {len(prepared_jobs)} 篇；论文并发 {max_workers}，"
                    f"LLM 全局并发由配置控制"
                )
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    future_jobs = {
                        pool.submit(
                            _process_batch_job,
                            job["bytes"],
                            job["paper_id"],
                            job["file"],
                            batch_mode,
                        ): {
                            **job,
                            "started_at": time.time(),
                        }
                        for job in prepared_jobs
                    }
                    pending = set(future_jobs)
                    while pending:
                        done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                        for fut in done:
                            job = future_jobs[fut]
                            try:
                                from src.application.batch_processing import completed_batch_result

                                q_result = fut.result()
                                completed_result = completed_batch_result(job, q_result)
                                if completed_result["counter"] == "success":
                                    success_n += 1
                                elif completed_result["counter"] == "skip_existing":
                                    skip_n += 1
                                    skip_existing_n += 1
                                elif completed_result["counter"] == "skip":
                                    skip_n += 1
                                else:
                                    fail_n += 1
                                if completed_result["preview_entry"]:
                                    st.session_state.batch_preview_results[job["paper_id"]] = completed_result["preview_entry"]
                                st.session_state.batch_results.append(completed_result["row"])
                            except Exception as exc:
                                fail_n += 1
                                st.session_state.batch_results.append({
                                    "no": job["no"],
                                    "file": job["file"],
                                    "paper_id": job["paper_id"],
                                    "status": "fail",
                                    "reason": str(exc)[:200],
                                })

                        completed = success_n + skip_n + fail_n
                        partial = sum(
                            min(0.9, (time.time() - future_jobs[fut]["started_at"]) / max(PIPELINE_TIMEOUT_SECONDS, 1))
                            for fut in pending
                        )
                        progress.progress(
                            min(1.0, (completed + partial) / len(batch_files)),
                            f"完成 {completed}/{len(batch_files)}；待完成 {len(pending)}",
                        )
                        if pending:
                            waiting_or_running = sorted(
                                (
                                    f"{future_jobs[fut]['no']}. {future_jobs[fut]['file']} "
                                    f"({time.time() - future_jobs[fut]['started_at']:.0f}s)"
                                    for fut in pending
                                )
                            )
                            status.info("待完成（排队或运行中）：\n" + "\n".join(waiting_or_running[:5]))
                        if st.session_state.batch_results:
                            import pandas as pd
                            results_view.dataframe(
                                pd.DataFrame(sorted(st.session_state.batch_results, key=lambda row: row.get("no", 0))),
                                width="stretch",
                                hide_index=True,
                            )

            progress.empty()
            status.empty()

            st.markdown(f"## ✅ 批量{batch_mode}完成")
            st.markdown(
                f"成功 **{success_n}** | "
                f"已入库跳过 **{skip_existing_n}** | "
                f"本批次重复 **{skip_batch_duplicate_n}** | "
                f"其他跳过 **{max(skip_n - skip_existing_n - skip_batch_duplicate_n, 0)}** | "
                f"失败 **{fail_n}**"
            )

            if st.session_state.batch_results:
                import pandas as pd
                df = pd.DataFrame(st.session_state.batch_results)
                st.dataframe(df, width="stretch", hide_index=True)

            _render_batch_preview_results()

            st.session_state.batch_trigger = False
        return

    # ── 单篇模式 ──
    if parse_clicked and pdf_bytes:
        from src.application.single_paper_processing import duplicate_reparse_decision

        dup_decision = duplicate_reparse_decision(
            paper_id=paper_id,
            mode=mode,
            existing_count=_qdrant_paper_count(paper_id),
            force_reparse_enabled=bool(st.session_state.get(f"force_reparse_{paper_id}")),
        )
        if dup_decision["should_prompt"]:
            force_key = dup_decision["force_key"]
            st.warning(dup_decision["message"])
            col_f1, col_f2, _ = st.columns([1, 1, 3])
            with col_f1:
                if st.button("强制重新解析", key=f"force_{paper_id}"):
                    st.session_state[force_key] = True
                    st.rerun()
            with col_f2:
                if st.button("取消", key=f"cancel_{paper_id}"):
                    return
            return

        st.session_state._pdf_bytes = pdf_bytes

        start_ts = time.time()
        try:
            from src.application.single_paper_processing import (
                execute_single_parse,
                single_parse_execution_plan,
            )

            parse_plan = single_parse_execution_plan(
                mode,
                timeout_seconds=PIPELINE_TIMEOUT_SECONDS,
                embedding_api_key_env=EMBEDDING_API_KEY_ENV,
            )
            timeout_banner = None
            if parse_plan["kind"] != "direct":
                timeout_banner = st.info(parse_plan["timeout_intro"])

            with st.spinner(parse_plan["spinner"]):
                parse_state = execute_single_parse(
                    pdf_bytes,
                    paper_id,
                    filename,
                    mode,
                    timeout_seconds=PIPELINE_TIMEOUT_SECONDS,
                    embedding_api_key_env=EMBEDDING_API_KEY_ENV,
                    run_mineru_preview=run_mineru_preview,
                    run_mineru_ingest=run_mineru_ingest,
                    run_preview=run_preview,
                    run_pipeline=run_pipeline,
                )

            if parse_state["status"] == "timeout":
                if timeout_banner is not None:
                    timeout_banner.error(parse_plan["timeout_error"])
                return
            if timeout_banner is not None:
                timeout_banner.empty()

            result = parse_state["result"]
            elapsed = parse_state["elapsed"]
            st.caption(f"✅ 解析完成，耗时 {elapsed:.1f} 秒")

            if is_legacy_auto_store_mode(mode):
                steps = st.empty()
                steps.info("📥 阶段 2/2: 入库到 Qdrant…")
                from src.application.single_paper_processing import store_legacy_auto_result

                store_r = store_legacy_auto_result(
                    result,
                    filename,
                    store_runner=store_parsed_chunks,
                    snapshot_builder=build_snapshot,
                    result_saver=save_result,
                    before_save=lambda: steps.info("💾 保存解析结果…"),
                )

                steps.empty()
                result["_elapsed"] = time.time() - start_ts
                from src.application.single_paper_processing import auto_store_feedback

                feedback = auto_store_feedback(mode=mode, paper_id=paper_id, store_result=store_r)
                st.session_state[feedback["stored_key"]] = feedback["store_result"]
                st.success(feedback["message"])
            elif is_mineru_auto_store_mode(mode):
                from src.application.single_paper_processing import auto_store_feedback

                feedback = auto_store_feedback(mode=mode, paper_id=paper_id, result=result)
                st.session_state[feedback["stored_key"]] = feedback["store_result"]
                if feedback["level"] == "warning":
                    st.warning(feedback["message"])
                elif feedback["level"] == "error":
                    st.error(feedback["message"])
                else:
                    st.success(feedback["message"])

            st.session_state.result = result
        except Exception as exc:
            st.error(f"解析失败：{exc}")
            logging.exception("parse error")
            return

    result = st.session_state.result

    if result is None:
        if st.session_state.get("batch_preview_results"):
            _render_batch_preview_results()
            st.divider()
        welcome_tabs = st.tabs(["🏠 欢迎", "🔍 语义检索", "📥 外源导入"])
        with welcome_tabs[0]:
            st.markdown("## 欢迎使用 SortPaper 📄")
            st.markdown(
                """
                **使用步骤：**
                1. 在左侧选择或上传 PDF 文件
                2. 选择解析模式（推荐使用 MinerU；历史自研解析入口用于展示和兼容）
                3. 点击 **🚀 开始解析**
                4. 在各标签页查看解析结果

                ---
                **MinerU 快速预览**：解析文本/表格/图片占位，不调用图片转文字。
                **MinerU 完整解析**：在快速预览基础上，按 Figure group 把图片转成文字。
                **MinerU 一键入库**：完整解析后直接写入 Qdrant。
                **历史快速预览 / 历史完整流水线**：保留自研 parser 展示入口，用于对照旧效果和回归排查。
                """
            )
        with welcome_tabs[1]:
            render_standalone_search_tab()
        with welcome_tabs[2]:
            render_external_import_tab()
        return

    # ── 结果展示 ──
    verdicts = result.get("verdicts", {})
    from src.application.result_actions import is_pipeline_result

    is_pipeline = is_pipeline_result(result)
    tabs = st.tabs(["📊 概览", "📝 文本块", "🖼️ 图片", "📋 表格", "📐 PDF 重建", "🔍 语义检索", "📥 外源导入"])

    with tabs[0]:
        render_overview(result)

        if is_pipeline:
            from src.application.result_actions import (
                enrich_result_quality,
                has_quality as result_has_quality,
                overwrite_result_snapshot,
                resolve_storage_state,
                save_result_snapshot,
                store_result_chunks,
            )

            quality = result.get("quality", {})
            has_quality = result_has_quality(result)
            storage_state = resolve_storage_state(
                result,
                session_stored_state=st.session_state.get(f"stored_{result.get('paper_id', '')}"),
                paper_count_lookup=_qdrant_paper_count,
            )
            stored_key = storage_state["stored_key"]
            stored_state = st.session_state.get(stored_key)
            is_stored = storage_state["is_stored"]

            st.divider()

            if not is_stored:
                st.info("📥 请先入库到 Qdrant；入库后可立即检索，质量分析可稍后补充")
                col_btn, _ = st.columns([1, 3])
                with col_btn:
                    if st.button("📥 入库到 Qdrant", width="stretch", type="primary",
                                 key=f"store_{result.get('paper_id', '')}"):
                        with st.spinner("正在检查重复并写入 Qdrant…"):
                            store_r = store_result_chunks(
                                result,
                                store_runner=store_parsed_chunks,
                            )
                        st.session_state[stored_key] = store_r
                        st.rerun()

            else:
                store_r = stored_state if isinstance(stored_state, dict) else {}
                if store_r.get("duplicate"):
                    st.warning(f"⚠️ 该论文已入库（{store_r.get('existing_count', '?')} 条记录），请删除后重新入库或使用覆盖功能。")
                else:
                    if store_r:
                        st.success(f"✅ 已入库 {store_r.get('stored', 0)} chunks + {store_r.get('degraded', 0)} degraded，可立即检索")
                    else:
                        st.success("✅ 已入库，可立即检索")

                if has_quality:
                    st.success(f"✅ 质量分析已完成：{quality.get('category', '?')} / credibility={quality.get('credibility', '?')}")
                    eval_label = "🔁 重新质量评估"
                else:
                    st.info("📊 质量分析待完成，可单独运行 Map-Reduce 补充元数据")
                    eval_label = "📊 一键质量评估"
                col_btn, _ = st.columns([1, 3])
                with col_btn:
                    if st.button(eval_label, width="stretch", type="primary",
                                 key=f"enrich_{result.get('paper_id', '')}"):
                        with st.spinner("正在从 Qdrant 读取 chunks 并执行 Map-Reduce（约 60~90 秒）…"):
                            enrich_state = enrich_result_quality(
                                result,
                                enrich_runner=evaluate_and_enrich_from_qdrant,
                            )
                        st.session_state.result = enrich_state["result"]
                        st.rerun()

            st.divider()

            # 保存按钮
            quality = result.get("quality", {})
            if quality.get("category"):
                col_save, _ = st.columns([1, 3])
                with col_save:
                    if st.button("💾 保存解析结果", width="stretch",
                                 key=f"save_{result.get('paper_id', '')}"):
                        fname = st.session_state.orig_filename or result.get("paper_id", "result")
                        save_state = save_result_snapshot(
                            result,
                            fname,
                            snapshot_builder=build_snapshot,
                            result_saver=save_result,
                        )
                        st.session_state.pending_save_filename = save_state["pending_save_filename"]
                        st.session_state.save_success_msg = save_state["success_message"]
                        st.rerun()

            if st.session_state.pending_save_filename:
                pfname = st.session_state.pending_save_filename
                st.warning(f"文件 `{pfname}.json` 已存在，是否覆盖？")
                col_ov, col_ca, _ = st.columns([1, 1, 3])
                with col_ov:
                    if st.button("覆盖", key=f"overwrite_{pfname}"):
                        fname = pfname or "result"
                        save_state = overwrite_result_snapshot(
                            result,
                            fname,
                            snapshot_builder=build_snapshot,
                            result_overwriter=overwrite_result,
                        )
                        st.session_state.pending_save_filename = save_state["pending_save_filename"]
                        st.session_state.save_success_msg = save_state["success_message"]
                        st.rerun()
                with col_ca:
                    if st.button("取消", key=f"cancel_save_{pfname}"):
                        st.session_state.pending_save_filename = ""
                        st.rerun()

            if st.session_state.save_success_msg:
                st.success(st.session_state.save_success_msg)
                if st.button("清除消息", key="clear_save_msg"):
                    st.session_state.save_success_msg = ""
                    st.rerun()

    with tabs[1]:
        text_chunks = [c for c in result.get("merged_chunks", []) if c.get("content_type") == "text"]
        render_text_tab(text_chunks, verdicts)

    with tabs[2]:
        render_image_tab(result)

    with tabs[3]:
        table_chunks = [c for c in result.get("merged_chunks", []) if c.get("content_type") == "table"]
        render_table_tab(table_chunks, verdicts)

    with tabs[4]:
        render_reconstruction_tab(result)

    with tabs[5]:
        render_search_tab(result.get("paper_id", ""))

    with tabs[6]:
        render_external_import_tab()


if __name__ == "__main__":
    main()
