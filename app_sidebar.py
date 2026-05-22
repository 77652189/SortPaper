"""
SortPaper Sidebar — 侧边栏组件。
"""
from __future__ import annotations

import logging

import streamlit as st

from app_config import (
    BATCH_ESTIMATED_MINUTES_PER_PAPER,
    BATCH_MAX_PAPER_WORKERS,
    EMBEDDING_API_KEY_ENV,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    RESULTS_DIR,
    SAMPLE_DIR,
)
from app_utils import build_paper_id

logger = logging.getLogger(__name__)
LOGO = "📄"

def sidebar() -> dict:
    """返回 {
        pdf_bytes, paper_id, mode, parse_clicked, filename,
        batch: {files: list[UploadedFile], pending: bool, mode: str} | None,
    }。"""
    result: dict = {
        "pdf_bytes": None, "paper_id": "", "mode": "快速预览",
        "parse_clicked": False, "filename": "",
        "batch": None,
    }

    with st.sidebar:
        st.title(f"{LOGO} SortPaper")
        st.caption("学术论文解析与检索工具")
        st.divider()

        # ── 文件来源 ──
        source = st.radio("PDF 来源", ["上传文件", "示例论文"], horizontal=True)

        pdf_bytes: bytes | None = None
        filename = ""
        uploaded_files: list = []

        if source == "上传文件":
            uploaded = st.file_uploader(
                "选择 PDF 文件",
                type="pdf",
                accept_multiple_files=True,
                label_visibility="collapsed",
                key="pdf_uploader",
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
                    "选择示例论文", options=samples,
                    format_func=lambda p: p.name, label_visibility="collapsed",
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
            st.caption(f"📎 {filename}")
            st.caption(f"ID: `{paper_id}`")
        elif is_batch:
            paper_id = "batch"
            filename = f"批量 {len(uploaded_files)} 篇"
            effective_batches = (len(uploaded_files) + BATCH_MAX_PAPER_WORKERS - 1) // BATCH_MAX_PAPER_WORKERS
            est_min = effective_batches * BATCH_ESTIMATED_MINUTES_PER_PAPER
            st.caption(f"📎 批量 {len(uploaded_files)} 篇")
            st.caption(f"预计耗时约 {est_min} 分钟（论文并发 {BATCH_MAX_PAPER_WORKERS}）")
        else:
            paper_id = ""

        result["pdf_bytes"] = pdf_bytes
        result["paper_id"] = paper_id
        result["filename"] = filename

        st.divider()

        # ── 解析模式 ──
        st.subheader("解析模式")
        mode = st.radio(
            "模式",
            ["快速预览", "完整流水线", "一键入库"],
            captions=[
                "解析文本/表格；仅表格 Judge",
                "解析文本/表格/图片 + Judge；不入库",
                "解析 → Judge → Qdrant 入库",
            ],
            label_visibility="collapsed",
        )
        result["mode"] = mode

        if mode == "快速预览":
            st.info(
                "⏱️ 快速解析文本/表格；仅表格质量诊断会调用 LLM Judge，图片只生成占位信息，不写入向量库。"
            )
        elif mode == "完整流水线":
            st.info(
                "⏱️ 单篇通常 **1 ~ 2 分钟**；批量时按论文并发处理。\n"
                "会解析文本、表格和图片并运行 LLM Judge，但不会自动写入向量库或执行 Map-Reduce。"
            )
        elif mode == "一键入库":
            st.info(
                "⏱️ 自动完成：解析 → Judge → Qdrant 入库；质量分析可稍后单独运行。\n"
                f"已入库论文会跳过；向量模型：`{EMBEDDING_PROVIDER}:{EMBEDDING_MODEL}`。"
            )

        st.divider()

        selected_count = len(uploaded_files) if source == "上传文件" else int(pdf_bytes is not None)
        can_parse = selected_count > 0
        button_label = "🚀 开始批量处理" if selected_count > 1 else "🚀 开始解析"
        parse_clicked = st.button(
            button_label, disabled=not can_parse,
            use_container_width=True, type="primary",
        )
        if not can_parse:
            st.caption("请先选择 PDF 文件")

        result["parse_clicked"] = parse_clicked
        if is_batch:
            result["batch"] = {
                "files": uploaded_files,
                "pending": parse_clicked,
                "mode": mode,
            }

        st.divider()
        _render_vector_library_ui()

    return result


def _render_batch_import() -> list:
    """渲染批量导入 UI，返回用户上传的 UploadedFile 列表。"""
    uploaded = st.file_uploader(
        "拖拽或选择 PDF 文件", type="pdf",
        accept_multiple_files=True, label_visibility="collapsed",
        key="batch_uploader",
    )
    if uploaded:
        st.caption(f"已选择 {len(uploaded)} 个 PDF 文件")
    else:
        st.caption("支持多选或拖拽 PDF 文件")
    return list(uploaded) if uploaded else []


def _render_vector_library_ui() -> None:
    """展示向量库文献列表 + 删除/清空功能。"""
    from app_pipeline import evaluate_and_enrich_from_qdrant
    from src.store.qdrant_store import QdrantStore

    st.subheader("向量库管理")

    try:
        store = QdrantStore()
        total = store.count
    except Exception:
        st.warning("Qdrant 未连接")
        return

    st.caption(f"共 {total} 条记录")

    papers = store.list_papers()
    if not papers:
        st.caption("暂无已录入文献")
    else:
        pending_papers = [p for p in papers if p.get("enrichment_status", "done") == "pending"]
        done_count = len(papers) - len(pending_papers)
        st.caption(f"论文 {len(papers)} 篇；待质量分析 {len(pending_papers)} 篇；已完成 {done_count} 篇")
        if st.button(
            "📊 一键质量分析全部待分析论文",
            disabled=not pending_papers,
            use_container_width=True,
            type="primary",
            key="enrich_all_pending_papers",
        ):
            progress = st.progress(0, "准备质量分析…")
            status_box = st.empty()
            success_n = 0
            fail_n = 0
            failures: list[str] = []
            total_pending = len(pending_papers)
            for idx, paper in enumerate(pending_papers, start=1):
                pid = paper["paper_id"]
                title = paper.get("paper_title", pid)
                progress.progress((idx - 1) / total_pending, f"质量分析 {idx}/{total_pending}: {title[:36]}")
                try:
                    result = evaluate_and_enrich_from_qdrant(pid)
                    if result.get("error"):
                        fail_n += 1
                        failures.append(f"{title}: {result.get('error')}")
                    else:
                        success_n += 1
                except Exception as exc:
                    fail_n += 1
                    failures.append(f"{title}: {exc}")
                status_box.caption(f"已完成 {idx}/{total_pending}；成功 {success_n}，失败 {fail_n}")
            progress.progress(1.0, "质量分析完成")
            if failures:
                st.warning(f"批量质量分析完成：成功 {success_n}，失败 {fail_n}")
                with st.expander("查看失败详情", expanded=False):
                    for item in failures[:20]:
                        st.caption(item)
                    if len(failures) > 20:
                        st.caption(f"其余 {len(failures) - 20} 条失败未展开显示")
            else:
                st.success(f"批量质量分析完成：成功 {success_n} 篇")
            st.rerun()

        for p in papers:
            pid = p["paper_id"]
            with st.expander(f"📄 {p['paper_title'][:50]}", expanded=False):
                st.caption(f"Hash: `{pid}`")
                st.caption(f"Chunks: {p['chunk_count']}")
                status = p.get("enrichment_status", "done")
                status_note = p.get("enrichment_status_note", "")
                if status == "pending":
                    label = "质量分析: 待分析"
                    if status_note:
                        label += f"（{status_note}）"
                else:
                    label = "质量分析: 已完成"
                st.caption(label)
                eval_label = "📊 一键质量评估" if status == "pending" else "🔁 重新质量评估"
                if st.button(eval_label, key=f"enrich_sidebar_{pid}", use_container_width=True):
                    with st.spinner("正在从 Qdrant 读取 chunks 并执行 Map-Reduce…"):
                        result = evaluate_and_enrich_from_qdrant(pid)
                    if result.get("error"):
                        st.error(f"质量评估失败：{result.get('error')}")
                    else:
                        st.success(
                            f"质量评估完成：{result.get('category', '?')} "
                            f"/ credibility={result.get('credibility', '?')}"
                        )
                        st.rerun()
                if st.button("🗑️ 删除此文献", key=f"del_{pid}", use_container_width=True):
                    try:
                        n = store.delete_by_paper_id(pid)
                        st.success(f"已删除 {n} 条记录")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"删除失败：{exc}")

    st.divider()
    # 清空按钮
    if "confirm_clear" not in st.session_state:
        st.session_state.confirm_clear = False

    if not st.session_state.confirm_clear:
        if st.button("🗑️ 清空全部向量库", use_container_width=True, type="secondary"):
            st.session_state.confirm_clear = True
            st.rerun()
    else:
        st.warning("⚠️ 将删除 Qdrant 中所有论文数据，不可恢复")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 确认清空", use_container_width=True, type="primary"):
                try:
                    before = store.count
                    store.clear()
                    st.session_state.confirm_clear = False
                    st.success(f"已清空（删除 {before} 条记录）")
                    st.rerun()
                except Exception as exc:
                    st.session_state.confirm_clear = False
                    st.error(f"清空失败：{exc}")
        with col2:
            if st.button("❌ 取消", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()


