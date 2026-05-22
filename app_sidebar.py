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
        batch: {files: list[Path], pending: bool} | None,
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
        source = st.radio("PDF 来源", ["上传文件", "示例论文", "📁 批量导入"], horizontal=True)

        pdf_bytes: bytes | None = None
        filename = ""

        if source == "📁 批量导入":
            # ── 批量导入 ──
            batch_files = _render_batch_import()
            result["batch"] = {"files": batch_files, "pending": False}
            if "batch_trigger" not in st.session_state:
                st.session_state.batch_trigger = False

            st.divider()

            # 一键入库按钮
            can_batch = bool(batch_files)
            if can_batch:
                effective_batches = (len(batch_files) + BATCH_MAX_PAPER_WORKERS - 1) // BATCH_MAX_PAPER_WORKERS
                est_min = effective_batches * BATCH_ESTIMATED_MINUTES_PER_PAPER
                st.caption(f"预计耗时约 {est_min} 分钟（论文并发 {BATCH_MAX_PAPER_WORKERS}）")
            batch_btn = st.button(
                "🚀 一键批量入库", disabled=not can_batch,
                use_container_width=True, type="primary",
            )
            if batch_btn and can_batch:
                st.session_state.batch_trigger = True
                result["batch"]["pending"] = True
            if not can_batch:
                st.caption("请先扫描并选择 PDF 文件")

            result["parse_clicked"] = False
            result["paper_id"] = "batch"
            result["filename"] = f"批量 {len(batch_files)} 篇"
            st.divider()
            _render_vector_library_ui()
            return result

        else:
            # ── 单篇模式（上传 / 示例）──
            if source == "上传文件":
                uploaded = st.file_uploader("选择 PDF 文件", type="pdf", label_visibility="collapsed")
                if uploaded:
                    pdf_bytes = uploaded.read()
                    filename = uploaded.name
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

            if pdf_bytes:
                paper_id = build_paper_id(pdf_bytes)
                st.caption(f"📎 {filename}")
                st.caption(f"ID: `{paper_id}`")
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
                captions=["仅解析，无 LLM（免费）", "Judge + 质量评估（手动入库）", "解析→评估→入库 全自动"],
                label_visibility="collapsed",
            )
            result["mode"] = mode

            if mode == "完整流水线":
                st.info(
                    "⏱️ 预计 **1 ~ 2 分钟** | LLM Judge + VisionParser\n"
                    "质量评估和入库请在解析完成后手动触发。\n"
                    f"需 `DEEPSEEK_API_KEY`；入库需 `{EMBEDDING_API_KEY_ENV}`（{EMBEDDING_PROVIDER}:{EMBEDDING_MODEL}）。",
                )
            elif mode == "一键入库":
                st.info(
                    "⏱️ 预计 **2 ~ 3 分钟**\n"
                    "自动完成：解析 → Judge → 质量评估 → Qdrant 入库。\n"
                    f"已入库的论文会检测重复；向量模型：`{EMBEDDING_PROVIDER}:{EMBEDDING_MODEL}`。",
                )

            st.divider()

            can_parse = pdf_bytes is not None
            parse_clicked = st.button(
                "🚀 开始解析", disabled=not can_parse,
                use_container_width=True, type="primary",
            )
            if not can_parse:
                st.caption("请先选择 PDF 文件")

            result["parse_clicked"] = parse_clicked

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
        for p in papers:
            pid = p["paper_id"]
            with st.expander(f"📄 {p['paper_title'][:50]}", expanded=False):
                st.caption(f"Hash: `{pid}`")
                st.caption(f"Chunks: {p['chunk_count']}")
                if st.button("🗑️ 删除此文献", key=f"del_{pid}"):
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


