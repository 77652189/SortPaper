"""
SortPaper — Streamlit 前端
功能：导入 PDF → 解析（预览 / 完整流水线）→ 查看结果
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from app_utils import (
    build_paper_id, _chunk_to_dict, build_snapshot, save_result, overwrite_result,
    qdrant_search, RESULTS_DIR,
)
from app_pipeline import (
    run_preview, run_pipeline, store_parsed_chunks,
    evaluate_parsed_chunks, _process_one_pdf,
)
from app_ui import (
    type_badge, verdict_badge, category_label,
    render_overview, render_chunk_card,
    render_reconstruction_tab, render_text_tab, render_table_tab, render_image_tab,
    render_search_tab, render_standalone_search_tab,
    render_saved_tab, render_saved_detail,
)
from app_sidebar import sidebar

logger = logging.getLogger(__name__)

# ─── 页面配置 ──────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SortPaper",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 常量 ──────────────────────────────────────────────────────────────────────

SAMPLE_DIR = Path("data/sample_papers")
LOGO = "📄"

# ─── 主界面 ────────────────────────────────────────────────────────────────────

def main() -> None:
    sb = sidebar()
    pdf_bytes: bytes | None = sb["pdf_bytes"]
    paper_id: str = sb["paper_id"]
    mode: str = sb["mode"]
    parse_clicked: bool = sb["parse_clicked"]
    filename: str = sb["filename"]
    batch_info: dict | None = sb.get("batch")

    if "result" not in st.session_state:
        st.session_state.result = None
    if "current_paper" not in st.session_state:
        st.session_state.current_paper = None
    if "orig_filename" not in st.session_state:
        st.session_state.orig_filename = ""
    if "selected_saved" not in st.session_state:
        st.session_state.selected_saved = None
    if "pending_save_filename" not in st.session_state:
        st.session_state.pending_save_filename = ""
    if "save_success_msg" not in st.session_state:
        st.session_state.save_success_msg = ""
    if "batch_results" not in st.session_state:
        st.session_state.batch_results: list[dict] = []

    if filename:
        st.session_state.orig_filename = filename

    if paper_id and paper_id != st.session_state.current_paper:
        st.session_state.result = None
        st.session_state.current_paper = paper_id
        st.session_state.selected_saved = None

    PIPELINE_TIMEOUT = 900

    # ── 批量模式 ──
    if batch_info and batch_info.get("pending"):
        batch_files = batch_info.get("files", [])
        if batch_files:
            progress = st.progress(0, "批量入库中…")
            status = st.empty()
            st.session_state.batch_results = []
            success_n = skip_n = fail_n = 0

            for i, pdf_file in enumerate(batch_files):
                pct = (i + 1) / len(batch_files)
                status.info(f"({i+1}/{len(batch_files)}) 处理: {pdf_file.name}")

                try:
                    file_bytes = pdf_file.read()
                    pid = build_paper_id(file_bytes)

                    from src.store.qdrant_store import QdrantStore
                    from qdrant_client.http import models as qdrant_models
                    dup_store = QdrantStore()
                    dup_count = dup_store.client.count(
                        collection_name=dup_store.collection,
                        count_filter=qdrant_models.Filter(
                            must=[qdrant_models.FieldCondition(
                                key="paper_id", match=qdrant_models.MatchValue(value=pid),
                            )],
                        ),
                    ).count
                    if dup_count > 0:
                        skip_n += 1
                        st.session_state.batch_results.append({
                            "file": pdf_file.name, "paper_id": pid, "status": "skip",
                            "reason": f"已存在 {dup_count} 条记录",
                        })
                        progress.progress(pct, f"({i+1}/{len(batch_files)}) {pdf_file.name} — 跳过（已入库）")
                        continue

                    t_start = time.time()
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        fut = pool.submit(_process_one_pdf, file_bytes, pid, pdf_file.name)
                        try:
                            q_result = fut.result(timeout=PIPELINE_TIMEOUT)
                        except FutureTimeoutError:
                            fail_n += 1
                            st.session_state.batch_results.append({
                                "file": pdf_file.name, "paper_id": pid, "status": "fail",
                                "reason": "超时",
                            })
                            progress.progress(pct, f"({i+1}/{len(batch_files)}) {pdf_file.name} — 超时")
                            continue

                    elapsed = time.time() - t_start
                    success_n += 1
                    st.session_state.batch_results.append({
                        "file": pdf_file.name, "paper_id": pid,
                        "status": "ok",
                        "category": q_result.get("category", "?"),
                        "chunks": q_result.get("chunks", 0),
                        "time": f"{elapsed:.0f}s",
                    })
                except Exception as exc:
                    fail_n += 1
                    st.session_state.batch_results.append({
                        "file": pdf_file.name, "paper_id": pid if 'pid' in dir() else "?",
                        "status": "fail", "reason": str(exc)[:200],
                    })

                progress.progress(pct)

            progress.empty()
            status.empty()

            st.markdown(f"## ✅ 批量处理完成")
            st.markdown(f"成功 **{success_n}** | 跳过（已入库）**{skip_n}** | 失败 **{fail_n}**")

            if st.session_state.batch_results:
                import pandas as pd
                df = pd.DataFrame(st.session_state.batch_results)
                st.dataframe(df, use_container_width=True, hide_index=True)

            st.session_state.batch_trigger = False
        return

    # ── 单篇模式 ──
    if parse_clicked and pdf_bytes:
        from src.store.qdrant_store import QdrantStore
        from qdrant_client.http import models as qdrant_models
        dup_store = QdrantStore()
        dup_count = dup_store.client.count(
            collection_name=dup_store.collection,
            count_filter=qdrant_models.Filter(
                must=[qdrant_models.FieldCondition(
                    key="paper_id", match=qdrant_models.MatchValue(value=paper_id),
                )],
            ),
        ).count
        if dup_count > 0 and mode != "一键入库":
            force_key = f"force_reparse_{paper_id}"
            if not st.session_state.get(force_key):
                st.warning(
                    f"⚠️ 该论文已存在 {dup_count} 条入库记录（paper_id: `{paper_id}`）。"
                )
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

        mode_key = "preview" if mode == "快速预览" else "pipeline"

        start_ts = time.time()
        try:
            if mode_key == "preview":
                with st.spinner("快速预览中…"):
                    result = run_preview(pdf_bytes)
            else:
                timeout_banner = st.info(
                    "⏳ 解析进行中…每次 API 调用最长等待 60 秒，"
                    f"总超时 **{PIPELINE_TIMEOUT // 60} 分钟**。请勿刷新页面。",
                )
                pipeline_spinner = "📄 阶段 1/3: 解析中（LLM Judge · VisionParser）…" if mode == "一键入库" else "解析中（LLM Judge · VisionParser）…"
                with st.spinner(pipeline_spinner):
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        fut = pool.submit(run_pipeline, pdf_bytes, paper_id, filename)
                        try:
                            result = fut.result(timeout=PIPELINE_TIMEOUT)
                        except FutureTimeoutError:
                            timeout_banner.error(
                                f"⏰ **解析超时**（已等待 {PIPELINE_TIMEOUT // 60} 分钟）\n\n"
                                "**可能原因：** DeepSeek / DashScope API 响应过慢或网络不稳定。\n\n"
                                "**建议：**\n"
                                "- 确认 `.env` 中 `DEEPSEEK_API_KEY` 与 `DASHSCOPE_API_KEY` 均有效且有余额\n"
                                "- 稍后重试，或改用「快速预览」模式跳过所有 LLM 调用"
                            )
                            return
                timeout_banner.empty()
            elapsed = time.time() - start_ts
            result["_elapsed"] = elapsed
            st.caption(f"✅ 解析完成，耗时 {elapsed:.1f} 秒")

            result["mode"] = "auto" if mode == "一键入库" else result.get("mode", "pipeline")

            if mode == "一键入库":
                steps = st.empty()
                steps.info("📊 阶段 2/3: 质量评估中（Map-Reduce）…")
                q_result = evaluate_parsed_chunks(result)
                result["quality"] = q_result

                steps.info("📥 阶段 3/3: 入库到 Qdrant…")
                store_parsed_chunks(result)

                steps.info("💾 保存解析结果…")
                snapshot = build_snapshot(result, filename)
                save_result(snapshot, filename)

                steps.empty()
                result["_elapsed"] = time.time() - start_ts
                stored_key = f"stored_{paper_id}"
                st.session_state[stored_key] = True
                st.success(
                    f"✅ 一键入库完成 | "
                    f"分类: {q_result.get('category', '?')} | "
                    f"可信度: {q_result.get('credibility', '?')} | "
                    f"已保存"
                )

            st.session_state.result = result
        except Exception as exc:
            st.error(f"解析失败：{exc}")
            logging.exception("parse error")
            return

    result = st.session_state.result

    if result is None:
        welcome_tabs = st.tabs(["🏠 欢迎", "🔍 语义检索", "💾 已保存"])
        with welcome_tabs[0]:
            st.markdown("## 欢迎使用 SortPaper 📄")
            st.markdown(
                """
                **使用步骤：**
                1. 在左侧选择或上传 PDF 文件
                2. 选择解析模式（快速预览 / 完整流水线）
                3. 点击 **🚀 开始解析**
                4. 在各标签页查看解析结果

                ---
                **快速预览**：仅调用本地解析器（PyMuPDF + pdfplumber），即时出结果，无需 API 配额。
                **完整流水线**：调用 LLM Judge 评估质量，图片调用 VisionParser 生成图片描述，最终写入 Qdrant 向量索引，支持语义检索。
                """
            )
        with welcome_tabs[1]:
            render_standalone_search_tab()
        with welcome_tabs[2]:
            render_saved_tab()
        return

    # ── 结果展示 ──
    verdicts = result.get("verdicts", {})
    is_pipeline = result.get("mode") in ("pipeline", "auto")
    is_manual_pipeline = result.get("mode") == "pipeline"

    tabs = st.tabs(["📊 概览", "📝 文本块", "🖼️ 图片", "📋 表格", "📐 PDF 重建", "🔍 语义检索", "💾 已保存"])

    with tabs[0]:
        render_overview(result)

        if is_manual_pipeline:
            quality = result.get("quality", {})
            has_quality = bool(quality.get("category"))
            stored_key = f"stored_{result.get('paper_id', '')}"
            is_stored = bool(st.session_state.get(stored_key))

            st.divider()

            if not has_quality:
                st.info("📊 请运行质量评估以获得分类、摘要和可信度")
                col_btn, _ = st.columns([1, 3])
                with col_btn:
                    if st.button("📊 质量评估", use_container_width=True, type="primary",
                                 key=f"eval_{result.get('paper_id', '')}"):
                        with st.spinner("正在执行 Map-Reduce 质量评估（约 60~90 秒）…"):
                            eval_result = evaluate_parsed_chunks(result)
                        result["quality"] = eval_result
                        st.session_state.result = result
                        st.rerun()

            elif not is_stored:
                st.info("📥 质量评估已完成，请入库到 Qdrant")
                col_btn, _ = st.columns([1, 3])
                with col_btn:
                    if st.button("📥 入库到 Qdrant", use_container_width=True, type="primary",
                                 key=f"store_{result.get('paper_id', '')}"):
                        with st.spinner("正在检查重复并写入 Qdrant…"):
                            store_r = store_parsed_chunks(result)
                        st.session_state[stored_key] = store_r
                        st.rerun()

            elif is_stored:
                store_r = st.session_state[stored_key]
                if store_r.get("duplicate"):
                    st.warning(f"⚠️ 该论文已入库（{store_r.get('existing_count', '?')} 条记录），请删除后重新入库或使用覆盖功能。")
                else:
                    st.success(f"✅ 已入库 {store_r.get('stored', 0)} chunks + {store_r.get('degraded', 0)} degraded")

            st.divider()

            # 保存按钮
            quality = result.get("quality", {})
            if quality.get("category"):
                col_save, _ = st.columns([1, 3])
                with col_save:
                    if st.button("💾 保存解析结果", use_container_width=True,
                                 key=f"save_{result.get('paper_id', '')}"):
                        fname = st.session_state.orig_filename or result.get("paper_id", "result")
                        snapshot = build_snapshot(result, fname)
                        saved_path = save_result(snapshot, fname)
                        if saved_path:
                            st.session_state.pending_save_filename = ""
                            st.session_state.save_success_msg = f"✅ 已保存: {saved_path}"
                            st.rerun()
                        else:
                            st.session_state.pending_save_filename = fname
                            st.rerun()

            if st.session_state.pending_save_filename:
                pfname = st.session_state.pending_save_filename
                st.warning(f"文件 `{pfname}.json` 已存在，是否覆盖？")
                col_ov, col_ca, _ = st.columns([1, 1, 3])
                with col_ov:
                    if st.button("覆盖", key=f"overwrite_{pfname}"):
                        fname = pfname or "result"
                        snapshot = build_snapshot(result, fname)
                        overwrite_result(snapshot, fname)
                        st.session_state.pending_save_filename = ""
                        st.session_state.save_success_msg = f"✅ 已覆盖保存: {fname}"
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
        render_saved_tab()

if __name__ == "__main__":
    main()
