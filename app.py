"""
SortPaper — Streamlit 前端
功能：导入 PDF → 解析（预览 / 完整流水线）→ 查看结果
"""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, TimeoutError as FutureTimeoutError, wait

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

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
from app_pipeline import (
    run_preview, run_pipeline, store_parsed_chunks,
    evaluate_and_enrich_from_qdrant, _process_one_pdf,
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

LOGO = "📄"


def _qdrant_paper_count(paper_id: str) -> int:
    from qdrant_client.http import models as qdrant_models
    from src.store.qdrant_store import QdrantStore

    store = QdrantStore()
    return store.client.count(
        collection_name=store.collection,
        count_filter=qdrant_models.Filter(
            must=[qdrant_models.FieldCondition(
                key="paper_id", match=qdrant_models.MatchValue(value=paper_id),
            )],
        ),
    ).count


def _existing_paper_reason(paper_id: str) -> str:
    qdrant_count = _qdrant_paper_count(paper_id)
    if qdrant_count > 0:
        return f"已入库 {qdrant_count} 条记录"
    return ""


def _embedding_api_ready() -> bool:
    return bool((os.getenv(EMBEDDING_API_KEY_ENV) or os.getenv(f"\ufeff{EMBEDDING_API_KEY_ENV}") or "").strip())


def _embedding_preflight_error() -> str:
    if not _embedding_api_ready():
        return f"缺少 {EMBEDDING_API_KEY_ENV}，无法生成向量入库"
    try:
        from src.store.qdrant_store import QdrantStore

        QdrantStore().embed("SortPaper embedding preflight")
    except Exception as exc:
        return f"Embedding 预检失败：{type(exc).__name__}: {str(exc)[:300]}"
    return ""


def _process_batch_job(file_bytes: bytes, paper_id: str, filename: str, mode: str) -> dict:
    t0 = time.time()
    if mode == "快速预览":
        result = run_preview(file_bytes)
        result["filename"] = filename
        chunks = (
            len(result.get("text_chunks", []))
            + len(result.get("table_chunks", []))
            + len(result.get("image_chunks", []))
        )
        return {
            "status": "ok",
            "reason": "",
            "category": "preview",
            "chunks": chunks,
            "stored": 0,
            "degraded": 0,
            "store_failed": 0,
            "store_attempted": 0,
            "excluded": 0,
            "parse_seconds": round(time.time() - t0, 1),
            "eval_seconds": 0,
            "store_seconds": 0,
            "total_seconds": round(time.time() - t0, 1),
            "preview_result": result,
        }

    if mode == "完整流水线":
        stage_start = time.time()
        result = run_pipeline(file_bytes, paper_id, filename)
        result["filename"] = filename
        parse_seconds = time.time() - stage_start
        save_result(build_snapshot(result, filename), filename)
        return {
            "status": "ok",
            "reason": "",
            "category": "parsed",
            "credibility": 0,
            "chunks": len(result.get("merged_chunks", [])),
            "stored": 0,
            "degraded": 0,
            "store_failed": 0,
            "store_attempted": 0,
            "excluded": 0,
            "parse_seconds": round(parse_seconds, 1),
            "eval_seconds": 0,
            "store_seconds": 0,
            "total_seconds": round(time.time() - t0, 1),
            "preview_result": result,
        }

    return _process_one_pdf(file_bytes, paper_id, filename)


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
    if "batch_preview_results" not in st.session_state:
        st.session_state.batch_preview_results = {}

    if filename:
        st.session_state.orig_filename = filename

    if paper_id and paper_id != st.session_state.current_paper:
        st.session_state.result = None
        st.session_state.current_paper = paper_id
        st.session_state.selected_saved = None

    # ── 批量模式 ──
    if batch_info and batch_info.get("pending"):
        batch_files = batch_info.get("files", [])
        batch_mode = str(batch_info.get("mode") or mode)
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
            for i, pdf_file in enumerate(batch_files):
                status.info(f"({i+1}/{len(batch_files)}) 预检查: {pdf_file.name}")
                try:
                    file_bytes = pdf_file.getvalue() if hasattr(pdf_file, "getvalue") else pdf_file.read()
                    pid = build_paper_id(file_bytes)
                    if pid in seen_paper_ids:
                        first_seen = seen_paper_ids[pid]
                        skip_n += 1
                        skip_batch_duplicate_n += 1
                        st.session_state.batch_results.append({
                            "no": i + 1, "file": pdf_file.name, "paper_id": pid,
                            "status": "skip",
                            "skip_type": "本批次重复",
                            "duplicate_of": f"{first_seen['no']}. {first_seen['file']}",
                            "reason": f"本批次重复（同第 {first_seen['no']} 个文件：{first_seen['file']}）",
                        })
                        progress.progress((i + 1) / len(batch_files), f"预检查 {i+1}/{len(batch_files)}")
                        continue
                    seen_paper_ids[pid] = {"no": i + 1, "file": pdf_file.name}
                    existing_reason = _existing_paper_reason(pid) if batch_mode == "一键入库" else ""
                    if existing_reason:
                        skip_n += 1
                        skip_existing_n += 1
                        st.session_state.batch_results.append({
                            "no": i + 1, "file": pdf_file.name, "paper_id": pid,
                            "status": "skip",
                            "skip_type": "已入库",
                            "duplicate_of": "",
                            "reason": existing_reason,
                        })
                        progress.progress((i + 1) / len(batch_files), f"预检查 {i+1}/{len(batch_files)}")
                        continue
                    prepared_jobs.append({
                        "no": i + 1,
                        "file": pdf_file.name,
                        "paper_id": pid,
                        "bytes": file_bytes,
                    })
                except Exception as exc:
                    fail_n += 1
                    st.session_state.batch_results.append({
                        "no": i + 1, "file": pdf_file.name, "paper_id": "?",
                        "status": "fail", "reason": f"预检查失败: {str(exc)[:160]}",
                    })
                progress.progress((i + 1) / len(batch_files), f"预检查 {i+1}/{len(batch_files)}")

            if st.session_state.batch_results:
                import pandas as pd
                results_view.dataframe(
                    pd.DataFrame(sorted(st.session_state.batch_results, key=lambda row: row.get("no", 0))),
                    width="stretch",
                    hide_index=True,
                )

            embedding_error = _embedding_preflight_error() if prepared_jobs and batch_mode == "一键入库" else ""
            if prepared_jobs and embedding_error:
                fail_n += len(prepared_jobs)
                reason = embedding_error
                for job in prepared_jobs:
                    st.session_state.batch_results.append({
                        "no": job["no"],
                        "file": job["file"],
                        "paper_id": job["paper_id"],
                        "status": "fail",
                        "reason": reason,
                    })
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
                                q_result = fut.result()
                                elapsed = float(q_result.get("total_seconds", 0) or 0)
                                result_status = str(q_result.get("status", "ok") or "ok")
                                if result_status in {"ok", "partial"}:
                                    success_n += 1
                                elif result_status == "skip":
                                    skip_n += 1
                                    reason_text = str(q_result.get("reason", ""))
                                    if reason_text.startswith("已入库"):
                                        skip_existing_n += 1
                                else:
                                    fail_n += 1
                                preview_result = q_result.pop("preview_result", None)
                                if preview_result:
                                    st.session_state.batch_preview_results[job["paper_id"]] = {
                                        "no": job["no"],
                                        "file": job["file"],
                                        "result": preview_result,
                                    }
                                st.session_state.batch_results.append({
                                    "no": job["no"],
                                    "file": job["file"],
                                    "paper_id": job["paper_id"],
                                    "status": result_status,
                                    "skip_type": "已入库" if result_status == "skip" and str(q_result.get("reason", "")).startswith("已入库") else "",
                                    "duplicate_of": "",
                                    "category": q_result.get("category", "?"),
                                    "chunks": q_result.get("chunks", 0),
                                    "stored": q_result.get("stored", 0),
                                    "degraded": q_result.get("degraded", 0),
                                    "store_failed": q_result.get("store_failed", 0),
                                    "store_attempted": q_result.get("store_attempted", 0),
                                    "excluded": q_result.get("excluded", 0),
                                    "parse": f"{float(q_result.get('parse_seconds', 0) or 0):.0f}s",
                                    "eval": f"{float(q_result.get('eval_seconds', 0) or 0):.0f}s",
                                    "store": f"{float(q_result.get('store_seconds', 0) or 0):.0f}s",
                                    "time": f"{elapsed:.0f}s",
                                    "reason": q_result.get("reason", ""),
                                })
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
                    f"总超时 **{PIPELINE_TIMEOUT_SECONDS // 60} 分钟**。请勿刷新页面。",
                )
                pipeline_spinner = (
                    "📄 阶段 1/2: 解析中（LLM Judge · VisionParser）…"
                    if mode == "一键入库"
                    else "解析中（LLM Judge · VisionParser）…"
                )
                with st.spinner(pipeline_spinner):
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        fut = pool.submit(run_pipeline, pdf_bytes, paper_id, filename)
                        try:
                            result = fut.result(timeout=PIPELINE_TIMEOUT_SECONDS)
                        except FutureTimeoutError:
                            timeout_banner.error(
                                f"⏰ **解析超时**（已等待 {PIPELINE_TIMEOUT_SECONDS // 60} 分钟）\n\n"
                                "**可能原因：** DeepSeek / OpenAI API 响应过慢或网络不稳定。\n\n"
                                "**建议：**\n"
                                f"- 确认 `.env` 中 `DEEPSEEK_API_KEY` 与 `{EMBEDDING_API_KEY_ENV}` 均有效且有余额\n"
                                "- 稍后重试，或改用「快速预览」模式跳过图片 VisionParser 与入库流程；表格质量仍会调用 LLM Judge"
                            )
                            return
                timeout_banner.empty()
            elapsed = time.time() - start_ts
            result["_elapsed"] = elapsed
            st.caption(f"✅ 解析完成，耗时 {elapsed:.1f} 秒")

            result["mode"] = "auto" if mode == "一键入库" else result.get("mode", "pipeline")

            if mode == "一键入库":
                steps = st.empty()
                steps.info("📥 阶段 2/2: 入库到 Qdrant…")
                store_r = store_parsed_chunks(result)

                steps.info("💾 保存解析结果…")
                snapshot = build_snapshot(result, filename)
                save_result(snapshot, filename)

                steps.empty()
                result["_elapsed"] = time.time() - start_ts
                stored_key = f"stored_{paper_id}"
                st.session_state[stored_key] = store_r
                st.success(
                    "✅ 一键入库完成，可立即检索；质量分析状态：待分析"
                )

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

        welcome_tabs = st.tabs(["🏠 欢迎", "🔍 语义检索", "💾 已保存"])
        with welcome_tabs[0]:
            st.markdown("## 欢迎使用 SortPaper 📄")
            st.markdown(
                """
                **使用步骤：**
                1. 在左侧选择或上传 PDF 文件
                2. 选择解析模式（快速预览 / 完整流水线 / 一键入库）
                3. 点击 **🚀 开始解析**
                4. 在各标签页查看解析结果

                ---
                **快速预览**：解析文本/表格；仅表格质量诊断会调用 LLM Judge，图片只生成占位信息，不写入向量库。
                **完整流水线**：解析文本、表格和图片，并运行 chunk-level Judge；不自动入库，也不执行 Map-Reduce 质量分析。
                **一键入库**：解析与 Judge 后立即写入 Qdrant，可马上检索；质量分析状态先标记为待分析，可之后单独补充。
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

    tabs = st.tabs(["📊 概览", "📝 文本块", "🖼️ 图片", "📋 表格", "📐 PDF 重建", "🔍 语义检索", "💾 已保存"])

    with tabs[0]:
        render_overview(result)

        if is_pipeline:
            quality = result.get("quality", {})
            has_quality = bool(quality.get("category"))
            stored_key = f"stored_{result.get('paper_id', '')}"
            stored_state = st.session_state.get(stored_key)
            try:
                is_stored = bool(stored_state) or _qdrant_paper_count(result.get("paper_id", "")) > 0
            except Exception:
                is_stored = bool(stored_state)

            st.divider()

            if not is_stored:
                st.info("📥 请先入库到 Qdrant；入库后可立即检索，质量分析可稍后补充")
                col_btn, _ = st.columns([1, 3])
                with col_btn:
                    if st.button("📥 入库到 Qdrant", width="stretch", type="primary",
                                 key=f"store_{result.get('paper_id', '')}"):
                        with st.spinner("正在检查重复并写入 Qdrant…"):
                            store_r = store_parsed_chunks(result)
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
                            eval_result = evaluate_and_enrich_from_qdrant(result.get("paper_id", ""))
                        result["quality"] = eval_result
                        st.session_state.result = result
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
