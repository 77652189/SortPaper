"""
SortPaper Pipeline — 解析、评估、入库编排。
纯业务逻辑，不渲染 UI。
"""
from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from app_config import EMBEDDING_API_KEY_ENV
from app_utils import (
    build_paper_id, _chunk_to_dict, _generate_chunk_description,
    build_snapshot, save_result, overwrite_result,
    load_saved_list, load_saved_detail, delete_saved,
    RESULTS_DIR,
)

logger = logging.getLogger(__name__)

_STORE_LOCKS: dict[str, threading.Lock] = {}
_STORE_LOCKS_GUARD = threading.Lock()


def _paper_store_lock(paper_id: str) -> threading.Lock:
    with _STORE_LOCKS_GUARD:
        lock = _STORE_LOCKS.get(paper_id)
        if lock is None:
            lock = threading.Lock()
            _STORE_LOCKS[paper_id] = lock
        return lock


def _embedding_content_for_chunk(chunk: dict) -> str:
    raw_content = str(chunk.get("raw_content", "") or "")
    if chunk.get("content_type") != "table":
        return raw_content
    try:
        from src.judge.table_judge import build_table_embedding_text
        return build_table_embedding_text(
            raw_content=raw_content,
            metadata=chunk.get("metadata", {}) or {},
        ) or raw_content
    except Exception:
        return raw_content

def run_preview(pdf_bytes: bytes) -> dict:
    """Preview parses full text/table structure and marks Vision work without calling it."""
    import sys
    sys.path.insert(0, ".")
    from src.graph.pipeline_graph import _group_by_section, _is_noise
    from src.judge.llm_judge import LLMJudge
    from src.parsers.layout_chunk import LayoutMerger
    from src.parsers.pymupdf_parser import PyMuPDFParser
    from src.parsers.table_parser import TableParser

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)

    try:
        t0_text = time.time()
        text_chunks = PyMuPDFParser(tmp_path).parse()
        wt_text = round(time.time() - t0_text, 1)

        t0_table = time.time()
        table_chunks = TableParser(
            tmp_path,
            enable_vision_fallback=False,
        ).parse(strategy="all")
        wt_table = round(time.time() - t0_table, 1)

        image_chunks, images = _extract_image_placeholders(tmp_path)
        merged_chunks = LayoutMerger.merge(text_chunks + table_chunks + image_chunks)

        verdicts: dict[str, dict] = {}
        t0_judge = time.time()
        pdf_path_str = str(tmp_path)
        noise_ids = {c.chunk_id for c in text_chunks if _is_noise(c.raw_content)}

        tasks: list[tuple] = []
        for sec in _group_by_section(text_chunks):
            if any(c.chunk_id not in noise_ids for c in sec):
                tasks.append(("text", sec))
        for chunk in table_chunks:
            if chunk.metadata.get("table_region_discovery_only"):
                verdicts[chunk.chunk_id] = {
                    "chunk_id": chunk.chunk_id,
                    "passed": True,
                    "score": chunk.metadata.get("table_region", {}).get("confidence", 1.0),
                    "feedback": "table region discovery only; structure is not judged",
                    "issue_type": "none",
                }
            else:
                tasks.append(("table", chunk))

        def _run_judge(task):
            kind, obj = task
            if kind == "text":
                section_chunks = obj
                combined = "\n\n".join(c.raw_content for c in section_chunks)
                verdict = LLMJudge().judge("text", combined, pdf_path_str)
                return [
                    (c.chunk_id, {
                        "chunk_id": c.chunk_id,
                        "passed": verdict.passed,
                        "score": verdict.score,
                        "feedback": verdict.feedback,
                    })
                    for c in section_chunks if c.chunk_id not in noise_ids
                ]

            chunk = obj
            verdict = LLMJudge().judge("table", chunk.raw_content, pdf_path_str)
            return [(chunk.chunk_id, {
                "chunk_id": chunk.chunk_id,
                "passed": verdict.passed,
                "score": verdict.score,
                "feedback": verdict.feedback,
                "issue_type": verdict.issue_type,
            })]

        if tasks:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(5, len(tasks))) as pool:
                for pairs in pool.map(_run_judge, tasks):
                    for cid, verdict in pairs:
                        verdicts[cid] = verdict

        jt_judge = round(time.time() - t0_judge, 1)
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "paper_id": build_paper_id(pdf_bytes),
        "_pdf_path": "",
        "text_chunks": [_chunk_to_dict(c) for c in text_chunks],
        "table_chunks": [_chunk_to_dict(c) for c in table_chunks],
        "image_chunks": [_chunk_to_dict(c) for c in image_chunks],
        "merged_chunks": [_chunk_to_dict(c) for c in merged_chunks],
        "images": images,
        "verdicts": verdicts,
        "status": "preview",
        "quality": {},
        "worker_timing": {"text": wt_text, "table": wt_table, "image": 0},
        "judge_timing": {"text_table": jt_judge},
        "merge_timing": 0,
        "desc_timing": 0,
        "mode": "preview",
    }


def _extract_image_placeholders(pdf_path: Path) -> tuple[list, list[dict]]:
    """Return image placeholder chunks and metadata without calling Vision."""
    import fitz  # type: ignore
    from src.parsers.layout_chunk import LayoutChunk, infer_column

    chunks = []
    images: list[dict] = []
    with fitz.open(str(pdf_path)) as doc:
        for page_index, page in enumerate(doc, start=1):
            page_width = float(page.rect.width)
            for image_index, info in enumerate(page.get_images(full=True)):
                xref = info[0]
                rects = page.get_image_rects(xref)
                if rects:
                    bbox = tuple(float(v) for v in rects[0])
                else:
                    bbox = (0.0, 0.0, 0.0, 0.0)
                width = int(info[2])
                height = int(info[3])
                images.append({
                    "page": page_index,
                    "xref": xref,
                    "width": width,
                    "height": height,
                    "bbox": bbox,
                    "vision_needed": True,
                })
                chunks.append(LayoutChunk(
                    content_type="image",
                    raw_content="[Preview placeholder] Image detected; GPT Vision parsing is needed here.",
                    page=page_index,
                    bbox=bbox,
                    column=infer_column(page_width, bbox[0], bbox[2]),
                    order_in_page=image_index,
                    metadata={
                        "parser": "preview_placeholder",
                        "image_index": image_index,
                        "xref": xref,
                        "width": width,
                        "height": height,
                        "vision_needed": True,
                    },
                ))
    return chunks, images


def run_pipeline(
    pdf_bytes: bytes,
    paper_id: str,
    filename: str = "",
    persist_pdf: bool = True,
) -> dict:
    """Parse pipeline: parsers + judge + merge（quality 和 store 由 Streamlit 按钮手动触发）。"""
    import sys
    sys.path.insert(0, ".")
    from src.graph.pipeline_graph import build_graph

    # 保存 PDF 副本供后续质量评估使用（用原始文件名，不用 hash）
    persistent_pdf: Path | None = None
    if persist_pdf:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = (filename or paper_id).replace("/", "_").replace("\\", "_")
        persistent_pdf = RESULTS_DIR / f"{safe_name}.pdf"
        persistent_pdf.write_bytes(pdf_bytes)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)

    try:
        pipeline = build_graph()
        initial = {
            "pdf_path": str(tmp_path),
            "paper_id": paper_id,
            "text_retries": 0, "table_retries": 0, "image_retries": 0,
            "status": "processing", "output_dir": "",
        }

        final: dict = dict(initial)
        node_cnt: dict[str, int] = {}
        for event in pipeline.stream(initial, config={"recursion_limit": 100},
                                     stream_mode="updates"):
            node_name, node_data = list(event.items())[0]
            for k, v in node_data.items():
                if isinstance(v, set):
                    final[k] = final.get(k, set()) | v
                elif isinstance(v, dict) and k in ("worker_timing", "judge_timing"):
                    # 与 PipelineState reducer 一致：保留各路线最慢耗时
                    a, b = final.get(k, {}), v
                    final[k] = {key: max(a.get(key, 0), b.get(key, 0)) for key in set(a) | set(b)}
                elif isinstance(v, dict):
                    final[k] = {**final.get(k, {}), **v}
                else:
                    final[k] = v

            # 终端进度
            node_cnt[node_name] = node_cnt.get(node_name, 0) + 1
            cnt = node_cnt[node_name]
            retry_tag = f" [retry {cnt-1}]" if cnt > 1 else ""
            wt = node_data.get("worker_timing", {})
            t = wt.get("text", wt.get("table", wt.get("image", 0))) if wt else 0

            if node_name == "text_worker":
                n = len(node_data.get("text_chunks", []))
                logger.info("[text] parsed %d chunks (%.1fs)%s", n, t, retry_tag)
            elif node_name == "table_worker":
                n = len(node_data.get("table_chunks", []))
                logger.info("[table] parsed %d tables (%.1fs)%s", n, t, retry_tag)
            elif node_name == "image_worker":
                n = len(node_data.get("image_chunks", []))
                logger.info("[image] described %d images (%.1fs)%s", n, t, retry_tag)
            elif node_name == "judge_text":
                v = node_data.get("text_verdicts", [])
                p = sum(1 for x in v if x.get("passed"))
                logger.info("[judge] text: %d/%d passed%s", p, len(v), retry_tag)
            elif node_name == "judge_table":
                v = node_data.get("table_verdicts", [])
                p = sum(1 for x in v if x.get("passed"))
                types = {}
                for x in v:
                    it = x.get("issue_type", "?")
                    types[it] = types.get(it, 0) + 1
                logger.info("[judge] table: %d/%d passed%s | issue_types=%s", p, len(v), retry_tag, types)
            elif node_name == "judge_image":
                v = node_data.get("image_verdicts", [])
                p = sum(1 for x in v if x.get("passed"))
                logger.info("[judge] image: %d/%d passed%s", p, len(v), retry_tag)
            elif node_name == "merge_chunks":
                n = len(node_data.get("merged_chunks", []))
                logger.info("[merge] %d merged chunks", n)
            elif node_name == "finalize":
                status = node_data.get("status", "done")
                logger.info("[pipeline] %s", status)
    finally:
        tmp_path.unlink(missing_ok=True)

    all_verdicts: dict[str, dict] = {}
    for v in (
        final.get("text_verdicts", [])
        + final.get("table_verdicts", [])
        + final.get("image_verdicts", [])
    ):
        if v.get("chunk_id"):
            all_verdicts[v["chunk_id"]] = v

    merged = final.get("merged_chunks", [])

    # 表/图：LLM 生成一句英文描述前缀（提升检索召回率），并行调用
    from concurrent.futures import ThreadPoolExecutor as _TPE
    t_desc = time.time()
    _targets = [(i, c) for i, c in enumerate(merged) if c.content_type in ("table", "image")]
    if _targets:
        def _describe_one(item):
            idx, chunk = item
            return idx, _generate_chunk_description(chunk.content_type, chunk.raw_content)
        with _TPE(max_workers=min(5, len(_targets))) as _pool:
            for idx, desc in _pool.map(_describe_one, _targets):
                if desc:
                    merged[idx].raw_content = f"{desc}\n\n{merged[idx].raw_content}"
    desc_timing = round(time.time() - t_desc, 1)

    return {
        "paper_id": paper_id,
        "_pdf_path": str(persistent_pdf) if persistent_pdf else "",
        "text_chunks": [_chunk_to_dict(c) for c in final.get("text_chunks", [])],
        "table_chunks": [_chunk_to_dict(c) for c in final.get("table_chunks", [])],
        "image_chunks": [_chunk_to_dict(c) for c in final.get("image_chunks", [])],
        "merged_chunks": [_chunk_to_dict(c) for c in merged],
        "verdicts": all_verdicts,
        "status": final.get("status", "unknown"),
        "quality": {},
        "worker_timing": final.get("worker_timing", {}),
        "judge_timing": final.get("judge_timing", {}),
        "merge_timing": final.get("merge_timing", 0),
        "desc_timing": desc_timing,
        "mode": "pipeline",
    }


def store_parsed_chunks(result: dict) -> dict:
    """手动入库：将已解析的 pipeline 结果存入 Qdrant。

    返回 {"stored": int, "failed": int, "duplicate": bool, ...}。
    """
    from src.store.qdrant_store import QdrantStore
    from qdrant_client.http import models

    store = QdrantStore()
    quality = result.get("quality", {})
    paper_id = result.get("paper_id", "?")
    verdicts = result.get("verdicts", {})
    merged = result.get("merged_chunks", [])
    chunk_contexts = quality.get("chunk_contexts", {})

    if not os.getenv(EMBEDDING_API_KEY_ENV, "").strip():
        attempted = 0
        for chunk in merged:
            verdict = verdicts.get(chunk.get("chunk_id", ""))
            if not verdict:
                continue
            if verdict.get("passed") or (
                chunk.get("content_type") == "table"
                and verdict.get("issue_type") != "false_positive"
            ):
                attempted += 1
        return {
            "stored": 0,
            "degraded": 0,
            "failed": attempted,
            "excluded": 0,
            "duplicate": False,
            "attempted": attempted,
            "missing_verdicts": 0,
            "error": f"缺少 {EMBEDDING_API_KEY_ENV}，无法生成向量入库",
        }

    # ── 重复检测 ──
    existing_count = store.client.count(
        collection_name=store.collection,
        count_filter=models.Filter(
            must=[models.FieldCondition(key="paper_id", match=models.MatchValue(value=paper_id))]
        ),
    ).count
    if existing_count > 0:
        logger.warning(
            "重复入库被阻止 | paper_id=%s | 已存在 %d 条记录 | 将跳过 %d 条新记录",
            paper_id, existing_count, len([c for c in merged if verdicts.get(c.get("chunk_id", ""), {}).get("passed")]),
        )
        return {
            "stored": 0,
            "degraded": 0,
            "failed": 0,
            "excluded": 0,
            "duplicate": True,
            "existing_count": existing_count,
            "attempted": 0,
        }

    stored = 0
    degraded = 0
    failed = 0
    excluded = 0
    excluded_tables = 0
    excluded_reasons: dict[str, int] = {}

    def _record_excluded(chunk: dict) -> None:
        nonlocal excluded, excluded_tables
        excluded += 1
        if chunk.get("content_type") == "table":
            excluded_tables += 1
        metadata = chunk.get("metadata", {}) or {}
        reason = str(
            metadata.get("storage_exclusion_reason")
            or metadata.get("unparseable_reason")
            or "excluded_from_storage"
        )
        excluded_reasons[reason] = excluded_reasons.get(reason, 0) + 1

    def _build_metadata(chunk: dict, v: dict, quality_label: str = "clean") -> dict:
        context = chunk_contexts.get(chunk.get("chunk_id", ""), "")
        raw_content = chunk.get("raw_content", "")
        display_content = f"{context}\n\n{raw_content}" if context else raw_content
        chunk_metadata = dict(chunk.get("metadata", {}) or {})
        return {
            **chunk_metadata,
            "page": chunk.get("page"),
            "bbox": chunk.get("bbox"),
            "column": chunk.get("column"),
            "order_in_page": chunk.get("order_in_page"),
            "global_order": chunk.get("global_order"),
            "chunk_id": chunk.get("chunk_id"),
            "content_type": chunk.get("content_type", "text"),
            "table_quality": quality_label if chunk.get("content_type") == "table" else "",
            "score": v.get("score"),
            "category": quality.get("category"),
            "classify_status": quality.get("classify_status"),
            "credibility": quality.get("credibility"),
            "fermentation_relevance": quality.get("fermentation_relevance"),
            "is_actionable": quality.get("is_actionable"),
            "paper_title": quality.get("paper_title"),
            "paper_summary": quality.get("paper_summary"),
            "target_products": quality.get("target_products", []),
            "organisms": quality.get("organisms", []),
            "classify_reason": quality.get("classify_reason", ""),
            "context": context,
        }, display_content

    attempted = 0
    missing_verdicts = 0

    for chunk in merged:
        cid = chunk.get("chunk_id", "")
        metadata = chunk.get("metadata", {}) or {}
        if chunk.get("content_type") == "table":
            try:
                from src.judge.table_judge import build_storage_decision
                storage_decision = build_storage_decision(
                    content_type="table",
                    metadata=metadata,
                )
                metadata.update(storage_decision)
                chunk["metadata"] = metadata
            except Exception:
                pass
        if metadata.get("excluded_from_storage"):
            _record_excluded(chunk)
            continue

        v = verdicts.get(cid)
        if not v:
            missing_verdicts += 1
            continue
        ct = chunk.get("content_type", "text")

        if v.get("passed"):
            # 通过 → 正常存储
            attempted += 1
            meta, display = _build_metadata(chunk, v, "clean")
            try:
                store.add(
                    paper_id=paper_id, worker_type=ct,
                    content=display,
                    embed_content=_embedding_content_for_chunk(chunk),
                    metadata=meta,
                )
                stored += 1
            except Exception as e:
                logger.error("入库失败 chunk_id=%s: %s", cid, e)
                failed += 1

        elif ct == "table" and v.get("issue_type") != "false_positive":
            # 表格解析质量差但非误检 → 降级存储（保留原始数据）
            attempted += 1
            meta, display = _build_metadata(chunk, v, "degraded")
            meta["judge_feedback"] = v.get("feedback", "")
            try:
                store.add(
                    paper_id=paper_id, worker_type=ct,
                    content=display,
                    embed_content=_embedding_content_for_chunk(chunk),
                    metadata=meta,
                )
                degraded += 1
            except Exception as e:
                logger.error("降级入库失败 chunk_id=%s: %s", cid, e)
                failed += 1

        elif ct == "table" and v.get("issue_type") == "false_positive":
            metadata.setdefault("excluded_from_storage", True)
            metadata.setdefault("storage_exclusion_reason", "quality judge marked table as false positive")
            chunk["metadata"] = metadata
            _record_excluded(chunk)

        # false_positive → 不入库（参考文献/图注等），解析候选仍保留在结果中

    store.save("")
    return {
        "stored": stored,
        "degraded": degraded,
        "failed": failed,
        "excluded": excluded,
        "excluded_tables": excluded_tables,
        "excluded_reasons": excluded_reasons,
        "attempted": attempted,
        "missing_verdicts": missing_verdicts,
    }



def evaluate_parsed_chunks(result: dict) -> dict:
    """质量评估：对已解析的 pipeline 结果执行分类 → Map-Reduce → chunk 上下文。"""
    from src.judge.paper_evaluator import PaperQualityEvaluator

    merged = result.get("merged_chunks", [])
    pdf_path = result.get("_pdf_path", "")

    if not merged:
        return {"skip": True, "error": "no merged chunks"}

    import types
    evaluator = PaperQualityEvaluator()
    chunk_objs = [types.SimpleNamespace(**c) for c in merged]
    return evaluator.evaluate(chunk_objs, pdf_path)


def _process_one_pdf(file_bytes: bytes, paper_id: str, filename: str) -> dict:
    """一条龙：解析 → 质量评估 → 入库。返回 quality dict。"""
    t0 = time.time()
    stage_start = t0
    logger.info("[batch] %s | pipeline start", filename)
    pipe_result = run_pipeline(file_bytes, paper_id, filename)
    parse_seconds = time.time() - stage_start
    logger.info("[batch] %s | pipeline done (%.0fs) | quality eval start", filename, parse_seconds)

    stage_start = time.time()
    q_result = evaluate_parsed_chunks(pipe_result)
    eval_seconds = time.time() - stage_start
    logger.info("[batch] %s | quality eval done (%.0fs) | store start", filename, time.time() - t0)

    stage_start = time.time()
    pipe_result["quality"] = q_result
    with _paper_store_lock(paper_id):
        store_result = store_parsed_chunks(pipe_result)
    store_seconds = time.time() - stage_start
    logger.info("[batch] %s | store done (%.0fs)", filename, time.time() - t0)
    stored = int(store_result.get("stored", 0) or 0)
    degraded = int(store_result.get("degraded", 0) or 0)
    store_failed = int(store_result.get("failed", 0) or 0)
    excluded = int(store_result.get("excluded", 0) or 0)
    attempted = int(store_result.get("attempted", 0) or 0)
    duplicate = bool(store_result.get("duplicate", False))
    if duplicate:
        status = "skip"
        reason = f"已入库 {store_result.get('existing_count', 0)} 条记录"
    elif store_failed:
        status = "partial" if stored or degraded else "fail"
        reason = str(store_result.get("error") or f"入库失败 {store_failed}/{attempted}")
    elif attempted == 0 and not excluded:
        status = "no_store"
        reason = f"没有可入库 chunk；缺少 verdict {store_result.get('missing_verdicts', 0)}"
    else:
        status = "ok"
        reason = ""
    return {
        "status": status,
        "reason": reason,
        "category": q_result.get("category", "?"),
        "credibility": q_result.get("credibility", 0),
        "chunks": len(pipe_result.get("merged_chunks", [])),
        "stored": stored,
        "degraded": degraded,
        "store_failed": store_failed,
        "excluded": excluded,
        "store_attempted": attempted,
        "parse_seconds": round(parse_seconds, 1),
        "eval_seconds": round(eval_seconds, 1),
        "store_seconds": round(store_seconds, 1),
        "total_seconds": round(time.time() - t0, 1),
    }

