"""
SortPaper Pipeline — 解析、评估、入库编排。
纯业务逻辑，不渲染 UI。
"""
from __future__ import annotations

import logging
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from app_utils import (
    build_paper_id, _chunk_to_dict, _generate_chunk_description,
    build_snapshot, save_result, overwrite_result,
    load_saved_list, load_saved_detail, delete_saved,
    RESULTS_DIR,
)

logger = logging.getLogger(__name__)

def run_preview(pdf_bytes: bytes) -> dict:
    """Quick preview: pdfplumber + PyMuPDF parsers + LLM Judge（无 camelot / Qwen-VL）。"""
    import sys
    sys.path.insert(0, ".")
    from src.parsers.pymupdf_parser import PyMuPDFParser
    from src.parsers.table_parser import TableParser
    from src.parsers.layout_chunk import LayoutMerger

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)

    try:
        t0_text = time.time()
        text_chunks = PyMuPDFParser(tmp_path).parse()
        wt_text = round(time.time() - t0_text, 1)

        t0_table = time.time()
        # 快速预览只跑 pdfplumber，跳过 camelot（camelot 全文档耗时15-30s）
        table_chunks = TableParser(tmp_path).parse(strategy="pdfplumber_only")
        wt_table = round(time.time() - t0_table, 1)

        print(f"[preview] before merge: {len(text_chunks)} text + {len(table_chunks)} table")
        for c in (text_chunks + table_chunks)[:5]:
            print(f"  p{c.page} col{c.column} y{c.bbox[1]:.1f}-{c.bbox[3]:.1f} | {c.raw_content[:50]}")
        merged_chunks = LayoutMerger.merge(text_chunks + table_chunks)
        print(f"[preview] after merge: {len(merged_chunks)} chunks")

        # 图片元数据（本地操作，无额外 API 费用）
        import fitz  # type: ignore
        doc = fitz.open(str(tmp_path))
        images = []
        for page_index in range(len(doc)):
            page = doc[page_index]
            img_list = page.get_images(full=True)
            for img_info in img_list:
                xref = img_info[0]
                bbox_list = page.get_image_rects(xref)
                bbox = tuple(bbox_list[0]) if bbox_list else None
                images.append({
                    "page": page_index + 1,
                    "xref": xref,
                    "width": img_info[2],
                    "height": img_info[3],
                    "bbox": bbox,
                })
        doc.close()

        # LLM Judge（DeepSeek，文本 section + 表格 chunk 全部并行）
        from concurrent.futures import ThreadPoolExecutor
        from src.judge.llm_judge import LLMJudge
        from src.graph.pipeline_graph import _is_noise, _group_by_section

        verdicts: dict[str, dict] = {}
        t0_judge = time.time()
        pdf_path_str = str(tmp_path)

        noise_ids = {c.chunk_id for c in text_chunks if _is_noise(c.raw_content)}

        # 构建所有任务（text section 和 table chunk 混合，统一并发）
        tasks: list[tuple] = []
        for sec in _group_by_section(text_chunks):
            if any(c.chunk_id not in noise_ids for c in sec):
                tasks.append(("text", sec))
        for chunk in table_chunks:
            tasks.append(("table", chunk))

        def _run_judge(task):
            kind, obj = task
            if kind == "text":
                section_chunks = obj
                combined = "\n\n".join(c.raw_content for c in section_chunks)
                v = LLMJudge().judge("text", combined, pdf_path_str)
                return [
                    (c.chunk_id, {"chunk_id": c.chunk_id, "passed": v.passed,
                                  "score": v.score, "feedback": v.feedback})
                    for c in section_chunks if c.chunk_id not in noise_ids
                ]
            else:
                chunk = obj
                v = LLMJudge().judge("table", chunk.raw_content, pdf_path_str)
                return [(chunk.chunk_id, {"chunk_id": chunk.chunk_id, "passed": v.passed,
                                          "score": v.score, "feedback": v.feedback,
                                          "issue_type": v.issue_type})]

        if tasks:
            with ThreadPoolExecutor(max_workers=min(5, len(tasks))) as pool:
                for pairs in pool.map(_run_judge, tasks):
                    for cid, v in pairs:
                        verdicts[cid] = v

        jt_judge = round(time.time() - t0_judge, 1)

    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "text_chunks": [_chunk_to_dict(c) for c in text_chunks],
        "table_chunks": [_chunk_to_dict(c) for c in table_chunks],
        "merged_chunks": [_chunk_to_dict(c) for c in merged_chunks],
        "images": images,
        "verdicts": verdicts,
        "mode": "preview",
        # 计时（供耗时面板展示）
        "worker_timing": {"text": wt_text, "table": wt_table},
        "judge_timing": {"text_table": jt_judge},
    }


def run_pipeline(pdf_bytes: bytes, paper_id: str, filename: str = "") -> dict:
    """Parse pipeline: parsers + judge + merge（quality 和 store 由 Streamlit 按钮手动触发）。"""
    import sys
    sys.path.insert(0, ".")
    from src.graph.pipeline_graph import build_graph

    # 保存 PDF 副本供后续质量评估使用（用原始文件名，不用 hash）
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
                print(f"  [text]   parsed {n} chunks ({t:.1f}s){retry_tag}", flush=True)
            elif node_name == "table_worker":
                n = len(node_data.get("table_chunks", []))
                print(f"  [table]  parsed {n} tables ({t:.1f}s){retry_tag}", flush=True)
            elif node_name == "image_worker":
                n = len(node_data.get("image_chunks", []))
                print(f"  [image]  described {n} images ({t:.1f}s){retry_tag}", flush=True)
            elif node_name == "judge_text":
                v = node_data.get("text_verdicts", [])
                p = sum(1 for x in v if x.get("passed"))
                print(f"  [judge]  text: {p}/{len(v)} passed{retry_tag}", flush=True)
            elif node_name == "judge_table":
                v = node_data.get("table_verdicts", [])
                p = sum(1 for x in v if x.get("passed"))
                types = {}
                for x in v:
                    it = x.get("issue_type", "?")
                    types[it] = types.get(it, 0) + 1
                print(f"  [judge]  table: {p}/{len(v)} passed{retry_tag} | issue_types: {types}", flush=True)
            elif node_name == "judge_image":
                v = node_data.get("image_verdicts", [])
                p = sum(1 for x in v if x.get("passed"))
                print(f"  [judge]  image: {p}/{len(v)} passed{retry_tag}", flush=True)
            elif node_name == "merge_chunks":
                n = len(node_data.get("merged_chunks", []))
                print(f"  [merge]  {n} merged chunks", flush=True)
            elif node_name == "finalize":
                status = node_data.get("status", "done")
                print(f"  [pipeline] {status}", flush=True)
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
        "_pdf_path": str(persistent_pdf),
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
        return {"stored": 0, "failed": 0, "duplicate": True, "existing_count": existing_count}

    stored = 0
    degraded = 0
    failed = 0

    def _build_metadata(chunk: dict, v: dict, quality_label: str = "clean") -> dict:
        context = chunk_contexts.get(chunk.get("chunk_id", ""), "")
        raw_content = chunk.get("raw_content", "")
        display_content = f"{context}\n\n{raw_content}" if context else raw_content
        return {
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

    for chunk in merged:
        cid = chunk.get("chunk_id", "")
        v = verdicts.get(cid)
        if not v:
            continue
        ct = chunk.get("content_type", "text")

        if v.get("passed"):
            # 通过 → 正常存储
            meta, display = _build_metadata(chunk, v, "clean")
            try:
                store.add(
                    paper_id=paper_id, worker_type=ct,
                    content=display,
                    embed_content=chunk.get("raw_content", ""),
                    metadata=meta,
                )
                stored += 1
            except Exception as e:
                logger.error("入库失败 chunk_id=%s: %s", cid, e)
                failed += 1

        elif ct == "table" and v.get("issue_type") != "false_positive":
            # 表格解析质量差但非误检 → 降级存储（保留原始数据）
            meta, display = _build_metadata(chunk, v, "degraded")
            meta["judge_feedback"] = v.get("feedback", "")
            try:
                store.add(
                    paper_id=paper_id, worker_type=ct,
                    content=display,
                    embed_content=chunk.get("raw_content", ""),
                    metadata=meta,
                )
                degraded += 1
            except Exception as e:
                logger.error("降级入库失败 chunk_id=%s: %s", cid, e)
                failed += 1

        # false_positive → 丢弃（参考文献/图注等）

    store.save("")
    return {"stored": stored, "degraded": degraded, "failed": failed}


def _generate_chunk_description(content_type: str, raw_content: str) -> str:
    """用 DeepSeek 为表格/图片生成一句英文描述（用于嵌入前缀）。

    table → 描述表中数据的内容和维度
    image → 描述图中的关键信息
    """
    import os
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    
    type_label = "table" if content_type == "table" else "figure/image"
    prompt = (
        f"Write ONE short English sentence summarizing what this {type_label} is about. "
        f"Focus on key concepts, entities, and data dimensions. Be specific.\n\n"
        f"Content:\n{raw_content[:2000]}"
    )

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("生成 %s 描述失败: %s", content_type, e)
        return ""


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
    import sys
    t0 = time.time()
    print(f"\n[batch] {filename} | pipeline start", flush=True); sys.stdout.flush()
    pipe_result = run_pipeline(file_bytes, paper_id, filename)
    print(f"[batch] {filename} | pipeline done ({time.time()-t0:.0f}s) | quality eval start", flush=True); sys.stdout.flush()
    q_result = evaluate_parsed_chunks(pipe_result)
    print(f"[batch] {filename} | quality eval done ({time.time()-t0:.0f}s) | store start", flush=True); sys.stdout.flush()
    pipe_result["quality"] = q_result
    store_parsed_chunks(pipe_result)
    print(f"[batch] {filename} | store done ({time.time()-t0:.0f}s)", flush=True); sys.stdout.flush()
    return {
        "category": q_result.get("category", "?"),
        "credibility": q_result.get("credibility", 0),
        "chunks": len(pipe_result.get("merged_chunks", [])),
    }

