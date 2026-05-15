"""
SortPaper GUI — Streamlit 前端
功能：导入 PDF → 解析（预览 / 完整流水线）→ 查看结果
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─── 页面配置 ──────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SortPaper",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 常量 ──────────────────────────────────────────────────────────────────────

SAMPLE_DIR = Path("data/sample_papers")
RESULTS_DIR = Path("data/results")
LOGO = "📄"

# ─── 保存/加载结果 ────────────────────────────────────────────────────────────


def build_snapshot(
    result: dict,
    filename: str,
) -> dict:
    """构造保存快照 JSON。"""
    quality = result.get("quality", {})
    verdicts = result.get("verdicts", {})
    chunks = result.get("merged_chunks", [])

    text_n = sum(1 for c in chunks if c.get("content_type") == "text")
    table_n = sum(1 for c in chunks if c.get("content_type") == "table")
    image_n = sum(1 for c in chunks if c.get("content_type") == "image")
    passed = sum(1 for v in verdicts.values() if v.get("passed"))
    failed = sum(1 for v in verdicts.values() if not v.get("passed"))

    return {
        "filename": filename,
        "paper_id": result.get("paper_id", ""),
        "saved_at": datetime.now().isoformat(),
        "quality": quality,
        "verdicts": verdicts,
        "chunks": chunks,
        "stats": {
            "total": len(chunks),
            "text": text_n,
            "table": table_n,
            "image": image_n,
            "passed": passed,
            "failed": failed,
            "status": result.get("status", "unknown"),
        },
    }


def save_result(snapshot: dict, filename: str) -> str | None:
    """保存到 data/results/{filename}.json。
    文件已存在时返回 None（由 UI 处理覆盖确认）。
    保存成功返回文件路径。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = filename.replace("/", "_").replace("\\", "_")
    target = RESULTS_DIR / f"{safe_name}.json"
    if target.exists():
        return None  # 已存在
    target.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(target)


def overwrite_result(snapshot: dict, filename: str) -> str:
    """强制覆盖保存。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = filename.replace("/", "_").replace("\\", "_")
    target = RESULTS_DIR / f"{safe_name}.json"
    target.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(target)


def load_saved_list() -> list[dict]:
    """列出所有已保存结果（仅元数据，不加载全量）。"""
    if not RESULTS_DIR.exists():
        return []
    items: list[dict] = []
    for f in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            q = data.get("quality", {})
            stt = data.get("stats", {})
            items.append({
                "file_stem": f.stem,
                "filename": data.get("filename", f.stem),
                "saved_at": data.get("saved_at", ""),
                "paper_title": q.get("paper_title", "?"),
                "category": q.get("category", "?"),
                "credibility": q.get("credibility", 0),
                "classify_status": q.get("classify_status", "?"),
                "paper_summary": q.get("paper_summary", ""),
                "total_chunks": stt.get("total", 0),
                "passed": stt.get("passed", 0),
                "failed": stt.get("failed", 0),
                "target_products": q.get("target_products", []),
                "organisms": q.get("organisms", []),
            })
        except Exception:
            continue
    return items


def load_saved_detail(file_stem: str) -> dict | None:
    """加载已保存文件的完整 JSON。"""
    target = RESULTS_DIR / f"{file_stem}.json"
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


def delete_saved(file_stem: str) -> bool:
    """删除已保存结果，返回是否成功。"""
    target = RESULTS_DIR / f"{file_stem}.json"
    if target.exists():
        target.unlink()
        return True
    return False


# ─── 工具函数 ──────────────────────────────────────────────────────────────────


def build_paper_id(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


def type_badge(ctype: str) -> str:
    colors = {"text": "🔵", "table": "🟠", "image": "🟢"}
    return colors.get(ctype, "⚪")


def verdict_badge(passed: bool) -> str:
    return "✅" if passed else "❌"


@st.cache_data(show_spinner=False)
def run_preview(pdf_bytes: bytes) -> dict:
    """Quick preview: parsers + OpenCV detection + LLM Judge（无 VisionParser / Qwen-VL）。"""
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
        table_chunks = TableParser(tmp_path).parse()
        wt_table = round(time.time() - t0_table, 1)

        print(f"[preview] before merge: {len(text_chunks)} text + {len(table_chunks)} table")
        for c in (text_chunks + table_chunks)[:5]:
            print(f"  p{c.page} col{c.column} y{c.bbox[1]:.1f}-{c.bbox[3]:.1f} | {c.raw_content[:50]}")
        merged_chunks = LayoutMerger.merge(text_chunks + table_chunks)
        print(f"[preview] after merge: {len(merged_chunks)} chunks")

        # image metadata + OpenCV 表格检测（均为本地操作，零额外 API 费用）
        import fitz  # type: ignore
        from src.parsers.opencv_table_detector import OpenCVTableDetector
        detector = OpenCVTableDetector()
        doc = fitz.open(str(tmp_path))
        images = []
        opencv_detections: dict[int, list[list[float]]] = {}
        for page_index in range(len(doc)):
            page = doc[page_index]
            # OpenCV 表格区域检测（纯本地形态学，毫秒级）
            bboxes_px = detector.detect_page(page)
            if bboxes_px:
                opencv_detections[page_index + 1] = [
                    list(detector.bbox_to_pdf_pts(b)) for b in bboxes_px
                ]
            # 图片元数据
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
        "opencv_detections": opencv_detections,
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
    """手动质量评估：对已解析的 pipeline 结果执行分类 → Map-Reduce → chunk 上下文。

    返回 quality_result dict，含 category / credibility / chunk_contexts 等。
    """
    from src.judge.paper_evaluator import PaperQualityEvaluator

    merged = result.get("merged_chunks", [])
    pdf_path = result.get("_pdf_path", "")

    if not merged:
        return {"skip": True, "error": "no merged chunks"}

    import types

    evaluator = PaperQualityEvaluator()
    # merged_chunks 已序列化为 dict，evaluator 需要属性访问（.content_type / .raw_content 等）
    chunk_objs = [types.SimpleNamespace(**c) for c in merged]
    quality = evaluator.evaluate(chunk_objs, pdf_path)
    return quality


def _chunk_to_dict(chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "content_type": chunk.content_type,
        "page": chunk.page,
        "column": chunk.column,
        "global_order": chunk.global_order,
        "raw_content": chunk.raw_content,
        "bbox": chunk.bbox,
        "metadata": chunk.metadata or {},
    }


def qdrant_search(query: str, paper_id: str = None, top_k: int = 5, rerank: bool = False) -> list[dict]:
    import sys
    sys.path.insert(0, ".")
    from src.store.qdrant_store import QdrantStore

    store = QdrantStore()
    filter_kwargs = {"paper_id": paper_id} if paper_id else None
    results = store.search(
        query=query,
        limit=top_k,
        filter_kwargs=filter_kwargs,
        rerank=rerank,
    )
    return [
        {
            "id": r["id"],
            "score": r["score"],
            "reranked": r.get("reranked", False),
            "payload": r["payload"],
        }
        for r in results
    ]


# ─── PDF 重建 ──────────────────────────────────────────────────────────────────


def render_reconstruction_tab(result: dict) -> None:
    """Chunk 重建可视化：空白画布上按 bbox 排列 chunk，每个框显示 global_order + 内容摘要。"""
    import fitz
    import textwrap
    from PIL import Image, ImageDraw, ImageFont

    merged = result.get("merged_chunks", [])
    if not merged:
        st.info("无 chunk 数据，无法重建")
        return

    color_mode = st.radio(
        "着色方式",
        ["按内容类型", "按栏位"],
        horizontal=True,
    )

    type_colors = {
        "text":  ("#2196F3", "#BBDEFB"),
        "table": ("#FF9800", "#FFE0B2"),
        "image": ("#4CAF50", "#C8E6C9"),
    }
    column_colors = {
        0: ("#1565C0", "#BBDEFB"),
        1: ("#C62828", "#FFCDD2"),
        2: ("#7B1FA2", "#E1BEE7"),
    }

    def get_color(chunk: dict) -> tuple[str, str]:
        if color_mode == "按内容类型":
            return type_colors.get(chunk.get("content_type", "text"), ("#888", "#ddd"))
        col = chunk.get("column", 0)
        return column_colors.get(col, ("#888", "#ddd"))

    pdf_file = st.session_state.get("_pdf_bytes")
    if not pdf_file:
        st.warning("PDF 文件数据丢失，请重新上传")
        return

    try:
        doc = fitz.open(stream=pdf_file, filetype="pdf")
        page_count = len(doc)
        pages = sorted(set(c["page"] for c in merged if 1 <= c["page"] <= page_count))
        if "recon_page" not in st.session_state:
            st.session_state.recon_page = pages[0]
        page_num = st.select_slider("选择页面", options=pages, key="recon_page")
        page_num = max(1, min(page_num, page_count))
        pw, ph = float(doc[page_num - 1].rect.width), float(doc[page_num - 1].rect.height)
        doc.close()
    except Exception as e:
        st.error(f"页面加载失败：{e}")
        return

    # 高分辨率画布（2x 缩放，保证字体清晰）
    scale = min(1560 / pw, 1800 / ph)
    canvas_w, canvas_h = int(pw * scale), int(ph * scale)
    img = Image.new("RGB", (canvas_w, canvas_h), (248, 249, 250))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("segoeui.ttf", 14)
        font_small = ImageFont.truetype("segoeui.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
        font_small = font

    page_chunks = [c for c in merged if c["page"] == page_num]
    page_chunks.sort(key=lambda c: c.get("bbox", (0, 0, 0, 0))[1] if c.get("bbox") else 0)

    for chunk in page_chunks:
        bbox = chunk.get("bbox")
        if not bbox:
            continue
        x0, y0, x1, y1 = bbox
        px0, py0 = x0 * scale, y0 * scale
        px1, py1 = x1 * scale, y1 * scale
        box_w, box_h = px1 - px0, py1 - py0

        if box_w < 8 or box_h < 8:
            continue

        outline, fill = get_color(chunk)
        draw.rectangle([px0, py0, px1, py1], outline=outline, fill=fill + "30", width=2)

        # global_order 标签（深底白字）
        g_order = chunk.get("global_order", chunk.get("order_in_page", 0))
        label = str(g_order)
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.rectangle([px0 + 2, py0 + 2, px0 + tw + 8, py0 + th + 6], fill=outline, width=0)
        draw.text((px0 + 5, py0 + 4), label, fill="white", font=font)

        # 内容摘要（多行自适应，尽量填满框的可用空间）
        content = chunk.get("raw_content", "").strip()
        ctype = chunk.get("content_type", "text")
        type_tag = {"text": "T", "table": "TBL", "image": "IMG"}.get(ctype, "?")

        if not content:
            # 无内容：只显示类型标签
            tag_text = f"[{type_tag}] (empty)"
            tbx = draw.textbbox((0, 0), tag_text, font=font_small)
            tw2, th2 = tbx[2] - tbx[0], tbx[3] - tbx[1]
            if box_h > th + th2 + 12:
                draw.text((px0 + 5, py0 + th + 10), tag_text, fill="#888", font=font_small)
            continue

        # 计算框内可用宽度 → 每行可容纳字符数
        text_area_left = px0 + 5
        text_area_width = box_w - 12
        avg_char_w = font_small.getlength("ABCDEFGH") / 8
        if avg_char_w > 0:
            chars_per_line = max(10, int(text_area_width / avg_char_w))
        else:
            chars_per_line = 60

        # 计算可用高度 → 可容纳行数（在 order 标签下方）
        lh = font_small.size + 4  # 行高
        available_h = box_h - th - 14  # th = order 标签高度
        max_lines = max(1, int(available_h / lh))

        # 多行换行，限制行数
        lines = textwrap.wrap(content, width=chars_per_line)
        display_lines = lines[:max_lines]

        if max_lines >= 2 and len(display_lines) >= 1:
            # 第一行带 [T] 标签，后续行继续显示内容
            lines_with_tag = [f"[{type_tag}] {display_lines[0]}"] + display_lines[1:]
            for i, line in enumerate(lines_with_tag):
                draw.text((text_area_left, py0 + th + 6 + i * lh), line, fill="#333", font=font_small)
        elif max_lines >= 1 and len(display_lines) >= 1:
            # 只能显示一行，带 [T] 标签
            line_text = f"[{type_tag}] {display_lines[0]}"
            tbx = draw.textbbox((0, 0), line_text, font=font_small)
            tw2, th2 = tbx[2] - tbx[0], tbx[3] - tbx[1]
            if box_h <= th + th2 + 10:
                # 框太小，只显示 [T]
                draw.text((text_area_left, py0 + th + 6), f"[{type_tag}]", fill="#555", font=font_small)
            else:
                draw.text((text_area_left, py0 + th + 6), line_text, fill="#333", font=font_small)

    st.image(img, use_container_width=True)

    # 图例
    if color_mode == "按内容类型":
        c1, c2, c3 = st.columns(3)
        for col, (ctype, (clr, _)) in zip([c1, c2, c3], type_colors.items()):
            col.markdown(
                f"<span style='display:inline-block;width:14px;height:14px;"                f"background:{clr};border-radius:3px;margin-right:6px;'></span>{ctype}",
                unsafe_allow_html=True,
            )
    else:
        c1, c2, c3 = st.columns(3)
        labels = {0: "左栏", 1: "右栏", 2: "通栏"}
        for col, (col_id, (clr, _)) in zip([c1, c2, c3], column_colors.items()):
            col.markdown(
                f"<span style='display:inline-block;width:14px;height:14px;"                f"background:{clr};border-radius:3px;margin-right:6px;'></span>{labels[col_id]}",
                unsafe_allow_html=True,
            )

    st.caption(f"共 {len(page_chunks)} 个 chunks | 页面 {page_num}")
def render_chunk_card(chunk: dict, verdict: dict | None = None, index: int = 0) -> None:
    ctype = chunk.get("content_type", "text")
    badge = type_badge(ctype)
    page = chunk.get("page", "?")
    cid = chunk.get("chunk_id", "")
    short_id = cid.split("_", 2)[-1] if "_" in cid else cid

    label = f"{badge} p{page} · {short_id}"
    if verdict:
        label = f"{verdict_badge(verdict['passed'])} {label}  score={verdict.get('score', 0):.2f}"

    with st.expander(label, expanded=False):
        content = chunk.get("raw_content", "")
        if ctype == "table":
            st.markdown(content)
        elif ctype == "image":
            st.markdown(f"**描述：** {content}")
        else:
            st.text(content[:1200] + ("…" if len(content) > 1200 else ""))

        col1, col2, col3 = st.columns(3)
        col1.caption(f"类型: {ctype}")
        col2.caption(f"页: {page}")
        col3.caption(f"列: {chunk.get('column', '?')}")

        if verdict:
            fb = verdict.get("feedback", "")
            if fb:
                st.caption(f"**Judge 意见：** {fb}")


def render_overview(result: dict) -> None:
    """纯展示，不包含任何会触发 st.rerun() 的按钮。"""
    text_n = len(result.get("text_chunks", []))
    table_n = len(result.get("table_chunks", []))
    img_n = len(result.get("images", result.get("image_chunks", [])))
    mode = result.get("mode", "preview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔵 文本块", text_n)
    c2.metric("🟠 表格块", table_n)
    c3.metric("🟢 图片", img_n)

    if mode == "pipeline":
        verdicts = result.get("verdicts", {})
        passed = sum(1 for v in verdicts.values() if v["passed"])
        failed = sum(1 for v in verdicts.values() if not v["passed"])
        c4.metric("Judge 通过", f"{passed} / {passed + failed}")

        status = result.get("status", "")
        quality = result.get("quality", {})
        has_quality = bool(quality.get("category"))

        if status == "done" and not has_quality:
            st.success("✅ 解析完成")
        elif status == "done" and has_quality:
            st.success("✅ 质量评估完成")
            st.caption(f"分类: {quality.get('category')} | 可信度: {quality.get('credibility', 0):.2f}")

        # 计时面板（展示，无按钮）
        quality = result.get("quality", {})
        timing = quality.get("timing", {})
        worker_timing = result.get("worker_timing", {})
        if timing or worker_timing:
            with st.expander("⏱️ 耗时统计", expanded=False):
                # 解析阶段
                wt_text = worker_timing.get("text", 0)
                wt_table = worker_timing.get("table", 0)
                wt_image = worker_timing.get("image", 0)
                parse_total = max(wt_text, wt_table, wt_image)  # Worker 并行，取最长
                st.caption("**解析阶段**（三个 Worker 并行）")
                pc1, pc2, pc3, pc4 = st.columns(4)
                pc1.metric("文本解析", f"{wt_text:.1f}s")
                pc2.metric("表格解析", f"{wt_table:.1f}s")
                pc3.metric("图片解析", f"{wt_image:.1f}s")
                pc4.metric("并行耗时", f"{parse_total:.1f}s")

                # Judge 阶段
                jt = result.get("judge_timing", {})
                if jt:
                    st.caption("**Judge 阶段**（text / table / image 分线并行）")
                    jc1, jc2, jc3, jc4 = st.columns(4)
                    jc1.metric("文本 Judge", f"{jt.get('text', 0):.1f}s")
                    jc2.metric("表格 Judge", f"{jt.get('table', 0):.1f}s")
                    jc3.metric("图片 Judge", f"{jt.get('image', 0):.1f}s")
                    jc4.metric("Judge 合计", f"{sum(jt.values()):.1f}s")

                # Merge 阶段
                mt = result.get("merge_timing", 0)
                if mt:
                    st.caption(f"**Merge 阶段**: {mt:.1f}s")

                # 描述生成阶段（表格/图片 DeepSeek 前缀，pipeline 结束后串行调用）
                dt = result.get("desc_timing", 0)
                if dt:
                    desc_n = sum(
                        1 for c in result.get("merged_chunks", [])
                        if c.get("content_type") in ("table", "image")
                    )
                    dc1, dc2 = st.columns(2)
                    dc1.metric("描述生成（表/图前缀）", f"{dt:.1f}s")
                    dc2.metric("  ", f"{desc_n} 个 chunk × DeepSeek 并行")

                # 图片子阶段（image_extract / image_describe）
                if worker_timing.get("image_describe", 0) > 0:
                    st.caption("图片解析详情：提取 {:.1f}s + VL描述 {:.1f}s".format(
                        worker_timing.get("image_extract", 0),
                        worker_timing.get("image_describe", 0),
                    ))

                # 总计
                total_elapsed = result.get("_elapsed", 0)
                if total_elapsed:
                    accounted = parse_total + sum(jt.values()) + mt + dt
                    st.caption(f"**总耗时 {total_elapsed:.0f} 秒** — 已统计 {accounted:.0f}s，其余为路由/通信开销")

                # 质量评估阶段
                t_cls = timing.get("classify", 0)
                t_map = timing.get("map", 0)
                t_red = timing.get("reduce", 0)
                if t_cls or t_map or t_red:
                    st.caption("**质量评估阶段**")
                    qc1, qc2, qc3, qc4 = st.columns(4)
                    qc1.metric("分类", f"{t_cls:.1f}s")
                    qc2.metric("Map", f"{t_map:.1f}s")
                    qc3.metric("Reduce", f"{t_red:.1f}s")
                    qc4.metric("合计", f"{t_cls + t_map + t_red:.1f}s")
    else:
        verdicts = result.get("verdicts", {})
        if verdicts:
            passed = sum(1 for v in verdicts.values() if v["passed"])
            total_v = len(verdicts)
            c4.metric("Judge 通过", f"{passed} / {total_v}")
        else:
            c4.metric("模式", "快速预览")
        st.info("ℹ️ 快速预览模式：PyMuPDF/pdfplumber 解析 + OpenCV 检测 + LLM Judge，无 VisionParser / Qwen-VL / 向量索引写入")

        # 计时面板（快速预览）
        worker_timing = result.get("worker_timing", {})
        judge_timing = result.get("judge_timing", {})
        if worker_timing or judge_timing:
            with st.expander("⏱️ 耗时统计", expanded=False):
                wt_text = worker_timing.get("text", 0)
                wt_table = worker_timing.get("table", 0)
                if wt_text or wt_table:
                    st.caption("**解析阶段**（文本+表格串行）")
                    pc1, pc2, pc3 = st.columns(3)
                    pc1.metric("文本解析", f"{wt_text:.1f}s")
                    pc2.metric("表格解析", f"{wt_table:.1f}s")
                    pc3.metric("合计", f"{wt_text + wt_table:.1f}s")

                jt_total = sum(judge_timing.values())
                if jt_total:
                    st.caption(f"**Judge 阶段**（文本+表格并行，含 DeepSeek 调用）: {jt_total:.1f}s")

                total_elapsed = result.get("_elapsed", 0)
                if total_elapsed:
                    accounted = wt_text + wt_table + jt_total
                    st.caption(f"**总耗时 {total_elapsed:.0f} 秒** — 已统计 {accounted:.0f}s，其余为 OpenCV 检测 / 路由开销")


def render_text_tab(chunks: list[dict], verdicts: dict) -> None:
    if not chunks:
        st.info("无文本块")
        return
    st.caption(f"共 {len(chunks)} 个文本块")
    for i, c in enumerate(chunks):
        verdict = verdicts.get(c["chunk_id"])
        render_chunk_card(c, verdict, i)


def _render_page_with_detections(pdf_bytes: bytes, page_idx: int, bboxes_pdf: list) -> object:
    """渲染单页缩略图并在检测到的表格区域上画红框，返回 PIL Image。"""
    import fitz  # type: ignore
    from PIL import Image, ImageDraw

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_idx - 1]
    mat = fitz.Matrix(1.5, 1.5)  # ~108 DPI，缩略图清晰度够用
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", (pix.w, pix.h), pix.samples)
    doc.close()

    draw = ImageDraw.Draw(img)
    scale = 1.5
    for bbox in bboxes_pdf:
        x0, y0, x1, y1 = [v * scale for v in bbox]
        draw.rectangle([x0, y0, x1, y1], outline="#e63946", width=3)
        draw.rectangle([x0 + 1, y0 + 1, x1 - 1, y1 - 1], outline="#e63946aa" if hasattr(draw, "alpha_composite") else "#e63946", width=1)

    return img


def render_table_tab(
    chunks: list[dict],
    verdicts: dict,
    opencv_detections: dict | None = None,
    pdf_bytes: bytes | None = None,
) -> None:
    # ── OpenCV 检测可视化（仅快速预览模式有数据）────────────────────────────
    if opencv_detections and pdf_bytes:
        total = sum(len(v) for v in opencv_detections.values())
        pages = sorted(opencv_detections.keys())
        with st.expander(
            f"🔍 OpenCV 检测到 {total} 处候选表格区域（共 {len(pages)} 页）",
            expanded=True,
        ):
            st.caption("红框 = 形态学横线聚类检测结果，仅用于定位，内容由旧引擎或视觉模型提取。")
            cols = st.columns(min(len(pages), 3))
            for i, page_idx in enumerate(pages):
                with cols[i % 3]:
                    try:
                        img = _render_page_with_detections(
                            pdf_bytes, page_idx, opencv_detections[page_idx]
                        )
                        st.image(
                            img,
                            caption=f"第 {page_idx} 页 · {len(opencv_detections[page_idx])} 处",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.warning(f"第 {page_idx} 页渲染失败: {e}")

    # ── 表格内容块 ────────────────────────────────────────────────────────────
    if not chunks:
        st.info("无表格块（已过滤所有误检伪表格）" if not opencv_detections else
                "旧引擎未提取到表格内容（OpenCV 已检测到区域，完整流水线可用视觉模型提取）")
        return
    st.caption(f"共 {len(chunks)} 个表格块")
    for i, c in enumerate(chunks):
        verdict = verdicts.get(c["chunk_id"])
        render_chunk_card(c, verdict, i)


def render_image_tab(result: dict) -> None:
    mode = result.get("mode", "preview")
    if mode == "preview":
        images = result.get("images", [])
        if not images:
            st.info("未检测到嵌入图片")
            return
        st.caption(f"检测到 {len(images)} 张嵌入图片（快速预览模式不调用 VisionParser）")
        for img in images:
            st.markdown(
                f"- **第 {img['page']} 页** — xref={img['xref']} | "
                f"尺寸 {img['width']}×{img['height']} px"
            )
    else:
        chunks = result.get("image_chunks", [])
        verdicts = result.get("verdicts", {})
        if not chunks:
            st.info("无图片块")
            return
        st.caption(f"共 {len(chunks)} 个图片块（已由 VisionParser 生成描述）")
        for i, c in enumerate(chunks):
            verdict = verdicts.get(c["chunk_id"])
            render_chunk_card(c, verdict, i)


def render_search_tab(paper_id: str) -> None:
    """语义检索标签页：当前论文内检索。"""
    try:
        from src.store.qdrant_store import QdrantStore
        store = QdrantStore()
        total = store.count
    except Exception:
        st.warning("Qdrant 服务未连接（localhost:6333），请先启动 Qdrant")
        return

    if total == 0:
        st.info("Qdrant 索引为空，请先解析论文并入库")
        return

    st.caption(f"当前索引共 {total} 条记录，当前论文: `{paper_id}`")

    sub_tabs = st.tabs(["🔍 手动检索", "🤖 Agent 检索"])

    with sub_tabs[0]:
        _render_manual_search(paper_id)

    with sub_tabs[1]:
        _render_agent_search(paper_id)


def render_standalone_search_tab() -> None:
    """独立检索标签页：不需要先解析论文，直接搜索 Qdrant 全库。"""
    try:
        from src.store.qdrant_store import QdrantStore
        store = QdrantStore()
        total = store.count
    except Exception:
        st.warning("Qdrant 服务未连接（localhost:6333），请先启动 Qdrant")
        return

    if total == 0:
        st.info("Qdrant 索引为空，请先解析论文并入库后再来检索。")
        return

    st.caption(f"全库共 {total} 条记录，跨论文检索")

    st_tabs = st.tabs(["🔍 手动检索", "🤖 Agent 检索"])

    with st_tabs[0]:
        _render_manual_search(paper_id=None)

    with st_tabs[1]:
        _render_agent_search(paper_id=None)


def _render_manual_search(paper_id: str | None = None) -> None:
    query = st.text_input("输入检索关键词或问题", placeholder="e.g. lacto-N-tetraose biosynthesis yield",
                          key="manual_query")
    top_k = st.slider("返回条数", 1, 10, 5, key="manual_top_k")
    use_rerank = st.checkbox("启用 Rerank（qwen3-rerank 二次排序）", value=True,
                              help="先 Hybrid 召回，再用 Rerank 模型精排", key="manual_rerank")
    if st.button("🔍 检索", key="manual_search_btn"):
        if not query.strip():
            st.warning("请输入查询内容")
            return
        spinner_text = "Hybrid 检索 + Rerank 中…" if use_rerank else "Hybrid 检索中…"
        with st.spinner(spinner_text):
            results = qdrant_search(query, paper_id, top_k, rerank=use_rerank)
        if not results:
            st.warning("未找到相关结果")
            return
        st.caption(f"找到 {len(results)} 条结果")
        for r in results:
            score = r.get("score", 0)
            payload = r.get("payload", {})
            wtype = payload.get("content_type", payload.get("worker_type", ""))
            content = payload.get("content", "")
            pg = payload.get("page", "?")
            paper_title = payload.get("paper_title", "?")
            reranked = "🔁 " if r.get("reranked") else ""
            paper_info = f" | {paper_title[:40]}" if paper_id is None else ""
            with st.expander(f"{reranked}📌 相似度={score:.4f} | {wtype} | p{pg}{paper_info}", expanded=True):
                if wtype == "table":
                    st.markdown(content)
                else:
                    st.text(content)


def _render_agent_search(paper_id: str | None = None) -> None:
    st.caption("Qwen 自主检索 + 综合建议。模型会自动选择关键词和过滤条件。" +
               ("（全库检索）" if paper_id is None else f"（限定论文: `{paper_id}`）"))
    question = st.text_area("输入你的问题", placeholder="e.g. HMO 发酵中乳糖补料导致乙酸积累，文献中有什么解决方案？",
                            key="agent_question", height=100)
    max_rounds = st.slider("最大检索轮次", 1, 3, 2, key="agent_rounds",
                            help="每轮可调用一次 search_literature，多轮可换角度再查")

    if st.button("🤖 Agent 检索", key="agent_search_btn", type="primary"):
        if not question.strip():
            st.warning("请输入问题")
            return

        from src.agent.literature_agent import LiteratureAgent

        agent = LiteratureAgent()
        with st.spinner("Agent 分析中（检索 + 综合建议）…"):
            result = agent.query(question, max_rounds=max_rounds)

        # 检索历史
        if result["search_history"]:
            with st.expander(f"🔍 检索过程（{len(result['search_history'])} 轮，共 {result['total_chunks']} 个 chunk）", expanded=False):
                for h in result["search_history"]:
                    st.caption(f"第 {h['round'] + 1} 轮: `{h['args'].get('query', '?')}`")
                    if h["args"].get("category"):
                        st.caption(f"  过滤: category={h['args']['category']}")
                    st.caption(f"  命中: {h['hits']} 条")

        # 综合建议
        st.markdown("### 🧠 综合建议")
        st.markdown(result["answer"])


# ─── 已保存结果 ────────────────────────────────────────────────────────────────


def category_label(cat: str) -> str:
    labels = {
        "fermentation_experiment": "🧪 发酵实验",
        "biosynthesis_review": "📖 合成生物学综述",
        "other": "📎 其他",
    }
    return labels.get(cat, cat)


def render_saved_tab() -> None:
    """已保存结果标签页：列表 + 详情。"""
    saved_list = load_saved_list()

    if not saved_list:
        st.info("暂无已保存的解析结果。运行完整流水线后点击\"💾 保存解析结果\"即可保存。")
        return

    # 如果当前选中项不在列表中（被删除），清除选中
    if st.session_state.selected_saved and st.session_state.selected_saved not in {
        s["file_stem"] for s in saved_list
    }:
        st.session_state.selected_saved = None

    # 列表 + 详情布局
    col_list, col_detail = st.columns([1, 2])

    with col_list:
        st.caption(f"共 {len(saved_list)} 条已保存结果")

        # 清空所有已保存
        if "confirm_clear_saved" not in st.session_state:
            st.session_state.confirm_clear_saved = False
        if not st.session_state.confirm_clear_saved:
            st.button("🗑️ 清空所有已保存", key="clear_all_saved", use_container_width=True,
                      on_click=lambda: setattr(st.session_state, "confirm_clear_saved", True))
        else:
            st.warning("⚠️ 将删除所有已保存结果（不含向量库数据）")
            c1, c2 = st.columns(2)
            if c1.button("✅ 确认", use_container_width=True):
                for item in saved_list:
                    delete_saved(item["file_stem"])
                st.session_state.confirm_clear_saved = False
                st.session_state.selected_saved = None
                st.rerun()
            if c2.button("❌ 取消", use_container_width=True):
                st.session_state.confirm_clear_saved = False
                st.rerun()

        for item in saved_list:
            title = item.get("paper_title", "?")
            cat = item.get("category", "?")
            cred = item.get("credibility", 0)
            passed = item.get("passed", 0)
            failed = item.get("failed", 0)
            saved_at = item.get("saved_at", "")[:16].replace("T", " ")

            # 卡片
            selected = st.session_state.selected_saved == item["file_stem"]
            card_style = "border: 2px solid #2196F3; border-radius: 8px; padding: 10px;" if selected else ""
            with st.container():
                if st.button(
                    f"{category_label(cat)}\n**{title[:60]}**\n可信度: {cred:.2f} | ✅{passed} ❌{failed}\n{saved_at}",
                    key=f"saved_{item['file_stem']}",
                    use_container_width=True,
                ):
                    st.session_state.selected_saved = item["file_stem"]
                    st.rerun()

    with col_detail:
        if st.session_state.selected_saved:
            detail = load_saved_detail(st.session_state.selected_saved)
            if detail:
                render_saved_detail(detail, st.session_state.selected_saved)
        else:
            st.info("← 选择一条已保存结果查看详情")


def render_saved_detail(detail: dict, file_stem: str) -> None:
    """已保存结果的详细视图。"""
    quality = detail.get("quality", {})
    stats = detail.get("stats", {})
    verdicts = detail.get("verdicts", {})
    chunks = detail.get("chunks", [])
    chunk_map = quality.get("chunk_map", {})

    st.subheader(quality.get("paper_title", file_stem))
    st.caption(f"保存时间: {detail.get('saved_at', '')[:19].replace('T', ' ')} | "
               f"文件: {detail.get('filename', file_stem)} | ID: `{detail.get('paper_id', '?')[:16]}`")

    if st.button("🗑️ 删除此结果", key=f"del_{file_stem}", type="secondary"):
        delete_saved(file_stem)
        st.session_state.selected_saved = None
        st.rerun()

    # 子标签页
    sub_tabs = st.tabs(["📊 概览", "📝 文本块", "📋 表格", "🖼️ 图片", "🔍 质量详情"])

    # ── 概览 ──
    with sub_tabs[0]:
        cat = quality.get("category", "?")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("分类", category_label(cat))
        c2.metric("可信度", f"{quality.get('credibility', 0):.2f}")
        c3.metric("发酵相关性", f"{quality.get('fermentation_relevance', 0):.2f}")
        c4.metric("自洽性", quality.get("classify_status", "?"))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("总 Chunks", stats.get("total", 0))
        c6.metric("✅ 通过", stats.get("passed", 0))
        c7.metric("❌ 失败", stats.get("failed", 0))
        c8.metric("状态", stats.get("status", "?"))

        # 计时
        timing = quality.get("timing", {})
        if timing:
            st.divider()
            st.caption("⏱️ 质量评估耗时")
            t_cls = timing.get("classify", 0)
            t_map = timing.get("map", 0)
            t_red = timing.get("reduce", 0)
            tc1, tc2, tc3, tc4 = st.columns(4)
            tc1.metric("分类", f"{t_cls:.1f}s")
            tc2.metric("Map", f"{t_map:.1f}s")
            tc3.metric("Reduce", f"{t_red:.1f}s")
            tc4.metric("合计", f"{t_cls + t_map + t_red:.1f}s")

        st.divider()

        if quality.get("paper_summary"):
            st.markdown("**论文摘要**")
            st.info(quality["paper_summary"])

        c9, c10 = st.columns(2)
        with c9:
            if quality.get("target_products"):
                st.markdown("**目标产物**")
                for p in quality["target_products"]:
                    st.markdown(f"- {p}")
        with c10:
            if quality.get("organisms"):
                st.markdown("**相关菌株**")
                for o in quality["organisms"]:
                    st.markdown(f"- {o}")

        if quality.get("classify_reason"):
            st.markdown("**分类依据**")
            st.caption(quality["classify_reason"])

        if quality.get("is_actionable"):
            st.success("✅ 包含可指导发酵实验的信息")

        if quality.get("key_evidence"):
            st.markdown("**关键原文证据**")
            for i, ev in enumerate(quality["key_evidence"], 1):
                with st.expander(f"证据 {i}"):
                    st.text(ev)

    # ── 文本块 ──
    with sub_tabs[1]:
        text_chunks = [c for c in chunks if c.get("content_type") == "text"]
        if not text_chunks:
            st.info("无文本块")
        else:
            st.caption(f"共 {len(text_chunks)} 个文本块")
            for c in text_chunks:
                cid = c.get("chunk_id", "")
                verdict = verdicts.get(cid, {})
                passed = verdict.get("passed", True)
                score = verdict.get("score", 0)
                v_badge = "✅" if passed else "❌"
                preview = c.get("raw_content", "")[:80].replace("\n", " ")

                with st.expander(
                    f"{v_badge} p{c.get('page', '?')} · {preview}… | score={score:.2f}",
                    expanded=False,
                ):
                    st.text(c.get("raw_content", "")[:1200])
                    # Map 结果
                    cm = chunk_map.get(cid)
                    if cm:
                        kps = cm.get("key_points", [])
                        ents = cm.get("entities", [])
                        claim = cm.get("contains_claim", False)
                        claim_detail = cm.get("claim_detail", "")
                        if kps:
                            st.caption("**关键要点**")
                            for kp in kps:
                                st.markdown(f"- {kp}")
                        if ents:
                            st.caption(f"**实体**: {', '.join(ents)}")
                        if claim:
                            st.info(f"📌 **声明**: {claim_detail}" if claim_detail else "📌 包含实验/方法声明")
                    # Verdict
                    fb = verdict.get("feedback", "")
                    if fb:
                        st.caption(f"**Judge 意见**: {fb}")

    # ── 表格 ──
    with sub_tabs[2]:
        table_chunks = [c for c in chunks if c.get("content_type") == "table"]
        if not table_chunks:
            st.info("无表格块")
        else:
            st.caption(f"共 {len(table_chunks)} 个表格块")
            for c in table_chunks:
                cid = c.get("chunk_id", "")
                verdict = verdicts.get(cid, {})
                passed = verdict.get("passed", True)
                score = verdict.get("score", 0)
                v_badge = "✅" if passed else "❌"
                with st.expander(
                    f"{v_badge} 表格 p{c.get('page', '?')} | score={score:.2f}",
                    expanded=False,
                ):
                    st.markdown(c.get("raw_content", ""))
                    fb = verdict.get("feedback", "")
                    if fb:
                        st.caption(f"**Judge 意见**: {fb}")

    # ── 图片 ──
    with sub_tabs[3]:
        img_chunks = [c for c in chunks if c.get("content_type") == "image"]
        if not img_chunks:
            st.info("无图片块")
        else:
            st.caption(f"共 {len(img_chunks)} 个图片块")
            for c in img_chunks:
                cid = c.get("chunk_id", "")
                verdict = verdicts.get(cid, {})
                passed = verdict.get("passed", True)
                score = verdict.get("score", 0)
                v_badge = "✅" if passed else "❌"
                with st.expander(
                    f"{v_badge} 图片 p{c.get('page', '?')} | score={score:.2f}",
                    expanded=False,
                ):
                    st.text(c.get("raw_content", "")[:1200])
                    fb = verdict.get("feedback", "")
                    if fb:
                        st.caption(f"**Judge 意见**: {fb}")

    # ── 质量详情 ──
    with sub_tabs[4]:
        classify_result = quality.get("classify_result", {})
        verify_result = quality.get("verify_result", {})

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**分类结果 (Prompt 1)**")
            st.json(classify_result)
        with c2:
            st.markdown("**验证结果 (Prompt 2)**")
            st.json(verify_result)

        if quality.get("classify_reason"):
            st.divider()
            st.markdown("**判断依据**")
            st.info(quality["classify_reason"])


# ─── 侧边栏 ────────────────────────────────────────────────────────────────────


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
                est_min = len(batch_files) * 5  # pipeline 2min + Map-Reduce ~3min
                st.caption(f"预计耗时约 {est_min} 分钟（每篇最长 15 分钟）")
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
                    "需 `DEEPSEEK_API_KEY` / `DASHSCOPE_API_KEY`。",
                )
            elif mode == "一键入库":
                st.info(
                    "⏱️ 预计 **2 ~ 3 分钟**\n"
                    "自动完成：解析 → Judge → 质量评估 → Qdrant 入库。\n"
                    "已入库的论文会检测重复。",
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


def _render_batch_import() -> list[Any]:
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


# ─── 批量处理辅助 ──────────────────────────────────────────────────────────────


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


# ─── 主界面 ────────────────────────────────────────────────────────────────────


def main() -> None:
    sb = sidebar()
    pdf_bytes: bytes | None = sb["pdf_bytes"]
    paper_id: str = sb["paper_id"]
    mode: str = sb["mode"]
    parse_clicked: bool = sb["parse_clicked"]
    filename: str = sb["filename"]
    batch_info: dict | None = sb.get("batch")

    # session state 管理
    if "result" not in st.session_state:
        st.session_state.result = None
    if "current_paper" not in st.session_state:
        st.session_state.current_paper = None
    if "orig_filename" not in st.session_state:
        st.session_state.orig_filename = ""
    # 已保存结果：选择查看
    if "selected_saved" not in st.session_state:
        st.session_state.selected_saved = None
    # 保存覆盖确认文件名
    if "pending_save_filename" not in st.session_state:
        st.session_state.pending_save_filename = ""
    # 保存成功消息
    if "save_success_msg" not in st.session_state:
        st.session_state.save_success_msg = ""
    # 批量处理进度
    if "batch_results" not in st.session_state:
        st.session_state.batch_results: list[dict] = []

    # 记住原始文件名
    if filename:
        st.session_state.orig_filename = filename

    # 切换论文时清除旧结果
    if paper_id and paper_id != st.session_state.current_paper:
        st.session_state.result = None
        st.session_state.current_paper = paper_id
        st.session_state.selected_saved = None

    PIPELINE_TIMEOUT = 900  # 每篇最长 15 分钟（pipeline ~2min + Map-Reduce ~5-10min + store）

    # ── 批量模式 ──
    if batch_info and batch_info.get("pending"):
        batch_files = batch_info.get("files", [])
        if batch_files:
            progress = st.progress(0, "批量入库中…")
            status = st.empty()
            st.session_state.batch_results = []
            success_n = 0
            skip_n = 0
            fail_n = 0

            for i, pdf_file in enumerate(batch_files):
                pct = (i + 1) / len(batch_files)
                status.info(f"({i+1}/{len(batch_files)}) 处理: {pdf_file.name}")

                try:
                    file_bytes = pdf_file.read()
                    pid = build_paper_id(file_bytes)

                    # 重复检测
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

                    # 一键入库：run_pipeline → quality eval → store（统一超时保护）
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
            # 不清除 batch_info，让用户可以看到结果后手动切换
        return

    # ── 单篇模式 ──
    if parse_clicked and pdf_bytes:
        # ── 重复检测 ──
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
                return  # 未确认，阻止解析

        st.session_state._pdf_bytes = pdf_bytes  # 供 PDF 重建使用

        if mode == "快速预览":
            mode_key = "preview"
        else:
            mode_key = "pipeline"  # 完整流水线 + 一键入库都用 pipeline

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
                                "**可能原因：** DeepSeek / DashScope / Qwen-VL API 响应过慢或网络不稳定。\n\n"
                                "**建议：**\n"
                                "- 确认 `.env` 中 `DEEPSEEK_API_KEY` 与 `DASHSCOPE_API_KEY` 均有效且有余额\n"
                                "- 稍后重试，或改用「快速预览」模式跳过所有 LLM 调用"
                            )
                            return
                timeout_banner.empty()
            elapsed = time.time() - start_ts
            result["_elapsed"] = elapsed
            st.caption(f"✅ 解析完成，耗时 {elapsed:.1f} 秒")

            # 标记解析模式（供结果展示判断）
            result["mode"] = "auto" if mode == "一键入库" else result.get("mode", "pipeline")

            # ── 一键入库：自动质量评估 + 入库 + 保存 ──
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
        # 欢迎页 + 已保存入口 + 独立检索
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
                **完整流水线**：调用 LLM Judge 评估质量，图片调用 VisionParser（qwen-vl-plus）生成描述，最终写入 Qdrant 向量索引，支持语义检索。
                """
            )
        with welcome_tabs[1]:
            render_standalone_search_tab()
        with welcome_tabs[2]:
            render_saved_tab()
        return

    # 结果展示
    verdicts = result.get("verdicts", {})
    is_pipeline = result.get("mode") in ("pipeline", "auto")
    is_manual_pipeline = result.get("mode") == "pipeline"  # 完整流水线（需手动分步）

    tabs = st.tabs(["📊 概览", "📝 文本块", "🖼️ 图片", "📋 表格", "📐 PDF 重建", "🔍 语义检索", "💾 已保存"])

    with tabs[0]:
        render_overview(result)

        # ── 完整流水线模式：阶段操作按钮（一键入库已自动完成，不显示）──
        if is_manual_pipeline:
            quality = result.get("quality", {})
            has_quality = bool(quality.get("category"))
            stored_key = f"stored_{result.get('paper_id', '')}"
            is_stored = bool(st.session_state.get(stored_key))

            st.divider()

            # 阶段 1: 解析完成 → 质量评估
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

            # 阶段 2: 质量评估完成 → 入库
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

            # 阶段 3: 入库完成
            elif is_stored:
                store_r = st.session_state[stored_key]
                if store_r.get("duplicate"):
                    st.warning(
                        f"⚠️ 重复入库被阻止：该论文已存在 {store_r['existing_count']} 条记录。"
                    )
                elif store_r.get("failed", 0) == 0:
                    st.success(f"✅ 入库完成！{store_r['stored']} 个 chunk 已写入 Qdrant")
                else:
                    st.warning(f"⚠️ 部分入库：{store_r['stored']} 成功, {store_r['failed']} 失败")

        # ── 流水线模式：保存到文件 ──
        if is_pipeline and result.get("status") in ("done",):
            st.divider()
            st.markdown("### 💾 保存解析结果到文件")
            save_filename = st.session_state.get("orig_filename", "paper") or "paper"
            save_name = st.text_input("保存文件名", value=save_filename, key="save_name_input").strip()

            col_save, _ = st.columns([1, 3])
            with col_save:
                if st.button("💾 保存结果", use_container_width=True, type="primary"):
                    safe_name = save_name.replace("/", "_").replace("\\", "_")
                    if (RESULTS_DIR / f"{safe_name}.json").exists():
                        st.session_state.pending_save_filename = save_name
                    else:
                        snapshot = build_snapshot(result, save_name)
                        save_result(snapshot, save_name)
                        st.session_state.save_success_msg = f"✅ 已保存到 data/results/{safe_name}.json"

            # 覆盖确认
            pending_name = st.session_state.get("pending_save_filename", "")
            if pending_name:
                safe_name = pending_name.replace("/", "_").replace("\\", "_")
                st.warning(f"⚠️ 已存在同名文件 \"{safe_name}.json\"，是否覆盖？")
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("✅ 确认覆盖", type="primary"):
                        snapshot = build_snapshot(result, pending_name)
                        overwrite_result(snapshot, pending_name)
                        st.session_state.pending_save_filename = ""
                        st.session_state.save_success_msg = f"✅ 已覆盖保存到 data/results/{safe_name}.json"
                        st.rerun()
                with col_no:
                    if st.button("❌ 取消"):
                        st.session_state.pending_save_filename = ""
                        st.rerun()

            # 保存成功提示
            if st.session_state.get("save_success_msg"):
                st.success(st.session_state.save_success_msg)
                st.session_state.save_success_msg = ""

    with tabs[1]:
        # 预览模式展示 merged_chunks（含 header 合并），流水线模式展示原始 text_chunks
        chunks_to_show = result.get("merged_chunks") if result.get("mode") == "preview" else result.get("text_chunks", [])
        render_text_tab(chunks_to_show, verdicts)

    with tabs[2]:
        render_image_tab(result)

    with tabs[3]:
        render_table_tab(
            result.get("table_chunks", []),
            verdicts,
            opencv_detections=result.get("opencv_detections"),
            pdf_bytes=pdf_bytes,
        )

    with tabs[4]:
        render_reconstruction_tab(result)

    with tabs[5]:
        if paper_id:
            render_search_tab(paper_id)
        else:
            st.info("无论文 ID，无法检索")

    with tabs[6]:
        render_saved_tab()


if __name__ == "__main__":
    main()
