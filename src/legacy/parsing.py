from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any

from src.application.pipeline_utils import (
    build_paper_id,
    chunk_to_dict as _chunk_to_dict,
    generate_chunk_description as _generate_chunk_description,
)

logger = logging.getLogger(__name__)


def run_legacy_preview(pdf_bytes: bytes) -> dict:
    """Legacy quick preview: PyMuPDF text + table parser + image placeholders."""
    import sys

    sys.path.insert(0, ".")
    from src.graph.pipeline_graph import _is_noise
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

        for chunk in text_chunks:
            if chunk.chunk_id in noise_ids:
                continue
            verdicts[chunk.chunk_id] = {
                "chunk_id": chunk.chunk_id,
                "passed": True,
                "score": 1.0,
                "feedback": "快速预览未调用文本 LLM Judge；完整流水线/入库时会逐 chunk 诊断。",
                "issue_type": "preview_text_not_judged",
                "judge_scope": "preview_skip",
            }

        tasks: list[tuple[str, Any]] = []
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
            _kind, chunk = task
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
        "text_chunks": [_chunk_to_dict(c) for c in text_chunks],
        "table_chunks": [_chunk_to_dict(c) for c in table_chunks],
        "image_chunks": [_chunk_to_dict(c) for c in image_chunks],
        "merged_chunks": [_chunk_to_dict(c) for c in merged_chunks],
        "images": images,
        "verdicts": verdicts,
        "status": "preview",
        "quality": {},
        "worker_timing": {"text": wt_text, "table": wt_table, "image": 0},
        "judge_timing": {"text": 0, "table": jt_judge},
        "merge_timing": 0,
        "desc_timing": 0,
        "mode": "preview",
    }


def _extract_image_placeholders(pdf_path: Path) -> tuple[list, list[dict]]:
    """Return image placeholder chunks and metadata without calling legacy VisionParser."""
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


def run_legacy_pipeline(
    pdf_bytes: bytes,
    paper_id: str,
    filename: str = "",
) -> dict:
    """Legacy LangGraph parser pipeline kept for display, compatibility, and regression checks."""
    import sys

    sys.path.insert(0, ".")
    from src.graph.pipeline_graph import build_graph

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)

    try:
        pipeline = build_graph()
        initial = {
            "pdf_path": str(tmp_path),
            "paper_id": paper_id,
            "text_retries": 0,
            "table_retries": 0,
            "image_retries": 0,
            "status": "processing",
            "output_dir": "",
        }

        final: dict = dict(initial)
        node_cnt: dict[str, int] = {}
        for event in pipeline.stream(initial, config={"recursion_limit": 100}, stream_mode="updates"):
            node_name, node_data = list(event.items())[0]
            for k, v in node_data.items():
                if isinstance(v, set):
                    final[k] = final.get(k, set()) | v
                elif isinstance(v, dict) and k in ("worker_timing", "judge_timing"):
                    a, b = final.get(k, {}), v
                    final[k] = {key: max(a.get(key, 0), b.get(key, 0)) for key in set(a) | set(b)}
                elif isinstance(v, dict):
                    final[k] = {**final.get(k, {}), **v}
                else:
                    final[k] = v

            node_cnt[node_name] = node_cnt.get(node_name, 0) + 1
            cnt = node_cnt[node_name]
            retry_tag = f" [retry {cnt - 1}]" if cnt > 1 else ""
            wt = node_data.get("worker_timing", {})
            t = wt.get("text", wt.get("table", wt.get("image", 0))) if wt else 0

            if node_name == "text_worker":
                logger.info("[legacy:text] parsed %d chunks (%.1fs)%s", len(node_data.get("text_chunks", [])), t, retry_tag)
            elif node_name == "table_worker":
                logger.info("[legacy:table] parsed %d tables (%.1fs)%s", len(node_data.get("table_chunks", [])), t, retry_tag)
            elif node_name == "image_worker":
                logger.info("[legacy:image] described %d images (%.1fs)%s", len(node_data.get("image_chunks", [])), t, retry_tag)
            elif node_name == "judge_text":
                verdicts = node_data.get("text_verdicts", [])
                passed = sum(1 for item in verdicts if item.get("passed"))
                logger.info("[legacy:judge] text: %d/%d passed%s", passed, len(verdicts), retry_tag)
            elif node_name == "judge_table":
                verdicts = node_data.get("table_verdicts", [])
                passed = sum(1 for item in verdicts if item.get("passed"))
                issue_types: dict[str, int] = {}
                for item in verdicts:
                    issue_type = item.get("issue_type", "?")
                    issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
                logger.info("[legacy:judge] table: %d/%d passed%s | issue_types=%s", passed, len(verdicts), retry_tag, issue_types)
            elif node_name == "judge_image":
                verdicts = node_data.get("image_verdicts", [])
                passed = sum(1 for item in verdicts if item.get("passed"))
                logger.info("[legacy:judge] image: %d/%d passed%s", passed, len(verdicts), retry_tag)
            elif node_name == "merge_chunks":
                logger.info("[legacy:merge] %d merged chunks", len(node_data.get("merged_chunks", [])))
            elif node_name == "finalize":
                logger.info("[legacy:pipeline] %s", node_data.get("status", "done"))
    finally:
        tmp_path.unlink(missing_ok=True)

    all_verdicts: dict[str, dict] = {}
    for verdict in (
        final.get("text_verdicts", [])
        + final.get("table_verdicts", [])
        + final.get("image_verdicts", [])
    ):
        if verdict.get("chunk_id"):
            all_verdicts[verdict["chunk_id"]] = verdict

    merged = final.get("merged_chunks", [])

    from concurrent.futures import ThreadPoolExecutor

    t_desc = time.time()
    targets = [(i, c) for i, c in enumerate(merged) if c.content_type in ("table", "image")]
    if targets:
        def _describe_one(item):
            idx, chunk = item
            return idx, _generate_chunk_description(chunk.content_type, chunk.raw_content)

        with ThreadPoolExecutor(max_workers=min(5, len(targets))) as pool:
            for idx, desc in pool.map(_describe_one, targets):
                if desc:
                    merged[idx].raw_content = f"{desc}\n\n{merged[idx].raw_content}"
    desc_timing = round(time.time() - t_desc, 1)

    return {
        "paper_id": paper_id,
        "filename": filename,
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
