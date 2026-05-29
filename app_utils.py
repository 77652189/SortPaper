"""
SortPaper 工具函数 — 纯函数，不依赖 Streamlit。
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime

from app_config import RESULTS_DIR

logger = logging.getLogger(__name__)


# ─── Paper ID ──────────────────────────────────────────────────────────────────

def build_paper_id(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


# ─── Chunk 序列化 ──────────────────────────────────────────────────────────────

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


# ─── Chunk 描述生成 ────────────────────────────────────────────────────────────

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
        from src.judge.llm_runtime import run_llm_call

        resp = run_llm_call(
            lambda: client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
                temperature=0.3,
            ),
            label="deepseek-chunk-description",
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("生成 %s 描述失败: %s", content_type, e)
        return ""


# ─── 保存/加载结果 ─────────────────────────────────────────────────────────────

def build_snapshot(result: dict, filename: str) -> dict:
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
        return None
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


# ─── Qdrant 检索 ─────────────────────────────────────────────────────────────

def qdrant_search(
    query: str,
    paper_id: str = None,
    top_k: int = 5,
    rerank: bool = False,
    filter_kwargs: dict | None = None,
    lexical_backfill: bool = True,
    query_rewrite: bool = False,
    multi_query: bool = False,
) -> list[dict]:
    import sys
    sys.path.insert(0, ".")
    from src.store.qdrant_store import QdrantStore
    from src.retrieval.multi_query import multi_query_search

    store = QdrantStore()
    filters = dict(filter_kwargs or {})
    if paper_id:
        filters["paper_id"] = paper_id
    started = time.monotonic()
    search_meta: dict = {}
    if query_rewrite or multi_query:
        multi_result = multi_query_search(
            store,
            query,
            limit=top_k,
            filter_kwargs=filters or None,
            rerank=rerank,
            lexical_backfill=lexical_backfill,
            use_query_rewrite=query_rewrite or multi_query,
            route_limit=4 if multi_query else 2,
        )
        results = multi_result.results
        search_meta = multi_result.metadata()
    else:
        results = store.search(
            query=query, limit=top_k,
            filter_kwargs=filters or None,
            rerank=rerank,
            lexical_backfill=lexical_backfill,
        )
    elapsed_ms = (time.monotonic() - started) * 1000
    logger.info(
        "Qdrant search | top_k=%s rerank=%s lexical_backfill=%s query_rewrite=%s multi_query=%s filters=%s hits=%s elapsed_ms=%.1f",
        top_k,
        rerank,
        lexical_backfill,
        query_rewrite,
        multi_query,
        sorted(filters.keys()),
        len(results),
        elapsed_ms,
    )
    return [
        {"id": r["id"], "score": r["score"],
         "reranked": r.get("reranked", False),
         "matched_routes": r.get("matched_routes", []),
         "matched_queries": r.get("matched_queries", []),
         "search_meta": search_meta,
         "payload": r["payload"]}
        for r in results
    ]
