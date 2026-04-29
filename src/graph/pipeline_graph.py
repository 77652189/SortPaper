"""
Pipeline 工作流图：基于 LayoutChunk 的并行解析 → 独立 Judge → 汇聚 → 存储流程。

并行 Fan-out/Fan-in 架构：
  coordinator
      │
      ├─────────────────────────────┬──────────────────────────────┐
      ▼                             ▼                               ▼
  text_worker                   table_worker                  image_worker
      │                             │                               │
      ▼                             ▼                               ▼
  judge_text                  judge_table                  judge_image
      │                             │                               │
      │◀─── retry < 3 ─────────────┤                               │
      │                             │◀──── retry < 3 ──────────────┤
      │                             │                               │
      │                             │◀────── retry < 3 ────────────┘
      │                             │                               │
      ├─────────────────────────────┴──────────────────────────────►│
      │                         (all pass)                        │
      ▼                                                            ▼
  merge_chunks (重建阅读顺序) ───────────────────────────▶ store ──▶ END
                                                              │
                                                    (any fail + retry >= 3)
                                                              │
                                                              ▼
                                                        finalize_failed
"""

from __future__ import annotations

import logging
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from src.judge.llm_judge import JudgeVerdict, LLMJudge
from src.parsers import LayoutChunk, LayoutMerger, PyMuPDFParser, TableParser, VisionParser
from src.store.faiss_store import FAISSStore

logger = logging.getLogger(__name__)


# ─── 工作流状态 ────────────────────────────────────────────────────────────────




class PipelineState(TypedDict, total=False):
    pdf_path: str
    paper_id: str
    # 三个 Worker 输出的 chunks（各自独立）
    text_chunks: list[LayoutChunk]
    table_chunks: list[LayoutChunk]
    image_chunks: list[LayoutChunk]
    # 各自 Judge 的判决结果
    text_verdicts: list[dict]
    table_verdicts: list[dict]
    image_verdicts: list[dict]
    # 各路线重试次数
    text_retries: int
    table_retries: int
    image_retries: int
    # 合并后的统一 chunks
    merged_chunks: list[LayoutChunk]
    status: str
    error: str | None
    output_dir: str


# ─── 常量 ──────────────────────────────────────────────────────────────────────


STORE_DIR = "faiss_index"
MAX_RETRIES = 3


# ─── 辅助函数 ──────────────────────────────────────────────────────────────────


def _failed_ids(verdicts: list[JudgeVerdict]) -> set[str]:
    return {v["chunk_id"] for v in verdicts if not v["passed"] and v["chunk_id"]}


def _build_feedback(verdicts: list[JudgeVerdict]) -> str:
    parts = [f"[{v['chunk_id']}] {v['feedback']}" for v in verdicts if not v["passed"] and v["chunk_id"]]
    return "\n".join(parts) if parts else ""


def _all_pass(state: PipelineState) -> bool:
    v = state.get
    text_ok = all(vp["passed"] for vp in v("text_verdicts", [])) and v("text_verdicts") != []
    table_ok = all(vp["passed"] for vp in v("table_verdicts", [])) and v("table_verdicts") != []
    image_ok = all(vp["passed"] for vp in v("image_verdicts", [])) and v("image_verdicts") != []
    return text_ok and table_ok and image_ok


def _any_failed(state: PipelineState) -> bool:
    """检查是否有任意路线存在失败的 chunks。"""
    v = state.get
    has_text_failed = any(not vp["passed"] for vp in v("text_verdicts", [])) and v("text_verdicts") != []
    has_table_failed = any(not vp["passed"] for vp in v("table_verdicts", [])) and v("table_verdicts") != []
    has_image_failed = any(not vp["passed"] for vp in v("image_verdicts", [])) and v("image_verdicts") != []
    return has_text_failed or has_table_failed or has_image_failed


# ─── 工作流节点 ───────────────────────────────────────────────────────────────


def coordinator_node(state: PipelineState) -> PipelineState:
    from pathlib import Path

    pdf_path = Path(state["pdf_path"])
    if not pdf_path.exists():
        return {**state, "status": "failed", "error": f"PDF not found: {pdf_path}"}
    return {
        **state,
        "status": "processing",
        "error": None,
        "text_retries": state.get("text_retries", 0),
        "table_retries": state.get("table_retries", 0),
        "image_retries": state.get("image_retries", 0),
        "output_dir": state.get("output_dir", STORE_DIR),
        "text_chunks": state.get("text_chunks", []),
        "table_chunks": state.get("table_chunks", []),
        "image_chunks": state.get("image_chunks", []),
        "text_verdicts": state.get("text_verdicts", []),
        "table_verdicts": state.get("table_verdicts", []),
        "image_verdicts": state.get("image_verdicts", []),
        "merged_chunks": [],
    }


# ─── Worker 节点（并行）─────────────────────────────────────────────────────────


def text_worker_node(state: PipelineState) -> PipelineState:
    failed_ids = _failed_ids(state.get("text_verdicts", []))
    feedback = _build_feedback(state.get("text_verdicts", [])) if failed_ids else None
    retries = state.get("text_retries", 0)

    if failed_ids:
        logger.warning(
            "text_worker retry %d/%d | paper=%s | failed=%s",
            retries + 1, MAX_RETRIES, state.get("paper_id", "?"), list(failed_ids),
        )

    parser = PyMuPDFParser(state["pdf_path"])
    new_chunks: list[LayoutChunk] = parser.parse(feedback=feedback)

    # 保留已 passed 的，更新失败的
    existing = {c.chunk_id: c for c in state.get("text_chunks", []) if c.chunk_id not in failed_ids}
    for chunk in new_chunks:
        existing[chunk.chunk_id] = chunk

    return {
        **state,
        "text_retries": retries + 1 if failed_ids else retries,
        "text_chunks": list(existing.values()),
    }


def table_worker_node(state: PipelineState) -> PipelineState:
    failed_ids = _failed_ids(state.get("table_verdicts", []))
    feedback = _build_feedback(state.get("table_verdicts", [])) if failed_ids else None
    retries = state.get("table_retries", 0)

    if failed_ids:
        logger.warning(
            "table_worker retry %d/%d | paper=%s | failed=%s",
            retries + 1, MAX_RETRIES, state.get("paper_id", "?"), list(failed_ids),
        )

    parser = TableParser(state["pdf_path"])
    new_chunks: list[LayoutChunk] = parser.parse(feedback=feedback)

    existing = {c.chunk_id: c for c in state.get("table_chunks", []) if c.chunk_id not in failed_ids}
    for chunk in new_chunks:
        existing[chunk.chunk_id] = chunk

    return {
        **state,
        "table_retries": retries + 1 if failed_ids else retries,
        "table_chunks": list(existing.values()),
    }


def image_worker_node(state: PipelineState) -> PipelineState:
    failed_ids = _failed_ids(state.get("image_verdicts", []))
    feedback = _build_feedback(state.get("image_verdicts", [])) if failed_ids else None
    retries = state.get("image_retries", 0)

    if failed_ids:
        logger.warning(
            "image_worker retry %d/%d | paper=%s | failed=%s",
            retries + 1, MAX_RETRIES, state.get("paper_id", "?"), list(failed_ids),
        )

    parser = VisionParser(state["pdf_path"])
    new_chunks: list[LayoutChunk] = parser.parse(feedback=feedback)

    existing = {c.chunk_id: c for c in state.get("image_chunks", []) if c.chunk_id not in failed_ids}
    for chunk in new_chunks:
        existing[chunk.chunk_id] = chunk

    return {
        **state,
        "image_retries": retries + 1 if failed_ids else retries,
        "image_chunks": list(existing.values()),
    }


# ─── Judge 节点（并行）────────────────────────────────────────────────────────


def _is_section_heading(text: str) -> bool:
    """启发式判断是否为 section 标题：全大写 / 以数字+点开头 / 长度较短。"""
    stripped = text.strip()
    if not stripped:
        return False
    # 数字+点开头，如 "1. Introduction" 或 "2.3 Methods"
    if stripped[:1].isdigit() and len(stripped) < 80:
        return True
    # 全大写且不超过60字符
    if stripped.upper() == stripped and len(stripped) < 60 and stripped.isalpha() is False:
        return stripped.replace(" ", "").isupper() and len(stripped) < 60
    return False


def _group_by_section(chunks: list[LayoutChunk]) -> list[list[LayoutChunk]]:
    """按标题行将 text_chunks 切分为 section 组，每组为一个列表。"""
    groups: list[list[LayoutChunk]] = []
    current: list[LayoutChunk] = []
    for chunk in chunks:
        if _is_section_heading(chunk.raw_content) and current:
            groups.append(current)
            current = [chunk]
        else:
            current.append(chunk)
    if current:
        groups.append(current)
    return groups if groups else [chunks]


def judge_text_node(state: PipelineState) -> PipelineState:
    chunks = state.get("text_chunks", [])
    if not chunks:
        return {**state, "text_verdicts": []}

    judge = LLMJudge()
    verdicts: list[JudgeVerdict] = []

    # Section 级别判断：同 section 内所有 chunks 拼接后一起送 judge，verdict 广播给 section 内所有 chunk
    sections = _group_by_section(chunks)
    for section_chunks in sections:
        combined = "\n\n".join(c.raw_content for c in section_chunks)
        v = judge.judge("text", combined, state["pdf_path"])
        for chunk in section_chunks:
            verdicts.append({"chunk_id": chunk.chunk_id, "passed": v.passed, "score": v.score, "feedback": v.feedback})

    return {**state, "text_verdicts": verdicts}


def judge_table_node(state: PipelineState) -> PipelineState:
    chunks = state.get("table_chunks", [])
    if not chunks:
        return {**state, "table_verdicts": [{"chunk_id": "", "passed": True, "score": 1.0, "feedback": "no tables"}]}

    judge = LLMJudge()
    verdicts: list[JudgeVerdict] = []
    for chunk in chunks:
        v = judge.judge(chunk.content_type, chunk.raw_content, state["pdf_path"])
        verdicts.append({"chunk_id": chunk.chunk_id, "passed": v.passed, "score": v.score, "feedback": v.feedback})
    return {**state, "table_verdicts": verdicts}


def judge_image_node(state: PipelineState) -> PipelineState:
    chunks = state.get("image_chunks", [])
    if not chunks:
        return {**state, "image_verdicts": [{"chunk_id": "", "passed": True, "score": 1.0, "feedback": "no images"}]}

    judge = LLMJudge()
    verdicts: list[JudgeVerdict] = []
    for chunk in chunks:
        v = judge.judge(chunk.content_type, chunk.raw_content, state["pdf_path"])
        verdicts.append({"chunk_id": chunk.chunk_id, "passed": v.passed, "score": v.score, "feedback": v.feedback})
    return {**state, "image_verdicts": verdicts}


# ─── 重试路由（各路线独立）────────────────────────────────────────────────────


def retry_text(state: PipelineState) -> Literal["text_worker", "merge_chunks"]:
    if (any(not v["passed"] for v in state.get("text_verdicts", []))
            and state.get("text_retries", 0) < MAX_RETRIES):
        return "text_worker"
    return "merge_chunks"


def retry_table(state: PipelineState) -> Literal["table_worker", "merge_chunks"]:
    if (any(not v["passed"] for v in state.get("table_verdicts", []))
            and state.get("table_retries", 0) < MAX_RETRIES):
        return "table_worker"
    return "merge_chunks"


def retry_image(state: PipelineState) -> Literal["image_worker", "merge_chunks"]:
    if (any(not v["passed"] for v in state.get("image_verdicts", []))
            and state.get("image_retries", 0) < MAX_RETRIES):
        return "image_worker"
    return "merge_chunks"


# ─── 汇聚节点 ──────────────────────────────────────────────────────────────────


def merge_chunks_node(state: PipelineState) -> PipelineState:
    """
    重建阅读顺序：合并三个 Worker 的 chunks，
    按 page → column → y0 → x0 排序，生成统一的 merged_chunks。
    排序后回填 global_order 和 nearby_context（前后各1个chunk摘要）。
    """
    all_chunks: list[LayoutChunk] = []
    all_chunks.extend(state.get("text_chunks", []))
    all_chunks.extend(state.get("table_chunks", []))
    all_chunks.extend(state.get("image_chunks", []))

    merged = LayoutMerger.merge(all_chunks)

    # 回填 nearby_context：前后各取100字符摘要存入 metadata
    for i, chunk in enumerate(merged):
        chunk.metadata["nearby_context"] = {
            "prev": merged[i - 1].raw_content[:100] if i > 0 else "",
            "next": merged[i + 1].raw_content[:100] if i < len(merged) - 1 else "",
        }

    return {**state, "merged_chunks": merged}


# ─── 最终路由 ──────────────────────────────────────────────────────────────────


def final_route(state: PipelineState) -> Literal["store", "finalize_failed", "text_worker", "table_worker", "image_worker"]:
    """
    三条路线全部 pass → store；任意路线重试耗尽 → finalize_failed。
    """
    text_failed = any(not v["passed"] for v in state.get("text_verdicts", []))
    table_failed = any(not v["passed"] for v in state.get("table_verdicts", []))
    image_failed = any(not v["passed"] for v in state.get("image_verdicts", []))

    any_failed = text_failed or table_failed or image_failed
    if not any_failed:
        return "store"

    # 有失败 → 检查是否还能重试
    if text_failed and state.get("text_retries", 0) < MAX_RETRIES:
        return "text_worker"
    if table_failed and state.get("table_retries", 0) < MAX_RETRIES:
        return "table_worker"
    if image_failed and state.get("image_retries", 0) < MAX_RETRIES:
        return "image_worker"

    return "finalize_failed"


# ─── Store 节点 ────────────────────────────────────────────────────────────────


def store_node(state: PipelineState) -> PipelineState:
    store = FAISSStore()
    out_dir = state.get("output_dir", STORE_DIR)

    from pathlib import Path

    if Path(out_dir).exists():
        try:
            store.load(out_dir)
        except Exception:
            pass

    verdicts_map: dict[str, JudgeVerdict] = {}
    for v in state.get("text_verdicts", []):
        verdicts_map[v["chunk_id"]] = v
    for v in state.get("table_verdicts", []):
        verdicts_map[v["chunk_id"]] = v
    for v in state.get("image_verdicts", []):
        verdicts_map[v["chunk_id"]] = v

    all_chunks: list[LayoutChunk] = []
    all_chunks.extend(state.get("text_chunks", []))
    all_chunks.extend(state.get("table_chunks", []))
    all_chunks.extend(state.get("image_chunks", []))

    for chunk in all_chunks:
        v = verdicts_map.get(chunk.chunk_id)
        if v and v["passed"]:
            store.add(
                paper_id=state["paper_id"],
                worker_type=chunk.content_type,
                content=chunk.raw_content,
                metadata={
                    **chunk.metadata,
                    "page": chunk.page,
                    "bbox": chunk.bbox,
                    "column": chunk.column,
                    "order_in_page": chunk.order_in_page,
                    "global_order": chunk.global_order,
                    "chunk_id": chunk.chunk_id,
                    "content_type": chunk.content_type,
                    "score": v["score"],
                },
            )

    store.save(out_dir)
    return {**state, "status": "done"}


def finalize_failed_node(state: PipelineState) -> PipelineState:
    logger.error(
        "Parse failed after max retries | paper=%s | text_retries=%d | table_retries=%d | image_retries=%d",
        state.get("paper_id", "?"),
        state.get("text_retries", 0),
        state.get("table_retries", 0),
        state.get("image_retries", 0),
    )
    return {**state, "status": "failed"}


# ─── 图构建 ────────────────────────────────────────────────────────────────────


def build_graph():
    graph = StateGraph(PipelineState)

    # 节点
    graph.add_node("coordinator", coordinator_node)
    graph.add_node("text_worker", text_worker_node)
    graph.add_node("table_worker", table_worker_node)
    graph.add_node("image_worker", image_worker_node)
    graph.add_node("judge_text", judge_text_node)
    graph.add_node("judge_table", judge_table_node)
    graph.add_node("judge_image", judge_image_node)
    graph.add_node("merge_chunks", merge_chunks_node)
    graph.add_node("store", store_node)
    graph.add_node("finalize_failed", finalize_failed_node)

    # 启动
    graph.add_edge(START, "coordinator")

    # Fan-out：coordinator 同时触发三条并行路线
    graph.add_edge("coordinator", "text_worker")
    graph.add_edge("coordinator", "table_worker")
    graph.add_edge("coordinator", "image_worker")

    # 各路线的 judge（并行）
    graph.add_edge("text_worker", "judge_text")
    graph.add_edge("table_worker", "judge_table")
    graph.add_edge("image_worker", "judge_image")

    # 各路线独立 retry 路由
    graph.add_conditional_edges("judge_text", retry_text)
    graph.add_conditional_edges("judge_table", retry_table)
    graph.add_conditional_edges("judge_image", retry_image)

    # Fan-in：三路线全部 pass 后汇聚到 merge_chunks
    # 然后由最终路由决定：全部 pass -> store；重试耗尽 -> finalize_failed
    graph.add_conditional_edges("merge_chunks", final_route)

    # 终结节点
    graph.add_edge("store", END)
    graph.add_edge("finalize_failed", END)

    return graph.compile()
