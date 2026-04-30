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
import operator
from typing import Any, Literal, TypedDict, Annotated

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
    # 三条路线是否已完成 judge（用于 fan-in 同步）
    # Annotated + operator.or_：并发写入时做集合并集（自动去重，不会累积）
    routes_done: Annotated[set[str], operator.or_]


# ─── 常量 ──────────────────────────────────────────────────────────────────────


STORE_DIR = "faiss_index"
MAX_RETRIES = 3


# ─── 辅助函数 ──────────────────────────────────────────────────────────────────


def _failed_ids(verdicts: list[dict]) -> set[str]:
    return {v["chunk_id"] for v in verdicts if not v["passed"] and v["chunk_id"]}


def _build_feedback(verdicts: list[dict]) -> str:
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
        "routes_done": set(),
    }


# ─── Worker 节点（并行）─────────────────────────────────────────────────────────


def text_worker_node(state: PipelineState) -> dict:
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

    # 只返回本节点实际修改的字段，避免并行 fan-out 时的写冲突
    return {
        "text_retries": retries + 1 if failed_ids else retries,
        "text_chunks": list(existing.values()),
    }


def table_worker_node(state: PipelineState) -> dict:
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
        "table_retries": retries + 1 if failed_ids else retries,
        "table_chunks": list(existing.values()),
    }


def image_worker_node(state: PipelineState) -> dict:
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
        "image_retries": retries + 1 if failed_ids else retries,
        "image_chunks": list(existing.values()),
    }


# ─── Judge 节点（并行）────────────────────────────────────────────────────────


def _is_section_heading(text: str) -> bool:
    """启发式判断是否为 section 标题。

    支持中英文论文的常见标题格式：
    - 数字+点编号： "1. Introduction", "2.3 Methods"
    - 中文数字编号： "一、引言", "1.1 相关工作"
    - 全大写英文标题（不含句号，较短）
    - 纯中文短行（无异议号结尾，长度 < 50）
    """
    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped) > 100:
        return False

    # 1. 数字+点开头（中英文均可能）
    if stripped[:1].isdigit():
        return True

    # 2. 中文数字编号开头：一、二、三、 or （一）（二）
    import re
    if re.match(r"^[一二三四五六七八九十]+[、．.]", stripped):
        return True
    if re.match(r"^（[一二三四五六七八九十]+）", stripped):
        return True

    # 3. 全大写英文标题（不含句号，且不全是数字/符号）
    if stripped.isascii() and stripped.upper() == stripped and not stripped.replace(" ", "").replace("-", "").isdigit():
        # 排除纯数字/纯符号，要求含有字母
        has_letter = any(c.isalpha() for c in stripped)
        return has_letter and len(stripped) < 60

    # 4. 纯中文短行（无异议号/逗号结尾，可能是标题）
    #    排除明显是正文的特征：含句号、逗号较多、或以"的"结尾
    if re.search(r"[\u4e00-\u9fff]", stripped):
        # 含中文
        if stripped[-1] not in "。，、；：！？":
            # 不以标点结尾，可能是标题
            if "。" not in stripped and len(stripped) < 50:
                return True

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


def judge_text_node(state: PipelineState) -> dict:
    chunks = state.get("text_chunks", [])

    if not chunks:
        return {"text_verdicts": [], "routes_done": {"text"}}

    judge = LLMJudge()
    verdicts: list[dict] = []

    # Section 级别判断：同 section 内所有 chunks 拼接后一起送 judge，verdict 广播给 section 内所有 chunk
    sections = _group_by_section(chunks)
    for section_chunks in sections:
        combined = "\n\n".join(c.raw_content for c in section_chunks)
        v = judge.judge("text", combined, state["pdf_path"])
        for chunk in section_chunks:
            verdicts.append({
                "chunk_id": chunk.chunk_id,
                "passed": v.passed,
                "score": v.score,
                "feedback": v.feedback,
            })

    return {"text_verdicts": verdicts, "routes_done": {"text"}}


def judge_table_node(state: PipelineState) -> dict:
    chunks = state.get("table_chunks", [])

    if not chunks:
        return {"table_verdicts": [{"chunk_id": "", "passed": True, "score": 1.0, "feedback": "no tables"}], "routes_done": {"table"}}

    judge = LLMJudge()
    verdicts: list[dict] = []
    for chunk in chunks:
        v = judge.judge(chunk.content_type, chunk.raw_content, state["pdf_path"])
        verdicts.append({
            "chunk_id": chunk.chunk_id,
            "passed": v.passed,
            "score": v.score,
            "feedback": v.feedback,
        })
    return {"table_verdicts": verdicts, "routes_done": {"table"}}


def judge_image_node(state: PipelineState) -> dict:
    chunks = state.get("image_chunks", [])

    if not chunks:
        return {"image_verdicts": [{"chunk_id": "", "passed": True, "score": 1.0, "feedback": "no images"}], "routes_done": {"image"}}

    judge = LLMJudge()
    verdicts: list[dict] = []
    for chunk in chunks:
        v = judge.judge(chunk.content_type, chunk.raw_content, state["pdf_path"])
        verdicts.append({
            "chunk_id": chunk.chunk_id,
            "passed": v.passed,
            "score": v.score,
            "feedback": v.feedback,
        })
    return {"image_verdicts": verdicts, "routes_done": {"image"}}


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

    return {"merged_chunks": merged}


# ─── 最终路由 ──────────────────────────────────────────────────────────────────


def final_route(state: PipelineState) -> Literal["store", "merge_chunks"]:
    """
    merge_chunks 汇聚后的终结路由。

    降级存储策略：无论是否有失败 chunk，retries 耗尽后都走 store_node。
    store_node 只存通过的 chunk；有失败时状态置为 "partial"，无失败时为 "done"。

    注意：routes_done 使用 operator.or_ 只能累加，第一轮完成后永远是满集，
    因此 fan-in 门控只在第一轮有效。retry 轮次的同步依赖 retry 耗尽检查。
    """
    done = state.get("routes_done", set())
    required = {"text", "table", "image"}
    if not required.issubset(done):
        # 第一轮尚未全部完成，继续等待
        return "merge_chunks"

    text_failed = any(not v["passed"] for v in state.get("text_verdicts", []))
    table_failed = any(not v["passed"] for v in state.get("table_verdicts", []))
    image_failed = any(not v["passed"] for v in state.get("image_verdicts", []))

    if not (text_failed or table_failed or image_failed):
        return "store"

    # 有失败：检查各失败路线的 retry 是否耗尽
    text_exhausted  = not text_failed  or state.get("text_retries",  0) >= MAX_RETRIES
    table_exhausted = not table_failed or state.get("table_retries", 0) >= MAX_RETRIES
    image_exhausted = not image_failed or state.get("image_retries", 0) >= MAX_RETRIES

    if text_exhausted and table_exhausted and image_exhausted:
        # retries 全部耗尽 → 降级存储：存通过的部分，记录哪些失败
        return "store"

    # retry 仍在进行，继续在 merge_chunks 轮询等待
    return "merge_chunks"


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

    # 直接遍历 merged_chunks（已按阅读顺序排序），而非重新聚合
    merged: list[LayoutChunk] = state.get("merged_chunks", [])

    for chunk in merged:
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

    # 降级存储：检查是否有失败 chunk，有则记录 warning 并置 partial 状态
    all_verdicts = (
        state.get("text_verdicts", [])
        + state.get("table_verdicts", [])
        + state.get("image_verdicts", [])
    )
    failed = [v for v in all_verdicts if not v.get("passed") and v.get("chunk_id")]
    if failed:
        logger.warning(
            "Stored with partial failures | paper=%s | stored=%d | failed=%d | failed_ids=%s",
            state.get("paper_id", "?"),
            store.index.ntotal if store.index else 0,
            len(failed),
            [v["chunk_id"] for v in failed],
        )
        return {"status": "partial"}

    return {"status": "done"}


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

    # Fan-in → 最终路由：全部通过 → store；retry 耗尽（含部分失败）→ store（降级存储）
    graph.add_conditional_edges("merge_chunks", final_route)

    # store 永远是终结节点（status=done 或 partial）
    graph.add_edge("store", END)

    return graph.compile()
