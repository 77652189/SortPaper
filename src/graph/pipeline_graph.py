"""
Pipeline 工作流图：基于 LayoutChunk 的并行解析 → 独立 Judge → 汇聚流程。

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
  merge_chunks (重建阅读顺序) ───────────────────────▶ finalize ──▶ END

quality_eval 和 store 已从图中移除，由 Streamlit 按钮手动触发。
"""

from __future__ import annotations

import logging
import operator
from typing import Any, Literal, TypedDict, Annotated

from langgraph.graph import END, START, StateGraph

from src.judge.llm_judge import JudgeVerdict, LLMJudge
from src.judge.paper_evaluator import PaperQualityEvaluator
from src.parsers import LayoutChunk, LayoutMerger, PyMuPDFParser, TableParser, VisionParser
from src.store.qdrant_store import QdrantStore
from src.judge.table_judge import build_storage_decision, build_table_embedding_text

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
    # 论文质量评估结果
    quality_result: dict
    # 各 Worker 解析耗时（秒）—— 并行写时合并
    worker_timing: Annotated[dict[str, float], lambda a, b: {k: max(a.get(k, 0), b.get(k, 0)) for k in set(a) | set(b)}]
    status: str
    error: str | None
    auto_store: bool  # 是否自动入库（默认 True）
    output_dir: str
    # 三条路线是否已完成 judge（用于 fan-in 同步）
    # Annotated + operator.or_：并发写入时做集合并集（自动去重，不会累积）
    routes_done: Annotated[set[str], operator.or_]


# ─── 常量 ──────────────────────────────────────────────────────────────────────


STORE_DIR = "faiss_index"
MAX_RETRIES = 1  # 全局上限（text 路径实际 0 次，见 retry_text）

# 图片描述润色阈值：Judge 评分低于此值时触发 DeepSeek 文字润色。
# 设为 0.9 意味着"绝大多数图片都会润色一次"，只有高质量描述才跳过。
IMAGE_POLISH_THRESHOLD = 0.9


# ─── 辅助函数 ──────────────────────────────────────────────────────────────────


def _failed_ids(verdicts: list[dict]) -> set[str]:
    return {v["chunk_id"] for v in verdicts if not v["passed"] and v["chunk_id"]}


def _needs_polish_ids(verdicts: list[dict]) -> set[str]:
    """返回需要润色的图片 chunk_id（Judge 评分 < IMAGE_POLISH_THRESHOLD）。

    与 _failed_ids 不同：即使 passed=True，只要分数未达到高质量阈值，
    也会触发一次 DeepSeek 文字润色，利用 Judge 反馈提升描述质量。
    """
    return {
        v["chunk_id"] for v in verdicts
        if v.get("score", 1.0) < IMAGE_POLISH_THRESHOLD and v.get("chunk_id")
    }


def _page_from_chunk_id(chunk_id: str) -> int | None:
    """从 chunk_id 中提取页码（格式: table_p9_col2_y...）。"""
    import re
    m = re.search(r"_p(\d+)_", chunk_id)
    return int(m.group(1)) if m else None


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
        "auto_store": state.get("auto_store", True),
        "text_chunks": state.get("text_chunks", []),
        "table_chunks": state.get("table_chunks", []),
        "image_chunks": state.get("image_chunks", []),
        "text_verdicts": state.get("text_verdicts", []),
        "table_verdicts": state.get("table_verdicts", []),
        "image_verdicts": state.get("image_verdicts", []),
        "merged_chunks": [],
        "quality_result": {},
        "worker_timing": state.get("worker_timing", {}),
        "routes_done": set(),
    }


# ─── Worker 节点（并行）─────────────────────────────────────────────────────────


def text_worker_node(state: PipelineState) -> dict:
    import time
    t0 = time.time()

    failed_ids = _failed_ids(state.get("text_verdicts", []))
    feedback = _build_feedback(state.get("text_verdicts", [])) if failed_ids else None
    retries = state.get("text_retries", 0)

    if failed_ids:
        logger.warning(
            "text_worker retry %d/%d | paper=%s | failed=%s",
            retries + 1, MAX_RETRIES, state.get("paper_id", "?"), list(failed_ids),
        )

    parser = PyMuPDFParser(state["pdf_path"])
    # 始终用 block 级解析；切换行级只会产生大量极短 chunk，judge 分数更低、失败更多
    new_chunks: list[LayoutChunk] = parser.parse()

    existing = {c.chunk_id: c for c in state.get("text_chunks", [])}
    for chunk in new_chunks:
        existing[chunk.chunk_id] = chunk

    # 只返回本节点实际修改的字段，避免并行 fan-out 时的写冲突
    return {
        "text_retries": retries + 1 if failed_ids else retries,
        "text_chunks": list(existing.values()),
        "worker_timing": {"text": round(time.time() - t0, 1)},
    }


def table_worker_node(state: PipelineState) -> dict:
    import time
    t0 = time.time()

    retries = state.get("table_retries", 0)
    prev_verdicts = state.get("table_verdicts", [])

    print(f"[debug] table_worker_node: retries={retries}", flush=True)

    if not prev_verdicts:
        # 首轮：关键词引导 + pdfplumber（含三线表/表头框几何提取）
        parser = TableParser(state["pdf_path"], enable_vision_fallback=False)
        new_chunks = parser.parse(strategy="all")
        existing: dict[str, LayoutChunk] = {}
        for chunk in new_chunks:
            existing[chunk.chunk_id] = chunk
        return {
            "table_retries": 0,
            "table_chunks": list(existing.values()),
            "worker_timing": {"table": round(time.time() - t0, 1)},
        }

    # retry：根据 issue_type 针对性重试
    failed_ids: set[str] = set()
    false_positive_ids: set[str] = set()
    print(f"[debug] table_worker retry | verdicts={len(prev_verdicts)}", flush=True)
    for v in prev_verdicts:
        cid = v.get("chunk_id", "")
        if v.get("passed") or not cid:
            continue
        issue = v.get("issue_type", "")
        print(f"[debug]   failed chunk: {cid} | issue_type={issue!r}", flush=True)
        failed_ids.add(cid)
        if issue == "false_positive":
            false_positive_ids.add(cid)
            print(f"[debug]   -> false_positive: discarding, skip retry", flush=True)
        elif issue == "structure_error":
            print("[debug]   -> structure_error: no alternate table parser configured", flush=True)
        elif issue == "missing_content":
            print("[debug]   -> missing_content: no alternate table parser configured", flush=True)

    logger.warning(
        "table_worker retry %d/%d | paper=%s | false_positive=%d",
        retries + 1, MAX_RETRIES, state.get("paper_id", "?"),
        len(false_positive_ids),
    )

    # 保留非失败 chunk
    existing = {c.chunk_id: c for c in state.get("table_chunks", [])
                if c.chunk_id not in failed_ids}

    print(f"[debug] retry result: keeping {len(existing)} chunks (discarded {len(failed_ids) - len(existing)} failed)", flush=True)

    return {
        "table_retries": retries + 1 if failed_ids else retries,
        "table_chunks": list(existing.values()),
        "worker_timing": {"table": round(time.time() - t0, 1)},
    }


def image_worker_node(state: PipelineState) -> dict:
    import time
    t0 = time.time()

    polish_ids = _needs_polish_ids(state.get("image_verdicts", []))
    retries = state.get("image_retries", 0)

    if polish_ids:
        # 润色路径：用 DeepSeek 根据 Judge 反馈改写描述（不重读图）。
        # 触发条件：Judge 评分 < IMAGE_POLISH_THRESHOLD（默认 0.9），
        # 即绝大多数图片都会经过一次润色，只有高质量描述才直接入库。
        logger.info(
            "image_worker polish %d/%d | paper=%s | polish=%d 张 (threshold=%.1f)",
            retries + 1, MAX_RETRIES, state.get("paper_id", "?"),
            len(polish_ids), IMAGE_POLISH_THRESHOLD,
        )
        existing = {c.chunk_id: c for c in state.get("image_chunks", [])}
        verdicts = state.get("image_verdicts", [])
        for vid in polish_ids:
            chunk = existing.get(vid)
            if chunk is None:
                continue
            fb = next((v.get("feedback", "") for v in verdicts if v.get("chunk_id") == vid), "")
            rewritten = _rewrite_image_desc(chunk.raw_content, fb)
            if rewritten:
                chunk.raw_content = rewritten

        return {
            "image_retries": retries + 1,
            "image_chunks": list(existing.values()),
            "worker_timing": {"image": round(time.time() - t0, 1)},
        }

    # 首次解析：VL 读图
    parser = VisionParser(state["pdf_path"])
    new_chunks, vis_timing = parser.parse()

    failed_ids = _failed_ids(state.get("image_verdicts", []))
    if failed_ids:
        existing: dict[str, LayoutChunk] = {}
    else:
        existing = {c.chunk_id: c for c in state.get("image_chunks", [])}
    for chunk in new_chunks:
        existing[chunk.chunk_id] = chunk

    return {
        "image_retries": retries + 1 if failed_ids else retries,
        "image_chunks": list(existing.values()),
        "worker_timing": {
            "image": round(time.time() - t0, 1),
            "image_extract": vis_timing.get("extract", 0),
            "image_describe": vis_timing.get("describe", 0),
        },
    }


def _rewrite_image_desc(original: str, feedback: str) -> str:
    """用 DeepSeek 根据 Judge 反馈改写图片描述（不重新读图）。"""
    import os
    from openai import OpenAI

    prompt = (
        "以下是一张学术论文图片的解析描述，以及质检员的反馈意见。"
        "请根据反馈修改描述，只描述图片中实际可见的视觉内容。"
        "保持原描述中有用的信息，删除编造或错误的部分。\n\n"
        f"原始描述：\n{original[:3000]}\n\n"
        f"质检反馈：{feedback}\n\n"
        "直接返回修改后的描述文字，不要额外说明。"
    )

    try:
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com/v1",
            timeout=30.0,
        )
        from src.judge.llm_runtime import run_llm_call

        resp = run_llm_call(
            lambda: client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.2,
            ),
            label="deepseek-image-rewrite",
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning("改写图片描述失败: %s", e)
        return ""


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


def _is_noise(text: str) -> bool:
    """判断是否为噪音文本（页码、页眉、过短碎片等），跳过 judge。"""
    stripped = text.strip()
    if not stripped:
        return True
    # 过短（低于 40 字符的碎片几乎都是页眉/页码/参考文献行，直接跳过 judge）
    if len(stripped) < 40:
        return True
    # 纯数字/符号
    if not any(c.isalpha() or '\u4e00' <= c <= '\u9fff' for c in stripped):
        return True
    # 常见页眉/页脚模式
    upper = stripped.upper().strip()
    if upper in ("REFERENCES", "ACKNOWLEDGEMENTS", "ACKNOWLEDGMENTS", "FUNDING"):
        return False  # 参考文献/致谢标题 → 保留
    noise_starts = ("DOI:", "HTTP", "WWW.", "COPYRIGHT", "©", "VOL.", "NO.", "PP.", "ISSN")
    if any(upper.startswith(p) for p in noise_starts):
        return True
    return False


def _group_by_section(chunks: list[LayoutChunk]) -> list[list[LayoutChunk]]:
    """按标题行将 text_chunks 切分为 section 组，每组为一个列表。
    噪音 chunk 被过滤，不参与分组。"""
    clean = [c for c in chunks if not _is_noise(c.raw_content)]
    if not clean:
        return []
    groups: list[list[LayoutChunk]] = []
    current: list[LayoutChunk] = []
    for chunk in clean:
        if _is_section_heading(chunk.raw_content) and current:
            groups.append(current)
            current = [chunk]
        else:
            current.append(chunk)
    if current:
        groups.append(current)
    return groups if groups else [clean]


def judge_text_node(state: PipelineState) -> dict:
    import time
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    chunks = state.get("text_chunks", [])

    if not chunks:
        return {"text_verdicts": [], "routes_done": {"text"}}

    prev_verdicts = {v["chunk_id"]: v for v in state.get("text_verdicts", [])}
    noise_ids: set[str] = {c.chunk_id for c in chunks if _is_noise(c.raw_content)}

    # 找出需要重判的 section（有未通过的非噪音 chunk）
    pending: list[list] = []
    for section_chunks in _group_by_section(chunks):
        need = [c for c in section_chunks
                if c.chunk_id not in noise_ids
                and not prev_verdicts.get(c.chunk_id, {}).get("passed")]
        if need:
            pending.append(section_chunks)

    def _judge_section(section_chunks):
        combined = "\n\n".join(c.raw_content for c in section_chunks)
        v = LLMJudge().judge("text", combined, state["pdf_path"])
        return section_chunks, v

    verdict_map: dict[str, dict] = dict(prev_verdicts)
    if pending:
        with ThreadPoolExecutor(max_workers=min(5, len(pending))) as pool:
            for section_chunks, v in pool.map(_judge_section, pending):
                for chunk in section_chunks:
                    if chunk.chunk_id not in noise_ids:
                        verdict_map[chunk.chunk_id] = {
                            "chunk_id": chunk.chunk_id,
                            "passed": v.passed,
                            "score": v.score,
                            "feedback": v.feedback,
                        }

    jt = state.get("judge_timing", {})
    jt["text"] = round(time.time() - t0, 1)
    return {"text_verdicts": list(verdict_map.values()), "routes_done": {"text"}, "judge_timing": jt}


def judge_table_node(state: PipelineState) -> dict:
    import time
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    chunks = state.get("table_chunks", [])

    if not chunks:
        return {"table_verdicts": [{"chunk_id": "", "passed": True, "score": 1.0, "feedback": "no tables"}], "routes_done": {"table"}}

    prev_verdicts = {v["chunk_id"]: v for v in state.get("table_verdicts", [])}
    verdict_map: dict[str, dict] = dict(prev_verdicts)
    for chunk in chunks:
        if chunk.metadata.get("table_region_discovery_only") and chunk.chunk_id not in verdict_map:
            verdict_map[chunk.chunk_id] = {
                "chunk_id": chunk.chunk_id,
                "passed": True,
                "score": chunk.metadata.get("table_region", {}).get("confidence", 1.0),
                "feedback": "table region discovery only; structure is not judged",
                "issue_type": "none",
            }

    need_judge = [
        c for c in chunks
        if not c.metadata.get("table_region_discovery_only")
        and not verdict_map.get(c.chunk_id, {}).get("passed")
    ]

    def _judge_one(chunk):
        v = LLMJudge().judge(chunk.content_type, chunk.raw_content, state["pdf_path"])
        return {"chunk_id": chunk.chunk_id, "passed": v.passed, "score": v.score,
                "feedback": v.feedback, "issue_type": v.issue_type}

    if need_judge:
        with ThreadPoolExecutor(max_workers=min(5, len(need_judge))) as pool:
            for nv in pool.map(_judge_one, need_judge):
                verdict_map[nv["chunk_id"]] = nv

    jt = state.get("judge_timing", {})
    jt["table"] = round(time.time() - t0, 1)
    return {"table_verdicts": list(verdict_map.values()), "routes_done": {"table"}, "judge_timing": jt}


def judge_image_node(state: PipelineState) -> dict:
    import time
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    chunks = state.get("image_chunks", [])

    if not chunks:
        return {"image_verdicts": [{"chunk_id": "", "passed": True, "score": 1.0, "feedback": "no images"}], "routes_done": {"image"}}

    prev_verdicts = {v["chunk_id"]: v for v in state.get("image_verdicts", [])}
    need_judge = [c for c in chunks if not prev_verdicts.get(c.chunk_id, {}).get("passed")]

    def _judge_one(chunk):
        v = LLMJudge().judge(chunk.content_type, chunk.raw_content, state["pdf_path"])
        return {"chunk_id": chunk.chunk_id, "passed": v.passed, "score": v.score,
                "feedback": v.feedback}

    verdict_map: dict[str, dict] = dict(prev_verdicts)
    if need_judge:
        with ThreadPoolExecutor(max_workers=min(5, len(need_judge))) as pool:
            for nv in pool.map(_judge_one, need_judge):
                verdict_map[nv["chunk_id"]] = nv

    jt = state.get("judge_timing", {})
    jt["image"] = round(time.time() - t0, 1)
    return {"image_verdicts": list(verdict_map.values()), "routes_done": {"image"}, "judge_timing": jt}


# ─── 重试路由（各路线独立）────────────────────────────────────────────────────


def retry_text(state: PipelineState) -> Literal["text_worker", "merge_chunks"]:
    # text 不复用反馈重试（line 级比 block 级更差），judge 只做质量门控
    return "merge_chunks"


def retry_table(state: PipelineState) -> Literal["table_worker", "merge_chunks"]:
    return "merge_chunks"


def retry_image(state: PipelineState) -> Literal["image_worker", "merge_chunks"]:
    # 有任意图片评分低于 IMAGE_POLISH_THRESHOLD 时触发润色（最多 1 次）。
    # 高于阈值（≥0.9）的图片直接入库，不再花费 API 调用。
    if (_needs_polish_ids(state.get("image_verdicts", []))
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
    import time
    t0 = time.time()
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

    mt = state.get("merge_timing", 0)
    return {"merged_chunks": merged, "merge_timing": round(mt + time.time() - t0, 1)}


# ─── 质量评估节点 ──────────────────────────────────────────────────────────────


def quality_eval_node(state: PipelineState) -> dict:
    """论文质量评估（分类 → Map-Reduce → chunk 上下文 → 入库）"""
    merged = state.get("merged_chunks", [])
    pdf_path = state.get("pdf_path", "")

    if not merged:
        return {"quality_result": {"skip": True}}

    from src.judge.paper_evaluator import PaperQualityEvaluator
    evaluator = PaperQualityEvaluator()
    result = evaluator.evaluate(merged, pdf_path)
    return {"quality_result": result}


# ─── 最终路由 ──────────────────────────────────────────────────────────────────


def final_route(state: PipelineState) -> Literal["finalize", "merge_chunks"]:
    """Fan-in 门控：等待三条路线全部完成后再结束。"""
    done = state.get("routes_done", set())
    required = {"text", "table", "image"}
    if not required.issubset(done):
        return "merge_chunks"
    return "finalize"


def finalize_node(state: PipelineState) -> PipelineState:
    """清理由 pipeline 负责的最终状态标记。"""
    return {"status": "done"}


# ─── Store 节点 ────────────────────────────────────────────────────────────────


def store_node(state: PipelineState) -> PipelineState:
    store = QdrantStore()
    quality = state.get("quality_result", {})

    # 用户关闭自动入库
    if not state.get("auto_store", True):
        logger.info("auto_store=False，跳过入库 | paper=%s", state.get("paper_id", "?"))
        return {"status": "pending_store"}

    # 分类为 other → 跳过入库
    if quality.get("skip"):
        logger.info("质量评估标记为跳过入库 | paper=%s | category=%s",
                     state.get("paper_id", "?"), quality.get("category"))
        return {"status": "skipped"}

    verdicts_map: dict[str, JudgeVerdict] = {}
    for v in state.get("text_verdicts", []):
        verdicts_map[v["chunk_id"]] = v
    for v in state.get("table_verdicts", []):
        verdicts_map[v["chunk_id"]] = v
    for v in state.get("image_verdicts", []):
        verdicts_map[v["chunk_id"]] = v

    merged: list[LayoutChunk] = state.get("merged_chunks", [])
    chunk_contexts: dict[str, str] = quality.get("chunk_contexts", {})
    excluded = 0
    excluded_tables = 0

    for chunk in merged:
        if chunk.content_type == "table":
            try:
                chunk.metadata.update(build_storage_decision(
                    content_type="table",
                    metadata=chunk.metadata,
                ))
            except Exception:
                pass
        if chunk.metadata.get("excluded_from_storage"):
            excluded += 1
            if chunk.content_type == "table":
                excluded_tables += 1
            logger.info(
                "跳过不可入库 chunk | paper=%s | chunk_id=%s | reason=%s",
                state.get("paper_id", "?"),
                chunk.chunk_id,
                chunk.metadata.get("storage_exclusion_reason")
                or chunk.metadata.get("unparseable_reason")
                or "excluded_from_storage",
            )
            continue

        v = verdicts_map.get(chunk.chunk_id)
        if v and v["passed"]:
            context = chunk_contexts.get(chunk.chunk_id, "")
            raw = chunk.raw_content
            display = f"{context}\n\n{raw}" if context else raw
            embed_content = (
                build_table_embedding_text(
                    raw_content=raw,
                    metadata=chunk.metadata,
                ) or raw
                if chunk.content_type == "table"
                else raw
            )

            metadata = {
                **chunk.metadata,
                "page": chunk.page,
                "bbox": chunk.bbox,
                "column": chunk.column,
                "order_in_page": chunk.order_in_page,
                "global_order": chunk.global_order,
                "chunk_id": chunk.chunk_id,
                "content_type": chunk.content_type,
                "score": v["score"],
                # 质量评估元数据（用于 Qdrant 过滤）
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
            }

            store.add(
                paper_id=state["paper_id"],
                worker_type=chunk.content_type,
                content=display,
                embed_content=embed_content,
                metadata=metadata,
            )

    store.save("")  # Qdrant 自动持久化

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
            store.count,
            len(failed),
            [v["chunk_id"] for v in failed],
        )
        return {
            "status": "partial",
            "stored": store.count,
            "failed": len(failed),
            "excluded": excluded,
            "excluded_tables": excluded_tables,
        }

    return {
        "status": "done",
        "excluded": excluded,
        "excluded_tables": excluded_tables,
    }


# ─── 图构建 ────────────────────────────────────────────────────────────────────


def build_graph():
    graph = StateGraph(PipelineState)

    # 节点（解析 + Judge + merge；quality 和 store 由 Streamlit 按钮手动触发）
    graph.add_node("coordinator", coordinator_node)
    graph.add_node("text_worker", text_worker_node)
    graph.add_node("table_worker", table_worker_node)
    graph.add_node("image_worker", image_worker_node)
    graph.add_node("judge_text", judge_text_node)
    graph.add_node("judge_table", judge_table_node)
    graph.add_node("judge_image", judge_image_node)
    graph.add_node("merge_chunks", merge_chunks_node)
    graph.add_node("finalize", finalize_node)

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

    # Fan-in 门控 → 等待全部完成 → finalize
    graph.add_conditional_edges("merge_chunks", final_route)
    graph.add_edge("finalize", END)

    return graph.compile()
