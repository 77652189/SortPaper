from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from src.judge.llm_judge import LLMJudge, SummaryReport
from src.parsers.pymupdf_parser import PyMuPDFParser
from src.parsers.table_parser import TableParser
from src.parsers.vision_parser import VisionParser
from src.store.faiss_store import FAISSStore


class WorkerResult(TypedDict):
    content: str
    metadata: dict[str, Any]


class JudgeVerdict(TypedDict):
    passed: bool
    score: float
    feedback: str


class PipelineState(TypedDict, total=False):
    pdf_path: str
    paper_id: str
    text_result: WorkerResult | None
    table_result: WorkerResult | None
    image_result: WorkerResult | None
    text_verdict: JudgeVerdict | None
    table_verdict: JudgeVerdict | None
    image_verdict: JudgeVerdict | None
    text_retries: int
    table_retries: int
    image_retries: int
    text_feedback_history: list[str]
    table_feedback_history: list[str]
    image_feedback_history: list[str]
    status: str
    error: str | None
    output_dir: str
    chunk_size: int
    chunk_overlap: int
    max_retries: int
    current_worker: str | None
    worker_order: list[str]
    summary: dict[str, Any] | None


STORE_DIR = "faiss_index"
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_MAX_RETRIES = 3


def coordinator_node(state: PipelineState) -> PipelineState:
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
        "text_feedback_history": state.get("text_feedback_history", []),
        "table_feedback_history": state.get("table_feedback_history", []),
        "image_feedback_history": state.get("image_feedback_history", []),
        "output_dir": state.get("output_dir", STORE_DIR),
        "chunk_size": state.get("chunk_size", DEFAULT_CHUNK_SIZE),
        "chunk_overlap": state.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
        "max_retries": state.get("max_retries", DEFAULT_MAX_RETRIES),
        "worker_order": state.get("worker_order", ["text", "table", "image"]),
        "current_worker": None,
        "summary": None,
    }


def route_after_coordinator(state: PipelineState) -> Literal["router", "finalize_failed"]:
    if state.get("status") == "failed":
        return "finalize_failed"
    return "router"


def router_node(state: PipelineState) -> PipelineState:
    if state.get("status") == "failed":
        return {**state, "current_worker": None}

    worker_order = state.get("worker_order", ["text", "table", "image"])
    for worker_type in worker_order:
        verdict = state.get(f"{worker_type}_verdict")
        if verdict is None:
            return {**state, "current_worker": worker_type}
    return {**state, "current_worker": None}


def route_from_router(
    state: PipelineState,
) -> Literal["text_worker", "table_worker", "image_worker", "store", "finalize_failed"]:
    if state.get("status") == "failed":
        return "finalize_failed"

    current_worker = state.get("current_worker")
    if current_worker == "text":
        return "text_worker"
    if current_worker == "table":
        return "table_worker"
    if current_worker == "image":
        return "image_worker"
    return "store"


def text_worker_node(state: PipelineState) -> PipelineState:
    previous_verdict = state.get("text_verdict") or {}
    feedback = previous_verdict.get("feedback")
    retries = state.get("text_retries", 0)
    history: list[str] = list(state.get("text_feedback_history") or [])
    if previous_verdict and not previous_verdict.get("passed") and feedback:
        retries += 1
        history.append(feedback)
    # 将完整历史拼接传给解析器，让它知道哪些方向已经试过
    combined_feedback = "\n".join(f"[第{i+1}次反馈] {fb}" for i, fb in enumerate(history)) if history else None
    result = PyMuPDFParser(state["pdf_path"]).parse(feedback=combined_feedback)
    return {
        **state,
        "text_retries": retries,
        "text_feedback_history": history,
        "text_result": {"content": result.content, "metadata": result.metadata},
    }


def judge_text_node(state: PipelineState) -> PipelineState:
    judge = LLMJudge()
    verdict = judge.judge("text", state.get("text_result", {}).get("content", ""), state["pdf_path"])
    return {
        **state,
        "text_verdict": {
            "passed": verdict.passed,
            "score": verdict.score,
            "feedback": verdict.feedback,
        },
    }


def table_worker_node(state: PipelineState) -> PipelineState:
    previous_verdict = state.get("table_verdict") or {}
    feedback = previous_verdict.get("feedback")
    retries = state.get("table_retries", 0)
    history: list[str] = list(state.get("table_feedback_history") or [])
    if previous_verdict and not previous_verdict.get("passed") and feedback:
        retries += 1
        history.append(feedback)
    combined_feedback = "\n".join(f"[第{i+1}次反馈] {fb}" for i, fb in enumerate(history)) if history else None
    result = TableParser(state["pdf_path"]).parse(feedback=combined_feedback)
    return {
        **state,
        "table_retries": retries,
        "table_feedback_history": history,
        "table_result": {"content": result.content, "metadata": result.metadata},
    }


def judge_table_node(state: PipelineState) -> PipelineState:
    content = state.get("table_result", {}).get("content", "")
    metadata = state.get("table_result", {}).get("metadata", {})
    if metadata.get("table_count", 0) == 0:
        return {
            **state,
            "table_verdict": {
                "passed": True,
                "score": 1.0,
                "feedback": "no tables detected",
            },
        }
    judge = LLMJudge()
    verdict = judge.judge("table", content, state["pdf_path"])
    return {
        **state,
        "table_verdict": {
            "passed": verdict.passed,
            "score": verdict.score,
            "feedback": verdict.feedback,
        },
    }


def image_worker_node(state: PipelineState) -> PipelineState:
    previous_verdict = state.get("image_verdict") or {}
    feedback = previous_verdict.get("feedback")
    retries = state.get("image_retries", 0)
    history: list[str] = list(state.get("image_feedback_history") or [])
    if previous_verdict and not previous_verdict.get("passed") and feedback:
        retries += 1
        history.append(feedback)
    combined_feedback = "\n".join(f"[第{i+1}次反馈] {fb}" for i, fb in enumerate(history)) if history else None
    result = VisionParser(state["pdf_path"]).parse(feedback=combined_feedback)
    return {
        **state,
        "image_retries": retries,
        "image_feedback_history": history,
        "image_result": {"content": result.content, "metadata": result.metadata},
    }


def judge_image_node(state: PipelineState) -> PipelineState:
    content = state.get("image_result", {}).get("content", "")
    metadata = state.get("image_result", {}).get("metadata", {})
    if metadata.get("image_count", 0) == 0:
        return {
            **state,
            "image_verdict": {
                "passed": True,
                "score": 1.0,
                "feedback": "no images detected",
            },
        }
    judge = LLMJudge()
    verdict = judge.judge("image", content, state["pdf_path"])
    return {
        **state,
        "image_verdict": {
            "passed": verdict.passed,
            "score": verdict.score,
            "feedback": verdict.feedback,
        },
    }


def store_node(state: PipelineState) -> PipelineState:
    store = FAISSStore()
    if Path(state["output_dir"]).exists():
        try:
            store.load(state["output_dir"])
        except Exception:
            pass

    for worker_type, result_key, verdict_key in (
        ("text", "text_result", "text_verdict"),
        ("table", "table_result", "table_verdict"),
        ("image", "image_result", "image_verdict"),
    ):
        result = state.get(result_key)
        verdict = state.get(verdict_key)
        if result and verdict and verdict.get("passed"):
            store.add_chunks(
                paper_id=state["paper_id"],
                worker_type=worker_type,
                content=result["content"],
                metadata=result["metadata"],
                chunk_size=state.get("chunk_size", DEFAULT_CHUNK_SIZE),
                chunk_overlap=state.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
            )

    store.save(state["output_dir"])
    return {**state, "status": "done"}


def summary_node(state: PipelineState) -> PipelineState:
    judge = LLMJudge()
    report: SummaryReport = judge.summarize(
        pdf_path=state["pdf_path"],
        text_verdict=state.get("text_verdict"),
        table_verdict=state.get("table_verdict"),
        image_verdict=state.get("image_verdict"),
        text_retries=state.get("text_retries", 0),
        table_retries=state.get("table_retries", 0),
        image_retries=state.get("image_retries", 0),
        status=state.get("status", "unknown"),
    )
    return {
        **state,
        "summary": {
            "status": report.status,
            "modules": report.modules,
            "stored_types": report.stored_types,
            "notes": report.notes,
        },
    }


def finalize_failed_node(state: PipelineState) -> PipelineState:
    return {**state, "status": "failed"}


def should_retry_text(state: PipelineState) -> Literal["text_worker", "router", "finalize_failed"]:
    verdict = state.get("text_verdict") or {}
    max_retries = state.get("max_retries", DEFAULT_MAX_RETRIES)
    if verdict.get("passed"):
        return "router"
    if state.get("text_retries", 0) < max_retries:
        return "text_worker"
    return "finalize_failed"


def should_retry_table(state: PipelineState) -> Literal["table_worker", "router", "finalize_failed"]:
    verdict = state.get("table_verdict") or {}
    max_retries = state.get("max_retries", DEFAULT_MAX_RETRIES)
    if verdict.get("passed"):
        return "router"
    if state.get("table_retries", 0) < max_retries:
        return "table_worker"
    return "finalize_failed"


def should_retry_image(state: PipelineState) -> Literal["image_worker", "router", "finalize_failed"]:
    verdict = state.get("image_verdict") or {}
    max_retries = state.get("max_retries", DEFAULT_MAX_RETRIES)
    if verdict.get("passed"):
        return "router"
    if state.get("image_retries", 0) < max_retries:
        return "image_worker"
    return "finalize_failed"


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("coordinator", coordinator_node)
    graph.add_node("router", router_node)
    graph.add_node("text_worker", text_worker_node)
    graph.add_node("judge_text", judge_text_node)
    graph.add_node("table_worker", table_worker_node)
    graph.add_node("judge_table", judge_table_node)
    graph.add_node("image_worker", image_worker_node)
    graph.add_node("judge_image", judge_image_node)
    graph.add_node("store", store_node)
    graph.add_node("summary", summary_node)
    graph.add_node("finalize_failed", finalize_failed_node)

    graph.add_edge(START, "coordinator")
    graph.add_conditional_edges("coordinator", route_after_coordinator)
    graph.add_conditional_edges("router", route_from_router)
    graph.add_edge("text_worker", "judge_text")
    graph.add_conditional_edges("judge_text", should_retry_text)
    graph.add_edge("table_worker", "judge_table")
    graph.add_conditional_edges("judge_table", should_retry_table)
    graph.add_edge("image_worker", "judge_image")
    graph.add_conditional_edges("judge_image", should_retry_image)
    graph.add_edge("store", "summary")
    graph.add_edge("summary", END)
    graph.add_edge("finalize_failed", END)

    return graph.compile()
