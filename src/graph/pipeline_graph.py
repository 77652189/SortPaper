from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from src.judge.llm_judge import LLMJudge
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
    status: str
    error: str | None
    output_dir: str


STORE_DIR = "faiss_index"


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
        "output_dir": state.get("output_dir", STORE_DIR),
    }


def route_after_coordinator(state: PipelineState) -> Literal["text_worker", "finalize_failed"]:
    if state.get("status") == "failed":
        return "finalize_failed"
    return "text_worker"


def text_worker_node(state: PipelineState) -> PipelineState:
    previous_verdict = state.get("text_verdict") or {}
    feedback = previous_verdict.get("feedback")
    retries = state.get("text_retries", 0)
    if previous_verdict and not previous_verdict.get("passed"):
        retries += 1
    result = PyMuPDFParser(state["pdf_path"]).parse(feedback=feedback)
    return {
        **state,
        "text_retries": retries,
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
    if previous_verdict and not previous_verdict.get("passed"):
        retries += 1
    result = TableParser(state["pdf_path"]).parse(feedback=feedback)
    return {
        **state,
        "table_retries": retries,
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
    if previous_verdict and not previous_verdict.get("passed"):
        retries += 1
    result = VisionParser(state["pdf_path"]).parse(feedback=feedback)
    return {
        **state,
        "image_retries": retries,
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
            store.add(state["paper_id"], worker_type, result["content"], result["metadata"])

    store.save(state["output_dir"])
    return {**state, "status": "done"}


def finalize_failed_node(state: PipelineState) -> PipelineState:
    return {**state, "status": "failed"}


def should_retry_text(state: PipelineState) -> Literal["text_worker", "table_worker", "finalize_failed"]:
    verdict = state.get("text_verdict") or {}
    if verdict.get("passed"):
        return "table_worker"
    if state.get("text_retries", 0) < 3:
        return "text_worker"
    return "finalize_failed"


def should_retry_table(state: PipelineState) -> Literal["table_worker", "image_worker", "finalize_failed"]:
    verdict = state.get("table_verdict") or {}
    if verdict.get("passed"):
        return "image_worker"
    if state.get("table_retries", 0) < 3:
        return "table_worker"
    return "finalize_failed"


def should_retry_image(state: PipelineState) -> Literal["image_worker", "store", "finalize_failed"]:
    verdict = state.get("image_verdict") or {}
    if verdict.get("passed"):
        return "store"
    if state.get("image_retries", 0) < 3:
        return "image_worker"
    return "finalize_failed"


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("coordinator", coordinator_node)
    graph.add_node("text_worker", text_worker_node)
    graph.add_node("judge_text", judge_text_node)
    graph.add_node("table_worker", table_worker_node)
    graph.add_node("judge_table", judge_table_node)
    graph.add_node("image_worker", image_worker_node)
    graph.add_node("judge_image", judge_image_node)
    graph.add_node("store", store_node)
    graph.add_node("finalize_failed", finalize_failed_node)

    graph.add_edge(START, "coordinator")
    graph.add_conditional_edges("coordinator", route_after_coordinator)
    graph.add_edge("text_worker", "judge_text")
    graph.add_conditional_edges("judge_text", should_retry_text)
    graph.add_edge("table_worker", "judge_table")
    graph.add_conditional_edges("judge_table", should_retry_table)
    graph.add_edge("image_worker", "judge_image")
    graph.add_conditional_edges("judge_image", should_retry_image)
    graph.add_edge("store", END)
    graph.add_edge("finalize_failed", END)

    return graph.compile()
