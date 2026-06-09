from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Callable

from app_modes import (
    is_auto_store_mode,
    is_legacy_auto_store_mode,
    is_legacy_pipeline_mode,
    is_legacy_preview_mode,
    is_mineru_auto_store_mode,
    is_mineru_full_mode,
    is_mineru_mode,
)


def duplicate_reparse_decision(
    *,
    paper_id: str,
    mode: str,
    existing_count: int,
    force_reparse_enabled: bool,
) -> dict:
    force_key = f"force_reparse_{paper_id}"
    should_prompt = (
        existing_count > 0
        and not is_auto_store_mode(mode)
        and not force_reparse_enabled
    )
    return {
        "force_key": force_key,
        "existing_count": existing_count,
        "should_prompt": should_prompt,
        "message": (
            f"⚠️ 该论文已存在 {existing_count} 条入库记录（paper_id: `{paper_id}`）。"
            if should_prompt
            else ""
        ),
    }


def single_parse_execution_plan(
    mode: str,
    *,
    timeout_seconds: int,
    embedding_api_key_env: str,
) -> dict:
    if is_mineru_mode(mode):
        return {"kind": "direct", "spinner": "MinerU 快速预览中..."}
    if is_mineru_full_mode(mode):
        return {"kind": "direct", "spinner": "MinerU 完整解析中（含 Figure group 图片转文字）..."}
    if is_mineru_auto_store_mode(mode):
        return {"kind": "direct", "spinner": "MinerU 完整解析并入库中..."}
    if is_legacy_preview_mode(mode):
        return {"kind": "direct", "spinner": "历史快速预览中..."}

    return {
        "kind": "legacy_pipeline",
        "spinner": (
            "阶段 1/2: 历史解析 + 入库准备中（LLM Judge · VisionParser）..."
            if is_legacy_auto_store_mode(mode)
            else "历史完整流水线解析中（LLM Judge · VisionParser）..."
        ),
        "timeout_intro": (
            "历史解析链路运行中...每次 API 调用最长等待 60 秒，"
            f"总超时 **{timeout_seconds // 60} 分钟**。请勿刷新页面。"
        ),
        "timeout_error": (
            f"**解析超时**（已等待 {timeout_seconds // 60} 分钟）\n\n"
            "**可能原因：** DeepSeek / OpenAI API 响应过慢或网络不稳定。\n\n"
            "**建议：**\n"
            f"- 确认 `.env` 中 `DEEPSEEK_API_KEY` 与 `{embedding_api_key_env}` 均有效且有余额\n"
            "- 稍后重试，或改用「MinerU 快速预览」或「历史快速预览」模式。"
        ),
    }


def run_direct_single_parse(
    pdf_bytes: bytes,
    paper_id: str,
    filename: str,
    mode: str,
    *,
    run_mineru_preview: Callable,
    run_mineru_ingest: Callable,
    run_preview: Callable,
) -> dict | None:
    if is_mineru_mode(mode):
        return run_mineru_preview(pdf_bytes, paper_id, filename)
    if is_mineru_full_mode(mode):
        return run_mineru_preview(
            pdf_bytes,
            paper_id,
            filename,
            describe_figure_groups=True,
        )
    if is_mineru_auto_store_mode(mode):
        return run_mineru_ingest(
            pdf_bytes,
            paper_id,
            filename,
            describe_figure_groups=True,
        )
    if is_legacy_preview_mode(mode):
        result = run_preview(pdf_bytes)
        result["mode"] = "legacy_preview"
        return result
    return None


def mark_legacy_pipeline_result(result: dict) -> dict:
    result["mode"] = "legacy_pipeline"
    return result


def finalize_single_parse_result(result: dict, *, mode: str, elapsed: float) -> dict:
    result["_elapsed"] = elapsed
    if is_legacy_auto_store_mode(mode):
        result["mode"] = "auto"
    elif is_mineru_auto_store_mode(mode):
        result["mode"] = "mineru_auto_store"
    elif is_legacy_pipeline_mode(mode):
        result["mode"] = result.get("mode", "legacy_pipeline")
    return result


def auto_store_feedback(
    *,
    mode: str,
    paper_id: str,
    store_result: dict | None = None,
    result: dict | None = None,
) -> dict | None:
    if is_legacy_auto_store_mode(mode):
        return {
            "stored_key": f"stored_{paper_id}",
            "store_result": store_result or {},
            "level": "success",
            "message": "✅ 一键入库完成，可立即检索；质量分析状态：待分析",
        }

    if not is_mineru_auto_store_mode(mode):
        return None

    mineru_store_result = store_result or ((result or {}).get("store_result", {}) or {})
    if mineru_store_result.get("duplicate"):
        return {
            "stored_key": f"stored_{paper_id}",
            "store_result": mineru_store_result,
            "level": "warning",
            "message": f"该论文已入库，现有 {mineru_store_result.get('existing_count', '?')} 条记录。",
        }
    if mineru_store_result.get("error"):
        return {
            "stored_key": f"stored_{paper_id}",
            "store_result": mineru_store_result,
            "level": "error",
            "message": f"MinerU 入库失败：{mineru_store_result.get('error')}",
        }
    return {
        "stored_key": f"stored_{paper_id}",
        "store_result": mineru_store_result,
        "level": "success",
        "message": (
            f"✅ MinerU 一键入库完成：stored={mineru_store_result.get('stored', 0)}，"
            f"degraded={mineru_store_result.get('degraded', 0)}；质量分析状态：待分析"
        ),
    }


def store_legacy_auto_result(
    result: dict,
    filename: str,
    *,
    store_runner: Callable[[dict], dict],
    snapshot_builder: Callable[[dict, str], dict],
    result_saver: Callable[[dict, str], object],
    before_save: Callable[[], None] | None = None,
) -> dict:
    store_result = store_runner(result)
    if before_save is not None:
        before_save()
    result_saver(snapshot_builder(result, filename), filename)
    return store_result


def execute_single_parse(
    pdf_bytes: bytes,
    paper_id: str,
    filename: str,
    mode: str,
    *,
    timeout_seconds: int,
    embedding_api_key_env: str,
    run_mineru_preview: Callable,
    run_mineru_ingest: Callable,
    run_preview: Callable,
    run_pipeline: Callable,
    clock: Callable[[], float] = time.time,
) -> dict:
    """Run the selected single-paper parse path without UI framework dependencies."""
    parse_plan = single_parse_execution_plan(
        mode,
        timeout_seconds=timeout_seconds,
        embedding_api_key_env=embedding_api_key_env,
    )
    started_at = clock()

    if parse_plan["kind"] == "direct":
        result = run_direct_single_parse(
            pdf_bytes,
            paper_id,
            filename,
            mode,
            run_mineru_preview=run_mineru_preview,
            run_mineru_ingest=run_mineru_ingest,
            run_preview=run_preview,
        )
    else:
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(run_pipeline, pdf_bytes, paper_id, filename)
        try:
            result = mark_legacy_pipeline_result(
                future.result(timeout=timeout_seconds)
            )
        except FutureTimeoutError:
            future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            return {
                "status": "timeout",
                "plan": parse_plan,
                "elapsed": clock() - started_at,
                "result": None,
            }
        else:
            pool.shutdown(wait=True)

    elapsed = clock() - started_at
    return {
        "status": "ok",
        "plan": parse_plan,
        "elapsed": elapsed,
        "result": finalize_single_parse_result(result, mode=mode, elapsed=elapsed),
    }
