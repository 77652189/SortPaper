from __future__ import annotations

import time
from typing import Callable

from app_modes import (
    is_auto_store_mode,
    is_legacy_pipeline_mode,
    is_legacy_preview_mode,
    is_mineru_auto_store_mode,
    is_mineru_full_mode,
    is_mineru_mode,
    normalize_mode,
)


def mineru_batch_timing_fields(result: dict, fallback_elapsed: float) -> dict:
    timing = result.get("worker_timing", {}) or {}
    store_seconds = float(timing.get("store", 0) or 0)
    total_seconds = float(timing.get("total", fallback_elapsed) or fallback_elapsed)
    parse_seconds = max(total_seconds - store_seconds, 0.0)
    return {
        "mineru_api_seconds": round(float(timing.get("mineru_api", 0) or 0), 3),
        "adapter_seconds": round(float(timing.get("adapter", 0) or 0), 3),
        "figure_vision_seconds": round(float(timing.get("figure_vision", 0) or 0), 3),
        "parse_seconds": round(parse_seconds, 3),
        "store_seconds": round(store_seconds, 3),
        "total_seconds": round(total_seconds, 3),
    }


def format_duration(value: object, *, skipped: bool = False) -> str:
    if skipped:
        return "未执行"
    seconds = float(value or 0)
    if 0 < seconds < 0.1:
        return "<0.1s"
    return f"{seconds:.1f}s"


def precheck_batch_file(
    *,
    no: int,
    filename: str,
    file_bytes: bytes,
    mode: str,
    seen_paper_ids: dict[str, dict],
    paper_id_builder: Callable[[bytes], str],
    existing_reason_lookup: Callable[[str], str],
) -> dict:
    try:
        paper_id = paper_id_builder(file_bytes)
    except Exception as exc:
        return {
            "kind": "fail",
            "row": {
                "no": no,
                "file": filename,
                "paper_id": "?",
                "status": "fail",
                "reason": f"预检查失败: {str(exc)[:160]}",
            },
        }

    if paper_id in seen_paper_ids:
        first_seen = seen_paper_ids[paper_id]
        return {
            "kind": "skip_batch_duplicate",
            "row": {
                "no": no,
                "file": filename,
                "paper_id": paper_id,
                "status": "skip",
                "skip_type": "本批次重复",
                "duplicate_of": f"{first_seen['no']}. {first_seen['file']}",
                "reason": f"本批次重复（同第 {first_seen['no']} 个文件：{first_seen['file']}）",
            },
        }

    seen_paper_ids[paper_id] = {"no": no, "file": filename}
    existing_reason = existing_reason_lookup(paper_id) if is_auto_store_mode(mode) else ""
    if existing_reason:
        return {
            "kind": "skip_existing",
            "row": {
                "no": no,
                "file": filename,
                "paper_id": paper_id,
                "status": "skip",
                "skip_type": "已入库",
                "duplicate_of": "",
                "reason": existing_reason,
            },
        }

    return {
        "kind": "prepared",
        "job": {
            "no": no,
            "file": filename,
            "paper_id": paper_id,
            "bytes": file_bytes,
        },
    }


def precheck_failure_row(no: int, filename: str, exc: Exception) -> dict:
    return {
        "no": no,
        "file": filename,
        "paper_id": "?",
        "status": "fail",
        "reason": f"预检查失败: {str(exc)[:160]}",
    }


def embedding_preflight_failure_rows(prepared_jobs: list[dict], reason: str) -> list[dict]:
    return [
        {
            "no": job["no"],
            "file": job["file"],
            "paper_id": job["paper_id"],
            "status": "fail",
            "reason": reason,
        }
        for job in prepared_jobs
    ]


def completed_batch_result(job: dict, q_result: dict) -> dict:
    result_data = dict(q_result)
    elapsed = float(result_data.get("total_seconds", 0) or 0)
    category = str(result_data.get("category", ""))
    is_mineru_result = category.startswith("mineru_")
    result_status = str(result_data.get("status", "ok") or "ok")
    reason = str(result_data.get("reason", ""))
    preview_result = result_data.get("preview_result")

    if result_status in {"ok", "partial"}:
        counter = "success"
    elif result_status == "skip":
        counter = "skip_existing" if reason.startswith("已入库") else "skip"
    else:
        counter = "fail"

    preview_entry = None
    if preview_result:
        preview_entry = {
            "no": job["no"],
            "file": job["file"],
            "result": preview_result,
        }

    return {
        "counter": counter,
        "preview_result": preview_result,
        "preview_entry": preview_entry,
        "row": {
            "no": job["no"],
            "file": job["file"],
            "paper_id": job["paper_id"],
            "status": result_status,
            "skip_type": "已入库" if result_status == "skip" and reason.startswith("已入库") else "",
            "duplicate_of": "",
            "category": category or "?",
            "chunks": result_data.get("chunks", 0),
            "stored": result_data.get("stored", 0),
            "degraded": result_data.get("degraded", 0),
            "store_failed": result_data.get("store_failed", 0),
            "store_attempted": result_data.get("store_attempted", 0),
            "excluded": result_data.get("excluded", 0),
            "parse": format_duration(result_data.get("parse_seconds", 0)),
            "mineru_api": format_duration(
                result_data.get("mineru_api_seconds", 0),
                skipped=not is_mineru_result,
            ),
            "adapter": format_duration(
                result_data.get("adapter_seconds", 0),
                skipped=not is_mineru_result,
            ),
            "figure_vl": format_duration(
                result_data.get("figure_vision_seconds", 0),
                skipped=category == "mineru_preview" or not is_mineru_result,
            ),
            "eval": format_duration(result_data.get("eval_seconds", 0)),
            "store": format_duration(
                result_data.get("store_seconds", 0),
                skipped=category != "mineru_stored" and is_mineru_result,
            ),
            "time": format_duration(elapsed),
            "reason": result_data.get("reason", ""),
        },
    }


def process_batch_job(
    file_bytes: bytes,
    paper_id: str,
    filename: str,
    mode: str,
    *,
    run_preview: Callable,
    run_pipeline: Callable,
    run_mineru_preview: Callable,
    run_mineru_ingest: Callable,
    fallback_runner: Callable,
    snapshot_builder: Callable,
    result_saver: Callable,
) -> dict:
    mode = normalize_mode(mode)
    started = time.time()
    if is_mineru_mode(mode):
        result = run_mineru_preview(file_bytes, paper_id, filename)
        result["filename"] = filename
        timing_fields = mineru_batch_timing_fields(result, time.time() - started)
        return {
            "status": "ok",
            "reason": "",
            "category": "mineru_preview",
            "chunks": len(result.get("merged_chunks", [])),
            "stored": 0,
            "degraded": 0,
            "store_failed": 0,
            "store_attempted": 0,
            "excluded": 0,
            "eval_seconds": 0,
            "preview_result": result,
            **timing_fields,
        }

    if is_mineru_full_mode(mode):
        result = run_mineru_preview(
            file_bytes,
            paper_id,
            filename,
            describe_figure_groups=True,
        )
        result["filename"] = filename
        timing_fields = mineru_batch_timing_fields(result, time.time() - started)
        return {
            "status": "ok",
            "reason": "",
            "category": "mineru_full",
            "chunks": len(result.get("merged_chunks", [])),
            "stored": 0,
            "degraded": 0,
            "store_failed": 0,
            "store_attempted": 0,
            "excluded": 0,
            "eval_seconds": 0,
            "preview_result": result,
            **timing_fields,
        }

    if is_mineru_auto_store_mode(mode):
        result = run_mineru_ingest(
            file_bytes,
            paper_id,
            filename,
            describe_figure_groups=True,
        )
        result["filename"] = filename
        store_result = result.get("store_result", {}) or {}
        timing_fields = mineru_batch_timing_fields(result, time.time() - started)
        return {
            "status": "partial" if store_result.get("error") else "ok",
            "reason": store_result.get("error", ""),
            "category": "mineru_stored",
            "chunks": len(result.get("merged_chunks", [])),
            "stored": store_result.get("stored", 0),
            "degraded": store_result.get("degraded", 0),
            "store_failed": store_result.get("failed", 0),
            "store_attempted": store_result.get("attempted", 0),
            "excluded": store_result.get("excluded", 0),
            "eval_seconds": 0,
            "preview_result": result,
            **timing_fields,
        }

    if is_legacy_preview_mode(mode):
        result = run_preview(file_bytes)
        result["filename"] = filename
        chunks = (
            len(result.get("text_chunks", []))
            + len(result.get("table_chunks", []))
            + len(result.get("image_chunks", []))
        )
        elapsed = time.time() - started
        return {
            "status": "ok",
            "reason": "",
            "category": "legacy_preview",
            "chunks": chunks,
            "stored": 0,
            "degraded": 0,
            "store_failed": 0,
            "store_attempted": 0,
            "excluded": 0,
            "parse_seconds": round(elapsed, 1),
            "eval_seconds": 0,
            "store_seconds": 0,
            "total_seconds": round(elapsed, 1),
            "preview_result": result,
        }

    if is_legacy_pipeline_mode(mode):
        stage_start = time.time()
        result = run_pipeline(file_bytes, paper_id, filename)
        result["filename"] = filename
        result["mode"] = "legacy_pipeline"
        parse_seconds = time.time() - stage_start
        result_saver(snapshot_builder(result, filename), filename)
        return {
            "status": "ok",
            "reason": "",
            "category": "legacy_parsed",
            "credibility": 0,
            "chunks": len(result.get("merged_chunks", [])),
            "stored": 0,
            "degraded": 0,
            "store_failed": 0,
            "store_attempted": 0,
            "excluded": 0,
            "parse_seconds": round(parse_seconds, 1),
            "eval_seconds": 0,
            "store_seconds": 0,
            "total_seconds": round(time.time() - started, 1),
            "preview_result": result,
        }

    return fallback_runner(file_bytes, paper_id, filename)
