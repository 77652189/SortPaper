from __future__ import annotations

from typing import Callable


PIPELINE_RESULT_MODES = {"pipeline", "legacy_pipeline", "auto"}


def is_pipeline_result(result: dict) -> bool:
    return result.get("mode") in PIPELINE_RESULT_MODES


def stored_state_key(paper_id: str) -> str:
    return f"stored_{paper_id}"


def has_quality(result: dict) -> bool:
    quality = result.get("quality", {}) or {}
    return bool(quality.get("category"))


def resolve_storage_state(
    result: dict,
    *,
    session_stored_state: object,
    paper_count_lookup: Callable[[str], int],
) -> dict:
    paper_id = result.get("paper_id", "")
    try:
        qdrant_count = paper_count_lookup(paper_id) if paper_id else 0
    except Exception:
        qdrant_count = 0
    return {
        "stored_key": stored_state_key(paper_id),
        "stored_state": session_stored_state,
        "is_stored": bool(session_stored_state) or qdrant_count > 0,
        "qdrant_count": qdrant_count,
    }


def apply_quality_result(result: dict, quality_result: dict) -> dict:
    result["quality"] = quality_result
    return result


def store_result_chunks(
    result: dict,
    *,
    store_runner: Callable[[dict], dict],
) -> dict:
    return store_runner(result)


def enrich_result_quality(
    result: dict,
    *,
    enrich_runner: Callable[[str], dict],
) -> dict:
    quality_result = enrich_runner(result.get("paper_id", ""))
    return {
        "quality": quality_result,
        "result": apply_quality_result(result, quality_result),
    }


def save_result_snapshot(
    result: dict,
    filename: str,
    *,
    snapshot_builder: Callable[[dict, str], dict],
    result_saver: Callable[[dict, str], object],
) -> dict:
    snapshot = snapshot_builder(result, filename)
    saved_path = result_saver(snapshot, filename)
    if saved_path:
        return {
            "saved": True,
            "saved_path": saved_path,
            "pending_save_filename": "",
            "success_message": f"✅ 已保存: {saved_path}",
        }
    return {
        "saved": False,
        "saved_path": "",
        "pending_save_filename": filename,
        "success_message": "",
    }


def overwrite_result_snapshot(
    result: dict,
    filename: str,
    *,
    snapshot_builder: Callable[[dict, str], dict],
    result_overwriter: Callable[[dict, str], object],
) -> dict:
    snapshot = snapshot_builder(result, filename)
    result_overwriter(snapshot, filename)
    return {
        "pending_save_filename": "",
        "success_message": f"✅ 已覆盖保存: {filename}",
    }
