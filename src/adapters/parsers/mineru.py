from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

from src.application.pipeline_utils import chunk_to_dict
from src.domain.pipeline_contracts import ParsedPaperResult


def extract_mineru_zip_url(result: dict[str, Any]) -> str:
    data = result.get("data")
    if not isinstance(data, dict):
        return ""
    extract_result = data.get("extract_result")
    if not isinstance(extract_result, list):
        return ""
    for item in extract_result:
        if not isinstance(item, dict):
            continue
        if str(item.get("state") or "").strip().lower() not in {"done", "success"}:
            error = str(item.get("err_msg") or "").strip()
            if error:
                raise ValueError(f"MinerU extraction failed: {error}")
        full_zip_url = str(item.get("full_zip_url") or "").strip()
        if full_zip_url:
            return full_zip_url
    return ""


def default_mineru_verdicts(chunks: list[dict]) -> dict[str, dict]:
    verdicts: dict[str, dict] = {}
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id") or "")
        if not chunk_id:
            continue
        metadata = chunk.get("metadata", {}) or {}
        verdicts[chunk_id] = {
            "chunk_id": chunk_id,
            "passed": True,
            "score": 1.0,
            "feedback": "MinerU VLM parser output accepted for storage; paper-level enrichment remains pending.",
            "issue_type": "mineru_parser_trusted",
            "judge_scope": "mineru_default",
            "parser": metadata.get("parser", "mineru_vlm"),
        }
    return verdicts


def run_preview(
    pdf_bytes: bytes,
    paper_id: str,
    filename: str = "",
    *,
    model_version: str | None = None,
    describe_figure_groups: bool = False,
    vision_model: str | None = None,
    force_refresh: bool = False,
    poll_interval_seconds: float = 10.0,
    poll_timeout_seconds: float = 600.0,
    cache_root: Path,
    figure_vision_max_workers: int,
) -> ParsedPaperResult:
    from src.parsers.mineru_adapter import MinerUZipParser, summarize_mineru_zip
    from src.parsers.mineru_api import MinerUClient

    started = time.time()
    cache_dir = cache_root / paper_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = cache_dir / (filename or f"{paper_id}.pdf")
    zip_path = cache_dir / "result.zip"
    response_path = cache_dir / "response.json"
    chunks_path = cache_dir / "chunks.json"
    figure_vision_cache_path = cache_dir / "figure_vision.json"

    pdf_path.write_bytes(pdf_bytes)
    api_seconds = 0.0
    used_cache = zip_path.exists() and not force_refresh
    if not used_cache:
        api_start = time.time()
        client = MinerUClient(model_version=model_version)
        batch = client.submit_local_files([pdf_path])
        result = client.poll_batch_result(
            batch.batch_id,
            interval_seconds=poll_interval_seconds,
            timeout_seconds=poll_timeout_seconds,
        )
        response_payload = {
            "batch_id": batch.batch_id,
            "file_count": len(batch.file_urls),
            "result": result,
        }
        response_path.write_text(json.dumps(response_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        full_zip_url = extract_mineru_zip_url(result)
        if not full_zip_url:
            raise ValueError(f"MinerU result did not include full_zip_url: {str(result)[:500]}")
        zip_response = requests.get(full_zip_url, timeout=120)
        zip_response.raise_for_status()
        zip_path.write_bytes(zip_response.content)
        api_seconds = time.time() - api_start

    adapter_start = time.time()
    chunks = MinerUZipParser(zip_path).parse()
    adapter_seconds = time.time() - adapter_start
    vision_seconds = 0.0
    if describe_figure_groups:
        from src.parsers.mineru_vision import describe_mineru_figure_groups

        vision_start = time.time()
        chunks = describe_mineru_figure_groups(
            chunks,
            zip_path,
            model=vision_model,
            cache_path=figure_vision_cache_path,
            max_workers=figure_vision_max_workers,
        )
        vision_seconds = time.time() - vision_start

    chunk_dicts = [chunk_to_dict(chunk) for chunk in chunks]
    chunks_path.write_text(json.dumps(chunk_dicts, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = summarize_mineru_zip(zip_path, chunks)
    text_chunks = [chunk for chunk in chunk_dicts if chunk.get("content_type") == "text"]
    table_chunks = [chunk for chunk in chunk_dicts if chunk.get("content_type") == "table"]
    image_chunks = [chunk for chunk in chunk_dicts if chunk.get("content_type") == "image"]
    result_mode = "mineru_full" if describe_figure_groups else "mineru_preview"

    return {
        "paper_id": paper_id,
        "filename": filename,
        "text_chunks": text_chunks,
        "table_chunks": table_chunks,
        "image_chunks": image_chunks,
        "merged_chunks": chunk_dicts,
        "verdicts": default_mineru_verdicts(chunk_dicts),
        "status": result_mode,
        "quality": {},
        "mode": result_mode,
        "worker_timing": {
            "mineru_api": round(api_seconds, 3),
            "adapter": round(adapter_seconds, 3),
            "figure_vision": round(vision_seconds, 3),
            "total": round(time.time() - started, 3),
        },
        "mineru": {
            "cache_dir": str(cache_dir),
            "pdf_path": str(pdf_path),
            "zip_path": str(zip_path),
            "response_path": str(response_path),
            "chunks_path": str(chunks_path),
            "figure_vision_cache_path": str(figure_vision_cache_path),
            "used_cache": used_cache,
            "figure_groups_described": describe_figure_groups,
            "summary": summary,
        },
    }
