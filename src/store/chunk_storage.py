from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any, Literal, NotRequired, TypedDict

from qdrant_client.http import models

import src.store.qdrant_store as qdrant_store

logger = logging.getLogger(__name__)


ContentType = Literal["text", "table", "image"]


class ChunkVerdict(TypedDict, total=False):
    chunk_id: str
    passed: bool
    score: float
    feedback: str
    issue_type: str


class StoredChunkInput(TypedDict, total=False):
    chunk_id: str
    content_type: ContentType
    raw_content: str
    page: int | None
    bbox: list[float] | tuple[float, float, float, float] | None
    column: int | None
    order_in_page: int | None
    global_order: int | None
    metadata: dict[str, Any]


class StoreParsedResult(TypedDict, total=False):
    paper_id: str
    filename: str
    paper_title: str
    title: str
    verdicts: dict[str, ChunkVerdict]
    merged_chunks: list[StoredChunkInput]


class StoreStats(TypedDict):
    stored: int
    degraded: int
    failed: int
    excluded: int
    attempted: int
    missing_verdicts: int
    error: str
    duplicate: NotRequired[bool]
    existing_count: NotRequired[int]
    excluded_tables: NotRequired[int]
    excluded_reasons: NotRequired[dict[str, int]]


StoreMetadata = dict[str, Any]
EmbedContentBuilder = Callable[[StoredChunkInput], str]


def store_parsed_result_chunks(
    result: StoreParsedResult,
    *,
    embedding_api_key_env: str,
    embed_content_for_chunk: EmbedContentBuilder,
) -> StoreStats:
    """Store parsed chunks in Qdrant while preserving the app_pipeline result contract."""
    store = qdrant_store.QdrantStore()
    paper_id = result.get("paper_id", "?")
    verdicts = result.get("verdicts", {})
    merged = result.get("merged_chunks", [])
    paper_title = str(
        result.get("filename")
        or result.get("paper_title")
        or result.get("title")
        or paper_id
    )

    if not _embedding_api_key_configured(embedding_api_key_env):
        attempted = count_attemptable_chunks_without_embedding(merged, verdicts)
        return {
            "stored": 0,
            "degraded": 0,
            "failed": attempted,
            "excluded": 0,
            "duplicate": False,
            "attempted": attempted,
            "missing_verdicts": 0,
            "error": f"缺少 {embedding_api_key_env}，无法生成向量入库",
        }

    existing_count = store.client.count(
        collection_name=store.collection,
        count_filter=models.Filter(
            must=[models.FieldCondition(key="paper_id", match=models.MatchValue(value=paper_id))]
        ),
    ).count
    if existing_count > 0:
        logger.warning(
            "重复入库被阻止 | paper_id=%s | 已存在 %d 条记录 | 将跳过 %d 条新记录",
            paper_id,
            existing_count,
            len([c for c in merged if verdicts.get(c.get("chunk_id", ""), {}).get("passed")]),
        )
        return {
            "stored": 0,
            "degraded": 0,
            "failed": 0,
            "excluded": 0,
            "duplicate": True,
            "existing_count": existing_count,
            "attempted": 0,
        }

    stats = _StorageStats()

    for chunk in merged:
        cid = chunk.get("chunk_id", "")
        metadata = apply_table_storage_decision(chunk)
        if metadata.get("excluded_from_storage"):
            stats.record_excluded(chunk)
            continue

        verdict = verdicts.get(cid)
        if not verdict:
            stats.missing_verdicts += 1
            continue
        content_type = chunk.get("content_type", "text")

        if verdict.get("passed"):
            stats.attempted += 1
            meta, display = build_store_metadata(chunk, verdict, paper_title, "clean")
            if _try_store_chunk(
                store,
                paper_id=paper_id,
                content_type=content_type,
                content=display,
                embed_content=embed_content_for_chunk(chunk),
                metadata=meta,
                chunk_id=cid,
                stats=stats,
            ):
                stats.stored += 1

        elif content_type == "table" and verdict.get("issue_type") != "false_positive":
            stats.attempted += 1
            meta, display = build_store_metadata(chunk, verdict, paper_title, "degraded")
            meta["judge_feedback"] = verdict.get("feedback", "")
            if _try_store_chunk(
                store,
                paper_id=paper_id,
                content_type=content_type,
                content=display,
                embed_content=embed_content_for_chunk(chunk),
                metadata=meta,
                chunk_id=cid,
                stats=stats,
                degraded=True,
            ):
                stats.degraded += 1

        elif content_type == "table" and verdict.get("issue_type") == "false_positive":
            metadata.setdefault("excluded_from_storage", True)
            metadata.setdefault("storage_exclusion_reason", "quality judge marked table as false positive")
            chunk["metadata"] = metadata
            stats.record_excluded(chunk)

    store.save("")
    return stats.to_result()


def count_attemptable_chunks_without_embedding(
    merged: list[StoredChunkInput],
    verdicts: dict[str, ChunkVerdict],
) -> int:
    attempted = 0
    for chunk in merged:
        verdict = verdicts.get(chunk.get("chunk_id", ""))
        if not verdict:
            continue
        if verdict.get("passed") or (
            chunk.get("content_type") == "table"
            and verdict.get("issue_type") != "false_positive"
        ):
            attempted += 1
    return attempted


def apply_table_storage_decision(chunk: StoredChunkInput) -> StoreMetadata:
    metadata = chunk.get("metadata", {}) or {}
    if chunk.get("content_type") != "table":
        return metadata
    try:
        from src.judge.table_judge import build_storage_decision

        storage_decision = build_storage_decision(
            content_type="table",
            metadata=metadata,
        )
        metadata.update(storage_decision)
        chunk["metadata"] = metadata
    except Exception:
        pass
    return metadata


def storage_exclusion_reason(chunk: StoredChunkInput) -> str:
    metadata = chunk.get("metadata", {}) or {}
    return str(
        metadata.get("storage_exclusion_reason")
        or metadata.get("unparseable_reason")
        or "excluded_from_storage"
    )


def build_store_metadata(
    chunk: StoredChunkInput,
    verdict: ChunkVerdict,
    paper_title: str,
    quality_label: str = "clean",
) -> tuple[StoreMetadata, str]:
    raw_content = chunk.get("raw_content", "")
    chunk_metadata = dict(chunk.get("metadata", {}) or {})
    return {
        **chunk_metadata,
        "page": chunk.get("page"),
        "bbox": chunk.get("bbox"),
        "column": chunk.get("column"),
        "order_in_page": chunk.get("order_in_page"),
        "global_order": chunk.get("global_order"),
        "chunk_id": chunk.get("chunk_id"),
        "content_type": chunk.get("content_type", "text"),
        "table_quality": quality_label if chunk.get("content_type") == "table" else "",
        "score": verdict.get("score"),
        "paper_title": paper_title,
        "raw_content": raw_content,
        "enrichment_status": "pending",
    }, raw_content


class _StorageStats:
    def __init__(self) -> None:
        self.stored = 0
        self.degraded = 0
        self.failed = 0
        self.excluded = 0
        self.excluded_tables = 0
        self.excluded_reasons: dict[str, int] = {}
        self.first_store_error = ""
        self.attempted = 0
        self.missing_verdicts = 0

    def record_excluded(self, chunk: StoredChunkInput) -> None:
        self.excluded += 1
        if chunk.get("content_type") == "table":
            self.excluded_tables += 1
        reason = storage_exclusion_reason(chunk)
        self.excluded_reasons[reason] = self.excluded_reasons.get(reason, 0) + 1

    def record_failed(self, exc: Exception) -> None:
        if not self.first_store_error:
            self.first_store_error = f"{type(exc).__name__}: {str(exc)[:300]}"
        self.failed += 1

    def to_result(self) -> StoreStats:
        return {
            "stored": self.stored,
            "degraded": self.degraded,
            "failed": self.failed,
            "excluded": self.excluded,
            "excluded_tables": self.excluded_tables,
            "excluded_reasons": self.excluded_reasons,
            "attempted": self.attempted,
            "missing_verdicts": self.missing_verdicts,
            "error": self.first_store_error,
        }


def _embedding_api_key_configured(env_name: str) -> bool:
    return bool((os.getenv(env_name) or os.getenv(f"\ufeff{env_name}") or "").strip())


def _try_store_chunk(
    store: Any,
    *,
    paper_id: str,
    content_type: str,
    content: str,
    embed_content: str,
    metadata: StoreMetadata,
    chunk_id: str,
    stats: _StorageStats,
    degraded: bool = False,
) -> bool:
    try:
        store.add(
            paper_id=paper_id,
            worker_type=content_type,
            content=content,
            embed_content=embed_content,
            metadata=metadata,
        )
        return True
    except Exception as exc:
        label = "降级入库失败" if degraded else "入库失败"
        logger.error("%s chunk_id=%s: %s", label, chunk_id, exc)
        stats.record_failed(exc)
        return False
