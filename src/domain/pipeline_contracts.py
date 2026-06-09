from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict


ContentType = Literal["text", "table", "image"]


class ChunkDict(TypedDict, total=False):
    chunk_id: str
    content_type: ContentType
    raw_content: str
    page: int | None
    bbox: list[float] | tuple[float, float, float, float] | None
    column: int | None
    order_in_page: int | None
    global_order: int | None
    metadata: dict[str, Any]


class VerdictDict(TypedDict, total=False):
    chunk_id: str
    passed: bool
    score: float
    feedback: str
    issue_type: str
    judge_scope: str
    parser: str


class ParsedPaperResult(TypedDict, total=False):
    paper_id: str
    filename: str
    text_chunks: list[ChunkDict]
    table_chunks: list[ChunkDict]
    image_chunks: list[ChunkDict]
    merged_chunks: list[ChunkDict]
    verdicts: dict[str, VerdictDict]
    status: str
    quality: dict[str, Any]
    mode: str
    worker_timing: dict[str, float]
    judge_timing: dict[str, float]
    merge_timing: float
    desc_timing: float
    mineru: dict[str, Any]
    store_result: dict[str, Any]


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


class BatchPaperResult(TypedDict, total=False):
    status: str
    reason: str
    category: str
    credibility: float
    chunks: int
    stored: int
    degraded: int
    store_failed: int
    excluded: int
    store_attempted: int
    parse_seconds: float
    eval_seconds: float
    store_seconds: float
    total_seconds: float
