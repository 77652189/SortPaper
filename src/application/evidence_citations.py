from __future__ import annotations

import re
from typing import Any


def source_id(index: int) -> str:
    return f"S{index + 1}"


def build_citation(payload: dict[str, Any], *, index: int) -> dict[str, Any]:
    sid = source_id(index)
    title = _clean_text(payload.get("paper_title") or payload.get("title") or "Untitled paper")
    page = payload.get("page")
    chunk_id = str(payload.get("chunk_id") or "")
    content_type = str(payload.get("content_type") or payload.get("worker_type") or "chunk")
    flags = []
    if payload.get("_paper_local_context"):
        flags.append("paper-local")
    if payload.get("_context_expansion"):
        flags.append("context")
    if payload.get("_neighbor_source_chunk_id"):
        flags.append(f"neighbor:{payload.get('_neighbor_source_chunk_id')}")
    return {
        "source_id": sid,
        "paper_title": title,
        "page": page if page not in (None, "") else "?",
        "chunk_id": chunk_id,
        "content_type": content_type,
        "flags": flags,
    }


def citation_label(payload: dict[str, Any], *, index: int) -> str:
    citation = build_citation(payload, index=index)
    chunk = f", chunk {citation['chunk_id']}" if citation["chunk_id"] else ""
    return (
        f"[{citation['source_id']}] {citation['paper_title']} "
        f"(p{citation['page']}, {citation['content_type']}{chunk})"
    )


def format_evidence_documents(
    chunks: list[dict[str, Any]],
    *,
    limit: int = 10,
    max_chars_per_chunk: int = 1800,
) -> str:
    documents: list[str] = []
    for index, chunk in enumerate(chunks[:limit]):
        label = citation_label(chunk, index=index)
        content = _clip_text(chunk.get("content") or chunk.get("raw_content") or "", max_chars=max_chars_per_chunk)
        documents.append(f"{label}\n{content}")
    return "\n---\n".join(documents)


def attach_source_summary(answer: str, chunks: list[dict[str, Any]], *, limit: int = 6) -> str:
    text = str(answer or "").strip()
    if not chunks:
        return text
    rows = []
    seen: set[tuple[str, str, str]] = set()
    for index, chunk in enumerate(chunks):
        citation = build_citation(chunk, index=index)
        key = (
            str(citation["paper_title"]),
            str(citation["page"]),
            str(citation["chunk_id"]),
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            f"- [{citation['source_id']}] {citation['paper_title']} | "
            f"p{citation['page']} | {citation['content_type']}"
        )
        if len(rows) >= limit:
            break
    if not rows:
        return text
    return text + "\n\n### Sources\n" + "\n".join(rows)


def build_source_summaries(
    chunks: list[dict[str, Any]],
    *,
    limit: int = 6,
    snippet_chars: int = 500,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for index, chunk in enumerate(chunks):
        citation = build_citation(chunk, index=index)
        key = (
            str(citation["paper_title"]),
            str(citation["page"]),
            str(citation["chunk_id"]),
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            **citation,
            "snippet": _clip_text(chunk.get("content") or chunk.get("raw_content") or "", max_chars=snippet_chars),
        })
        if len(rows) >= limit:
            break
    return rows


def format_tool_result(result: dict[str, Any], *, index: int) -> dict[str, Any]:
    payload = result.get("payload") or {}
    return {
        "source_id": source_id(index),
        "score": round(float(result.get("score") or 0.0), 4),
        "content": payload.get("content", ""),
        "paper_title": payload.get("paper_title", ""),
        "page": payload.get("page", "?"),
        "content_type": payload.get("content_type", "text"),
        "chunk_id": payload.get("chunk_id", ""),
        "matched_routes": result.get("matched_routes", []),
        "selective_rerank_reason": result.get("selective_rerank_reason"),
    }


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clip_text(value: Any, *, max_chars: int) -> str:
    text = _clean_text(value)
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rsplit(" ", 1)[0].strip()
    return clipped or text[:max_chars].strip()
