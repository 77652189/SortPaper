from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.application.search import search_chunks
from src.application.y103_retrieval_cases import y103_rag_smoke_cases


SearchRunner = Callable[..., list[dict[str, Any]]]


def load_cases(path: Path | None = None) -> list[dict[str, Any]]:
    if path is None:
        return y103_rag_smoke_cases()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("RAG smoke cases must be a JSON list")
    return [dict(item) for item in payload if isinstance(item, dict)]


def run_smoke_cases(
    cases: list[dict[str, Any]],
    *,
    top_k: int,
    default_profile: str,
    search_runner: SearchRunner = search_chunks,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for case in cases:
        query = str(case.get("query") or "").strip()
        if not query:
            continue
        profile = str(case.get("profile") or default_profile)
        case_started = time.perf_counter()
        hits = search_runner(
            query,
            top_k=top_k,
            retrieval_profile=profile,
        )
        elapsed_ms = (time.perf_counter() - case_started) * 1000
        results.append({
            "id": case.get("id") or query[:40],
            "topic": case.get("topic", ""),
            "query": query,
            "profile": profile,
            "hit_count": len(hits),
            "elapsed_ms": elapsed_ms,
            "top_hits": [_summarize_hit(hit) for hit in hits[: min(top_k, 5)]],
        })
    total_elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "case_count": len(results),
        "top_k": top_k,
        "default_profile": default_profile,
        "elapsed_ms": total_elapsed_ms,
        "library_hint": _library_hint(results),
        "results": results,
    }


def _summarize_hit(hit: dict[str, Any]) -> dict[str, Any]:
    payload = hit.get("payload") or {}
    return {
        "score": hit.get("score"),
        "paper_title": payload.get("paper_title", ""),
        "paper_id": payload.get("paper_id", ""),
        "chunk_id": payload.get("chunk_id", ""),
        "page": payload.get("page", ""),
        "content_type": payload.get("content_type") or payload.get("worker_type") or "",
        "reranked": bool(hit.get("reranked")),
        "selective_rerank_reason": hit.get("selective_rerank_reason") or "",
        "matched_routes": hit.get("matched_routes") or [],
    }


def _library_hint(results: list[dict[str, Any]]) -> dict[str, Any]:
    hits = [
        hit
        for result in results
        for hit in result.get("top_hits", [])
        if isinstance(hit, dict)
    ]
    paper_ids = sorted({str(hit.get("paper_id") or "") for hit in hits if str(hit.get("paper_id") or "")})
    paper_titles = sorted({str(hit.get("paper_title") or "") for hit in hits if str(hit.get("paper_title") or "")})
    placeholder_count = sum(1 for hit in hits if _is_placeholder_hit(hit))
    only_placeholder_hits = bool(hits) and placeholder_count == len(hits)
    note = ""
    if only_placeholder_hits:
        note = "当前 smoke 只命中占位/重置样本文档，只能证明链路可跑，不能判断真实召回质量。"
    elif not hits:
        note = "当前 smoke 没有命中结果，请先确认 Qdrant 中已有入库论文。"
    return {
        "top_hit_count": len(hits),
        "unique_paper_count_in_hits": len(paper_ids),
        "unique_paper_titles_in_hits": paper_titles[:10],
        "placeholder_hit_count": placeholder_count,
        "only_placeholder_hits": only_placeholder_hits,
        "note": note,
    }


def _is_placeholder_hit(hit: dict[str, Any]) -> bool:
    title = str(hit.get("paper_title") or "").lower()
    paper_id = str(hit.get("paper_id") or "").lower()
    return "default-reset-smoke" in title or paper_id.startswith("codex-default-reset-smoke")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small PaperSort RAG smoke set.")
    parser.add_argument("--cases", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--profile", default="evidence", choices=["quick", "evidence", "agent"])
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    report = run_smoke_cases(
        cases,
        top_k=max(1, args.top_k),
        default_profile=args.profile,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
