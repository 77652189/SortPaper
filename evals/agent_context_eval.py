from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

from evals.retrieval_eval import (
    DEFAULT_KS,
    RetrievalCase,
    build_cases,
    first_nearby_rank,
    first_rank,
    round_floats,
    scroll_payloads,
)
from src.store.qdrant_store import QdrantStore


@dataclass(frozen=True)
class AgentContextResult:
    case: RetrievalCase
    search_chunk_rank: int | None
    search_nearby_rank: int | None
    context_chunk_rank: int | None
    context_nearby_rank: int | None
    search_chunks: list[str]
    context_chunks: list[str]
    added_context_chunks: int


def evaluate_context_case(
    store: QdrantStore,
    case: RetrievalCase,
    *,
    search_top_k: int,
    synthesis_limit: int,
    lexical_backfill: bool = False,
    expand_neighbor_context: bool = True,
    neighbor_total_limit: int = 10,
) -> AgentContextResult:
    results = store.search(
        case.query,
        limit=search_top_k,
        rerank=False,
        lexical_backfill=lexical_backfill,
    )
    context_results = (
        store.expand_neighbor_context(
            results,
            per_result_limit=2,
            total_limit=neighbor_total_limit,
        )
        if expand_neighbor_context
        else []
    )
    search_payloads = [item.get("payload") or {} for item in results]
    context_payloads = [
        item.get("payload") or {}
        for item in (results + context_results)[:synthesis_limit]
    ]
    search_chunks = [str(payload.get("chunk_id") or "") for payload in search_payloads]
    context_chunks = [str(payload.get("chunk_id") or "") for payload in context_payloads]

    expected_chunks = set(case.expected_chunk_ids)
    search_chunk_rank = (
        first_rank(search_chunks, expected_chunks)
        if expected_chunks
        else None
    )
    context_chunk_rank = (
        first_rank(context_chunks, expected_chunks)
        if expected_chunks
        else None
    )
    search_nearby_rank = (
        first_nearby_rank(search_payloads, case.expected_locations)
        if expected_chunks
        else None
    )
    context_nearby_rank = (
        first_nearby_rank(context_payloads, case.expected_locations)
        if expected_chunks
        else None
    )
    return AgentContextResult(
        case=case,
        search_chunk_rank=search_chunk_rank,
        search_nearby_rank=search_nearby_rank,
        context_chunk_rank=context_chunk_rank,
        context_nearby_rank=context_nearby_rank,
        search_chunks=search_chunks,
        context_chunks=context_chunks,
        added_context_chunks=len(context_results),
    )


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def summarize_context_results(
    results: list[AgentContextResult],
    *,
    ks: tuple[int, ...],
) -> dict[str, Any]:
    chunk_results = [result for result in results if result.case.expected_chunk_ids]
    if not chunk_results:
        return {"cases": len(results), "chunk_cases": 0}

    summary: dict[str, Any] = {
        "cases": len(results),
        "chunk_cases": len(chunk_results),
        "search_chunk_mrr": mean(reciprocal_rank(result.search_chunk_rank) for result in chunk_results),
        "search_nearby_mrr": mean(reciprocal_rank(result.search_nearby_rank) for result in chunk_results),
        "context_chunk_mrr": mean(reciprocal_rank(result.context_chunk_rank) for result in chunk_results),
        "context_nearby_mrr": mean(reciprocal_rank(result.context_nearby_rank) for result in chunk_results),
        "avg_added_context_chunks": mean(result.added_context_chunks for result in results),
    }
    for k in ks:
        summary[f"search_chunk_hit@{k}"] = mean(
            1.0 if result.search_chunk_rank and result.search_chunk_rank <= k else 0.0
            for result in chunk_results
        )
        summary[f"search_nearby_hit@{k}"] = mean(
            1.0 if result.search_nearby_rank and result.search_nearby_rank <= k else 0.0
            for result in chunk_results
        )
        summary[f"context_chunk_hit@{k}"] = mean(
            1.0 if result.context_chunk_rank and result.context_chunk_rank <= k else 0.0
            for result in chunk_results
        )
        summary[f"context_nearby_hit@{k}"] = mean(
            1.0 if result.context_nearby_rank and result.context_nearby_rank <= k else 0.0
            for result in chunk_results
        )
    return summary


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    store = QdrantStore()
    payloads = scroll_payloads(store, max_points=args.max_points)
    cases = build_cases(
        payloads,
        max_cases=args.max_cases,
        seed=args.seed,
        min_content_chars=args.min_content_chars,
        content_types=set(args.content_types) if args.content_types else None,
    )
    if not cases:
        raise SystemExit("No agent context eval cases could be generated from Qdrant payloads.")

    results = [
        evaluate_context_case(
            store,
            case,
            search_top_k=args.search_top_k,
            synthesis_limit=args.synthesis_limit,
            lexical_backfill=args.lexical_backfill,
            expand_neighbor_context=args.expand_neighbor_context,
            neighbor_total_limit=args.neighbor_total_limit,
        )
        for case in cases
    ]
    return {
        "config": {
            "content_types": args.content_types,
            "expand_neighbor_context": args.expand_neighbor_context,
            "lexical_backfill": args.lexical_backfill,
            "max_cases": args.max_cases,
            "max_points": args.max_points,
            "min_content_chars": args.min_content_chars,
            "neighbor_total_limit": args.neighbor_total_limit,
            "search_top_k": args.search_top_k,
            "seed": args.seed,
            "synthesis_limit": args.synthesis_limit,
        },
        "summary": round_floats(summarize_context_results(results, ks=args.ks)),
        "cases": [asdict(case) for case in cases],
        "results": [asdict(result) for result in results],
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether Agent synthesis context contains exact or nearby evidence chunks.",
    )
    parser.add_argument("--max-cases", type=int, default=60)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--min-content-chars", type=int, default=180)
    parser.add_argument("--content-types", nargs="+", default=["text", "table"])
    parser.add_argument("--lexical-backfill", action="store_true")
    parser.add_argument("--expand-neighbor-context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--neighbor-total-limit", type=int, default=10)
    parser.add_argument("--search-top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--synthesis-limit", type=int, default=10)
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    args.ks = tuple(sorted(set(args.ks)))
    report = run_eval(args)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote agent context eval report: {args.out}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
