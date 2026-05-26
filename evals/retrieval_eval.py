from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
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

from src.store.qdrant_store import QdrantStore
from src.retrieval.multi_query import multi_query_search
from src.retrieval.query_rewrite import QueryRewrite, rewrite_query


DEFAULT_KS = (1, 3, 5, 10, 20)

STOPWORDS = {
    "about",
    "after",
    "aaaaa",
    "also",
    "and",
    "are",
    "article",
    "based",
    "been",
    "biosynthesis_review",
    "between",
    "both",
    "can",
    "cell",
    "cells",
    "data",
    "different",
    "doi",
    "during",
    "each",
    "effect",
    "fermentation_experiment",
    "figure",
    "from",
    "has",
    "have",
    "high",
    "into",
    "its",
    "may",
    "method",
    "mol",
    "more",
    "not",
    "paper",
    "pdf",
    "pubmed",
    "crossref",
    "production",
    "result",
    "results",
    "show",
    "shown",
    "study",
    "table",
    "text",
    "that",
    "the",
    "their",
    "these",
    "this",
    "through",
    "chunk",
    "using",
    "was",
    "were",
    "when",
    "which",
    "with",
}


@dataclass(frozen=True)
class RetrievalCase:
    id: str
    case_type: str
    query: str
    expected_paper_ids: list[str]
    expected_chunk_ids: list[str]
    expected_category: str | None = None
    source_title: str = ""
    expected_locations: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class CaseResult:
    case: RetrievalCase
    paper_rank: int | None
    chunk_rank: int | None
    nearby_rank: int | None
    top_papers: list[str]
    top_chunks: list[str]
    top_titles: list[str]
    top_scores: list[float]
    elapsed_ms: float = 0.0
    query_used: str = ""
    query_rewrite: dict[str, Any] | None = None


def normalize_list(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        parts = re.split(r"[,;|/]\s*", value)
        return [part.strip() for part in parts if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def clean_title(title: str) -> str:
    title = Path(title).stem
    title = re.sub(r"[_-]+", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def content_text(payload: dict[str, Any]) -> str:
    return str(
        payload.get("content")
        or payload.get("raw_content")
        or payload.get("paper_summary")
        or ""
    )


def extract_keywords(text: str, *, limit: int = 10) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9'_-]{2,}", text)
    counts: Counter[str] = Counter()
    for token in tokens:
        lowered = token.strip("_-").lower()
        if len(lowered) < 3 or lowered in STOPWORDS:
            continue
        if lowered.isdigit():
            continue
        counts[lowered] += 1
    return [word for word, _count in counts.most_common(limit)]


def scroll_payloads(store: QdrantStore, *, max_points: int | None = None) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    offset = None
    while True:
        batch, offset = store.client.scroll(
            collection_name=store.collection,
            limit=500,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for point in batch:
            payload = point.payload or {}
            if payload.get("paper_id") and payload.get("chunk_id"):
                payloads.append(payload)
                if max_points and len(payloads) >= max_points:
                    return payloads
        if offset is None:
            break
    return payloads


def build_cases(
    payloads: list[dict[str, Any]],
    *,
    max_cases: int,
    seed: int,
    min_content_chars: int = 180,
    content_types: set[str] | None = None,
) -> list[RetrievalCase]:
    rng = random.Random(seed)
    by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        by_paper[str(payload.get("paper_id"))].append(payload)

    cases: list[RetrievalCase] = []
    papers = list(by_paper.items())
    rng.shuffle(papers)

    for paper_id, chunks in papers:
        first = chunks[0]
        title = str(first.get("paper_title") or paper_id)
        title_query = clean_title(title)
        if len(title_query) >= 12:
            cases.append(RetrievalCase(
                id=f"title:{paper_id}",
                case_type="title",
                query=title_query,
                expected_paper_ids=[paper_id],
                expected_chunk_ids=[],
                expected_category=first.get("category"),
                source_title=title,
            ))

        metadata_case = metadata_case_for_paper(paper_id, chunks, title)
        if metadata_case:
            cases.append(metadata_case)

        chunk_cases = chunk_cases_for_paper(
            paper_id,
            chunks,
            title=title,
            min_content_chars=min_content_chars,
            content_types=content_types,
            rng=rng,
        )
        cases.extend(chunk_cases[:2])

        if len(cases) >= max_cases * 2:
            break

    rng.shuffle(cases)
    return cases[:max_cases]


def metadata_case_for_paper(
    paper_id: str,
    chunks: list[dict[str, Any]],
    title: str,
) -> RetrievalCase | None:
    products: list[str] = []
    organisms: list[str] = []
    categories: Counter[str] = Counter()
    summary = ""
    for payload in chunks:
        products.extend(normalize_list(payload.get("target_products")))
        organisms.extend(normalize_list(payload.get("organisms")))
        if payload.get("category"):
            categories[str(payload["category"])] += 1
        if not summary and payload.get("paper_summary"):
            summary = str(payload["paper_summary"])

    terms = []
    terms.extend(unique_keep_order(products)[:3])
    terms.extend(unique_keep_order(organisms)[:3])
    terms.extend(extract_keywords(summary, limit=6))
    if len(terms) < 4:
        return None

    query = " ".join(terms[:10])
    expected_category = categories.most_common(1)[0][0] if categories else None
    return RetrievalCase(
        id=f"metadata:{paper_id}",
        case_type="metadata",
        query=query,
        expected_paper_ids=[paper_id],
        expected_chunk_ids=[],
        expected_category=expected_category,
        source_title=title,
    )


def chunk_cases_for_paper(
    paper_id: str,
    chunks: list[dict[str, Any]],
    *,
    title: str,
    min_content_chars: int,
    content_types: set[str] | None,
    rng: random.Random,
) -> list[RetrievalCase]:
    candidates = []
    for payload in chunks:
        if content_types and str(payload.get("content_type") or "") not in content_types:
            continue
        text = content_text(payload)
        if len(text) < min_content_chars:
            continue
        keywords = extract_keywords(text, limit=10)
        if len(keywords) < 5:
            continue
        candidates.append((payload, keywords))
    rng.shuffle(candidates)

    cases = []
    for payload, keywords in candidates[:2]:
        chunk_id = str(payload.get("chunk_id"))
        query_terms = keywords[:8]
        products = normalize_list(payload.get("target_products"))
        organisms = normalize_list(payload.get("organisms"))
        if products:
            query_terms.insert(0, products[0])
        if organisms:
            query_terms.insert(1, organisms[0])
        cases.append(RetrievalCase(
            id=f"chunk:{paper_id}:{chunk_id}",
            case_type="chunk",
            query=" ".join(unique_keep_order(query_terms)[:10]),
            expected_paper_ids=[paper_id],
            expected_chunk_ids=[chunk_id],
            expected_category=payload.get("category"),
            source_title=title,
            expected_locations=[location_from_payload(payload)],
        ))
    return cases


def location_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": str(payload.get("paper_id") or ""),
        "chunk_id": str(payload.get("chunk_id") or ""),
        "page": payload.get("page"),
        "global_order": payload.get("global_order"),
        "order_in_page": payload.get("order_in_page"),
        "column": payload.get("column"),
        "bbox": payload.get("bbox"),
        "page_width": payload.get("page_width"),
        "page_height": payload.get("page_height"),
    }


def unique_keep_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        normalized = str(value).strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def first_rank(values: list[str], expected: set[str]) -> int | None:
    for index, value in enumerate(values, start=1):
        if value in expected:
            return index
    return None


def first_nearby_rank(
    payloads: list[dict[str, Any]],
    expected_locations: list[dict[str, Any]],
) -> int | None:
    if not expected_locations:
        return None
    for index, payload in enumerate(payloads, start=1):
        if any(is_nearby_payload(payload, expected) for expected in expected_locations):
            return index
    return None


def is_nearby_payload(
    payload: dict[str, Any],
    expected: dict[str, Any],
    *,
    global_order_window: int = 2,
    order_in_page_window: int = 2,
    bbox_center_ratio: float = 0.12,
) -> bool:
    if str(payload.get("paper_id") or "") != str(expected.get("paper_id") or ""):
        return False
    if str(payload.get("chunk_id") or "") == str(expected.get("chunk_id") or ""):
        return True

    expected_global = as_int(expected.get("global_order"))
    actual_global = as_int(payload.get("global_order"))
    if expected_global is not None and actual_global is not None:
        return abs(actual_global - expected_global) <= global_order_window

    if expected.get("page") is None or payload.get("page") is None:
        return False
    if str(payload.get("page")) != str(expected.get("page")):
        return False

    expected_order = as_int(expected.get("order_in_page"))
    actual_order = as_int(payload.get("order_in_page"))
    if expected_order is not None and actual_order is not None:
        return abs(actual_order - expected_order) <= order_in_page_window

    expected_column = expected.get("column")
    actual_column = payload.get("column")
    if expected_column not in (None, "") and actual_column not in (None, ""):
        if str(expected_column) != str(actual_column):
            return False

    return bbox_centers_are_close(payload, expected, center_ratio=bbox_center_ratio)


def as_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def bbox_centers_are_close(
    payload: dict[str, Any],
    expected: dict[str, Any],
    *,
    center_ratio: float,
) -> bool:
    actual_bbox = normalize_bbox(payload.get("bbox"))
    expected_bbox = normalize_bbox(expected.get("bbox"))
    if not actual_bbox or not expected_bbox:
        return False

    width = as_float(payload.get("page_width")) or as_float(expected.get("page_width"))
    height = as_float(payload.get("page_height")) or as_float(expected.get("page_height"))
    if not width or not height:
        return False

    ax, ay = bbox_center(actual_bbox)
    ex, ey = bbox_center(expected_bbox)
    distance = math.dist((ax / width, ay / height), (ex / width, ey / height))
    return distance <= center_ratio


def as_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        parsed = float(value)
        return parsed if parsed > 0 else None
    except (TypeError, ValueError):
        return None


def normalize_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return tuple(float(item) for item in value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def evaluate_case(
    store: QdrantStore,
    case: RetrievalCase,
    *,
    top_k: int,
    rerank: bool,
    lexical_backfill: bool = False,
    neighbor_backfill: bool = False,
    strategy: str = "standard",
    paper_limit: int = 8,
    per_paper_limit: int = 4,
    lead_paper_limit: int = 12,
    rewritten_query: QueryRewrite | None = None,
    multi_query: bool = False,
) -> CaseResult:
    query = rewritten_query.normalized_query if rewritten_query and rewritten_query.normalized_query else case.query
    started = time.perf_counter()
    if multi_query:
        multi_result = multi_query_search(
            store,
            case.query,
            limit=top_k,
            rerank=rerank,
            lexical_backfill=lexical_backfill,
            neighbor_backfill=neighbor_backfill,
            rewritten_query=rewritten_query,
            route_limit=4,
        )
        results = multi_result.results
        query = " | ".join(route.query for route in multi_result.routes)
    elif strategy == "two-stage":
        results = store.search_evidence(
            query,
            limit=top_k,
            rerank=rerank,
            lexical_backfill=lexical_backfill,
            neighbor_backfill=neighbor_backfill,
        )
    elif strategy == "paper-local":
        results = store.search_paper_local_evidence(
            query,
            limit=top_k,
            rerank=rerank,
            lexical_backfill=lexical_backfill,
            neighbor_backfill=neighbor_backfill,
            paper_limit=paper_limit,
            per_paper_limit=per_paper_limit,
            lead_paper_limit=lead_paper_limit,
        )
    else:
        results = store.search(
            query,
            limit=top_k,
            rerank=rerank,
            lexical_backfill=lexical_backfill,
            neighbor_backfill=neighbor_backfill,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000
    top_papers = [str((item.get("payload") or {}).get("paper_id", "")) for item in results]
    top_chunks = [str((item.get("payload") or {}).get("chunk_id", "")) for item in results]
    top_payloads = [item.get("payload") or {} for item in results]
    top_titles = [str((item.get("payload") or {}).get("paper_title", "")) for item in results]
    top_scores = [float(item.get("score") or 0.0) for item in results]

    paper_rank = first_rank(top_papers, set(case.expected_paper_ids))
    chunk_rank = (
        first_rank(top_chunks, set(case.expected_chunk_ids))
        if case.expected_chunk_ids
        else None
    )
    nearby_rank = (
        first_nearby_rank(top_payloads, case.expected_locations)
        if case.expected_chunk_ids
        else None
    )
    return CaseResult(
        case=case,
        paper_rank=paper_rank,
        chunk_rank=chunk_rank,
        nearby_rank=nearby_rank,
        top_papers=top_papers,
        top_chunks=top_chunks,
        top_titles=top_titles,
        top_scores=top_scores,
        elapsed_ms=elapsed_ms,
        query_used=query,
        query_rewrite=rewritten_query.to_dict() if rewritten_query else None,
    )


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def summarize(
    results: list[CaseResult],
    *,
    ks: tuple[int, ...],
    include_by_type: bool = True,
) -> dict[str, Any]:
    if not results:
        return {}

    summary: dict[str, Any] = {
        "cases": len(results),
        "paper_mrr": mean(reciprocal_rank(result.paper_rank) for result in results),
    }
    elapsed_values = [result.elapsed_ms for result in results]
    summary["elapsed_ms_avg"] = mean(elapsed_values)
    summary["elapsed_ms_p50"] = percentile(elapsed_values, 0.50)
    summary["elapsed_ms_p95"] = percentile(elapsed_values, 0.95)
    chunk_results = [result for result in results if result.case.expected_chunk_ids]
    if chunk_results:
        summary["chunk_cases"] = len(chunk_results)
        summary["chunk_mrr"] = mean(
            reciprocal_rank(result.chunk_rank) for result in chunk_results
        )
        summary["nearby_chunk_mrr"] = mean(
            reciprocal_rank(result.nearby_rank) for result in chunk_results
        )

    for k in ks:
        summary[f"paper_hit@{k}"] = mean(
            1.0 if result.paper_rank and result.paper_rank <= k else 0.0
            for result in results
        )
        if chunk_results:
            summary[f"chunk_hit@{k}"] = mean(
                1.0 if result.chunk_rank and result.chunk_rank <= k else 0.0
                for result in chunk_results
            )
            summary[f"nearby_chunk_hit@{k}"] = mean(
                1.0 if result.nearby_rank and result.nearby_rank <= k else 0.0
                for result in chunk_results
            )

    if include_by_type:
        by_type: dict[str, list[CaseResult]] = defaultdict(list)
        for result in results:
            by_type[result.case.case_type].append(result)
        summary["by_type"] = {
            case_type: summarize(case_results, ks=ks, include_by_type=False)
            for case_type, case_results in sorted(by_type.items())
        }
    return summary


def round_floats(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return value
        return round(value, 4)
    if isinstance(value, dict):
        return {key: round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_floats(item) for item in value]
    return value


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
        raise SystemExit("No retrieval eval cases could be generated from Qdrant payloads.")

    top_k = max(args.ks)
    rewrites: dict[str, QueryRewrite] = {}
    if args.query_rewrite or args.multi_query:
        for case in cases:
            rewrites[case.id] = rewrite_query(
                case.query,
                model=args.query_rewrite_model,
                use_llm=True,
            )
    results = [
        evaluate_case(
            store,
            case,
            top_k=top_k,
            rerank=args.rerank,
            lexical_backfill=args.lexical_backfill,
            neighbor_backfill=args.neighbor_backfill,
            strategy=args.strategy,
            paper_limit=args.paper_limit,
            per_paper_limit=args.per_paper_limit,
            lead_paper_limit=args.lead_paper_limit,
            rewritten_query=rewrites.get(case.id),
            multi_query=args.multi_query,
        )
        for case in cases
    ]
    report = {
        "config": {
            "content_types": args.content_types,
            "max_cases": args.max_cases,
            "max_points": args.max_points,
            "min_content_chars": args.min_content_chars,
            "lexical_backfill": args.lexical_backfill,
            "neighbor_backfill": args.neighbor_backfill,
            "rerank": args.rerank,
            "seed": args.seed,
            "strategy": args.strategy,
            "paper_limit": args.paper_limit,
            "per_paper_limit": args.per_paper_limit,
            "lead_paper_limit": args.lead_paper_limit,
            "query_rewrite": args.query_rewrite,
            "query_rewrite_model": args.query_rewrite_model,
            "multi_query": args.multi_query,
            "top_k": top_k,
        },
        "summary": round_floats(summarize(results, ks=args.ks)),
        "cases": [asdict(case) for case in cases],
        "results": [asdict(result) for result in results],
    }
    return report


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate weakly-labelled retrieval cases from Qdrant and evaluate recall/ranking.",
    )
    parser.add_argument("--max-cases", type=int, default=60)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--min-content-chars", type=int, default=180)
    parser.add_argument("--content-types", nargs="+", default=["text", "table"])
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--lexical-backfill", action="store_true")
    parser.add_argument("--neighbor-backfill", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--strategy", choices=["standard", "two-stage", "paper-local"], default="standard")
    parser.add_argument("--paper-limit", type=int, default=8)
    parser.add_argument("--per-paper-limit", type=int, default=4)
    parser.add_argument("--lead-paper-limit", type=int, default=12)
    parser.add_argument("--query-rewrite", action="store_true")
    parser.add_argument("--query-rewrite-model", default=None)
    parser.add_argument("--multi-query", action="store_true")
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
        print(f"Wrote retrieval eval report: {args.out}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
