from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.retrieval_eval import (  # noqa: E402
    CaseResult,
    build_cases,
    first_nearby_rank,
    first_rank,
    round_floats,
    scroll_payloads,
    summarize,
)
from src.store.qdrant_store import QdrantStore  # noqa: E402


FIELD_WEIGHTS: dict[str, float] = {
    "entities": 12.0,
    "seo": 10.0,
    "visual": 8.0,
    "title": 6.0,
    "summary": 4.0,
    "context": 3.0,
    "content": 1.0,
}


def normalize_text(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, (list, tuple, set)):
        value = " ".join(str(item) for item in value if str(item).strip())
    return QdrantStore._normalize_search_text(str(value))


def payload_field_texts(payload: dict[str, Any]) -> dict[str, str]:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    entities = [
        payload.get("target_products"),
        payload.get("organisms"),
        payload.get("genes"),
        payload.get("enzymes"),
        payload.get("metrics"),
        payload.get("aliases"),
        payload.get("seo_terms"),
        payload.get("seo_metric_terms"),
    ]
    visual = [
        metadata.get("caption"),
        metadata.get("table_caption"),
        metadata.get("figure_caption"),
        metadata.get("vision_group_description"),
        metadata.get("mineru_visual_content"),
        payload.get("vision_group_description"),
        payload.get("mineru_visual_content"),
    ]
    return {
        "title": normalize_text(payload.get("paper_title")),
        "entities": normalize_text(entities),
        "seo": normalize_text([
            payload.get("seo_anchor_text"),
            payload.get("seo_section"),
            payload.get("seo_evidence_type"),
        ]),
        "visual": normalize_text(visual),
        "summary": normalize_text(payload.get("paper_summary")),
        "context": normalize_text(payload.get("context")),
        "content": normalize_text(
            payload.get("search_text")
            or payload.get("content")
            or payload.get("raw_content")
        ),
    }


def build_document_frequency(docs: list[dict[str, str]]) -> dict[str, int]:
    df: dict[str, int] = {}
    for fields in docs:
        seen: set[str] = set()
        for text in fields.values():
            seen.update(text.split())
        for term in seen:
            df[term] = df.get(term, 0) + 1
    return df


def term_idf(term: str, *, total_docs: int, df: dict[str, int]) -> float:
    return math.log((1 + total_docs) / (1 + df.get(term, 0))) + 1.0


def field_term_score(text: str, term: str) -> float:
    if not text or not term:
        return 0.0
    occurrences = text.count(term)
    if occurrences <= 0:
        return 0.0
    return min(3.0, 1.0 + 0.25 * (occurrences - 1))


def fielded_score(
    fields: dict[str, str],
    query_terms: list[str],
    *,
    total_docs: int,
    df: dict[str, int],
) -> float:
    score = 0.0
    for field_name, text in fields.items():
        weight = FIELD_WEIGHTS.get(field_name, 1.0)
        for term in query_terms:
            score += weight * field_term_score(text, term) * term_idf(term, total_docs=total_docs, df=df)

    phrase = " ".join(query_terms[:4])
    if len(query_terms) >= 2 and phrase:
        for field_name, text in fields.items():
            if phrase in text:
                score += FIELD_WEIGHTS.get(field_name, 1.0) * 2.0
    return score


def query_terms(query: str) -> list[str]:
    terms = QdrantStore._query_relevance_terms(query)
    return [term for term in terms if len(term) >= 2]


def evaluate_case_fielded(
    case: Any,
    payloads: list[dict[str, Any]],
    docs: list[dict[str, str]],
    *,
    df: dict[str, int],
    top_k: int,
) -> CaseResult:
    terms = query_terms(case.query)
    started = time.perf_counter()
    scored: list[tuple[float, int]] = []
    for index, fields in enumerate(docs):
        score = fielded_score(fields, terms, total_docs=len(docs), df=df)
        if score > 0:
            scored.append((score, index))
    scored.sort(key=lambda item: item[0], reverse=True)
    ranked = scored[:top_k]
    elapsed_ms = (time.perf_counter() - started) * 1000

    top_payloads = [payloads[index] for _score, index in ranked]
    top_papers = [str(payload.get("paper_id") or "") for payload in top_payloads]
    top_chunks = [str(payload.get("chunk_id") or "") for payload in top_payloads]
    return CaseResult(
        case=case,
        paper_rank=first_rank(top_papers, set(case.expected_paper_ids)),
        chunk_rank=(
            first_rank(top_chunks, set(case.expected_chunk_ids))
            if case.expected_chunk_ids
            else None
        ),
        nearby_rank=(
            first_nearby_rank(top_payloads, case.expected_locations)
            if case.expected_chunk_ids
            else None
        ),
        top_papers=top_papers,
        top_chunks=top_chunks,
        top_titles=[str(payload.get("paper_title") or "") for payload in top_payloads],
        top_scores=[float(score) for score, _index in ranked],
        elapsed_ms=elapsed_ms,
        query_used=case.query,
        query_rewrite=None,
    )


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
    docs = [payload_field_texts(payload) for payload in payloads]
    df = build_document_frequency(docs)
    top_k = max(args.ks)
    results = [
        evaluate_case_fielded(case, payloads, docs, df=df, top_k=top_k)
        for case in cases
    ]
    return {
        "config": {
            "strategy": "fielded-lexical",
            "field_weights": FIELD_WEIGHTS,
            "content_types": args.content_types,
            "max_cases": args.max_cases,
            "max_points": args.max_points,
            "min_content_chars": args.min_content_chars,
            "seed": args.seed,
            "top_k": top_k,
        },
        "summary": round_floats(summarize(results, ks=args.ks)),
        "cases": [asdict(case) for case in cases],
        "results": [asdict(result) for result in results],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline fielded lexical retrieval evaluation.")
    parser.add_argument("--max-cases", type=int, default=60)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--min-content-chars", type=int, default=180)
    parser.add_argument("--content-types", nargs="+", default=["text", "table"])
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 30, 50, 100])
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.ks = tuple(sorted(set(args.ks)))
    report = run_eval(args)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote fielded lexical eval report: {args.out}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
