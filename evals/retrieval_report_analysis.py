from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FailureBucket:
    name: str
    count: int
    examples: list[dict[str, Any]]


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def chunk_results(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        result
        for result in report.get("results", [])
        if result.get("case", {}).get("expected_chunk_ids")
    ]


def rank_within(result: dict[str, Any], rank_key: str, k: int) -> bool:
    rank = result.get(rank_key)
    return isinstance(rank, int) and rank <= k


def short_example(result: dict[str, Any], *, top_n: int = 8) -> dict[str, Any]:
    case = result.get("case", {})
    return {
        "id": case.get("id", ""),
        "query": case.get("query", ""),
        "source_title": case.get("source_title", ""),
        "expected_paper": (case.get("expected_paper_ids") or [""])[0],
        "expected_chunk": (case.get("expected_chunk_ids") or [""])[0],
        "paper_rank": result.get("paper_rank"),
        "chunk_rank": result.get("chunk_rank"),
        "nearby_rank": result.get("nearby_rank"),
        "top_papers": list(dict.fromkeys(result.get("top_papers", [])))[:top_n],
        "top_chunks": result.get("top_chunks", [])[:top_n],
    }


def classify_failures(
    report: dict[str, Any],
    *,
    k: int,
    example_limit: int = 5,
) -> dict[str, Any]:
    rows = chunk_results(report)
    paper_miss: list[dict[str, Any]] = []
    paper_hit_exact_miss_nearby_hit: list[dict[str, Any]] = []
    paper_hit_exact_and_nearby_miss: list[dict[str, Any]] = []
    exact_deep_hit: list[dict[str, Any]] = []

    for result in rows:
        paper_hit = rank_within(result, "paper_rank", k)
        exact_hit = rank_within(result, "chunk_rank", k)
        nearby_hit = rank_within(result, "nearby_rank", k)
        example = short_example(result)
        if not paper_hit:
            paper_miss.append(example)
        elif not exact_hit and nearby_hit:
            paper_hit_exact_miss_nearby_hit.append(example)
        elif not exact_hit and not nearby_hit:
            paper_hit_exact_and_nearby_miss.append(example)
        elif exact_hit and result.get("chunk_rank", 0) > 10:
            exact_deep_hit.append(example)

    buckets = [
        FailureBucket("paper_miss", len(paper_miss), paper_miss[:example_limit]),
        FailureBucket(
            "paper_hit_exact_miss_nearby_hit",
            len(paper_hit_exact_miss_nearby_hit),
            paper_hit_exact_miss_nearby_hit[:example_limit],
        ),
        FailureBucket(
            "paper_hit_exact_and_nearby_miss",
            len(paper_hit_exact_and_nearby_miss),
            paper_hit_exact_and_nearby_miss[:example_limit],
        ),
        FailureBucket("exact_hit_only_after_top10", len(exact_deep_hit), exact_deep_hit[:example_limit]),
    ]
    return {
        "chunk_cases": len(rows),
        "k": k,
        "exact_hit": sum(1 for result in rows if rank_within(result, "chunk_rank", k)),
        "nearby_hit": sum(1 for result in rows if rank_within(result, "nearby_rank", k)),
        "paper_hit": sum(1 for result in rows if rank_within(result, "paper_rank", k)),
        "buckets": [asdict(bucket) for bucket in buckets],
    }


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    ks: list[int],
) -> dict[str, Any]:
    base_summary = baseline.get("summary", {})
    cand_summary = candidate.get("summary", {})
    metrics = ["paper_hit", "chunk_hit", "nearby_chunk_hit"]
    deltas: dict[str, Any] = {}
    for metric in metrics:
        for k in ks:
            key = f"{metric}@{k}"
            if key in base_summary and key in cand_summary:
                deltas[key] = round(float(cand_summary[key]) - float(base_summary[key]), 4)
    for key in ["paper_mrr", "chunk_mrr", "nearby_chunk_mrr", "elapsed_ms_p50", "elapsed_ms_p95"]:
        if key in base_summary and key in cand_summary:
            deltas[key] = round(float(cand_summary[key]) - float(base_summary[key]), 4)
    return {
        "baseline_config": baseline.get("config", {}),
        "candidate_config": candidate.get("config", {}),
        "deltas": deltas,
    }


def analyze_report(
    report_path: Path,
    *,
    k: int,
    example_limit: int = 5,
    baseline_path: Path | None = None,
    compare_ks: list[int] | None = None,
) -> dict[str, Any]:
    report = load_report(report_path)
    analysis = {
        "report": str(report_path),
        "config": report.get("config", {}),
        "summary": report.get("summary", {}),
        "failure_analysis": classify_failures(report, k=k, example_limit=example_limit),
    }
    if baseline_path:
        baseline = load_report(baseline_path)
        analysis["comparison"] = compare_reports(
            baseline,
            report,
            ks=compare_ks or [1, 3, 5, 10, 30, 50, 100],
        )
    return analysis


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze retrieval_eval JSON reports offline.")
    parser.add_argument("report", type=Path)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--example-limit", type=int, default=5)
    parser.add_argument("--compare-ks", type=int, nargs="+", default=[1, 3, 5, 10, 30, 50, 100])
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    analysis = analyze_report(
        args.report,
        baseline_path=args.baseline,
        k=args.k,
        example_limit=args.example_limit,
        compare_ks=args.compare_ks,
    )
    text = json.dumps(analysis, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote retrieval report analysis: {args.out}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
