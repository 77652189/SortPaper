from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.retrieval_eval import parse_args as parse_retrieval_args
from evals.retrieval_eval import run_eval


CORE_METRICS = (
    "cases",
    "chunk_cases",
    "paper_hit@10",
    "chunk_mrr",
    "nearby_chunk_mrr",
    "chunk_hit@1",
    "chunk_hit@10",
    "chunk_hit@30",
    "chunk_hit@50",
    "chunk_hit@100",
    "nearby_chunk_hit@10",
    "nearby_chunk_hit@30",
    "nearby_chunk_hit@50",
    "nearby_chunk_hit@100",
    "elapsed_ms_p50",
    "elapsed_ms_p95",
    "selective_rerank_trigger_rate",
)


def _retrieval_argv(args: argparse.Namespace, weight: float) -> list[str]:
    argv = [
        "--max-cases",
        str(args.max_cases),
        "--min-content-chars",
        str(args.min_content_chars),
        "--seed",
        str(args.seed),
        "--strategy",
        args.strategy,
        "--case-profile",
        args.case_profile,
        "--fielded-lexical-weight",
        str(weight),
        "--ks",
        *[str(item) for item in args.ks],
    ]
    if args.max_points is not None:
        argv.extend(["--max-points", str(args.max_points)])
    if args.content_types:
        argv.extend(["--content-types", *args.content_types])
    if args.lexical_backfill:
        argv.append("--lexical-backfill")
    if args.neighbor_backfill:
        argv.append("--neighbor-backfill")
    if args.rerank:
        argv.append("--rerank")
    if args.selective_rerank:
        argv.append("--selective-rerank")
    if args.rerank_top_n is not None:
        argv.extend(["--rerank-top-n", str(args.rerank_top_n)])
    if args.strategy == "paper-local":
        argv.extend([
            "--paper-limit",
            str(args.paper_limit),
            "--per-paper-limit",
            str(args.per_paper_limit),
            "--lead-paper-limit",
            str(args.lead_paper_limit),
        ])
    return argv


def _compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: summary.get(key) for key in CORE_METRICS if key in summary}


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for weight in args.weights:
        retrieval_args = parse_retrieval_args(_retrieval_argv(args, weight))
        retrieval_args.ks = tuple(sorted(set(retrieval_args.ks)))
        report = run_eval(retrieval_args)
        if args.reports_dir:
            safe_weight = str(weight).replace(".", "p").replace("-", "neg")
            run_path = args.reports_dir / f"{args.prefix}_fielded_{safe_weight}.json"
            run_path.parent.mkdir(parents=True, exist_ok=True)
            run_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            run_path = None
        runs.append({
            "fielded_lexical_weight": weight,
            "report_path": str(run_path) if run_path else "",
            "summary": _compact_summary(report.get("summary", {})),
        })
    return {
        "config": {
            "weights": args.weights,
            "max_cases": args.max_cases,
            "max_points": args.max_points,
            "min_content_chars": args.min_content_chars,
            "content_types": args.content_types,
            "case_profile": args.case_profile,
            "ks": args.ks,
            "strategy": args.strategy,
            "lexical_backfill": args.lexical_backfill,
            "neighbor_backfill": args.neighbor_backfill,
            "rerank": args.rerank,
            "selective_rerank": args.selective_rerank,
            "rerank_top_n": args.rerank_top_n,
            "seed": args.seed,
        },
        "runs": runs,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval_eval over fielded lexical weight values.")
    parser.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.01, 0.03, 0.05, 0.1])
    parser.add_argument("--max-cases", type=int, default=60)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--min-content-chars", type=int, default=180)
    parser.add_argument("--content-types", nargs="+", default=["text", "table"])
    parser.add_argument("--case-profile", choices=["default", "hard"], default="default")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 30, 50, 100])
    parser.add_argument("--strategy", choices=["standard", "two-stage", "paper-local"], default="standard")
    parser.add_argument("--lexical-backfill", action="store_true")
    parser.add_argument("--neighbor-backfill", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--selective-rerank", action="store_true")
    parser.add_argument("--rerank-top-n", type=int, default=None)
    parser.add_argument("--paper-limit", type=int, default=8)
    parser.add_argument("--per-paper-limit", type=int, default=4)
    parser.add_argument("--lead-paper-limit", type=int, default=12)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports" / "weight_sweep")
    parser.add_argument("--prefix", default="retrieval_eval")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "retrieval_weight_sweep.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.ks = tuple(sorted(set(args.ks)))
    report = run_sweep(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote sweep report: {args.out}")
    print(json.dumps(report["runs"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
