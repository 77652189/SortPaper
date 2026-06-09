from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

from app_pipeline import run_mineru_preview, store_parsed_chunks
from app_utils import build_paper_id
from src.store.qdrant_store import QdrantStore


STOPWORDS = {
    "abstract",
    "accepted",
    "after",
    "also",
    "and",
    "article",
    "author",
    "batch",
    "between",
    "biologia",
    "bioingenieria",
    "corresponding",
    "culture",
    "cultures",
    "departamento",
    "different",
    "effect",
    "figure",
    "from",
    "into",
    "high",
    "introduction",
    "kinetic",
    "mexico",
    "more",
    "only",
    "paper",
    "production",
    "received",
    "revised",
    "result",
    "results",
    "study",
    "table",
    "that",
    "the",
    "their",
    "these",
    "this",
    "under",
    "used",
    "using",
    "version",
    "are",
    "was",
    "were",
    "with",
    "xico",
}


DEFAULT_QUERIES = [
    {
        "id": "title",
        "query": "Engineering Escherichia coli for high-titer biosynthesis of lacto-N-tetraose",
        "must_have_any": ["Engineering Escherichia coli", "lacto-N-tetraose"],
    },
    {
        "id": "lnt_titer",
        "query": "What LNT titer did the engineered E. coli strain achieve?",
        "must_have_any": ["LNT titer", "lacto-N-tetraose", "g/L"],
    },
    {
        "id": "pathway_genes",
        "query": "Which LNT pathway genes were engineered in E. coli?",
        "must_have_any": ["lgtA", "wbgO", "galE", "galT", "galK"],
    },
    {
        "id": "table_strains",
        "query": "Table of engineered strains and LNT II LNT titers",
        "must_have_any": ["Table 2", "LNT II titer", "LNT titer"],
    },
    {
        "id": "figure_group",
        "query": "Figure 4 optimization of UDP-Gal UDP-GlcNAc module for LNT biosynthesis",
        "must_have_any": ["Figure 4", "UDP-Gal", "UDP-GlcNAc"],
    },
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate MinerU preview -> storage -> retrieval on an isolated Qdrant collection."
    )
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("--filename", default="")
    parser.add_argument("--paper-id", default="")
    parser.add_argument("--collection", default="papers_mineru_eval")
    parser.add_argument("--model-version", default="vlm")
    parser.add_argument("--describe-figure-groups", action="store_true")
    parser.add_argument("--vision-model", default=None)
    parser.add_argument("--reset-collection", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query-set", choices=["lnt", "auto"], default="lnt")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "mineru_ingest_eval_lnt.json")
    return parser.parse_args(argv)


def _reset_eval_collection(collection: str) -> None:
    if not collection.startswith("papers_mineru_eval"):
        raise SystemExit("--reset-collection is only allowed for collections starting with papers_mineru_eval")
    store = QdrantStore(collection=collection)
    collections = {item.name for item in store.client.get_collections().collections}
    if collection in collections:
        store.client.delete_collection(collection_name=collection)


def _patch_storage_collection(collection: str) -> None:
    import src.store.chunk_storage as chunk_storage
    import src.store.qdrant_store as qdrant_store

    base_store = QdrantStore

    class EvalQdrantStore(base_store):
        def __init__(self) -> None:
            super().__init__(collection=collection)

    qdrant_store.QdrantStore = EvalQdrantStore
    chunk_storage.qdrant_store.QdrantStore = EvalQdrantStore


def _chunk_summary(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(chunks),
        "by_content_type": dict(Counter(str(chunk.get("content_type") or "") for chunk in chunks)),
        "by_mineru_type": dict(Counter(str((chunk.get("metadata") or {}).get("mineru_type") or "") for chunk in chunks)),
        "pages": sorted({chunk.get("page") for chunk in chunks if chunk.get("page") is not None}),
        "with_bbox": sum(1 for chunk in chunks if chunk.get("bbox")),
        "figure_groups": len({
            str((chunk.get("metadata") or {}).get("figure_group_id") or "")
            for chunk in chunks
            if (chunk.get("metadata") or {}).get("figure_group_id")
        }),
    }


def _hit_text(payload: dict[str, Any]) -> str:
    return "\n".join(
        str(payload.get(key) or "")
        for key in ("content", "raw_content", "search_text", "figure_caption", "mineru_visual_content")
    )


def _evaluate_queries(store: QdrantStore, paper_id: str, *, top_k: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in DEFAULT_QUERIES:
        started = time.time()
        hits = store.search(
            str(case["query"]),
            limit=top_k,
            filter_kwargs={"paper_id": paper_id},
            lexical_backfill=True,
            neighbor_backfill=True,
            rerank=False,
        )
        elapsed_ms = round((time.time() - started) * 1000, 1)
        expected = [str(item).lower() for item in case["must_have_any"]]
        matched_rank = None
        matched_term = ""
        previews = []
        for rank, hit in enumerate(hits, start=1):
            payload = hit.get("payload") or {}
            text = _hit_text(payload)
            lower = text.lower()
            if matched_rank is None:
                for term in expected:
                    if term in lower:
                        matched_rank = rank
                        matched_term = term
                        break
            previews.append({
                "rank": rank,
                "score": hit.get("score"),
                "chunk_id": payload.get("chunk_id"),
                "content_type": payload.get("content_type"),
                "page": payload.get("page"),
                "parser": payload.get("parser"),
                "preview": text[:240].replace("\n", " "),
            })
        results.append({
            "id": case["id"],
            "query": case["query"],
            "must_have_any": case["must_have_any"],
            "hit_count": len(hits),
            "matched_rank": matched_rank,
            "matched_term": matched_term,
            "elapsed_ms": elapsed_ms,
            "top_hits": previews,
        })
    return results


def _auto_queries(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    text_chunks = [
        chunk for chunk in chunks
        if chunk.get("content_type") == "text" and len(str(chunk.get("raw_content") or "")) >= 80
    ]
    if text_chunks:
        title = str(text_chunks[0].get("raw_content") or "").strip()
        title_terms = _extract_keywords(title, limit=4)
        cases.append({
            "id": "auto_title",
            "query": title[:180],
            "must_have_any": title_terms or [title[:40]],
        })

    selected: list[dict[str, Any]] = []
    for chunk in text_chunks[1:]:
        text = str(chunk.get("raw_content") or "")
        keywords = _extract_keywords(text, limit=6)
        if len(keywords) >= 3:
            selected.append({
                "id": f"auto_chunk_{len(selected) + 1}",
                "query": " ".join(keywords[:5]),
                "must_have_any": keywords[:5],
            })
        if len(selected) >= 4:
            break
    cases.extend(selected)

    image_chunks = [
        chunk for chunk in chunks
        if chunk.get("content_type") == "image"
    ]
    if image_chunks:
        context = " ".join(
            " ".join(str((chunk.get("metadata") or {}).get(key) or "") for key in ("figure_caption", "mineru_visual_content"))
            for chunk in image_chunks
        )
        keywords = _extract_keywords(context, limit=6)
        if keywords:
            cases.append({
                "id": "auto_visual",
                "query": " ".join(keywords[:5]),
                "must_have_any": keywords[:5],
            })
    return cases[:6]


def _extract_keywords(text: str, *, limit: int) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9'_-]{2,}", text)
    counts: Counter[str] = Counter()
    for token in tokens:
        cleaned = token.strip("_-")
        lowered = cleaned.lower()
        if len(lowered) < 3 or lowered in STOPWORDS or lowered.isdigit():
            continue
        if lowered.endswith("'s"):
            lowered = lowered[:-2]
        counts[cleaned] += 1
    return [word for word, _count in counts.most_common(limit)]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    pdf_bytes = args.pdf_path.read_bytes()
    paper_id = args.paper_id or build_paper_id(pdf_bytes)
    filename = args.filename or args.pdf_path.name

    if args.reset_collection:
        _reset_eval_collection(args.collection)
    _patch_storage_collection(args.collection)

    preview = run_mineru_preview(
        pdf_bytes,
        paper_id,
        filename,
        model_version=args.model_version,
        describe_figure_groups=args.describe_figure_groups,
        vision_model=args.vision_model,
        force_refresh=False,
    )
    store_result = store_parsed_chunks(preview)
    store = QdrantStore(collection=args.collection)
    chunks = preview.get("merged_chunks", [])
    global DEFAULT_QUERIES
    if args.query_set == "auto":
        DEFAULT_QUERIES = _auto_queries(chunks)
    query_results = _evaluate_queries(store, paper_id, top_k=args.top_k) if DEFAULT_QUERIES else []
    report = {
        "pdf_path": str(args.pdf_path),
        "paper_id": paper_id,
        "collection": args.collection,
        "used_cache": bool((preview.get("mineru") or {}).get("used_cache")),
        "chunk_summary": _chunk_summary(chunks),
        "mineru_summary": (preview.get("mineru") or {}).get("summary", {}),
        "store_result": store_result,
        "query_set": args.query_set,
        "query_summary": {
            "cases": len(query_results),
            "matched": sum(1 for item in query_results if item["matched_rank"] is not None),
            "hit_counts": [item["hit_count"] for item in query_results],
        },
        "query_results": query_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "paper_id": paper_id,
        "collection": args.collection,
        "used_cache": report["used_cache"],
        "chunks": report["chunk_summary"],
        "store_result": store_result,
        "query_summary": report["query_summary"],
        "out": str(args.out),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
