from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

from src.store.qdrant_store import QdrantStore
from src.store.search_text import build_search_text


def backfill_search_text(
    *,
    apply: bool,
    batch_size: int,
    max_points: int | None,
    force: bool,
) -> dict[str, int]:
    store = QdrantStore()
    scanned = updated = unchanged = skipped_empty = 0
    offset = None

    while True:
        points, offset = store.client.scroll(
            collection_name=store.collection,
            limit=batch_size,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for point in points:
            payload = point.payload or {}
            scanned += 1
            search_text = build_search_text(payload)
            if not search_text:
                skipped_empty += 1
                continue
            if not force and str(payload.get("search_text") or "") == search_text:
                unchanged += 1
                continue
            updated += 1
            if apply:
                store.client.set_payload(
                    collection_name=store.collection,
                    payload={"search_text": search_text},
                    points=[point.id],
                )

            if max_points and scanned >= max_points:
                return {
                    "scanned": scanned,
                    "updated": updated,
                    "unchanged": unchanged,
                    "skipped_empty": skipped_empty,
                    "applied": int(apply),
                }

        if offset is None:
            break
        if max_points and scanned >= max_points:
            break

    if apply and updated:
        if hasattr(store, "_invalidate_lexical_cache"):
            store._invalidate_lexical_cache()
        else:
            store._lexical_document_cache = None
            store._lexical_index_cache = None
    return {
        "scanned": scanned,
        "updated": updated,
        "unchanged": unchanged,
        "skipped_empty": skipped_empty,
        "applied": int(apply),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill clean search_text into existing Qdrant payloads without re-embedding.",
    )
    parser.add_argument("--apply", action="store_true", help="Write search_text payloads. Omit for dry-run.")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--force", action="store_true", help="Rewrite existing search_text even if unchanged.")
    parser.add_argument("--max-points", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    result = backfill_search_text(
        apply=args.apply,
        batch_size=args.batch_size,
        max_points=args.max_points,
        force=args.force,
    )
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"{mode} search_text backfill: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
