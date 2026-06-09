from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parsers.mineru_adapter import MinerUZipParser


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a MinerU result zip into LayoutChunk JSON.")
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-merge", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    chunks = MinerUZipParser(args.zip_path, merge=not args.no_merge).parse()
    payload = [chunk.to_dict() for chunk in chunks]
    summary = {
        "chunks": len(chunks),
        "by_content_type": dict(Counter(chunk.content_type for chunk in chunks)),
        "by_mineru_type": dict(Counter(str(chunk.metadata.get("mineru_type") or "") for chunk in chunks)),
        "pages": sorted({chunk.page for chunk in chunks}),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["out"] = str(args.out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
