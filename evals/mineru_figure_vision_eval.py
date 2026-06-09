from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
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

from src.parsers.mineru_adapter import MinerUZipParser
from src.parsers.mineru_vision import describe_mineru_figure_groups


KEYWORD_HINTS = {
    "figure_p2": ["LNT", "lacto-N-tetraose", "lgtA", "wbgO", "pathway"],
    "figure_p4": ["LNT", "LNT II", "titer", "strain", "shake-flask"],
    "figure_p5": ["lgtA", "wbgO", "UDP-Gal", "UDP-GlcNAc", "plasmid"],
    "figure_p6": ["UDP-Gal", "UDP-GlcNAc", "galE", "galT", "galK"],
    "figure_p7": ["UDP-Gal", "ugd", "gcd", "knockout", "LNT"],
    "figure_p8": ["fed-batch", "bioreactor", "LNT", "titer", "lacto-N-tetraose"],
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MinerU figure-group VL reparsing on a cached result zip."
    )
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--call-vision", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-images-per-group", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "mineru_figure_vision_eval_lnt.json")
    return parser.parse_args(argv)


def _figure_groups(chunks: list[Any]) -> OrderedDict[str, list[Any]]:
    groups: OrderedDict[str, list[Any]] = OrderedDict()
    for chunk in chunks:
        metadata = chunk.metadata or {}
        if chunk.content_type != "image" or metadata.get("parser") != "mineru_vlm":
            continue
        group_id = str(metadata.get("figure_group_id") or "").strip()
        if group_id:
            groups.setdefault(group_id, []).append(chunk)
    return groups


def _context_text(group_chunks: list[Any]) -> str:
    parts: list[str] = []
    for chunk in group_chunks:
        metadata = chunk.metadata or {}
        parts.append(str(chunk.raw_content or ""))
        for key in ("figure_caption", "mineru_visual_content", "image_caption", "chart_caption"):
            value = metadata.get(key)
            if isinstance(value, list):
                parts.extend(str(item) for item in value)
            elif value:
                parts.append(str(value))
    return "\n".join(parts)


def _keyword_hints(group_id: str, context: str) -> list[str]:
    hints: list[str] = []
    for prefix, values in KEYWORD_HINTS.items():
        if group_id.startswith(prefix):
            hints.extend(values)
    tokens = re.findall(r"\b(?:LNT|LNT\s+II|UDP-Gal|UDP-GlcNAc|lgtA|wbgO|galE|galT|galK|ugd|gcd)\b", context)
    for token in tokens:
        if token not in hints:
            hints.append(token)
    return hints[:8]


def _keyword_hits(description: str, hints: list[str]) -> list[str]:
    lowered = description.lower()
    hits = []
    for hint in hints:
        if hint.lower() in lowered:
            hits.append(hint)
    return hits


def _group_report(group_id: str, group_chunks: list[Any]) -> dict[str, Any]:
    first = group_chunks[0]
    metadata = first.metadata or {}
    descriptions = [
        str((chunk.metadata or {}).get("vision_group_description") or "").strip()
        for chunk in group_chunks
    ]
    description = next((item for item in descriptions if item), "")
    context = _context_text(group_chunks)
    hints = _keyword_hints(group_id, context)
    image_paths = sorted({
        str((chunk.metadata or {}).get("img_path") or "")
        for chunk in group_chunks
        if (chunk.metadata or {}).get("img_path")
    })
    return {
        "group_id": group_id,
        "figure_label": metadata.get("figure_label"),
        "page": first.page,
        "member_chunks": [chunk.chunk_id for chunk in group_chunks],
        "image_count": len(image_paths),
        "image_paths": image_paths,
        "attempted": bool(metadata.get("vision_group_attempted")),
        "succeeded": bool(metadata.get("vision_group_succeeded")),
        "cache_hit": bool(metadata.get("vision_group_cache_hit")),
        "description_chars": len(description),
        "keyword_hints": hints,
        "keyword_hits": _keyword_hits(description, hints),
        "description_preview": description[:1000],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    chunks = MinerUZipParser(args.zip_path).parse()
    before_groups = _figure_groups(chunks)
    if args.call_vision:
        chunks = describe_mineru_figure_groups(
            chunks,
            args.zip_path,
            model=args.model,
            max_images_per_group=args.max_images_per_group,
            max_workers=args.max_workers,
            cache_path=args.cache_path,
        )
    groups = _figure_groups(chunks)
    group_reports = [_group_report(group_id, members) for group_id, members in groups.items()]
    succeeded = sum(1 for item in group_reports if item["succeeded"])
    report = {
        "zip_path": str(args.zip_path),
        "call_vision": args.call_vision,
        "model": args.model,
        "cache_path": str(args.cache_path) if args.cache_path else "",
        "max_workers": args.max_workers,
        "group_count": len(group_reports),
        "visual_chunk_count": sum(len(members) for members in before_groups.values()),
        "succeeded_groups": succeeded,
        "all_groups_succeeded": bool(group_reports) and succeeded == len(group_reports),
        "groups": group_reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "zip_path": str(args.zip_path),
        "call_vision": args.call_vision,
        "group_count": report["group_count"],
        "visual_chunk_count": report["visual_chunk_count"],
        "succeeded_groups": report["succeeded_groups"],
        "all_groups_succeeded": report["all_groups_succeeded"],
        "out": str(args.out),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
