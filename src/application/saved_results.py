from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def safe_result_name(filename: str) -> str:
    return filename.replace("/", "_").replace("\\", "_")


def build_snapshot(result: dict, filename: str, *, saved_at: str | None = None) -> dict:
    quality = result.get("quality", {})
    verdicts = result.get("verdicts", {})
    chunks = result.get("merged_chunks", [])

    text_n = sum(1 for chunk in chunks if chunk.get("content_type") == "text")
    table_n = sum(1 for chunk in chunks if chunk.get("content_type") == "table")
    image_n = sum(1 for chunk in chunks if chunk.get("content_type") == "image")
    passed = sum(1 for verdict in verdicts.values() if verdict.get("passed"))
    failed = sum(1 for verdict in verdicts.values() if not verdict.get("passed"))

    return {
        "filename": filename,
        "paper_id": result.get("paper_id", ""),
        "saved_at": saved_at or datetime.now().isoformat(),
        "quality": quality,
        "verdicts": verdicts,
        "chunks": chunks,
        "stats": {
            "total": len(chunks),
            "text": text_n,
            "table": table_n,
            "image": image_n,
            "passed": passed,
            "failed": failed,
            "status": result.get("status", "unknown"),
        },
    }


def save_result(snapshot: dict, filename: str, *, results_dir: Path) -> str | None:
    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / f"{safe_result_name(filename)}.json"
    if target.exists():
        return None
    target.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(target)


def overwrite_result(snapshot: dict, filename: str, *, results_dir: Path) -> str:
    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / f"{safe_result_name(filename)}.json"
    target.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(target)


def saved_list(*, results_dir: Path) -> list[dict]:
    if not results_dir.exists():
        return []
    items: list[dict] = []
    for path in sorted(results_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            quality = data.get("quality", {})
            stats = data.get("stats", {})
            items.append({
                "file_stem": path.stem,
                "filename": data.get("filename", path.stem),
                "saved_at": data.get("saved_at", ""),
                "paper_title": quality.get("paper_title", "?"),
                "category": quality.get("category", "?"),
                "credibility": quality.get("credibility", 0),
                "classify_status": quality.get("classify_status", "?"),
                "paper_summary": quality.get("paper_summary", ""),
                "total_chunks": stats.get("total", 0),
                "passed": stats.get("passed", 0),
                "failed": stats.get("failed", 0),
                "target_products": quality.get("target_products", []),
                "organisms": quality.get("organisms", []),
            })
        except Exception:
            continue
    return items


def saved_detail(file_stem: str, *, results_dir: Path) -> dict | None:
    target = results_dir / f"{file_stem}.json"
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


def delete_saved(file_stem: str, *, results_dir: Path) -> bool:
    target = results_dir / f"{file_stem}.json"
    if target.exists():
        target.unlink()
        return True
    return False
