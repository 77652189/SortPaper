from __future__ import annotations


def run_preview(pdf_bytes: bytes) -> dict:
    from src.legacy import run_legacy_preview

    return run_legacy_preview(pdf_bytes)


def run_pipeline(pdf_bytes: bytes, paper_id: str, filename: str = "") -> dict:
    from src.legacy import run_legacy_pipeline

    return run_legacy_pipeline(pdf_bytes, paper_id, filename)
