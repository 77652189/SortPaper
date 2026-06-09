from __future__ import annotations

from src.application import saved_results


def test_build_snapshot_counts_chunks_and_verdicts() -> None:
    snapshot = saved_results.build_snapshot(
        {
            "paper_id": "paper-1",
            "status": "ok",
            "quality": {"paper_title": "Paper", "category": "fermentation_experiment"},
            "verdicts": {
                "c1": {"passed": True},
                "c2": {"passed": False},
            },
            "merged_chunks": [
                {"chunk_id": "c1", "content_type": "text"},
                {"chunk_id": "c2", "content_type": "table"},
                {"chunk_id": "c3", "content_type": "image"},
            ],
        },
        "paper.pdf",
        saved_at="2026-01-01T00:00:00",
    )

    assert snapshot["filename"] == "paper.pdf"
    assert snapshot["paper_id"] == "paper-1"
    assert snapshot["saved_at"] == "2026-01-01T00:00:00"
    assert snapshot["stats"] == {
        "total": 3,
        "text": 1,
        "table": 1,
        "image": 1,
        "passed": 1,
        "failed": 1,
        "status": "ok",
    }


def test_save_overwrite_list_detail_and_delete_saved_result(tmp_path) -> None:
    snapshot = {
        "filename": "paper.pdf",
        "saved_at": "2026-01-01T00:00:00",
        "quality": {
            "paper_title": "Paper",
            "category": "fermentation_experiment",
            "credibility": 0.8,
            "classify_status": "ok",
            "paper_summary": "summary",
            "target_products": ["LNT"],
            "organisms": ["E. coli"],
        },
        "stats": {"total": 2, "passed": 1, "failed": 1},
    }

    saved_path = saved_results.save_result(snapshot, "nested/paper", results_dir=tmp_path)
    duplicate_path = saved_results.save_result(snapshot, "nested/paper", results_dir=tmp_path)
    overwritten_path = saved_results.overwrite_result(
        {**snapshot, "filename": "paper-v2.pdf"},
        "nested/paper",
        results_dir=tmp_path,
    )
    saved_list = saved_results.saved_list(results_dir=tmp_path)
    detail = saved_results.saved_detail("nested_paper", results_dir=tmp_path)
    deleted = saved_results.delete_saved("nested_paper", results_dir=tmp_path)

    assert saved_path == str(tmp_path / "nested_paper.json")
    assert duplicate_path is None
    assert overwritten_path == saved_path
    assert saved_list == [{
        "file_stem": "nested_paper",
        "filename": "paper-v2.pdf",
        "saved_at": "2026-01-01T00:00:00",
        "paper_title": "Paper",
        "category": "fermentation_experiment",
        "credibility": 0.8,
        "classify_status": "ok",
        "paper_summary": "summary",
        "total_chunks": 2,
        "passed": 1,
        "failed": 1,
        "target_products": ["LNT"],
        "organisms": ["E. coli"],
    }]
    assert detail["filename"] == "paper-v2.pdf"
    assert deleted is True
    assert saved_results.saved_detail("nested_paper", results_dir=tmp_path) is None


def test_saved_list_ignores_invalid_json(tmp_path) -> None:
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")

    assert saved_results.saved_list(results_dir=tmp_path) == []
