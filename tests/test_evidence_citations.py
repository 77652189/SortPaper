from __future__ import annotations

from src.application.evidence_citations import (
    attach_source_summary,
    build_source_summaries,
    citation_label,
    format_evidence_documents,
    format_tool_result,
)


def test_evidence_documents_use_stable_source_ids() -> None:
    chunks = [
        {
            "paper_title": "Paper A",
            "page": 3,
            "chunk_id": "c1",
            "content_type": "table",
            "content": "Table reports 3.42 g/L.",
        }
    ]

    assert citation_label(chunks[0], index=0) == "[S1] Paper A (p3, table, chunk c1)"
    docs = format_evidence_documents(chunks)
    assert "[S1] Paper A" in docs
    assert "3.42 g/L" in docs


def test_attach_source_summary_deduplicates_sources() -> None:
    chunks = [
        {"paper_title": "Paper A", "page": 1, "chunk_id": "c1", "content_type": "text"},
        {"paper_title": "Paper A", "page": 1, "chunk_id": "c1", "content_type": "text"},
        {"paper_title": "Paper B", "page": 2, "chunk_id": "c2", "content_type": "table"},
    ]

    answer = attach_source_summary("answer", chunks)

    assert "### Sources" in answer
    assert answer.count("Paper A") == 1
    assert "Paper B" in answer


def test_format_tool_result_includes_source_id_and_chunk_id() -> None:
    item = {
        "score": 0.87654,
        "payload": {
            "content": "Evidence",
            "paper_title": "Paper A",
            "page": 4,
            "content_type": "text",
            "chunk_id": "c4",
        },
    }

    formatted = format_tool_result(item, index=2)

    assert formatted["source_id"] == "S3"
    assert formatted["score"] == 0.8765
    assert formatted["chunk_id"] == "c4"


def test_build_source_summaries_include_snippets_and_deduplicate() -> None:
    sources = build_source_summaries(
        [
            {
                "paper_title": "Paper A",
                "page": 3,
                "chunk_id": "c1",
                "content_type": "text",
                "content": "RBS optimization improved expression.",
            },
            {
                "paper_title": "Paper A",
                "page": 3,
                "chunk_id": "c1",
                "content_type": "text",
                "content": "duplicate",
            },
        ]
    )

    assert len(sources) == 1
    assert sources[0]["source_id"] == "S1"
    assert sources[0]["snippet"] == "RBS optimization improved expression."
