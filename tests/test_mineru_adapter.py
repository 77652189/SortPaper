from __future__ import annotations

import json
import zipfile
from pathlib import Path

from src.parsers.mineru_adapter import MinerUZipParser, html_table_to_markdown


def _write_mineru_zip(path: Path, items: list[dict]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("sample_content_list.json", json.dumps(items, ensure_ascii=False))


def test_html_table_to_markdown() -> None:
    markdown = html_table_to_markdown("<table><tr><th>A</th><th>B</th></tr><tr><td>x</td><td>y</td></tr></table>")

    assert markdown == "| A | B |\n| --- | --- |\n| x | y |"


def test_mineru_zip_parser_maps_items_to_layout_chunks(tmp_path: Path) -> None:
    zip_path = tmp_path / "mineru.zip"
    _write_mineru_zip(
        zip_path,
        [
            {"type": "header", "text": "noise", "bbox": [0, 0, 100, 10], "page_idx": 0},
            {"type": "text", "text": "Abstract text", "bbox": [10, 20, 300, 80], "page_idx": 0},
            {
                "type": "table",
                "table_caption": ["Table 1. Strains"],
                "table_body": "<table><tr><td>strain</td><td>titer</td></tr><tr><td>B1</td><td>1.2 g/L</td></tr></table>",
                "table_footnote": ["foot"],
                "img_path": "images/table.jpg",
                "bbox": [20, 100, 500, 240],
                "page_idx": 0,
            },
            {
                "type": "chart",
                "content": "| x | y |",
                "chart_caption": ["Figure 1."],
                "img_path": "images/chart.jpg",
                "bbox": [40, 260, 500, 420],
                "page_idx": 0,
            },
            {
                "type": "image",
                "content": "",
                "image_caption": [],
                "img_path": "images/icon.jpg",
                "bbox": [1, 1, 20, 20],
                "page_idx": 0,
            },
            {
                "type": "image",
                "content": "",
                "image_caption": [],
                "img_path": "images/logo.jpg",
                "bbox": [1, 1, 80, 80],
                "page_idx": 0,
            },
            {"type": "list", "list_items": ["ref one", "ref two"], "bbox": [10, 20, 300, 80], "page_idx": 1},
        ],
    )

    chunks = MinerUZipParser(zip_path, merge=False).parse()

    assert [chunk.content_type for chunk in chunks] == ["text", "table", "image", "text"]
    assert chunks[0].page == 1
    assert chunks[0].raw_content == "Abstract text"
    assert chunks[1].raw_content.startswith("Table 1. Strains")
    assert "| strain | titer |" in chunks[1].raw_content
    assert chunks[1].metadata["parser"] == "mineru_vlm"
    assert chunks[1].metadata["mineru_type"] == "table"
    assert chunks[2].metadata["img_path"] == "images/chart.jpg"
    assert chunks[2].metadata["vision_needed"] is True
    assert chunks[2].metadata["mineru_visual_content"] == "| x | y |"
    assert chunks[2].raw_content.startswith("[Vision reparse needed]")
    assert chunks[2].metadata["figure_group_id"] == "figure_p1_001"
    assert chunks[2].metadata["figure_label"] == "Figure 1"
    assert chunks[2].metadata["figure_group_role"] == "caption_owner"
    assert chunks[3].page == 2
    assert chunks[3].raw_content == "ref one\nref two"


def test_mineru_zip_parser_keeps_large_uncaptioned_image(tmp_path: Path) -> None:
    zip_path = tmp_path / "mineru.zip"
    _write_mineru_zip(
        zip_path,
        [
            {
                "type": "image",
                "content": "",
                "image_caption": [],
                "img_path": "images/figure.jpg",
                "bbox": [0, 0, 200, 200],
                "page_idx": 0,
            }
        ],
    )

    chunks = MinerUZipParser(zip_path, merge=False).parse()

    assert len(chunks) == 1
    assert chunks[0].content_type == "image"


def test_mineru_zip_parser_uses_content_list_not_v2(tmp_path: Path) -> None:
    zip_path = tmp_path / "mineru.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("sample_content_list_v2.json", json.dumps([{"bad": True}]))
        archive.writestr(
            "sample_content_list.json",
            json.dumps([{"type": "text", "text": "ok", "bbox": [1, 2, 3, 4], "page_idx": 0}]),
        )

    chunks = MinerUZipParser(zip_path, merge=False).parse()

    assert len(chunks) == 1
    assert chunks[0].raw_content == "ok"


def test_mineru_zip_parser_attaches_section_heading_to_next_text(tmp_path: Path) -> None:
    zip_path = tmp_path / "mineru.zip"
    _write_mineru_zip(
        zip_path,
        [
            {"type": "text", "text": "INTRODUCTION", "text_level": 1, "bbox": [10, 10, 120, 30], "page_idx": 0},
            {
                "type": "text",
                "text": "Human milk oligosaccharides are important.",
                "bbox": [10, 40, 300, 90],
                "page_idx": 0,
            },
            {"type": "text", "text": "METHODS", "text_level": 1, "bbox": [10, 100, 120, 120], "page_idx": 0},
            {
                "type": "table",
                "table_body": "<table><tr><td>A</td><td>B</td></tr><tr><td>x</td><td>y</td></tr></table>",
                "bbox": [10, 140, 300, 260],
                "page_idx": 0,
            },
        ],
    )

    chunks = MinerUZipParser(zip_path, merge=False).parse()

    assert len(chunks) == 3
    assert chunks[0].raw_content.startswith("INTRODUCTION\n\nHuman milk")
    assert chunks[0].metadata["attached_heading"] == "INTRODUCTION"
    assert chunks[1].raw_content == "METHODS"
    assert chunks[2].content_type == "table"


def test_mineru_zip_parser_groups_visuals_when_caption_arrives_late(tmp_path: Path) -> None:
    zip_path = tmp_path / "mineru.zip"
    _write_mineru_zip(
        zip_path,
        [
            {
                "type": "chart",
                "content": "panel A",
                "chart_caption": [],
                "img_path": "images/a.jpg",
                "bbox": [10, 10, 210, 110],
                "page_idx": 0,
            },
            {
                "type": "chart",
                "content": "panel B",
                "chart_caption": ["Figure 2. Combined caption."],
                "img_path": "images/b.jpg",
                "bbox": [10, 120, 210, 220],
                "page_idx": 0,
            },
            {
                "type": "image",
                "content": "",
                "image_caption": [],
                "img_path": "images/c.jpg",
                "bbox": [230, 20, 430, 120],
                "page_idx": 0,
            },
            {
                "type": "image",
                "content": "",
                "image_caption": [],
                "img_path": "images/far.jpg",
                "bbox": [10, 420, 210, 520],
                "page_idx": 0,
            },
            {
                "type": "chart",
                "content": "other page",
                "chart_caption": [],
                "img_path": "images/other-page.jpg",
                "bbox": [10, 10, 210, 110],
                "page_idx": 1,
            },
        ],
    )

    chunks = MinerUZipParser(zip_path, merge=False).parse()

    first, second, third, fourth, fifth = chunks
    assert first.metadata["figure_group_id"] == "figure_p1_001"
    assert second.metadata["figure_group_id"] == "figure_p1_001"
    assert third.metadata["figure_group_id"] == "figure_p1_001"
    assert first.metadata["figure_group_role"] == "part"
    assert second.metadata["figure_group_role"] == "caption_owner"
    assert third.metadata["figure_group_role"] == "part"
    assert first.metadata["figure_group_bbox"] == [10.0, 10.0, 430.0, 220.0]
    assert first.metadata["figure_group_child_ids"] == [first.chunk_id, second.chunk_id, third.chunk_id]
    assert fourth.metadata["figure_group_id"].startswith("figure_p1_ungrouped")
    assert fourth.metadata["figure_group_role"] == "single"
    assert fifth.metadata["figure_group_id"].startswith("figure_p2_ungrouped")
    assert fifth.metadata["figure_group_role"] == "single"
