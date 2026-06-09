from __future__ import annotations

from src.store.search_text import build_search_text


def test_build_search_text_cleans_noise_and_expands_aliases() -> None:
    payload = {
        "paper_title": "2021_Example_LNTII_paper(科研通-ablesci.com).pdf",
        "content": "[标题] text | chunk 1/2 E. coli improves LNTII through lgtA.",
        "target_products": ["LNTII"],
        "organisms": ["E. coli"],
        "paper_summary": "[分类] biosynthesis_review",
    }

    text = build_search_text(payload).lower()

    assert "pdf" not in text
    assert "text | chunk" not in text
    assert "biosynthesis_review" not in text
    assert "lacto-n-triose ii" in text
    assert "escherichia coli" in text
    assert "lgta" in text


def test_build_search_text_uses_table_caption_and_cells() -> None:
    payload = {
        "content_type": "table",
        "content": "| Strain | Titer |\n| --- | --- |\n| E. coli | 2-FL |",
        "table_caption": "Table 1. 2-FL production",
    }

    text = build_search_text(payload)

    assert "Table 1" in text
    assert "Strain" in text
    assert "Titer" in text
    assert "2-fucosyllactose" in text


def test_build_search_text_uses_figure_vision_and_seo_fields() -> None:
    payload = {
        "content_type": "image",
        "content": "[Vision reparse needed]",
        "figure_caption": "Figure 2. UDP-Gal pathway for LNT biosynthesis",
        "vision_group_description": "The chart shows lgtA and wbgO supporting LNT production.",
        "seo_terms": ["figure chart evidence", "2.13 g/L"],
        "seo_anchor_text": "Results | Figure 2 | figure chart evidence",
        "aliases": ["HMO"],
    }

    text = build_search_text(payload).lower()

    assert "udp-galactose" in text
    assert "lacto-n-tetraose" in text
    assert "lgta" in text
    assert "wbgo" in text
    assert "human milk oligosaccharide" in text
    assert "2.13 g/l" in text
