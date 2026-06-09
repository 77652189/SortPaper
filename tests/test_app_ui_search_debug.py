from __future__ import annotations

from app_ui import _strip_markdown_source_summary, result_route_debug_bits, search_meta_debug_lines


def test_search_meta_debug_lines_describe_rewrite_routes() -> None:
    lines = search_meta_debug_lines(
        {
            "routes": [
                {"source": "raw", "query": "LNT II 为什么残留过高？"},
                {
                    "source": "normalized",
                    "query": "lacto-N-triose II accumulation engineering strategy",
                },
            ],
            "rewrite": {
                "normalized_query": "lacto-N-triose II accumulation engineering strategy",
                "evidence_preference": "table",
                "products": ["LNT II"],
                "organisms": ["E. coli"],
                "genes": ["lgtA"],
                "metrics": ["accumulation"],
            },
            "context_query": "lacto-N-triose II accumulation LNT II lgtA",
            "search_hits": 5,
            "context_hits": 3,
            "elapsed_ms": 123.456,
        }
    )

    assert lines[0].startswith("检索路由: raw:")
    assert "normalized:" in lines[0]
    assert lines[1] == "标准化 query: lacto-N-triose II accumulation engineering strategy"
    assert lines[2] == "证据偏好: table"
    assert lines[3] == "改写实体: 产物: LNT II | 菌株: E. coli | 基因: lgtA | 指标: accumulation"
    assert lines[4] == "上下文定位 query: lacto-N-triose II accumulation LNT II lgtA"
    assert lines[5] == "命中: 5 条 | 上下文补充: 3 条"
    assert lines[6] == "多路召回耗时: 123.5 ms"


def test_search_meta_debug_lines_ignores_empty_meta() -> None:
    assert search_meta_debug_lines({}) == []
    assert search_meta_debug_lines(None) == []


def test_search_meta_debug_lines_describe_route_policy() -> None:
    lines = search_meta_debug_lines(
        {
            "routes": [{"source": "raw", "query": "paper title"}],
            "route_policy": {
                "reason": "compact_generic_query",
                "requested_limit": 4,
                "effective_limit": 2,
            },
        }
    )

    assert "route policy: compact_generic_query (2/4)" in lines


def test_result_route_debug_bits_clip_queries() -> None:
    bits = result_route_debug_bits(
        {
            "matched_routes": ["raw", "normalized"],
            "matched_queries": [
                "lacto-N-triose II " + "very-long-token " * 20,
                "LNT II lgtA RBS",
            ],
        }
    )

    assert bits[0] == "命中路由: raw, normalized"
    assert bits[1].startswith("命中 query: lacto-N-triose II")
    assert "..." in bits[1]
    assert "LNT II lgtA RBS" in bits[1]


def test_strip_markdown_source_summary_removes_compat_sources_block() -> None:
    answer = "正文回答\n\n### Sources\n- [S1] Paper A | p1 | text"

    assert _strip_markdown_source_summary(answer) == "正文回答"
