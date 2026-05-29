from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from src.retrieval.query_rewrite import (
    DEFAULT_QUERY_REWRITE_MODEL,
    QueryRewrite,
    build_context_query,
    normalize_rewrite_payload,
    rewrite_query,
)


def test_normalize_rewrite_payload_expands_aliases_and_intents() -> None:
    rewrite = normalize_rewrite_payload(
        {
            "normalized_query": "",
            "products": ["LNT II"],
            "organisms": [],
            "genes": ["lgtA"],
            "metrics": ["accumulation"],
            "intents": [],
            "evidence_preference": "invalid",
        },
        raw_query="LNT II 为什么残留过高，改造逻辑是什么？",
    )

    assert "lacto-N-triose II" in rewrite.products
    assert "engineering_strategy" in rewrite.intents
    assert "mechanism" in rewrite.intents
    assert rewrite.evidence_preference == "any"
    assert "lgtA" in rewrite.normalized_query
    assert "lacto-N-tetraose" not in rewrite.products


def test_normalize_rewrite_payload_flattens_nested_aliases() -> None:
    rewrite = normalize_rewrite_payload(
        {
            "normalized_query": "lacto-N-triose II accumulation",
            "products": [],
            "aliases": [["LNT II", "lacto-N-triose II"]],
        },
        raw_query="LNT II accumulation",
    )

    assert "['LNT II', 'lacto-N-triose II']" not in rewrite.aliases
    assert "LNT II" in rewrite.aliases
    assert "lacto-N-triose II" in rewrite.products
    assert "lacto-N-tetraose" not in rewrite.products


def test_build_context_query_keeps_structured_rewrite_terms() -> None:
    rewrite = QueryRewrite(
        raw_query="LNT II 为什么残留过高？",
        normalized_query="lacto-N-triose II accumulation engineering strategy",
        products=["LNT II", "lacto-N-triose II"],
        organisms=["E. coli"],
        genes=["lgtA"],
        enzymes=["beta-1,3-N-acetylglucosaminyltransferase"],
        metrics=["accumulation"],
        aliases=["LNTII", "LNT II"],
        variants=[
            "LNT II lgtA RBS",
            "lacto-N-triose II biosynthesis pathway",
            "ignored overflow variant",
        ],
    )

    context_query = build_context_query(rewrite, fallback_query="raw query")

    assert context_query.startswith("lacto-N-triose II accumulation engineering strategy")
    assert "LNT II" in context_query
    assert "E. coli" in context_query
    assert "lgtA" in context_query
    assert "beta-1,3-N-acetylglucosaminyltransferase" in context_query
    assert "LNT II lgtA RBS" in context_query
    assert "ignored overflow variant" not in context_query
    assert context_query.count("lacto-N-triose II") == 2


def test_build_context_query_falls_back_without_rewrite() -> None:
    assert build_context_query(None, fallback_query="raw query") == "raw query"


def test_build_context_query_skips_terms_already_in_longer_terms() -> None:
    rewrite = QueryRewrite(
        raw_query="raw query",
        normalized_query="lacto-N-triose II accumulation engineering strategy",
        products=["lacto-N-triose II"],
        metrics=["accumulation"],
        aliases=["LNT II"],
        variants=["LNT II lgtA RBS"],
    )

    context_query = build_context_query(rewrite, fallback_query="raw query")

    assert context_query == (
        "lacto-N-triose II accumulation engineering strategy "
        "LNT II LNT II lgtA RBS"
    )


def test_build_context_query_limits_terms_and_length() -> None:
    rewrite = QueryRewrite(
        raw_query="raw query",
        normalized_query="alpha beta gamma",
        products=["product-one", "product-two", "product-three"],
        organisms=["organism-one"],
        variants=["variant one", "variant two"],
    )

    limited = build_context_query(
        rewrite,
        fallback_query="raw query",
        max_terms=3,
        max_chars=32,
    )

    assert limited == "alpha beta gamma product-one"
    assert "product-two" not in limited
    assert len(limited) <= 32


def test_build_context_query_clips_long_fallback_query() -> None:
    query = build_context_query(None, fallback_query="one two three four", max_chars=13)

    assert query == "one two"


def test_rewrite_query_uses_deepseek_v4_flash_by_default(monkeypatch) -> None:
    import src.retrieval.query_rewrite as query_rewrite

    captured: dict[str, object] = {}

    class FakeCompletions:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                '{"normalized_query":"lacto-N-triose II accumulation engineering strategy",'
                                '"products":["LNT II"],"organisms":["E. coli"],"genes":["lgtA"],'
                                '"enzymes":[],"metrics":["accumulation"],'
                                '"intents":["mechanism"],"evidence_preference":"text",'
                                '"aliases":["LNTII"],"variants":[]}'
                            )
                        )
                    )
                ]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions)

    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    rewrite = query_rewrite.rewrite_query("LNT II 为什么残留过高？")

    assert captured["model"] == DEFAULT_QUERY_REWRITE_MODEL
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["client_kwargs"]["api_key"] == "deepseek-key"
    assert rewrite.normalized_query == "lacto-N-triose II accumulation engineering strategy"
    assert "lacto-N-triose II" in rewrite.products


def test_rewrite_query_missing_deepseek_key_fails_early(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("\ufeffDEEPSEEK_API_KEY", raising=False)

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        rewrite_query("LNT II 为什么残留过高？")
