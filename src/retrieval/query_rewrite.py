from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from src.judge.llm_runtime import run_llm_call
from src.runtime_env import load_project_env

load_project_env()

logger = logging.getLogger(__name__)


DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_QUERY_REWRITE_MODEL = "deepseek-v4-flash"

ALIAS_GROUPS = (
    ("LNT II", "LNTII", "lacto-N-triose II"),
    ("LNT", "lacto-N-tetraose"),
    ("LNnT", "lacto-N-neotetraose"),
    ("2'-FL", "2-FL", "2-fucosyllactose"),
    ("3-FL", "3-fucosyllactose"),
    ("E. coli", "Escherichia coli"),
)

VALID_EVIDENCE_PREFERENCES = {"any", "text", "table"}


@dataclass(frozen=True)
class QueryRewrite:
    raw_query: str
    normalized_query: str
    products: list[str] = field(default_factory=list)
    organisms: list[str] = field(default_factory=list)
    genes: list[str] = field(default_factory=list)
    enzymes: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)
    evidence_preference: str = "any"
    aliases: list[str] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rewrite_query(
    query: str,
    *,
    model: str | None = None,
    use_llm: bool = True,
) -> QueryRewrite:
    """Rewrite a user question into a stable retrieval query object."""
    raw_query = str(query or "").strip()
    if not raw_query:
        return QueryRewrite(raw_query="", normalized_query="")
    if not use_llm:
        return normalize_rewrite_payload({"raw_query": raw_query}, raw_query=raw_query)

    payload = _rewrite_with_deepseek(raw_query, model=model or _env_value(
        "SORTPAPER_QUERY_REWRITE_MODEL",
        DEFAULT_QUERY_REWRITE_MODEL,
    ))
    return normalize_rewrite_payload(payload, raw_query=raw_query)


def _rewrite_with_deepseek(query: str, *, model: str) -> dict[str, Any]:
    from openai import OpenAI

    api_key = deepseek_api_key()
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 未设置，无法使用查询改写")

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL, timeout=30.0)
    response = run_llm_call(
        lambda: client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _compact_rewrite_system_prompt()},
                {"role": "user", "content": query},
            ],
            temperature=0.1,
            max_tokens=2000,
            response_format={"type": "json_object"},
        ),
        label="deepseek-query-rewrite",
    )
    content = response.choices[0].message.content or ""
    return _parse_json_object(content)


def normalize_rewrite_payload(payload: dict[str, Any], *, raw_query: str) -> QueryRewrite:
    aliases = _dedupe_strings(_as_list(payload.get("aliases")) + _rule_aliases(raw_query, payload))
    products = _dedupe_strings(_as_list(payload.get("products")) + _aliases_for_kind(aliases, "product"))
    organisms = _dedupe_strings(_as_list(payload.get("organisms")) + _aliases_for_kind(aliases, "organism"))
    genes = _dedupe_strings(_as_list(payload.get("genes")))
    enzymes = _dedupe_strings(_as_list(payload.get("enzymes")))
    metrics = _dedupe_strings(_as_list(payload.get("metrics")))
    intents = _dedupe_strings(_as_list(payload.get("intents")) + _rule_intents(raw_query))

    evidence_preference = str(payload.get("evidence_preference") or "").strip().lower()
    if evidence_preference not in VALID_EVIDENCE_PREFERENCES:
        evidence_preference = _rule_evidence_preference(raw_query)

    normalized_query = str(payload.get("normalized_query") or "").strip()
    if not normalized_query:
        normalized_query = _build_normalized_query(
            raw_query=raw_query,
            products=products,
            organisms=organisms,
            genes=genes,
            enzymes=enzymes,
            metrics=metrics,
            intents=intents,
            aliases=aliases,
        )

    variants = _dedupe_strings(
        [normalized_query]
        + _as_list(payload.get("variants"))
        + _rule_variants(normalized_query, products, organisms, genes, metrics, intents)
    )

    return QueryRewrite(
        raw_query=raw_query,
        normalized_query=normalized_query,
        products=products,
        organisms=organisms,
        genes=genes,
        enzymes=enzymes,
        metrics=metrics,
        intents=intents,
        evidence_preference=evidence_preference,
        aliases=aliases,
        variants=variants[:4],
    )


def deepseek_api_key() -> str:
    return _env_value("DEEPSEEK_API_KEY")


def _rewrite_system_prompt() -> str:
    return (
        "你是 PaperSort 的轻量查询改写器。任务是把中文或英文生物制造/文献检索问题"
        "改写成稳定的英文检索结构。只输出 JSON，不要 Markdown。\n"
        "JSON 字段必须包含：raw_query, normalized_query, products, organisms, genes, enzymes, "
        "metrics, intents, evidence_preference, aliases, variants。\n"
        "要求：\n"
        "- normalized_query 用英文科学术语，适合向量/词面检索。\n"
        "- products/organisms/genes/enzymes/metrics/intents/aliases/variants 都是字符串数组。\n"
        "- evidence_preference 只能是 any、text、table。\n"
        "- variants 最多 5 条，覆盖实体、机制、路径、实验指标或表格证据角度。\n"
        "- 常见别名要展开，例如 LNTII -> lacto-N-triose II，E. coli -> Escherichia coli。\n"
        "- 不确定的字段输出空数组，不要编造论文结论。\n"
        "- 输出紧凑 JSON，字段值保持简短，避免长段解释。"
    )


def _compact_rewrite_system_prompt() -> str:
    return (
        "You are PaperSort's query normalizer for scientific literature retrieval. "
        "Return one compact JSON object only, no Markdown and no explanations. "
        "Required keys: raw_query, normalized_query, products, organisms, genes, enzymes, "
        "metrics, intents, evidence_preference, aliases, variants. "
        "Rules: normalized_query must be short English scientific search text; "
        "array fields must contain short strings only; evidence_preference must be any, text, or table; "
        "variants must contain at most 3 short search queries; expand aliases such as "
        "LNTII to lacto-N-triose II and E. coli to Escherichia coli; "
        "use empty arrays when uncertain; never invent paper findings."
    )


def _parse_json_object(value: str) -> dict[str, Any]:
    text = str(value or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            logger.warning("Query rewrite JSON parse failed: %s", text[:120])
            return {}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("Query rewrite JSON object parse failed: %s", text[:120])
            return {}
    return payload if isinstance(payload, dict) else {}


def _rule_aliases(raw_query: str, payload: dict[str, Any]) -> list[str]:
    text = " ".join([raw_query] + _flatten_strings(list(payload.values())))
    normalized = text.lower()
    aliases: list[str] = []
    for group in ALIAS_GROUPS:
        if any(_alias_in_text(alias, normalized) for alias in group):
            aliases.extend(group)
    return aliases


def _alias_in_text(alias: str, normalized_text: str) -> bool:
    normalized_alias = re.sub(r"[^a-z0-9]+", " ", alias.lower()).strip()
    normalized_query = re.sub(r"[^a-z0-9]+", " ", normalized_text).strip()
    if normalized_alias == "lnt":
        return bool(re.search(r"(?<![a-z0-9])lnt(?!\s+ii)(?![a-z0-9])", normalized_query))
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(normalized_alias)}(?![a-z0-9])", normalized_query))


def _aliases_for_kind(aliases: list[str], kind: str) -> list[str]:
    alias_set = {item.lower() for item in aliases}
    if kind == "organism" and ("e. coli" in alias_set or "escherichia coli" in alias_set):
        return ["E. coli", "Escherichia coli"]
    if kind == "product":
        products: list[str] = []
        for group in ALIAS_GROUPS[:-1]:
            if any(alias.lower() in alias_set for alias in group):
                products.extend(group)
        return products
    return []


def _rule_intents(raw_query: str) -> list[str]:
    rules = {
        "mechanism": ("why", "mechanism", "原因", "为什么", "机理", "机制"),
        "engineering_strategy": ("strategy", "engineering", "改造", "策略", "逻辑"),
        "pathway": ("pathway", "route", "通路", "路径", "途径"),
        "production": ("production", "yield", "titer", "产量", "滴度", "提高"),
        "comparison": ("compare", "versus", "vs", "对比", "比较"),
    }
    lowered = raw_query.lower()
    return [intent for intent, keys in rules.items() if any(key in lowered for key in keys)]


def _rule_evidence_preference(raw_query: str) -> str:
    lowered = raw_query.lower()
    if any(key in lowered for key in ("table", "表格", "titer", "yield", "产量", "滴度")):
        return "table"
    return "any"


def _rule_variants(
    normalized_query: str,
    products: list[str],
    organisms: list[str],
    genes: list[str],
    metrics: list[str],
    intents: list[str],
) -> list[str]:
    variants: list[str] = []
    core_entities = " ".join(products[:2] + organisms[:1] + genes[:3]).strip()
    if core_entities:
        variants.append(core_entities)
    if products and ("mechanism" in intents or "pathway" in intents):
        variants.append(" ".join(products[:2] + ["biosynthesis pathway mechanism"]))
    if products and "engineering_strategy" in intents:
        variants.append(" ".join(products[:2] + genes[:2] + ["metabolic engineering strategy"]))
    if products and metrics:
        variants.append(" ".join(products[:2] + metrics[:3]))
    variants.append(normalized_query)
    return variants


def _build_normalized_query(
    *,
    raw_query: str,
    products: list[str],
    organisms: list[str],
    genes: list[str],
    enzymes: list[str],
    metrics: list[str],
    intents: list[str],
    aliases: list[str],
) -> str:
    parts = products + organisms + genes + enzymes + metrics + intents + aliases
    if not parts:
        return raw_query
    return " ".join(_dedupe_strings(parts))


def _as_list(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return _flatten_strings(list(value))
    return [str(value)]


def _flatten_strings(values: list[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (list, tuple, set)):
            result.extend(_flatten_strings(list(value)))
        elif isinstance(value, dict):
            result.extend(_flatten_strings(list(value.values())))
        else:
            text = str(value).strip()
            if text:
                result.append(text)
    return result


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _env_value(name: str, default: str = "") -> str:
    return (os.getenv(name) or os.getenv(f"\ufeff{name}") or default).strip()
