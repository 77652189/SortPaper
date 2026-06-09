from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from src.domain import result_policy


QUERY_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "what", "how",
    "are", "was", "were", "into", "over", "high", "problem",
    "pdf", "text", "chunk",
}

FIELD_WEIGHTS = {
    "entities": 12.0,
    "seo": 10.0,
    "visual": 8.0,
    "title": 6.0,
    "summary": 4.0,
    "context": 3.0,
    "content": 1.0,
}


def expand_scientific_aliases(value: str) -> str:
    text = str(value)
    text = re.sub(r"2\s*[\u2032\u2019']?\s*-?\s*fl\b", "2'-FL 2-FL 2-fucosyllactose", text, flags=re.IGNORECASE)
    text = re.sub(r"3\s*[\u2032\u2019']?\s*-?\s*fl\b", "3-FL 3-fucosyllactose", text, flags=re.IGNORECASE)
    text = re.sub(r"\blnt\s*ii\b|\blntii\b", "LNT II lacto-N-triose II", text, flags=re.IGNORECASE)
    text = re.sub(r"\blnnt\b", "LNnT lacto-N-neotetraose", text, flags=re.IGNORECASE)
    text = re.sub(r"\blnt\b(?!\s*ii)", "LNT lacto-N-tetraose", text, flags=re.IGNORECASE)
    text = re.sub(r"\be\.\s*coli\b", "E. coli Escherichia coli", text, flags=re.IGNORECASE)
    text = re.sub(r"\budp\s*-\s*gal\b", "UDP-Gal UDP-galactose", text, flags=re.IGNORECASE)
    text = re.sub(r"\budp\s*-\s*glcnac\b", "UDP-GlcNAc UDP-N-acetylglucosamine", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhmos?\b", "HMO human milk oligosaccharide human milk oligosaccharides", text, flags=re.IGNORECASE)
    return text


def normalize_search_text(value: str) -> str:
    text = expand_scientific_aliases(str(value)).lower()
    text = text.replace("\u00ad", "")
    for dash in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015", "\u2212"):
        text = text.replace(dash, "-")
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_rerank_title(value: str) -> str:
    text = Path(str(value)).stem
    text = re.sub(r"\([^)]*ablesci\.com\)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text.replace("_", " ")).strip()
    return expand_scientific_aliases(text)


def clean_rerank_field(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"\b(?:text|table|image)\s*\|\s*chunk\s+\d+(?:/\d+)?", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bbiosynthesis_review\b|\bfermentation_experiment\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:pdf|text|chunk)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return expand_scientific_aliases(text)


def clean_rerank_list(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        items = re.split(r"[,;|/]\s*", value)
    elif isinstance(value, (list, tuple, set)):
        items = [str(item) for item in value]
    else:
        items = [str(value)]
    cleaned = [
        expand_scientific_aliases(clean_rerank_field(item))
        for item in items
        if str(item).strip()
    ]
    return ", ".join(dict.fromkeys(item for item in cleaned if item))


def rerank_document_text(payload: dict[str, Any]) -> str:
    search_text = str(payload.get("search_text") or "").strip()
    if search_text:
        return search_text
    fields = [
        clean_rerank_title(str(payload.get("paper_title") or "")),
        clean_rerank_list(payload.get("target_products")),
        clean_rerank_list(payload.get("organisms")),
        clean_rerank_list(payload.get("genes")),
        clean_rerank_list(payload.get("enzymes")),
        clean_rerank_list(payload.get("metrics")),
        clean_rerank_list(payload.get("seo_terms")),
        clean_rerank_field(payload.get("seo_anchor_text", "")),
        clean_rerank_field(payload.get("context", "")),
        clean_rerank_field(payload.get("paper_summary", "")),
        clean_rerank_field(payload.get("content", "")),
    ]
    return "\n".join(str(field) for field in fields if str(field).strip())


def has_lnt_ii_anchor(normalized_text: str) -> bool:
    return (
        "lnt ii" in normalized_text
        or "lacto n triose ii" in normalized_text
        or "lacto-n-triose ii" in normalized_text
    )


def query_requires_strict_anchor(query: str) -> bool:
    normalized_query = normalize_search_text(query)
    return "lnt ii" in normalized_query or "lnt-ii" in normalized_query


def query_relevance_terms(query: str) -> dict[str, float]:
    normalized = normalize_search_text(query)
    terms: dict[str, float] = {}

    def add(term: str, weight: float) -> None:
        key = normalize_search_text(term)
        if key:
            terms[key] = max(terms.get(key, 0.0), weight)

    for token in re.findall(r"[a-z0-9][a-z0-9']{2,}", normalized):
        if token not in QUERY_STOPWORDS:
            add(token, 0.04)
    for phrase in re.findall(
        r"\b[a-z0-9][a-z0-9']*\s+(?:i|ii|iii|iv|v|vi|vii|viii|ix|x)\b",
        normalized,
    ):
        add(phrase, 0.18)

    if "lnt ii" in normalized or "lnt-ii" in normalized:
        add("lnt ii", 0.30)
        add("lacto-n-triose ii", 0.30)
        add("lacto n triose ii", 0.30)
        add("lgtA", 0.20)
        add("RBS", 0.18)
    if "e coli" in normalized or "escherichia coli" in normalized:
        add("escherichia coli", 0.16)
        add("e. coli", 0.16)
    if "metabolic" in normalized or "carbon flux" in normalized:
        add("metabolic", 0.08)
        add("carbon flux", 0.12)
    if "pathway" in normalized or "biosynthesis" in normalized:
        add("pathway", 0.08)
        add("biosynthesis pathway", 0.10)
    if "intermediate" in normalized:
        add("intermediate", 0.10)
    if "accumulation" in normalized or "accumulated" in normalized:
        add("accumulation", 0.10)
        add("accumulated", 0.10)
    if "engineering" in normalized or "strategy" in normalized:
        add("engineering", 0.08)
        add("strategy", 0.08)
    return terms


def lexical_match_score(query: str, text: str) -> float:
    return lexical_match_score_from_terms(
        normalize_search_text(text),
        query_terms=query_relevance_terms(query),
        requires_strict_anchor=query_requires_strict_anchor(query),
    )


def lexical_match_score_from_terms(
    normalized_text: str,
    *,
    query_terms: dict[str, float],
    requires_strict_anchor: bool,
) -> float:
    if requires_strict_anchor and not has_lnt_ii_anchor(normalized_text):
        return 0.0
    score = 0.0
    for term, weight in query_terms.items():
        if term == "lnt":
            continue
        if term in normalized_text:
            score += weight
    if "fermentation_experiment" in normalized_text or "fermentation experiment" in normalized_text:
        score += 0.12
    return score


def lexical_idf_match_score_from_terms(
    normalized_text: str,
    *,
    query_terms: dict[str, float],
    requires_strict_anchor: bool,
    lexical_index: dict[str, Any],
) -> float:
    if requires_strict_anchor and not has_lnt_ii_anchor(normalized_text):
        return 0.0
    score = 0.0
    for term, weight in query_terms.items():
        if term == "lnt":
            continue
        if term in normalized_text:
            score += weight * lexical_term_idf(term, lexical_index)
    if "fermentation_experiment" in normalized_text or "fermentation experiment" in normalized_text:
        score += 0.12
    return score


def lexical_term_idf(term: str, lexical_index: dict[str, Any]) -> float:
    index_terms = lexical_index_terms(term)
    if not index_terms:
        return 1.0
    total = max(len(lexical_index.get("entries") or []), 1)
    inverted = lexical_index.get("inverted") or {}
    frequencies = [
        len(inverted.get(index_term, set()))
        for index_term in index_terms
        if index_term in inverted
    ]
    document_frequency = min(frequencies) if frequencies else total
    return 1.0 + math.log((total + 1) / (document_frequency + 1))


def lexical_index_terms(value: str) -> set[str]:
    normalized = normalize_search_text(value)
    terms: set[str] = set()
    for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", normalized):
        if len(token) < 2 or token in QUERY_STOPWORDS:
            continue
        terms.add(token)
    return terms


def lexical_candidate_entries(
    *,
    query_terms: dict[str, float],
    lexical_index: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    term_postings: dict[str, set[int]] = {}
    inverted = lexical_index.get("inverted") or {}
    for term in query_terms:
        for index_term in lexical_index_terms(term):
            postings = inverted.get(index_term, set())
            if postings:
                term_postings[index_term] = postings

    if not term_postings:
        return []

    target_pool_size = max(limit * 8, 1000)
    max_postings_per_term = max(target_pool_size * 2, 2000)
    ranked_postings = [
        item
        for item in sorted(term_postings.items(), key=lambda item: len(item[1]))
        if len(item[1]) <= max_postings_per_term
    ]
    if not ranked_postings:
        return []

    candidate_offsets: set[int] = set()
    selected_terms = 0
    for _term, postings in ranked_postings:
        candidate_offsets.update(postings)
        selected_terms += 1
        if len(candidate_offsets) >= target_pool_size and selected_terms >= 3:
            break
        if selected_terms >= 12:
            break

    entries = lexical_index.get("entries") or []
    return [
        entries[offset]
        for offset in sorted(candidate_offsets)
        if 0 <= offset < len(entries)
    ]


def lexical_document_index(entries: list[dict[str, Any]]) -> dict[str, Any]:
    inverted: dict[str, set[int]] = {}
    for offset, entry in enumerate(entries):
        for term in lexical_index_terms(entry.get("normalized_text", "")):
            inverted.setdefault(term, set()).add(offset)
    return {"entries": entries, "inverted": inverted}


def lexical_candidates_from_entries(
    entries: list[dict[str, Any]],
    *,
    query_terms: dict[str, float],
    requires_strict_anchor: bool,
    score_fn: Any = lexical_match_score_from_terms,
    limit: int,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for entry in entries:
        lexical_score = score_fn(
            entry["normalized_text"],
            query_terms=query_terms,
            requires_strict_anchor=requires_strict_anchor,
        )
        if lexical_score <= 0:
            continue
        scored.append({
            "id": entry["id"],
            "score": lexical_score,
            "payload": entry["payload"],
            "lexical_score": lexical_score,
        })

    scored.sort(key=lambda item: item["lexical_score"], reverse=True)
    return scored[:limit]


def has_required_query_anchor(query: str, normalized_text: str) -> bool:
    if query_requires_strict_anchor(query):
        return has_lnt_ii_anchor(normalized_text)
    return True


def fielded_lexical_texts(payload: dict[str, Any]) -> dict[str, str]:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    entities = [
        payload.get("target_products"),
        payload.get("organisms"),
        payload.get("genes"),
        payload.get("enzymes"),
        payload.get("metrics"),
        payload.get("aliases"),
        payload.get("seo_terms"),
        payload.get("seo_metric_terms"),
    ]
    visual = [
        metadata.get("caption"),
        metadata.get("table_caption"),
        metadata.get("table_footnote"),
        metadata.get("figure_caption"),
        metadata.get("figure_label"),
        metadata.get("vision_group_description"),
        metadata.get("mineru_visual_content"),
        payload.get("vision_group_description"),
        payload.get("mineru_visual_content"),
    ]
    return {
        "title": normalize_field_value(payload.get("paper_title")),
        "entities": normalize_field_value(entities),
        "seo": normalize_field_value([
            payload.get("seo_anchor_text"),
            payload.get("seo_section"),
            payload.get("seo_evidence_type"),
        ]),
        "visual": normalize_field_value(visual),
        "summary": normalize_field_value(payload.get("paper_summary")),
        "context": normalize_field_value(payload.get("context")),
        "content": normalize_field_value(
            payload.get("search_text")
            or payload.get("content")
            or payload.get("raw_content")
        ),
    }


def normalize_field_value(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, (list, tuple, set)):
        value = " ".join(str(item) for item in value if str(item).strip())
    return normalize_search_text(str(value))


def fielded_lexical_score(
    payload: dict[str, Any],
    *,
    query_terms: dict[str, float],
    requires_strict_anchor: bool,
    lexical_index: dict[str, Any],
) -> float:
    fields = fielded_lexical_texts(payload)
    normalized_all = " ".join(fields.values())
    if requires_strict_anchor and not has_lnt_ii_anchor(normalized_all):
        return 0.0

    score = 0.0
    for field, text in fields.items():
        if not text:
            continue
        field_weight = FIELD_WEIGHTS.get(field, 1.0)
        for term, query_weight in query_terms.items():
            if term == "lnt":
                continue
            if term not in text:
                continue
            occurrences = min(text.count(term), 3)
            term_score = query_weight * lexical_term_idf(term, lexical_index)
            score += field_weight * term_score * (1.0 + 0.25 * (occurrences - 1))
    return score


def apply_fielded_lexical_boost(
    query: str,
    results: list[dict[str, Any]],
    *,
    weight: float,
    lexical_index: dict[str, Any],
) -> list[dict[str, Any]]:
    if weight <= 0 or not results:
        return results
    query_terms = query_relevance_terms(query)
    if not query_terms:
        return results
    requires_strict_anchor = query_requires_strict_anchor(query)
    for item in results:
        payload = item.get("payload") or {}
        fielded_score = fielded_lexical_score(
            payload,
            query_terms=query_terms,
            requires_strict_anchor=requires_strict_anchor,
            lexical_index=lexical_index,
        )
        if fielded_score <= 0:
            continue
        item["fielded_lexical_score"] = fielded_score
        item["score_before_fielded_lexical"] = float(item.get("score", 0.0) or 0.0)
        item["score"] = item["score_before_fielded_lexical"] + min(fielded_score * weight, 1.5)
    return sorted(results, key=lambda item: item.get("score", 0.0), reverse=True)


def apply_query_relevance_boost(query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    boost_cap = 1.35 if query_requires_strict_anchor(query) else 0.35
    for item in results:
        payload = item.get("payload") or {}
        lexical_score = lexical_match_score(query, rerank_document_text(payload))
        if lexical_score <= 0:
            continue
        item["lexical_score"] = lexical_score
        item["rerank_score"] = item.get("score", 0.0)
        item["score"] = float(item.get("score", 0.0) or 0.0) + min(lexical_score, boost_cap)
    return sorted(results, key=lambda item: item.get("score", 0.0), reverse=True)


def content_type_preference_bonus(
    payload: dict[str, Any],
    *,
    content_type_preference: str | None,
) -> float:
    preference = str(content_type_preference or "").strip().lower()
    if preference not in {"text", "table"}:
        return 0.0
    content_type = str(
        payload.get("content_type") or payload.get("worker_type") or ""
    ).strip().lower()
    return 0.08 if content_type == preference else 0.0


def paper_local_context_from_entries(
    entries: list[dict[str, Any]],
    *,
    ranked_papers: list[str],
    existing_results: list[dict[str, Any]],
    query_terms: dict[str, float],
    requires_strict_anchor: bool,
    lexical_index: dict[str, Any],
    per_paper_limit: int,
    total_limit: int,
    content_type_preference: str | None,
) -> list[dict[str, Any]]:
    paper_rank = {paper_id: index + 1 for index, paper_id in enumerate(ranked_papers)}
    existing_keys = {result_policy.result_key(item) for item in existing_results}
    evidence_by_paper: dict[str, list[dict[str, Any]]] = {
        paper_id: [] for paper_id in ranked_papers
    }

    for entry in entries:
        payload = entry.get("payload") or {}
        paper_id = str(payload.get("paper_id") or "")
        rank = paper_rank.get(paper_id)
        if rank is None or result_policy.result_key(entry) in existing_keys:
            continue
        lexical_score = lexical_idf_match_score_from_terms(
            entry["normalized_text"],
            query_terms=query_terms,
            requires_strict_anchor=requires_strict_anchor,
            lexical_index=lexical_index,
        )
        if lexical_score <= 0:
            continue
        preference_bonus = content_type_preference_bonus(
            payload,
            content_type_preference=content_type_preference,
        )
        score = lexical_score + preference_bonus + (0.03 / rank)
        evidence_by_paper[paper_id].append({
            "id": entry["id"],
            "score": score,
            "payload": payload,
            "lexical_score": lexical_score,
            "content_type_preference_bonus": preference_bonus,
            "paper_rank": rank,
            "context_expansion": True,
            "paper_local_context": True,
        })

    groups: list[list[dict[str, Any]]] = []
    for paper_id in ranked_papers:
        group = sorted(
            evidence_by_paper.get(paper_id, []),
            key=lambda item: item.get("score", 0.0),
            reverse=True,
        )
        groups.append(group[:per_paper_limit])

    context = result_policy.deduplicate_results(
        result_policy.score_rank_evidence_groups(groups)
    )
    return context[:total_limit]


def selective_rerank_reason(query: str, results: list[dict[str, Any]], *, limit: int) -> str:
    if len(results) < 2 or limit <= 0:
        return ""
    normalized_query = normalize_search_text(query)
    evidence_terms = {
        "which",
        "what",
        "find",
        "evidence",
        "passage",
        "reports",
        "report",
        "table",
        "figure",
        "image",
        "g l",
        "mg l",
    }
    if any(term in normalized_query for term in evidence_terms):
        return "evidence_query"
    return ""
