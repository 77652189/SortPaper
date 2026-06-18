from __future__ import annotations

import re
from datetime import date
from typing import Any

from src.application.external_briefing import BriefClassificationDecision
from src.domain.external_papers import DailyBriefItem, ExternalPaperCandidate


BRIEF_TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "AI/数据方法": ("artificial intelligence", "machine learning", "deep learning", "generative ai", "algorithm", "prediction", "model"),
    "发酵/代谢工程": ("fermentation", "metabolic engineering", "biosynthesis", "bioreactor", "strain", "escherichia coli", "enzyme"),
    "微生物/感染": ("microbi", "bacteria", "infection", "pathogen", "antimicrobial", "malassezia", "biofilm"),
    "临床研究": ("clinical trial", "randomized", "double-blind", "patient", "cohort", "dose", "placebo"),
    "药物/天然产物": ("drug", "natural product", "compound", "therapeutic", "inhibitor", "pharmac"),
    "营养/乳铁蛋白": ("lactoferrin", "milk", "neonatal", "nutrition", "chitosan", "oligosaccharide"),
    "综述/观点": ("review", "perspective", "prospects", "insights", "overview", "meta-analysis"),
}


def candidate_priority_score(candidate: ExternalPaperCandidate) -> int:
    score = 0
    if candidate.pdf_access_status == "verified":
        score += 30
    elif candidate.pdf_url:
        score += 16
    if candidate.is_open_access:
        score += 10
    if candidate.abstract:
        score += 8
    if candidate.pmid or candidate.pmcid:
        score += 6
    if candidate.doi:
        score += 4
    if (candidate.year or 0) >= date.today().year - 1:
        score += 5
    return score


def daily_brief_item(
    candidate: ExternalPaperCandidate,
    decision: BriefClassificationDecision | None = None,
) -> DailyBriefItem:
    decision = decision or BriefClassificationDecision()
    topics = [decision.category_name] if decision.category_id != "other" else ["其他/不纳入Y103"]
    priority, reason = candidate_priority(candidate, decision)
    return DailyBriefItem(
        candidate_id=candidate.candidate_id,
        title=candidate.title,
        journal=candidate.journal,
        published_date=candidate.published_date,
        year=candidate.year,
        sources=list(candidate.sources),
        doi=candidate.doi,
        pmid=candidate.pmid,
        pmcid=candidate.pmcid,
        landing_url=candidate.landing_url,
        pdf_url=candidate.pdf_url,
        pdf_access_status=candidate.pdf_access_status,
        is_open_access=candidate.is_open_access,
        topics=topics,
        classification_id=decision.category_id,
        classification_name=decision.category_name,
        classification_confidence=decision.confidence,
        classification_status=decision.status,
        classification_reason=decision.reason,
        judge_reason=decision.judge_reason,
        abstract=candidate.abstract,
        abstract_translation_zh=candidate.abstract_translation_zh,
        abstract_translation_model=candidate.abstract_translation_model,
        abstract_translation_error=candidate.abstract_translation_error,
        brief=candidate_brief(candidate, decision),
        priority=priority,
        priority_reason=reason,
        recommended_action=candidate_recommended_action(candidate, priority),
        evidence_level=candidate_evidence_level(candidate),
    )


def candidate_topics(candidate: ExternalPaperCandidate) -> list[str]:
    haystack = " ".join([
        candidate.title,
        candidate.abstract,
        candidate.journal,
        " ".join(candidate.authors),
    ]).lower()
    topics = [
        topic for topic, keywords in BRIEF_TOPIC_KEYWORDS.items()
        if any(keyword in haystack for keyword in keywords)
    ]
    return topics or ["其他"]


def candidate_priority(
    candidate: ExternalPaperCandidate,
    decision: BriefClassificationDecision,
) -> tuple[str, str]:
    if decision.category_id == "other":
        return "low", decision.reason or "不纳入 Y103，默认不作为简报重点"
    score = candidate_priority_score(candidate)
    reasons = []
    if decision.category_id != "other":
        score += 10
        reasons.append(f"命中Y103分类：{decision.category_name}")
    if decision.status.startswith("judge_"):
        score += 6
        reasons.append("已通过Judge校验")
    if decision.confidence >= 0.75:
        score += 4
        reasons.append("分类置信度较高")
    if candidate.pdf_access_status == "verified":
        reasons.append("PDF 已可访问")
    elif candidate.pdf_url:
        reasons.append("存在待校验 PDF 链接")
    if candidate.abstract:
        reasons.append("有摘要可用于初筛")
    if candidate.pmid or candidate.pmcid:
        reasons.append("有 PubMed/PMC 标识")
    if score >= 38:
        return "high", "；".join(reasons) or "元数据完整度较高"
    if score >= 18:
        return "medium", "；".join(reasons) or "可继续观察"
    return "low", "；".join(reasons) or "元数据较少或暂不可下载"


def candidate_brief(candidate: ExternalPaperCandidate, decision: BriefClassificationDecision) -> str:
    lead = first_sentence(candidate.abstract) or "该候选目前主要依据标题和来源元数据进行初筛。"
    source = ", ".join(candidate.sources) or "未知来源"
    class_text = decision.category_name or "其他/不纳入Y103"
    journal = f"；期刊/来源：{candidate.journal}" if candidate.journal else ""
    reason = f" 分类依据：{decision.reason}" if decision.reason else ""
    return f"{lead} Y103分类：{class_text}；来源：{source}{journal}。{reason}"


def first_sentence(text: str) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return ""
    match = re.search(r"(.{20,260}?[。.!?])(?:\s|$)", clean)
    if match:
        return match.group(1).strip()
    return clean[:260].strip()


def candidate_recommended_action(candidate: ExternalPaperCandidate, priority: str) -> str:
    if candidate.import_status == "imported":
        return "已入库，可在向量库中检索"
    if candidate.pdf_access_status == "verified":
        return "优先导入 PDF"
    if candidate.pdf_url and candidate.pdf_access_status == "unknown":
        return "先做 PDF 预检"
    if candidate.pdf_access_status == "unavailable":
        return "保留元数据，等待替代来源或人工查看原文"
    if priority == "high":
        return "重点关注，尝试补充全文来源"
    return "保留候选，后续按主题筛选"


def candidate_evidence_level(candidate: ExternalPaperCandidate) -> str:
    text = " ".join([candidate.title, candidate.abstract]).lower()
    if any(keyword in text for keyword in ("randomized", "double-blind", "clinical trial")):
        return "临床试验"
    if "meta-analysis" in text or "systematic review" in text:
        return "系统综述/Meta"
    if "review" in text or "prospects" in text or "overview" in text:
        return "综述/观点"
    if candidate.abstract:
        return "研究论文/预印本"
    return "仅元数据"


def pdf_bucket(pdf_url: str, status: str) -> str:
    if status == "verified":
        return "PDF可访问"
    if status == "unavailable":
        return "PDF不可访问"
    if pdf_url:
        return "PDF待预检"
    return "无PDF链接"


def count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "").strip()
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def daily_brief_highlights(items: list[DailyBriefItem], *, total_candidates: int) -> list[str]:
    highlights = [f"本次窗口内共有 {total_candidates} 条候选，纳入简报 {len(items)} 条。"]
    high_count = sum(1 for item in items if item.priority == "high")
    verified_count = sum(1 for item in items if item.pdf_access_status == "verified")
    unavailable_count = sum(1 for item in items if item.pdf_access_status == "unavailable")
    if high_count:
        highlights.append(f"{high_count} 条被标为高优先级，建议优先查看。")
    if verified_count:
        highlights.append(f"{verified_count} 条已有可访问 PDF，可直接进入导入前处理。")
    if unavailable_count:
        highlights.append(f"{unavailable_count} 条 PDF 暂不可访问，可先作为元数据候选保留。")
    topic_counts = count_values(topic for item in items for topic in item.topics)
    if topic_counts:
        top_topics = "、".join(f"{topic}({count})" for topic, count in list(topic_counts.items())[:3])
        highlights.append(f"主要主题集中在：{top_topics}。")
    return highlights
