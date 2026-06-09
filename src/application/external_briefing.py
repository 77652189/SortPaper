from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

from src.adapters.llm.deepseek import DeepSeekChatClient
from src.application.settings import (
    DAILY_BRIEF_JUDGE_MODEL,
    DAILY_BRIEF_MAX_WORKERS,
    DAILY_BRIEF_SMALL_MODEL,
)
from src.domain.y103_topics import Y103_TOPICS, Y103Topic
from src.domain.external_papers import ExternalPaperCandidate
from src.ports.llm import ChatCompletionClient


BriefProgressCallback = Callable[[dict[str, Any]], None]


Y103Category = Y103Topic


@dataclass
class BriefClassificationDecision:
    category_id: str = "other"
    category_name: str = "其他/不纳入Y103"
    confidence: float = 0.0
    status: str = "rule_only"
    reason: str = ""
    judge_reason: str = ""
    key_evidence: list[str] = field(default_factory=list)


Y103_CATEGORIES: tuple[Y103Category, ...] = Y103_TOPICS

_CATEGORY_BY_ID = {category.category_id: category for category in Y103_CATEGORIES}
_GENERIC_PICHIA_KEYWORDS = {"pichia", "pastoris", "komagataella"}


def classify_candidates_for_brief(
    candidates: list[ExternalPaperCandidate],
    *,
    use_llm: bool = False,
    small_llm_client: ChatCompletionClient | None = None,
    judge_llm_client: ChatCompletionClient | None = None,
    small_model: str = DAILY_BRIEF_SMALL_MODEL,
    judge_model: str = DAILY_BRIEF_JUDGE_MODEL,
    max_workers: int = DAILY_BRIEF_MAX_WORKERS,
    progress_callback: BriefProgressCallback | None = None,
) -> dict[str, BriefClassificationDecision]:
    classifier = MetadataBriefClassifier(
        use_llm=use_llm,
        small_llm_client=small_llm_client,
        judge_llm_client=judge_llm_client,
        small_model=small_model,
        judge_model=judge_model,
    )
    decisions: dict[str, BriefClassificationDecision] = {}
    if not candidates:
        return decisions

    worker_count = max(1, min(max_workers, len(candidates)))
    _emit(progress_callback, event="start", total=len(candidates), completed=0)
    completed = 0
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {pool.submit(classifier.classify, candidate): candidate for candidate in candidates}
        for future in as_completed(futures):
            candidate = futures[future]
            try:
                decision = future.result()
            except Exception as exc:
                decision = classifier.rule_classify(candidate)
                decision.status = "failed_fallback_rule"
                decision.judge_reason = f"{type(exc).__name__}: {str(exc)[:300]}"
            decisions[candidate.candidate_id] = decision
            completed += 1
            _emit(
                progress_callback,
                event="candidate_done",
                total=len(candidates),
                completed=completed,
                candidate_id=candidate.candidate_id,
                title=candidate.title,
                category_id=decision.category_id,
                category_name=decision.category_name,
                status=decision.status,
            )
    _emit(progress_callback, event="done", total=len(candidates), completed=completed)
    return decisions


class MetadataBriefClassifier:
    def __init__(
        self,
        *,
        use_llm: bool = False,
        small_llm_client: ChatCompletionClient | None = None,
        judge_llm_client: ChatCompletionClient | None = None,
        small_model: str = DAILY_BRIEF_SMALL_MODEL,
        judge_model: str = DAILY_BRIEF_JUDGE_MODEL,
    ) -> None:
        self.use_llm = use_llm
        self.small_llm_client = small_llm_client
        self.judge_llm_client = judge_llm_client
        self.small_model = small_model
        self.judge_model = judge_model

    def classify(self, candidate: ExternalPaperCandidate) -> BriefClassificationDecision:
        fallback = self.rule_classify(candidate)
        if not self.use_llm:
            return fallback
        try:
            small = self._small_model_classify(candidate, fallback=fallback)
        except Exception as exc:
            fallback.status = "small_failed_fallback_rule"
            fallback.judge_reason = f"{type(exc).__name__}: {str(exc)[:300]}"
            return fallback
        try:
            return self._judge(candidate, small)
        except Exception as exc:
            small.status = "judge_failed_small_only"
            small.judge_reason = f"{type(exc).__name__}: {str(exc)[:300]}"
            return small

    def rule_classify(self, candidate: ExternalPaperCandidate) -> BriefClassificationDecision:
        text = _candidate_text(candidate)
        scored: list[tuple[int, Y103Category]] = []
        for category in Y103_CATEGORIES:
            if not _category_gate(text, category):
                continue
            matched_keywords = [
                keyword for keyword in category.keywords
                if _keyword_in_text(text, keyword)
            ]
            score = sum(
                1 for keyword in matched_keywords
                if keyword.lower() not in _GENERIC_PICHIA_KEYWORDS
            )
            if category.category_id == "03" and {"pichia", "pastoris"}.intersection(text.split()) and (
                "lactoferrin" in text or "osteopontin" in text
            ):
                score += 4
            if category.category_id == "13" and any(
                marker in text
                for marker in ("recombinant", "heterologous", "expression", "expressing", "expressed")
            ):
                score += 2
            if score:
                scored.append((score, category))
        if not scored:
            return BriefClassificationDecision(
                category_id="other",
                category_name="其他/不纳入Y103",
                confidence=0.25,
                status="rule_only",
                reason="候选元数据未命中Y103分类标准中的核心关键词。",
            )
        score, category = sorted(scored, key=lambda item: (item[0], item[1].category_id), reverse=True)[0]
        confidence = min(0.9, 0.45 + score * 0.12)
        return BriefClassificationDecision(
            category_id=category.category_id,
            category_name=category.name,
            confidence=round(confidence, 2),
            status="rule_only",
            reason=f"规则命中 {score} 个分类关键词。",
            key_evidence=_keyword_evidence(text, category.keywords),
        )

    def _small_model_classify(
        self,
        candidate: ExternalPaperCandidate,
        *,
        fallback: BriefClassificationDecision,
    ) -> BriefClassificationDecision:
        client = self.small_llm_client or DeepSeekChatClient(timeout=30.0)
        payload = _extract_json(client.complete(
            system_prompt="你是文献元数据快速分类助手，只输出JSON。",
            user_prompt=_small_classifier_prompt(candidate, fallback),
            model=self.small_model,
            temperature=0.0,
            max_tokens=900,
            response_format={"type": "json_object"},
            label="daily-brief-small-classifier",
        ))
        return _decision_from_payload(payload, default_status="small_only", fallback=fallback)

    def _judge(
        self,
        candidate: ExternalPaperCandidate,
        small_decision: BriefClassificationDecision,
    ) -> BriefClassificationDecision:
        client = self.judge_llm_client or self.small_llm_client or DeepSeekChatClient(timeout=60.0)
        payload = _extract_json(client.complete(
            system_prompt="你是严谨的Y103文献分类审稿人，只输出JSON。",
            user_prompt=_judge_prompt(candidate, small_decision),
            model=self.judge_model,
            temperature=0.0,
            max_tokens=1200,
            response_format={"type": "json_object"},
            label="daily-brief-judge",
        ))
        accepted = bool(payload.get("accepted", False))
        if accepted:
            small_decision.status = "judge_confirmed"
            small_decision.judge_reason = str(payload.get("reason") or "")
            if payload.get("confidence") is not None:
                small_decision.confidence = _safe_float(payload.get("confidence"), small_decision.confidence)
            return small_decision
        revised = _decision_from_payload(payload, default_status="judge_revised", fallback=small_decision)
        revised.judge_reason = str(payload.get("reason") or payload.get("judge_reason") or "")
        return revised


def y103_standard_text() -> str:
    lines = []
    for category in Y103_CATEGORIES:
        lines.append(f"{category.category_id}. {category.name}: {category.description}")
    lines.append("other. 其他/不纳入Y103: 与上述分类均不匹配，或元数据不足以判断。")
    return "\n".join(lines)


def _small_classifier_prompt(candidate: ExternalPaperCandidate, fallback: BriefClassificationDecision) -> str:
    return f"""请按Y103分类标准对候选论文做快速分类。

Y103分类标准：
{y103_standard_text()}

候选论文元数据：
{_candidate_prompt_block(candidate)}

规则初判：{fallback.category_id} {fallback.category_name}，理由：{fallback.reason}

只返回JSON：
{{
  "category_id": "01-16或other",
  "category_name": "分类名称",
  "confidence": 0到1,
  "reason": "一句话理由",
  "key_evidence": ["1-3条来自标题/摘要/期刊的证据"]
}}
"""


def _judge_prompt(candidate: ExternalPaperCandidate, small_decision: BriefClassificationDecision) -> str:
    return f"""请审查小模型对候选论文的Y103分类是否可靠。

Y103分类标准：
{y103_standard_text()}

候选论文元数据：
{_candidate_prompt_block(candidate)}

小模型分类结果：
{json.dumps({
    "category_id": small_decision.category_id,
    "category_name": small_decision.category_name,
    "confidence": small_decision.confidence,
    "reason": small_decision.reason,
    "key_evidence": small_decision.key_evidence,
}, ensure_ascii=False)}

请优先检查：分类是否落在Y103标准内、是否把综述误判为实验、是否仅因出现单个关键词而过度匹配。
只返回JSON：
{{
  "accepted": true或false,
  "category_id": "接受时可复用小模型category_id；不接受时给出修正分类",
  "category_name": "分类名称",
  "confidence": 0到1,
  "reason": "judge判断理由",
  "key_evidence": ["1-3条证据"]
}}
"""


def _candidate_prompt_block(candidate: ExternalPaperCandidate) -> str:
    authors = ", ".join(candidate.authors[:8])
    return "\n".join([
        f"标题：{candidate.title}",
        f"摘要：{candidate.abstract[:1800]}",
        f"期刊/来源：{candidate.journal}",
        f"作者：{authors}",
        f"日期：{candidate.published_date}",
        f"DOI：{candidate.doi}",
        f"PMID：{candidate.pmid}",
        f"PMCID：{candidate.pmcid}",
        f"外部来源：{', '.join(candidate.sources)}",
    ])


def _candidate_text(candidate: ExternalPaperCandidate) -> str:
    return " ".join([
        candidate.title,
        candidate.abstract,
        candidate.journal,
        " ".join(candidate.authors),
    ]).lower()


def _category_gate(text: str, category: Y103Category) -> bool:
    has_pichia = any(marker in text for marker in ("pichia", "komagataella", "pastoris", "????"))
    has_milk_protein = any(marker in text for marker in ("lactoferrin", "osteopontin", "????", "????"))
    if category.category_id == "03":
        return has_pichia and has_milk_protein and any(
            _keyword_in_text(text, marker)
            for marker in ("recombinant", "heterologous", "expression", "expressing", "secretion", "strain", "production")
        )
    if category.category_id == "02":
        return has_milk_protein and any(
            _keyword_in_text(text, marker)
            for marker in ("review", "overview", "meta-analysis", "systematic review")
        )
    if category.category_id == "04":
        return has_milk_protein and any(
            _keyword_in_text(text, marker)
            for marker in ("glycosylation", "phosphorylation", "post-translational", "ptm")
        )
    if category.category_id == "15":
        return has_milk_protein and any(
            _keyword_in_text(text, marker)
            for marker in ("purification", "separation", "chromatography", "fermentation broth")
        )
    return has_pichia


def _keyword_evidence(text: str, keywords: tuple[str, ...]) -> list[str]:
    return [keyword for keyword in keywords if _keyword_in_text(text, keyword)][:3]


def _keyword_in_text(text: str, keyword: str) -> bool:
    normalized = keyword.lower().strip()
    if not normalized:
        return False
    if re.fullmatch(r"[a-z0-9'-]+", normalized):
        return bool(re.search(rf"(?<![a-z0-9'-]){re.escape(normalized)}(?![a-z0-9'-])", text))
    return normalized in text


def _decision_from_payload(
    payload: dict[str, Any],
    *,
    default_status: str,
    fallback: BriefClassificationDecision,
) -> BriefClassificationDecision:
    category_id = str(payload.get("category_id") or fallback.category_id or "other").strip()
    category = _CATEGORY_BY_ID.get(category_id)
    if category is None and category_id != "other":
        category_id = fallback.category_id if fallback.category_id in _CATEGORY_BY_ID else "other"
        category = _CATEGORY_BY_ID.get(category_id)
    return BriefClassificationDecision(
        category_id=category_id,
        category_name=category.name if category else str(payload.get("category_name") or "其他/不纳入Y103"),
        confidence=_safe_float(payload.get("confidence"), fallback.confidence),
        status=default_status,
        reason=str(payload.get("reason") or fallback.reason),
        judge_reason=str(payload.get("judge_reason") or ""),
        key_evidence=[
            str(item) for item in payload.get("key_evidence", fallback.key_evidence) or []
            if str(item).strip()
        ][:3],
    )


def _extract_json(message: Any) -> dict[str, Any]:
    text = str(message or "").strip()
    for prefix in ("```json", "```"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("LLM response JSON is not an object")
    return value


def _safe_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


def _emit(callback: BriefProgressCallback | None, **event: Any) -> None:
    if callback is None:
        return
    try:
        callback(event)
    except Exception:
        return
