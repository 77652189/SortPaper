from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import Any

from src.domain.external_papers import DailyBrief, DailyBriefItem, ExternalPaperCandidate, utc_now_iso
from src.ports.llm import ChatCompletionClient


TRANSLATION_SYSTEM_PROMPT = (
    "你是严谨的生命科学论文摘要翻译助手。"
    "请把英文论文摘要翻译成自然、准确、适合中文科研读者快速判断价值的中文。"
    "保留基因名、菌株名、蛋白名、缩写、单位和专有名词，不添加原文没有的信息。"
)


def translate_daily_brief_abstracts(
    brief: DailyBrief,
    *,
    llm_client: ChatCompletionClient,
    model: str,
    max_abstract_chars: int = 4000,
    max_workers: int = 4,
) -> DailyBrief:
    if not brief.items:
        return brief
    worker_count = max(1, min(max_workers, len(brief.items)))
    if worker_count == 1:
        translated_items = [
            translate_daily_brief_item_abstract(
                item,
                llm_client=llm_client,
                model=model,
                max_abstract_chars=max_abstract_chars,
            )
            for item in brief.items
        ]
    else:
        translated_items = list(brief.items)
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {
                pool.submit(
                    translate_daily_brief_item_abstract,
                    item,
                    llm_client=llm_client,
                    model=model,
                    max_abstract_chars=max_abstract_chars,
                ): index
                for index, item in enumerate(brief.items)
            }
            for future in as_completed(futures):
                translated_items[futures[future]] = future.result()
    brief.items = translated_items
    return brief


def translate_daily_brief_item_abstract(
    item: DailyBriefItem,
    *,
    llm_client: ChatCompletionClient,
    model: str,
    max_abstract_chars: int = 4000,
) -> DailyBriefItem:
    abstract = _clean_abstract(item.abstract)
    if not abstract or item.abstract_translation_zh:
        return item
    prompt = _translation_prompt(
        title=item.title,
        abstract=abstract[:max(max_abstract_chars, 1)],
    )
    try:
        translated = llm_client.complete(
            system_prompt=TRANSLATION_SYSTEM_PROMPT,
            user_prompt=prompt,
            model=model,
            temperature=0.1,
            max_tokens=1200,
            label="daily-brief-abstract-translation",
        )
    except Exception as exc:
        return replace(
            item,
            abstract_translation_model=model,
            abstract_translation_error=f"{type(exc).__name__}: {str(exc)[:240]}",
        )
    return replace(
        item,
        abstract_translation_zh=_clean_translation(translated),
        abstract_translation_model=model,
        abstract_translation_error="",
    )


def translate_candidate_abstracts(
    candidates: list[ExternalPaperCandidate],
    *,
    llm_client: ChatCompletionClient,
    model: str,
    max_abstract_chars: int = 4000,
    max_workers: int = 4,
    batch_size: int = 4,
) -> list[ExternalPaperCandidate]:
    """Translate missing candidate abstracts while preserving input order."""
    if not candidates:
        return []
    pending = [candidate for candidate in candidates if candidate_needs_abstract_translation(candidate)]
    if not pending:
        return list(candidates)

    batches = list(_batches(pending, max(int(batch_size or 1), 1)))
    worker_count = max(1, min(int(max_workers or 1), len(batches)))
    translated_by_id: dict[str, ExternalPaperCandidate] = {}

    if worker_count == 1:
        for batch in batches:
            for translated in translate_candidate_abstract_batch(
                batch,
                llm_client=llm_client,
                model=model,
                max_abstract_chars=max_abstract_chars,
            ):
                translated_by_id[translated.candidate_id] = translated
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {
                pool.submit(
                    translate_candidate_abstract_batch,
                    batch,
                    llm_client=llm_client,
                    model=model,
                    max_abstract_chars=max_abstract_chars,
                ): batch
                for batch in batches
            }
            for future in as_completed(futures):
                for translated in future.result():
                    translated_by_id[translated.candidate_id] = translated

    return [
        translated_by_id.get(candidate.candidate_id, candidate)
        for candidate in candidates
    ]


def translate_candidate_abstract_batch(
    candidates: list[ExternalPaperCandidate],
    *,
    llm_client: ChatCompletionClient,
    model: str,
    max_abstract_chars: int = 4000,
) -> list[ExternalPaperCandidate]:
    pending = [candidate for candidate in candidates if candidate_needs_abstract_translation(candidate)]
    if not pending:
        return list(candidates)
    if len(pending) == 1:
        translated = translate_candidate_abstract(
            pending[0],
            llm_client=llm_client,
            model=model,
            max_abstract_chars=max_abstract_chars,
        )
        return [translated if item.candidate_id == translated.candidate_id else item for item in candidates]

    payload = [
        {
            "candidate_id": candidate.candidate_id,
            "title": candidate.title,
            "abstract": _clean_abstract(candidate.abstract)[:max(max_abstract_chars, 1)],
        }
        for candidate in pending
    ]
    try:
        message = llm_client.complete(
            system_prompt=TRANSLATION_SYSTEM_PROMPT,
            user_prompt=_batch_translation_prompt(payload),
            model=model,
            temperature=0.1,
            max_tokens=max(1200, min(6000, 900 * len(payload))),
            response_format={"type": "json_object"},
            label="external-candidate-abstract-translation-batch",
        )
        translations = _parse_batch_translations(message)
    except Exception as exc:
        return [
            _candidate_translation_error(candidate, model, exc)
            if candidate.candidate_id in {item.candidate_id for item in pending}
            else candidate
            for candidate in candidates
        ]

    translated_by_id = {
        candidate.candidate_id: _apply_candidate_translation(
            candidate,
            model=model,
            translation=translations.get(candidate.candidate_id, ""),
        )
        for candidate in pending
    }
    return [
        translated_by_id.get(candidate.candidate_id, candidate)
        for candidate in candidates
    ]


def translate_candidate_abstract(
    candidate: ExternalPaperCandidate,
    *,
    llm_client: ChatCompletionClient,
    model: str,
    max_abstract_chars: int = 4000,
) -> ExternalPaperCandidate:
    abstract = _clean_abstract(candidate.abstract)
    if not candidate_needs_abstract_translation(candidate):
        return candidate
    prompt = _translation_prompt(
        title=candidate.title,
        abstract=abstract[:max(max_abstract_chars, 1)],
    )
    try:
        translated = llm_client.complete(
            system_prompt=TRANSLATION_SYSTEM_PROMPT,
            user_prompt=prompt,
            model=model,
            temperature=0.1,
            max_tokens=1200,
            label="external-candidate-abstract-translation",
        )
    except Exception as exc:
        return _candidate_translation_error(candidate, model, exc)
    return _apply_candidate_translation(candidate, model=model, translation=translated)


def translate_candidate_abstracts_for_repository(
    repo: Any,
    candidate_ids: list[str],
    *,
    llm_client: ChatCompletionClient,
    model: str,
    max_abstract_chars: int = 4000,
    max_workers: int = 4,
    batch_size: int = 4,
) -> dict[str, Any]:
    unique_ids = list(dict.fromkeys(str(candidate_id) for candidate_id in candidate_ids if str(candidate_id)))
    candidates = [
        candidate for candidate_id in unique_ids
        if (candidate := repo.get_candidate(candidate_id)) is not None
    ]
    pending_ids = {
        candidate.candidate_id
        for candidate in candidates
        if candidate_needs_abstract_translation(candidate)
    }
    if not pending_ids:
        return {
            "requested_count": len(unique_ids),
            "found_count": len(candidates),
            "skipped_count": len(candidates),
            "translated_count": 0,
            "failed_count": 0,
            "model": model,
            "candidate_ids": [],
            "errors": [],
        }

    translated = translate_candidate_abstracts(
        candidates,
        llm_client=llm_client,
        model=model,
        max_abstract_chars=max_abstract_chars,
        max_workers=max_workers,
        batch_size=batch_size,
    )

    translated_count = 0
    failed_count = 0
    updated_ids: list[str] = []
    errors: list[dict[str, str]] = []
    for candidate in translated:
        if candidate.candidate_id not in pending_ids:
            continue
        if hasattr(repo, "update_candidate"):
            repo.update_candidate(candidate)
        updated_ids.append(candidate.candidate_id)
        if candidate.abstract_translation_zh and not candidate.abstract_translation_error:
            translated_count += 1
        else:
            failed_count += 1
            if candidate.abstract_translation_error:
                errors.append({
                    "candidate_id": candidate.candidate_id,
                    "error": candidate.abstract_translation_error,
                })

    return {
        "requested_count": len(unique_ids),
        "found_count": len(candidates),
        "skipped_count": len(candidates) - len(pending_ids),
        "translated_count": translated_count,
        "failed_count": failed_count,
        "model": model,
        "batch_size": max(int(batch_size or 1), 1),
        "max_workers": max(int(max_workers or 1), 1),
        "candidate_ids": updated_ids,
        "errors": errors,
    }


def apply_cached_candidate_translations_to_brief(brief: DailyBrief, repo: Any) -> DailyBrief:
    """Copy stored candidate translations into a freshly generated brief without LLM calls."""
    if not hasattr(repo, "get_candidate"):
        return brief
    items: list[DailyBriefItem] = []
    for item in brief.items:
        candidate = repo.get_candidate(item.candidate_id)
        if candidate is None:
            items.append(item)
            continue
        items.append(replace(
            item,
            abstract_translation_zh=item.abstract_translation_zh or candidate.abstract_translation_zh,
            abstract_translation_model=item.abstract_translation_model or candidate.abstract_translation_model,
            abstract_translation_error=item.abstract_translation_error or candidate.abstract_translation_error,
        ))
    brief.items = items
    return brief


def candidate_needs_abstract_translation(candidate: ExternalPaperCandidate) -> bool:
    abstract = _clean_abstract(candidate.abstract)
    if not abstract:
        return False
    if not candidate.abstract_translation_zh:
        return True
    stored_hash = str(candidate.abstract_translation_source_hash or "").strip()
    return bool(stored_hash and stored_hash != _abstract_hash(abstract))


def _translation_prompt(*, title: str, abstract: str) -> str:
    return "\n".join([
        "请翻译以下论文摘要，只输出中文翻译本身。",
        f"标题：{title}",
        f"英文摘要：{abstract}",
    ])


def _batch_translation_prompt(items: list[dict[str, str]]) -> str:
    return "\n".join([
        "请批量翻译以下论文摘要。",
        "只返回 JSON，不要输出额外文字。格式如下：",
        '{"translations":[{"candidate_id":"...","translation_zh":"中文翻译"}]}',
        "待翻译条目：",
        json.dumps(items, ensure_ascii=False, indent=2),
    ])


def _parse_batch_translations(message: Any) -> dict[str, str]:
    text = str(message or "").strip()
    for prefix in ("```json", "```"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    payload = json.loads(text)
    records = payload.get("translations", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("translation response does not contain a translations list")
    translations: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        candidate_id = str(record.get("candidate_id") or "").strip()
        translation = _clean_translation(str(record.get("translation_zh") or record.get("translation") or ""))
        if candidate_id and translation:
            translations[candidate_id] = translation
    return translations


def _apply_candidate_translation(
    candidate: ExternalPaperCandidate,
    *,
    model: str,
    translation: str,
) -> ExternalPaperCandidate:
    cleaned = _clean_translation(translation)
    if not cleaned:
        return replace(
            candidate,
            abstract_translation_model=model,
            abstract_translation_error="empty translation response",
            updated_at=utc_now_iso(),
        )
    abstract = _clean_abstract(candidate.abstract)
    return replace(
        candidate,
        abstract_translation_zh=cleaned,
        abstract_translation_model=model,
        abstract_translation_error="",
        abstract_translation_source_hash=_abstract_hash(abstract),
        abstract_translated_at=utc_now_iso(),
        updated_at=utc_now_iso(),
    )


def _candidate_translation_error(
    candidate: ExternalPaperCandidate,
    model: str,
    exc: Exception,
) -> ExternalPaperCandidate:
    return replace(
        candidate,
        abstract_translation_model=model,
        abstract_translation_error=f"{type(exc).__name__}: {str(exc)[:240]}",
        updated_at=utc_now_iso(),
    )


def _abstract_hash(value: str) -> str:
    return hashlib.sha1(_clean_abstract(value).encode("utf-8")).hexdigest()


def _batches(items: list[ExternalPaperCandidate], batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index:index + batch_size]


def _clean_abstract(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clean_translation(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^中文翻译[:：]\s*", "", text)
    return text


__all__ = [
    "apply_cached_candidate_translations_to_brief",
    "candidate_needs_abstract_translation",
    "translate_candidate_abstract",
    "translate_candidate_abstract_batch",
    "translate_candidate_abstracts",
    "translate_candidate_abstracts_for_repository",
    "translate_daily_brief_abstracts",
    "translate_daily_brief_item_abstract",
]
