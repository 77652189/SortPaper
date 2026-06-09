from __future__ import annotations

from typing import Any


def result_key(result: dict[str, Any]) -> tuple[str, str]:
    payload = result.get("payload") or {}
    return (
        str(payload.get("paper_id") or ""),
        str(payload.get("chunk_id") or result.get("id") or ""),
    )


def deduplicate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen = set()
    for result in results:
        key = result_key(result)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped


def rank_papers_from_results(results: list[dict[str, Any]], *, limit: int) -> list[str]:
    ranked: list[str] = []
    seen = set()
    for result in results:
        paper_id = str((result.get("payload") or {}).get("paper_id") or "")
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        ranked.append(paper_id)
        if len(ranked) >= limit:
            break
    return ranked


def append_unique_by_id(
    results: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen = {item["id"] for item in results}
    for item in candidates:
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        results.append(item)
    return results


def backfill_anchor_count(limit: int) -> int:
    if limit <= 0:
        return 0
    if limit <= 5:
        return min(limit, 3)
    return min(limit, max(5, limit // 2))


def restore_anchor_candidates(
    results: list[dict[str, Any]],
    anchor_results: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0 or not anchor_results:
        return results[:limit]

    final = deduplicate_results(results)[:limit]
    final_keys = {result_key(item) for item in final}
    anchor_keys = {result_key(item) for item in anchor_results}
    missing_anchors = [
        item for item in anchor_results
        if result_key(item) not in final_keys
    ]
    if not missing_anchors:
        return final

    for anchor in missing_anchors:
        anchor_key = result_key(anchor)
        restored = dict(anchor)
        restored["anchor_restored"] = True
        if len(final) < limit:
            final.append(restored)
            final_keys.add(anchor_key)
            continue

        replace_index = next(
            (
                index for index in range(len(final) - 1, -1, -1)
                if result_key(final[index]) not in anchor_keys
            ),
            None,
        )
        if replace_index is None:
            break
        final[replace_index] = restored
        final_keys.add(anchor_key)

    return final[:limit]


def protect_prefix_candidates(
    results: list[dict[str, Any]],
    protected_results: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    if not protected_results:
        return deduplicate_results(results)[:limit]
    protected = deduplicate_results(protected_results)[:limit]
    protected_keys = {result_key(item) for item in protected}
    tail = [
        item for item in deduplicate_results(results)
        if result_key(item) not in protected_keys
    ]
    return (protected + tail)[:limit]


def interleave_ranked_groups(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    interleaved: list[dict[str, Any]] = []
    max_len = max((len(group) for group in groups), default=0)
    for index in range(max_len):
        for group in groups:
            if index < len(group):
                interleaved.append(group[index])
    return interleaved


def prioritize_evidence_groups(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if not groups:
        return []
    representatives = [group[0] for group in groups if group]
    lead_rest = groups[0][1:] if groups[0] else []
    other_rest = [group[1:] for group in groups[1:] if len(group) > 1]
    return representatives + lead_rest + interleave_ranked_groups(other_rest)


def score_rank_evidence_groups(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    ranked = [item for group in groups for item in group]
    ranked.sort(
        key=lambda item: (
            float(item.get("score") or 0.0),
            float(item.get("lexical_score") or 0.0),
            -int(item.get("paper_rank") or 9999),
        ),
        reverse=True,
    )
    return ranked
