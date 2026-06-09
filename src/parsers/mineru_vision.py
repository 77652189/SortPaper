from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import mimetypes
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.parsers.config import VISION as CFG
from src.parsers.layout_chunk import LayoutChunk
from src.parsers.openai_vision import call_openai_vision_images

logger = logging.getLogger(__name__)


def describe_mineru_figure_groups(
    chunks: list[LayoutChunk],
    zip_path: str | Path,
    *,
    model: str | None = None,
    max_images_per_group: int = 8,
    cache_path: str | Path | None = None,
    max_workers: int = 1,
) -> list[LayoutChunk]:
    """Describe MinerU visual chunks by figure group instead of per image."""
    groups = _figure_groups(chunks)
    if not groups:
        return chunks
    model_name = model or CFG.VISION_MODEL
    cache = _load_cache(cache_path)
    cache_dirty = False
    tasks: list[_VisionTask] = []

    with zipfile.ZipFile(zip_path) as archive:
        for group_id, group_chunks in groups.items():
            cache_key = _cache_key(group_id, group_chunks, model_name=model_name)
            cached = _cached_description(cache, cache_key, model_name=model_name)
            if cached:
                _mark_group(group_chunks, attempted=True, succeeded=True, description=cached, cache_hit=True)
                continue
            images = _group_images(archive, group_chunks, max_images=max_images_per_group)
            if not images:
                _mark_group(group_chunks, attempted=False, succeeded=False, description="")
                continue
            prompt = _figure_group_prompt(group_id, group_chunks)
            tasks.append(_VisionTask(
                group_id=group_id,
                group_chunks=group_chunks,
                images=images,
                prompt=prompt,
                model_name=model_name,
                cache_key=cache_key,
            ))

    for task, description in _run_vision_tasks(tasks, max_workers=max_workers):
        _mark_group(
            task.group_chunks,
            attempted=True,
            succeeded=bool(description),
            description=description,
            cache_hit=False,
        )
        if description:
            _store_cached_description(cache, task.cache_key, model_name=model_name, description=description)
            cache_dirty = True
    if cache_dirty:
        _save_cache(cache_path, cache)
    return chunks


@dataclass
class _VisionTask:
    group_id: str
    group_chunks: list[LayoutChunk]
    images: list[dict[str, str]]
    prompt: str
    model_name: str
    cache_key: str


def _run_vision_tasks(
    tasks: list[_VisionTask],
    *,
    max_workers: int,
) -> list[tuple[_VisionTask, str]]:
    if not tasks:
        return []
    workers = max(1, min(int(max_workers or 1), len(tasks)))
    if workers == 1:
        return [(task, _call_group_vision(task)) for task in tasks]
    results: list[tuple[_VisionTask, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_task = {pool.submit(_call_group_vision, task): task for task in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                description = future.result()
            except Exception as exc:
                logger.warning("MinerU figure group vision failed for %s: %s", task.group_id, exc)
                description = ""
            results.append((task, description))
    return results


def _call_group_vision(task: _VisionTask) -> str:
    try:
        return call_openai_vision_images(
            images=task.images,
            prompt=task.prompt,
            model=task.model_name,
            timeout=90.0,
            max_output_tokens=2048,
        )
    except Exception as exc:
        logger.warning("MinerU figure group vision failed for %s: %s", task.group_id, exc)
        return ""


def _load_cache(cache_path: str | Path | None) -> dict[str, Any]:
    if not cache_path:
        return {"version": 1, "groups": {}}
    path = Path(cache_path)
    if not path.exists():
        return {"version": 1, "groups": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("MinerU figure vision cache is unreadable: %s", path)
        return {"version": 1, "groups": {}}
    if not isinstance(payload, dict):
        return {"version": 1, "groups": {}}
    groups = payload.get("groups")
    if not isinstance(groups, dict):
        payload["groups"] = {}
    payload.setdefault("version", 1)
    return payload


def _save_cache(cache_path: str | Path | None, cache: dict[str, Any]) -> None:
    if not cache_path:
        return
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _cached_description(cache: dict[str, Any], cache_key: str, *, model_name: str) -> str:
    entry = (cache.get("groups") or {}).get(cache_key)
    if not isinstance(entry, dict):
        return ""
    if entry.get("model") != model_name:
        return ""
    return str(entry.get("description") or "").strip()


def _store_cached_description(
    cache: dict[str, Any],
    cache_key: str,
    *,
    model_name: str,
    description: str,
) -> None:
    groups = cache.setdefault("groups", {})
    groups[cache_key] = {
        "model": model_name,
        "description": description.strip(),
    }


def _cache_key(group_id: str, group_chunks: list[LayoutChunk], *, model_name: str) -> str:
    payload = {
        "group_id": group_id,
        "model": model_name,
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "page": chunk.page,
                "bbox": list(chunk.bbox),
                "img_path": str((chunk.metadata or {}).get("img_path") or ""),
                "figure_caption": str((chunk.metadata or {}).get("figure_caption") or ""),
                "mineru_visual_content": str((chunk.metadata or {}).get("mineru_visual_content") or ""),
            }
            for chunk in group_chunks
        ],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _figure_groups(chunks: list[LayoutChunk]) -> OrderedDict[str, list[LayoutChunk]]:
    groups: OrderedDict[str, list[LayoutChunk]] = OrderedDict()
    for chunk in chunks:
        metadata = chunk.metadata or {}
        if chunk.content_type != "image":
            continue
        if metadata.get("parser") != "mineru_vlm":
            continue
        group_id = str(metadata.get("figure_group_id") or "").strip()
        if not group_id:
            continue
        groups.setdefault(group_id, []).append(chunk)
    return groups


def _group_images(
    archive: zipfile.ZipFile,
    group_chunks: list[LayoutChunk],
    *,
    max_images: int,
) -> list[dict[str, str]]:
    images: list[dict[str, str]] = []
    seen: set[str] = set()
    for chunk in group_chunks:
        img_path = str((chunk.metadata or {}).get("img_path") or "").strip()
        if not img_path or img_path in seen:
            continue
        seen.add(img_path)
        try:
            image_bytes = archive.read(img_path)
        except KeyError:
            logger.warning("MinerU image not found in zip: %s", img_path)
            continue
        image_b64, mime_type = _encode_image_for_vision(image_bytes, img_path)
        if image_b64:
            images.append({"image_b64": image_b64, "mime_type": mime_type})
        if len(images) >= max_images:
            break
    return images


def _encode_image_for_vision(image_bytes: bytes, image_path: str) -> tuple[str, str]:
    ext_mime = mimetypes.guess_type(image_path)[0] or "image/png"
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        max_dim = 1400
        if width > max_dim or height > max_dim:
            ratio = max_dim / max(width, height)
            img = img.resize((int(width * ratio), int(height * ratio)), Image.LANCZOS)
        if img.mode not in {"RGB", "L"}:
            img = img.convert("RGB")
        output = io.BytesIO()
        img.save(output, format="JPEG", quality=88)
        return base64.b64encode(output.getvalue()).decode("utf-8"), "image/jpeg"
    except Exception:
        return base64.b64encode(image_bytes).decode("utf-8"), ext_mime


def _figure_group_prompt(group_id: str, group_chunks: list[LayoutChunk]) -> str:
    first = group_chunks[0]
    metadata = first.metadata or {}
    caption = str(metadata.get("figure_caption") or "").strip()
    label = str(metadata.get("figure_label") or group_id).strip()
    parts = []
    for index, chunk in enumerate(group_chunks, start=1):
        item_meta = chunk.metadata or {}
        visual_text = str(item_meta.get("mineru_visual_content") or "").strip()
        item_caption = " ".join(
            value
            for key in ("image_caption", "chart_caption")
            for value in _string_list(item_meta.get(key))
        )
        parts.append(
            f"{index}. type={item_meta.get('mineru_type', 'image')}; "
            f"page={chunk.page}; bbox={list(chunk.bbox)}; "
            f"caption={item_caption[:500]}; mineru_text={visual_text[:500]}"
        )

    return (
        "You are analyzing figures from a scientific paper. The attached images are parts of the same figure group. "
        "Describe the whole figure as one retrieval-ready chunk.\n\n"
        "Requirements:\n"
        "1. Identify visible panels such as A, B, C and describe each panel separately when possible.\n"
        "2. Preserve experimental entities, pathway names, genes, strains, analytes, axes, units, and trends.\n"
        "3. Use the caption and MinerU text as context, but do not invent details that are not visible or stated.\n"
        "4. Output concise English prose suitable for semantic retrieval.\n\n"
        f"Figure label: {label}\n"
        f"Figure caption: {caption}\n"
        "MinerU visual item context:\n"
        + "\n".join(parts)
    )


def _mark_group(
    group_chunks: list[LayoutChunk],
    *,
    attempted: bool,
    succeeded: bool,
    description: str,
    cache_hit: bool = False,
) -> None:
    child_ids = [chunk.chunk_id for chunk in group_chunks]
    for chunk in group_chunks:
        metadata: dict[str, Any] = chunk.metadata
        metadata["vision_group_attempted"] = attempted
        metadata["vision_group_succeeded"] = succeeded
        metadata["vision_group_cache_hit"] = cache_hit
        metadata["vision_group_child_ids"] = child_ids
        if description:
            metadata["vision_group_description"] = description
            metadata["vision_needed"] = False
            chunk.raw_content = _merge_description(chunk, description)


def _merge_description(chunk: LayoutChunk, description: str) -> str:
    metadata = chunk.metadata or {}
    label = str(metadata.get("figure_label") or metadata.get("figure_group_id") or "Figure").strip()
    original = str(chunk.raw_content or "").strip()
    if original and original != description:
        return f"{label} group vision description:\n{description.strip()}\n\nOriginal MinerU visual placeholder:\n{original}"
    return description.strip()


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []
