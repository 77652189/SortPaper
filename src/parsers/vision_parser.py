"""
图像解析器：使用 PyMuPDF 提取图片 + Qwen-VL 描述，包装为 LayoutChunk。
每个图片作为一个 chunk，携带 bbox 坐标和描述文本。

两阶段：先用 PyMuPDF 提取所有图片数据（快），再并发调 Qwen-VL（慢）。
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.parsers.layout_chunk import LayoutChunk, infer_column
from src.parsers.config import VISION as CFG

MAX_VISION_WORKERS = 4  # Qwen-VL 并发数

logger = logging.getLogger(__name__)

# （面积/图注阈值已移至 src/parsers/config.py VisionParserConfig）


class VisionParser:
    def __init__(self, pdf_path: str | Path, model: str = "qwen-vl-plus") -> None:
        self.pdf_path = str(pdf_path)
        self.model = model

    def parse(self, feedback: str | None = None) -> tuple[list[LayoutChunk], dict[str, float]]:
        """
        提取 PDF 所有图片并生成描述，返回 (LayoutChunk 列表, 耗时统计)。

        两阶段：
        1. 串行提取：PyMuPDF 获取图片数据（本机操作，极快）
        2. 并发描述：ThreadPoolExecutor 并发调 Qwen-VL API
        """
        import fitz

        t0 = time.time()

        # ── Phase 1: 提取所有图片数据 ──
        image_tasks: list[dict] = []  # {image_bytes, ext, caption, page, bbox, ...}

        with fitz.open(self.pdf_path) as doc:
            for page_index, page in enumerate(doc, start=1):
                page_width = float(page.rect.width)
                text_blocks = page.get_text("blocks")
                image_entries = page.get_image_info(xrefs=True)

                for img_idx, image_info in enumerate(image_entries):
                    xref = image_info.get("xref")
                    if not xref:
                        continue
                    bbox = self._normalize_bbox(image_info.get("bbox"))
                    if bbox is None:
                        continue
                    x0, y0, x1, y1 = bbox
                    if (x1 - x0) * (y1 - y0) < CFG.MIN_AREA_PT2:
                        continue
                    extracted = doc.extract_image(xref)
                    image_bytes = extracted.get("image")
                    if not image_bytes:
                        continue
                    img_w = extracted.get("width", 0)
                    img_h = extracted.get("height", 0)
                    if img_w * img_h < CFG.MIN_PIXEL_AREA:
                        continue
                    caption = self._find_caption(bbox, text_blocks)
                    ext = extracted.get("ext", "png")
                    column = infer_column(page_width, x0, x1)
                    image_tasks.append({
                        "image_bytes": image_bytes,
                        "ext": ext,
                        "caption": caption,
                        "page": page_index,
                        "bbox": bbox,
                        "column": column,
                        "xref": xref,
                        "img_idx": img_idx,
                    })

        t_extract = time.time() - t0

        # ── Phase 2: 并发 VL 描述 ──
        t_desc_start = time.time()
        chunk_slots: list[LayoutChunk | None] = [None] * len(image_tasks)

        if not image_tasks:
            return [], {"extract": round(t_extract, 1), "describe": 0, "total": round(t_extract, 1)}

        # 为每个任务构造 LayoutChunk 模板（除 raw_content 外已完整）
        templates: list[LayoutChunk] = []
        for t in image_tasks:
            raw = f"【图注】{t['caption']}" if t["caption"] else ""
            templates.append(LayoutChunk(
                content_type="image",
                raw_content=raw,
                page=t["page"],
                bbox=t["bbox"],
                column=t["column"],
                order_in_page=t["img_idx"],
                metadata={
                    "parser": "qwen-vl-plus",
                    "caption": t["caption"],
                    "image_index": t["img_idx"],
                    "xref": t["xref"],
                },
            ))

        works = min(MAX_VISION_WORKERS, len(image_tasks))
        with ThreadPoolExecutor(max_workers=works) as pool:
            futures = {
                pool.submit(
                    VisionParser._describe_single,
                    self.model,
                    image_tasks[i]["image_bytes"],
                    image_tasks[i]["ext"],
                    image_tasks[i]["caption"],
                    feedback,
                ): i
                for i in range(len(image_tasks))
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    desc = future.result()
                except Exception as exc:
                    logger.warning("VL describe task %d failed: %s", idx, exc)
                    desc = ""
                if desc:
                    chunk = templates[idx]
                    chunk.raw_content = (
                        f"{desc}\n\n【图注】{image_tasks[idx]['caption']}"
                        if image_tasks[idx]["caption"] else desc
                    )
                    chunk_slots[idx] = chunk

        chunks = [c for c in chunk_slots if c is not None]
        t_describe = time.time() - t_desc_start
        timing = {
            "extract": round(t_extract, 1),
            "describe": round(t_describe, 1),
            "total": round(time.time() - t0, 1),
        }
        return chunks, timing

    @staticmethod
    def _describe_single(
        model: str, image_bytes: bytes, ext: str, caption: str, feedback: str | None,
    ) -> str:
        """静态方法：调用 Qwen-VL，可安全用于 ThreadPoolExecutor。
        大图自动缩放至 1024px 以内，减少 API 耗时。"""
        import base64
        import io
        from dashscope import MultiModalConversation

        # 大图缩放：最长边超过 1024px 时等比缩小
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes))
            w, h = img.size
            max_dim = 1024
            if w > max_dim or h > max_dim:
                ratio = max_dim / max(w, h)
                new_size = (int(w * ratio), int(h * ratio))
                img = img.resize(new_size, Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                image_bytes = buf.getvalue()
                ext = "jpeg"
        except Exception:
            pass  # PIL 不可用或格式不支持时保持原图

        mime = f"image/{ext.lower()}" if ext else "image/png"
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        prompt = "描述此论文图片的关键内容，并结合图注总结可检索信息。"
        if caption:
            prompt += f" 图注：{caption}"
        if feedback:
            prompt += f" 评审反馈：{feedback}"

        try:
            response = MultiModalConversation.call(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"image": f"data:{mime};base64,{image_b64}"},
                        {"text": prompt},
                    ],
                }],
                timeout=60,
            )
        except Exception:
            logger.exception("Qwen-VL call failed")
            return ""

        http_status = getattr(response, "status_code", None)
        if http_status is not None and http_status != 200:
            logger.error("Qwen-VL API error %s: %s", getattr(response, "code", ""), getattr(response, "message", ""))
            return ""

        if not hasattr(response, "output") or not response.output:
            return ""

        try:
            content = response.output.choices[0].message.content
            if isinstance(content, list):
                return "\n".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                ).strip()
            return str(content).strip()
        except Exception:
            logger.exception("Qwen-VL response parse failed")
            return ""

    @staticmethod
    def _normalize_bbox(bbox: Any) -> tuple[float, float, float, float] | None:
        """确保 bbox 格式为 (x0, y0, x1, y1)。"""
        if bbox is None:
            return None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _find_caption(
        bbox: tuple[float, float, float, float] | None,
        text_blocks: list[tuple[Any, ...]],
    ) -> str:
        """在图片下方查找图注文字。"""
        if bbox is None:
            return ""
        x0, y0, x1, y1 = bbox
        candidates: list[tuple[float, str]] = []

        for block in text_blocks:
            bx0, by0, bx1, _by1, text, *_rest = block
            cleaned = str(text).strip()
            if not cleaned:
                continue
            horizontal_overlap = min(x1, float(bx1)) - max(x0, float(bx0))
            if horizontal_overlap <= 0:
                continue
            if float(by0) >= y1 and float(by0) - y1 <= CFG.CAPTION_SEARCH_RANGE:
                candidates.append((float(by0) - y1, cleaned))

        candidates.sort(key=lambda item: item[0])
        return " ".join(text for _dist, text in candidates[:CFG.CAPTION_CANDIDATES_MAX]).strip()
