"""
图像解析器：使用 PyMuPDF 提取图片 + Qwen-VL 描述，包装为 LayoutChunk。
每个图片作为一个 chunk，携带 bbox 坐标和描述文本。
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

from src.parsers.layout_chunk import LayoutChunk, infer_column
from src.parsers.config import VISION as CFG

logger = logging.getLogger(__name__)

# （面积/图注阈值已移至 src/parsers/config.py VisionParserConfig）


class VisionParser:
    def __init__(self, pdf_path: str | Path, model: str = "qwen-vl-max") -> None:
        self.pdf_path = str(pdf_path)
        self.model = model

    def parse(self, feedback: str | None = None) -> list[LayoutChunk]:
        """
        提取 PDF 所有图片并生成描述，返回 LayoutChunk 列表。
        前置粗筛：bbox 面积 < _MIN_AREA_PT2 或像素面积 < _MIN_PIXEL_AREA 的图片跳过 VL 调用。
        图注合并进 raw_content，便于检索时同时命中描述和图注。
        """
        import fitz

        chunks: list[LayoutChunk] = []

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

                    # 前置粗筛：PDF 坐标面积过小则跳过
                    x0, y0, x1, y1 = bbox
                    if (x1 - x0) * (y1 - y0) < CFG.MIN_AREA_PT2:
                        continue

                    extracted = doc.extract_image(xref)
                    image_bytes = extracted.get("image")
                    if not image_bytes:
                        continue

                    # 像素面积粗筛
                    img_w = extracted.get("width", 0)
                    img_h = extracted.get("height", 0)
                    if img_w * img_h < CFG.MIN_PIXEL_AREA:
                        continue

                    caption = self._find_caption(bbox, text_blocks)
                    img_ext = extracted.get("ext", "png")
                    description = self._describe_image(image_bytes, img_ext, caption, feedback)
                    if not description:
                        continue

                    # 将图注原文追加到 raw_content，便于精确检索
                    raw_content = f"{description}\n\n【图注】{caption}" if caption else description

                    column = infer_column(page_width, x0, x1)

                    chunks.append(
                        LayoutChunk(
                            content_type="image",
                            raw_content=raw_content,
                            page=page_index,
                            bbox=bbox,
                            column=column,
                            order_in_page=img_idx,
                            metadata={
                                "parser": "qwen-vl-max",
                                "caption": caption,
                                "image_index": img_idx,
                                "xref": xref,
                            },
                        )
                    )

        return chunks

    def _describe_image(
        self, image_bytes: bytes, ext: str, caption: str, feedback: str | None
    ) -> str:
        """调用 Qwen-VL 生成图片描述。失败时记录日志并返回空字符串。"""
        from dashscope import MultiModalConversation

        mime = f"image/{ext.lower()}" if ext else "image/png"
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        prompt = CFG.VL_PROMPT
        if caption:
            prompt += f" 图注：{caption}"
        if feedback:
            prompt += f" 评审反馈：{feedback}"

        try:
            response = MultiModalConversation.call(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"image": f"data:{mime};base64,{image_b64}"},
                            {"text": prompt},
                        ],
                    }
                ],
            )
        except Exception:
            logger.exception("Qwen-VL call failed")
            return ""

        # DashScope 成功时 response.code 为空字符串，真正的 HTTP 状态在 response.status_code
        http_status = getattr(response, "status_code", None)
        if http_status is not None and http_status != 200:
            logger.error("Qwen-VL API error %s: %s", getattr(response, "code", ""), getattr(response, "message", ""))
            return ""

        if not hasattr(response, "output") or not response.output:
            logger.error("Qwen-VL returned empty output")
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
