from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ParseResult:
    content: str
    metadata: dict[str, Any]


class VisionParser:
    def __init__(self, pdf_path: str | Path, model: str = "qwen-vl-max") -> None:
        self.pdf_path = str(pdf_path)
        self.model = model

    def parse(self, feedback: str | None = None) -> ParseResult:
        import fitz

        descriptions: list[str] = []
        image_count = 0

        with fitz.open(self.pdf_path) as doc:
            for page_index, page in enumerate(doc, start=1):
                text_blocks = page.get_text("blocks")
                image_entries = page.get_image_info(xrefs=True)
                for image_index, image_info in enumerate(image_entries, start=1):
                    xref = image_info.get("xref")
                    if not xref:
                        continue
                    extracted = doc.extract_image(xref)
                    image_bytes = extracted.get("image")
                    if not image_bytes:
                        continue
                    caption = self._find_caption(image_info, text_blocks)
                    description = self._describe_image(image_bytes, caption, feedback)
                    descriptions.append(
                        f"## Page {page_index} Image {image_index}\nCaption: {caption or 'N/A'}\nDescription: {description}"
                    )
                    image_count += 1

        return ParseResult(
            content="\n\n".join(descriptions),
            metadata={
                "image_count": image_count,
                "parser": "qwen-vl-max",
                "feedback_used": bool(feedback),
            },
        )

    def _describe_image(self, image_bytes: bytes, caption: str, feedback: str | None) -> str:
        from dashscope import MultiModalConversation

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        prompt = "描述此生物论文图片的关键内容，并结合图注总结可检索信息。"
        if caption:
            prompt += f" 图注：{caption}"
        if feedback:
            prompt += f" 评审反馈：{feedback}"

        response = MultiModalConversation.call(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"image": f"data:image/png;base64,{image_b64}"},
                        {"text": prompt},
                    ],
                }
            ],
        )

        content = response.output.choices[0].message.content
        if isinstance(content, list):
            return "\n".join(part.get("text", "") for part in content if isinstance(part, dict)).strip()
        return str(content).strip()

    @staticmethod
    def _find_caption(image_info: dict[str, Any], text_blocks: list[tuple[Any, ...]]) -> str:
        bbox = image_info.get("bbox") or ()
        if len(bbox) != 4:
            return ""
        x0, y0, x1, y1 = [float(value) for value in bbox]
        candidates: list[tuple[float, str]] = []
        for block in text_blocks:
            bx0, by0, bx1, _by1, text, *_rest = block
            cleaned = str(text).strip()
            if not cleaned:
                continue
            horizontal_overlap = min(x1, float(bx1)) - max(x0, float(bx0))
            if horizontal_overlap <= 0:
                continue
            if float(by0) >= y1 and float(by0) - y1 <= 120:
                candidates.append((float(by0) - y1, cleaned))
        candidates.sort(key=lambda item: item[0])
        return " ".join(text for _distance, text in candidates[:2]).strip()
