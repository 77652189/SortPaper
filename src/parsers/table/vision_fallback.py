from __future__ import annotations

import base64
import logging

from src.parsers.config import TABLE_PARSER as CFG

logger = logging.getLogger(__name__)


def parse_table_with_vision(
    pdf_path: str,
    page_index: int,
    bbox: tuple[float, float, float, float],
) -> list[list[str]]:
    """Render a table bbox and ask GPT Vision for Markdown table rows."""
    try:
        import fitz
        from src.parsers.openai_vision import call_openai_vision

        with fitz.open(pdf_path) as doc:
            page = doc[page_index]
            dpi = CFG.VISION_FALLBACK_DPI
            mat = fitz.Matrix(dpi / 72, dpi / 72)

            x0, y0, x1, y1 = bbox
            margin = CFG.VISION_CAPTURE_MARGIN
            clip = fitz.Rect(
                max(0, x0 - margin),
                max(0, y0 - margin),
                min(page.rect.width, x1 + margin),
                min(page.rect.height, y1 + margin),
            )
            pix = page.get_pixmap(matrix=mat, clip=clip)
            img_bytes = pix.tobytes("png")

        image_b64 = base64.b64encode(img_bytes).decode("utf-8")
        text = call_openai_vision(
            image_b64=image_b64,
            mime_type="image/png",
            prompt=_table_prompt(),
            model=CFG.VISION_FALLBACK_MODEL,
            timeout=120.0,
            max_output_tokens=4096,
        )
        if not text:
            return []

        return markdown_table_to_rows(text.strip())

    except Exception:
        logger.exception("Vision table fallback failed")
        return []


def markdown_table_to_rows(markdown: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in markdown.split("\n"):
        line = line.strip()
        if not line or not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.split("|")]
        if cells and not cells[0]:
            cells = cells[1:]
        if cells and not cells[-1]:
            cells = cells[:-1]
        if all(set(cell) <= {"-", ":", " "} or cell == "" for cell in cells):
            continue
        if cells:
            rows.append(cells)
    return rows


def _table_prompt() -> str:
    return (
        "精确转录图片中的学术论文表格为 Markdown 格式。\n\n"
        "规则：\n"
        "1. 第一行为列名。\n"
        "2. 保留原文中的上下标、斜体、希腊字母等排版格式（用 Markdown 表示）。\n"
        "3. 每个逻辑行只对应表格中的一行，不要把多行数据错误合并。\n"
        "4. 同一单元格内有多个数值（上下排列）时用 ' / ' 连接，保留括号或方括号中的单位/条件标注。\n"
        "5. 空单元格填 '—'。\n"
        "6. 表格底部的脚注或说明文字（以特殊符号开头的短行）原样保留。\n"
        "7. 不要包含表格上方或下方的正文段落。\n"
        "8. 不要编造、省略或改写任何数据。\n\n"
        "只返回 Markdown 表格和注释，不加额外说明。"
    )
