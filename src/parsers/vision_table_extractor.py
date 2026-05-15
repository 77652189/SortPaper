"""
视觉模型表格内容提取器。

职责拆分为两个阶段（便于线程安全）：
  render_crop(page, bbox_detect_px) → bytes
      在调用方持有 fitz.Document 的串行阶段调用，将表格区域裁剪为 PNG/JPEG 字节。

  call_api(image_bytes) → str
      无状态，可在 ThreadPoolExecutor 中并发调用，返回 Markdown 表格字符串。

坐标换算说明：
  detect_dpi=150 时，1px = 72/150 ≈ 0.48 pt
  clip 参数使用 fitz.Rect（PDF点，左上角原点），与 detect_dpi 像素坐标的换算：
    clip_pt = bbox_px * (72 / detect_dpi)
"""

from __future__ import annotations

import base64
import io
import logging

from src.parsers.config import OPENCV_TABLE as CFG

logger = logging.getLogger(__name__)

_EXTRACT_PROMPT = (
    "Extract the table from this image and return it as a Markdown table. "
    "Preserve all superscripts, subscripts, and special characters exactly as they appear. "
    "If merged cells exist, duplicate the content across the affected rows or columns. "
    "Output ONLY the Markdown table — no explanations, no surrounding text."
)


class VisionTableExtractor:
    """
    两阶段表格提取器：裁剪渲染（串行）→ API调用（可并发）。
    """

    def __init__(
        self,
        model: str = CFG.VISION_MODEL,
        detect_dpi: int = CFG.DETECT_DPI,
        extract_dpi: int = CFG.EXTRACT_DPI,
        margin_ratio: float = CFG.MARGIN_RATIO,
        max_dim: int = CFG.MAX_IMAGE_DIM,
    ) -> None:
        self.model = model
        self.detect_dpi = detect_dpi
        self.extract_dpi = extract_dpi
        self.margin_ratio = margin_ratio
        self.max_dim = max_dim

    # ── Phase 1：裁剪渲染（需要 fitz.Page，必须串行调用） ────────────────────

    def render_crop(
        self,
        page,
        bbox_detect_px: tuple[int, int, int, int],
    ) -> bytes:
        """
        以 extract_dpi 高分辨率渲染表格裁剪区域，返回 PNG/JPEG 字节。

        Args:
            page: fitz.Page 对象
            bbox_detect_px: 表格在 detect_dpi 坐标系下的像素 bbox (x1,y1,x2,y2)

        Returns:
            图片字节（JPEG，如果 PIL 不可用则回退到 PNG）。失败时返回空字节。
        """
        import fitz

        x1, y1, x2, y2 = bbox_detect_px
        pts = 72.0 / self.detect_dpi  # 检测像素 → PDF点

        # 计算带边距的裁剪区域（PDF点坐标）
        margin_x = (x2 - x1) * self.margin_ratio * pts
        margin_y = (y2 - y1) * self.margin_ratio * pts

        clip = fitz.Rect(
            max(0.0, x1 * pts - margin_x),
            max(0.0, y1 * pts - margin_y),
            min(page.rect.width,  x2 * pts + margin_x),
            min(page.rect.height, y2 * pts + margin_y),
        )

        mat = fitz.Matrix(self.extract_dpi / 72, self.extract_dpi / 72)
        try:
            pix = page.get_pixmap(matrix=mat, clip=clip)
        except Exception:
            logger.exception("渲染裁剪区域失败 bbox=%s", bbox_detect_px)
            return b""

        return self._pix_to_bytes(pix)

    # ── Phase 2：API调用（无 fitz，可并发） ──────────────────────────────────

    def call_api(self, image_bytes: bytes) -> str:
        """
        将图片字节发送给 Qwen-VL-Max，返回提取的 Markdown 表格字符串。
        失败时返回空字符串。
        """
        if not image_bytes:
            return ""

        image_bytes = self._maybe_resize(image_bytes)
        return self._call_dashscope(image_bytes)

    # ── 内部方法 ─────────────────────────────────────────────────────────────

    def _pix_to_bytes(self, pix) -> bytes:
        """将 PyMuPDF Pixmap 转为 JPEG 字节（PIL 可用）或 PNG 字节（回退）。"""
        try:
            from PIL import Image
            mode = "RGB" if pix.n == 3 else ("RGBA" if pix.n == 4 else "L")
            img = Image.frombytes(mode, (pix.w, pix.h), pix.samples)
            if mode == "RGBA":
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            return buf.getvalue()
        except Exception:
            # PIL 不可用时直接用 PyMuPDF 输出 PNG
            try:
                return pix.tobytes("png")
            except Exception:
                logger.exception("Pixmap转字节失败")
                return b""

    def _maybe_resize(self, image_bytes: bytes) -> bytes:
        """最长边超过 max_dim 时等比缩放，降低 API 延迟和成本。"""
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes))
            w, h = img.size
            if w <= self.max_dim and h <= self.max_dim:
                return image_bytes
            ratio = self.max_dim / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            return buf.getvalue()
        except Exception:
            return image_bytes

    def _call_dashscope(self, image_bytes: bytes) -> str:
        """调用 DashScope MultiModalConversation API。"""
        from dashscope import MultiModalConversation

        # 判断 MIME 类型（JPEG 以 0xFF 0xD8 开头）
        ext = "jpeg" if len(image_bytes) >= 2 and image_bytes[:2] == b"\xff\xd8" else "png"
        mime = f"image/{ext}"
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        try:
            response = MultiModalConversation.call(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"image": f"data:{mime};base64,{b64}"},
                        {"text": _EXTRACT_PROMPT},
                    ],
                }],
                timeout=60,
            )
        except Exception:
            logger.exception("Qwen-VL表格提取API调用失败")
            return ""

        http_status = getattr(response, "status_code", None)
        if http_status is not None and http_status != 200:
            logger.error(
                "Qwen-VL API错误 code=%s msg=%s",
                getattr(response, "code", ""),
                getattr(response, "message", ""),
            )
            return ""

        try:
            content = response.output.choices[0].message.content
            if isinstance(content, list):
                return "\n".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                ).strip()
            return str(content).strip()
        except Exception:
            logger.exception("Qwen-VL响应解析失败")
            return ""
