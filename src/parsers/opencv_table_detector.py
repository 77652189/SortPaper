"""
表格横线检测器（双阶段并行策略）。

Stage A — fitz 向量路径检测：
  从 page.get_drawings() 提取深色水平笔画。
  天然过滤填充矩形（图表 bar）和浅灰格线。

Stage B — OpenCV 像素检测：
  将页面渲染为灰度图，用形态学操作提取水平线段。
  能检测到 Stage A 看不到的线（三线表在某些 PDF 中不存在向量路径）。

两个阶段同时运行，结果合并去重（IoU > 0.5 视为同一区域）。

过滤层（两个阶段结果合并后统一应用）：
  1. 页眉/页脚区域排除
  2. 嵌入栅格图片重叠检测（过滤 TIFF/PNG 嵌入的图表）
  3. 文字内容最低阈值（MIN_TEXT_CHARS，兜底）

MIN_LINES_PER_TABLE = 3：三线表恰好 3 条线；两条线的摘要框/图表边框被过滤。

返回值：detect_dpi 像素坐标 bbox 列表 [(x1,y1,x2,y2), ...]。
"""

from __future__ import annotations

import logging

import numpy as np

from src.parsers.config import OPENCV_TABLE as CFG

logger = logging.getLogger(__name__)


class OpenCVTableDetector:
    """
    PDF 页面表格候选区域检测器。

    detect_page(page) → [(x1,y1,x2,y2)...] in detect_dpi 像素坐标。
    bbox_to_pdf_pts(bbox_px) → (x0,y0,x1,y1) in PDF 点坐标。
    """

    def __init__(self, detect_dpi: int = CFG.DETECT_DPI) -> None:
        self.detect_dpi = detect_dpi
        self._scale = detect_dpi / 72.0  # PDF点 → 像素

    # ── 公共接口 ─────────────────────────────────────────────────────────────

    def detect_page(self, page) -> list[tuple[int, int, int, int]]:
        """
        检测单页中的表格区域（Stage A + Stage B 并行，合并去重）。

        Args:
            page: fitz.Page 对象（PyMuPDF）

        Returns:
            bbox列表，格式 (x1, y1, x2, y2)，detect_dpi像素坐标，左上角原点。
        """
        # Stage A: 向量路径检测
        segs_pt = self._detect_from_drawings(page)
        bboxes_a = (
            self._cluster_lines_pt(segs_pt, page.rect.width, page.rect.height)
            if segs_pt
            else []
        )
        if bboxes_a:
            logger.debug("Stage A 检测到 %d 个候选区域", len(bboxes_a))

        # Stage B: OpenCV 像素检测（同时运行，捕获 Stage A 遗漏的情况）
        bboxes_b = self._detect_from_pixels(page)
        if bboxes_b:
            logger.debug("Stage B 检测到 %d 个候选区域", len(bboxes_b))

        # 合并去重
        raw_bboxes = self._merge_bboxes(bboxes_a, bboxes_b)

        # 过滤：嵌入栅格图片区域 + 无文字区域
        result = []
        for b in raw_bboxes:
            bbox_pt = self.bbox_to_pdf_pts(b)
            if self._overlaps_raster_image(page, bbox_pt):
                logger.debug("过滤栅格图片区域 bbox_pt=(%.0f,%.0f,%.0f,%.0f)", *bbox_pt)
                continue
            if not self._has_text_content(page, bbox_pt):
                logger.debug("过滤无文字区域 bbox_pt=(%.0f,%.0f,%.0f,%.0f)", *bbox_pt)
                continue
            result.append(b)
        return result

    def bbox_to_pdf_pts(
        self, bbox_px: tuple[int, int, int, int]
    ) -> tuple[float, float, float, float]:
        """
        将 detect_dpi 像素坐标转换为 fitz/PDF 点坐标（左上角原点，单位 pt）。

        fitz 与 pdfplumber 的 within_bbox 均使用相同的坐标系，
        因此本函数的输出可直接传给两者。
        """
        pts = 72.0 / self.detect_dpi
        x1, y1, x2, y2 = bbox_px
        return (x1 * pts, y1 * pts, x2 * pts, y2 * pts)

    # ── Stage A：向量路径检测 ─────────────────────────────────────────────────

    def _detect_from_drawings(self, page) -> list[tuple[float, float, float]]:
        """
        从页面向量绘图路径中提取深色水平线段。

        过滤规则：
        - 跳过有 fill 的路径（图表 bar、彩色背景）
        - 跳过浅色笔画（brightness > 0.5，即灰色格线）
        - 跳过非近似水平的路径（高度 > 4pt）
        - 跳过宽度不足 MIN_LINE_WIDTH_RATIO 的路径
        - 跳过页眉/页脚区域

        Returns:
            (y_center_pt, x0_pt, x1_pt) 列表，PDF 点坐标。
        """
        page_rect = page.rect
        min_w_pt = CFG.MIN_LINE_WIDTH_RATIO * page_rect.width
        header_y = page_rect.height * CFG.HEADER_ZONE_RATIO
        footer_y = page_rect.height * (1.0 - CFG.FOOTER_ZONE_RATIO)

        results: list[tuple[float, float, float]] = []
        for d in page.get_drawings():
            # 跳过填充区域（图表 bar、彩色矩形）
            if d.get("fill") is not None:
                continue

            # 只保留深色笔画（过滤浅灰格线）
            color = d.get("color")
            if color is not None and isinstance(color, (tuple, list)) and len(color) >= 3:
                brightness = (color[0] + color[1] + color[2]) / 3.0
                if brightness > 0.5:
                    continue

            r = d.get("rect")
            if r is None:
                continue

            x0, y0, x1, y1 = r.x0, r.y0, r.x1, r.y1
            width = x1 - x0
            height = abs(y1 - y0)

            if width < min_w_pt:
                continue
            if height > 4.0:  # 非近似水平（4pt 宽容度）
                continue

            y_center = (y0 + y1) / 2.0
            if not (header_y <= y_center <= footer_y):
                continue  # 页眉/页脚区域

            results.append((y_center, x0, x1))

        return results

    def _cluster_lines_pt(
        self,
        segs_pt: list[tuple[float, float, float]],
        page_w_pt: float,
        page_h_pt: float,
    ) -> list[tuple[int, int, int, int]]:
        """将 pt 单位线段转换为像素后聚类，返回像素 bbox 列表。"""
        s = self._scale
        segs_px = [(int(y * s), int(x0 * s), int(x1 * s)) for (y, x0, x1) in segs_pt]
        return self._cluster_to_bboxes(segs_px, int(page_w_pt * s), int(page_h_pt * s))

    # ── Stage B：OpenCV 像素检测 ──────────────────────────────────────────────

    def _detect_from_pixels(self, page) -> list[tuple[int, int, int, int]]:
        """OpenCV像素级横线检测，捕获向量路径里找不到的表格线。"""
        import cv2
        import fitz

        mat = fitz.Matrix(self.detect_dpi / 72, self.detect_dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)

        _, binary = cv2.threshold(img, CFG.BINARIZE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

        h_kernel_w = max(CFG.H_KERNEL_MIN_WIDTH_PX, int(pix.w * CFG.H_KERNEL_PAGE_RATIO))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_w, 1))
        h_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

        if CFG.DILATE_H_PX > 1:
            dilate_k = cv2.getStructuringElement(
                cv2.MORPH_RECT, (CFG.DILATE_H_PX, CFG.DILATE_V_PX)
            )
            h_mask = cv2.dilate(h_mask, dilate_k, iterations=1)

        contours, _ = cv2.findContours(h_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_w = max(CFG.MIN_LINE_WIDTH_PX, int(pix.w * CFG.MIN_LINE_WIDTH_RATIO))
        line_segs: list[tuple[int, int, int]] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < min_w:
                continue
            line_segs.append((y + h // 2, x, x + w))

        if not line_segs:
            return []

        # 页眉/页脚区域排除
        header_zone_y = int(pix.h * CFG.HEADER_ZONE_RATIO)
        footer_zone_y = int(pix.h * (1.0 - CFG.FOOTER_ZONE_RATIO))
        line_segs = [seg for seg in line_segs if header_zone_y <= seg[0] <= footer_zone_y]

        if not line_segs:
            return []

        return self._cluster_to_bboxes(line_segs, pix.w, pix.h)

    # ── 合并去重 ──────────────────────────────────────────────────────────────

    def _merge_bboxes(
        self,
        bboxes_a: list[tuple[int, int, int, int]],
        bboxes_b: list[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int]]:
        """合并两个 bbox 列表，IoU > 0.5 的视为同一区域，去重保留 Stage A 版本。"""
        if not bboxes_a:
            return list(bboxes_b)
        if not bboxes_b:
            return list(bboxes_a)

        merged = list(bboxes_a)
        for b in bboxes_b:
            if not any(self._iou(b, existing) > 0.5 for existing in merged):
                merged.append(b)
        return merged

    @staticmethod
    def _iou(b1: tuple, b2: tuple) -> float:
        ix1 = max(b1[0], b2[0])
        iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2])
        iy2 = min(b1[3], b2[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (area1 + area2 - inter)

    # ── 过滤方法 ──────────────────────────────────────────────────────────────

    def _overlaps_raster_image(self, page, bbox_pt: tuple) -> bool:
        """若候选区域与嵌入栅格图片高度重叠（≥ IMAGE_OVERLAP_RATIO），返回 True。"""
        import fitz

        x0, y0, x1, y1 = bbox_pt
        detect_rect = fitz.Rect(x0, y0, x1, y1)
        detect_area = detect_rect.width * detect_rect.height
        if detect_area <= 0:
            return False
        for img_info in page.get_image_info(xrefs=False):
            img_rect = fitz.Rect(img_info["bbox"])
            overlap = detect_rect & img_rect
            if not overlap.is_empty:
                overlap_area = overlap.width * overlap.height
                if overlap_area / detect_area >= CFG.IMAGE_OVERLAP_RATIO:
                    return True
        return False

    def _has_text_content(self, page, bbox_pt: tuple) -> bool:
        """检查区域是否含足够文字，用于兜底过滤纯图形区域。"""
        import fitz

        x0, y0, x1, y1 = bbox_pt
        text = page.get_text("text", clip=fitz.Rect(x0, y0, x1, y1)).strip()
        return len(text) >= CFG.MIN_TEXT_CHARS

    # ── 聚类核心 ──────────────────────────────────────────────────────────────

    def _cluster_to_bboxes(
        self,
        line_segs: list[tuple[int, int, int]],
        page_w: int,
        page_h: int,
    ) -> list[tuple[int, int, int, int]]:
        """
        将水平线段聚类为表格候选bbox。

        聚类规则（贪心）：
        1. 按 y 坐标排序
        2. 对每条线段，寻找已存在的簇：若与簇中任一线段的 x 重叠度 > X_OVERLAP_RATIO
           且与簇中最近线段的 y 距离 < max_y_gap_px，则加入该簇
        3. 无法加入任何簇时，新建簇
        4. 簇中横线数 < MIN_LINES_PER_TABLE（=3）或高度不足时丢弃
        """
        line_segs_sorted = sorted(line_segs, key=lambda s: s[0])
        max_y_gap_px = int(CFG.MAX_TABLE_HEIGHT_PT * self._scale)
        min_height_px = int(CFG.MIN_TABLE_HEIGHT_PT * self._scale)

        clusters: list[list[tuple[int, int, int]]] = []

        for seg in line_segs_sorted:
            y, x1, x2 = seg
            placed = False
            for cluster in clusters:
                last_y = max(s[0] for s in cluster)
                if y - last_y > max_y_gap_px:
                    continue
                for cs_y, cs_x1, cs_x2 in cluster:
                    overlap = min(x2, cs_x2) - max(x1, cs_x1)
                    shorter = min(x2 - x1, cs_x2 - cs_x1)
                    if shorter > 0 and overlap / shorter >= CFG.X_OVERLAP_RATIO:
                        cluster.append(seg)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                clusters.append([seg])

        bboxes: list[tuple[int, int, int, int]] = []
        for cluster in clusters:
            if len(cluster) < CFG.MIN_LINES_PER_TABLE:  # 少于3条线直接丢弃
                continue
            y_vals = [s[0] for s in cluster]
            x1_vals = [s[1] for s in cluster]
            x2_vals = [s[2] for s in cluster]

            bx1 = max(0, min(x1_vals))
            by1 = max(0, min(y_vals))
            bx2 = min(page_w, max(x2_vals))
            by2 = min(page_h, max(y_vals))

            if by2 - by1 < min_height_px:
                continue

            bboxes.append((bx1, by1, bx2, by2))
            logger.debug(
                "检测到表格区域 bbox_px=(%d,%d,%d,%d) lines=%d",
                bx1, by1, bx2, by2, len(cluster),
            )

        return bboxes
