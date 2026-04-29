"""
统一分块数据结构：LayoutChunk
每个 chunk 携带内容类型、位置信息（bbox）、页码、栏位，
保证跨 Worker 合并后仍能重建原始阅读顺序。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# ─── 内容类型枚举 ──────────────────────────────────────────────────────────────


ContentType = Literal["text", "table", "image"]


# ─── 统一栏位推断 ──────────────────────────────────────────────────────────────


def infer_column(page_width: float, x0: float, x1: float | None = None) -> int:
    """推断 bbox 属于左栏(0) / 右栏(1) / 通栏(2)。

    - page_width <= 400：单栏版式，直接返回通栏
    - x1 provided 且 (x1-x0) > 60% page_width：跨栏元素，返回通栏
    - 否则按 x0 与中线比较
    """
    if page_width <= 400:
        return 2
    if x1 is not None and (x1 - x0) > 0.6 * page_width:
        return 2
    x_mid = page_width / 2
    return 0 if x0 < x_mid else 1


# ─── 核心数据结构 ──────────────────────────────────────────────────────────────


@dataclass
class LayoutChunk:
    """
    统一分块单元。

    - content_type: 文字/表格/图片，决定后续如何存储和检索
    - bbox: 原始 PDF 中的坐标 (x0, y0, x1, y1)，用于排序和定位
    - column: 0=左栏, 1=右栏, 2=通栏（由 page_width + x0 推断）
    - order_in_page: 本页内的阅读顺序（按 y0 排序后得到）
    - raw_content: 解析器原始输出内容（纯文本/Markdown/图片描述）
    - metadata: 各解析器的附加信息（行高、字体、行列数等）
    """
    content_type: ContentType
    raw_content: str
    page: int
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    column: int                               # 0=left, 1=right, 2=full_width
    order_in_page: int                        # 本页阅读顺序
    chunk_id: str = ""                        # 全局唯一 ID
    global_order: int = -1                    # 跨页全局阅读顺序（merge 后填写）
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            x0, y0, x1, y1 = self.bbox
            self.chunk_id = (
                f"{self.content_type}_p{self.page}_"
                f"col{self.column}_y{int(y0)}_x{int(x0)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content_type": self.content_type,
            "raw_content": self.raw_content,
            "page": self.page,
            "bbox": self.bbox,
            "column": self.column,
            "order_in_page": self.order_in_page,
            "global_order": self.global_order,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LayoutChunk:
        return cls(
            chunk_id=d["chunk_id"],
            content_type=d["content_type"],
            raw_content=d["raw_content"],
            page=d["page"],
            bbox=tuple(d["bbox"]),
            column=d["column"],
            order_in_page=d["order_in_page"],
            global_order=d.get("global_order", -1),
            metadata=d.get("metadata", {}),
        )


# ─── LayoutDeduplicator ───────────────────────────────────────────────────────


class LayoutDeduplicator:
    """
    基于 bbox 重叠检测 + 内容相似度的去重器。

    策略：
    1. 计算两两 chunk 之间的 IoU
    2. IoU > IOU_THRESHOLD 且内容高度相似 → 判定为重复
    3. 保留优先级：table > text > image（同区域数据类优先）
    4. 如果内容不相似（IoU 高但是不同内容，如图片+周围文字），保留两者
    """

    # IoU 阈值：超过此值认为空间上重叠
    IOU_THRESHOLD = 0.5
    # 内容相似度阈值：超过此值认为是同一内容
    CONTENT_SIMILARITY_THRESHOLD = 0.8

    # 优先级：数字越小优先级越高
    PRIORITY: dict[str, int] = {
        "table": 0,  # 表格数据优先
        "text": 1,   # 纯文本次之
        "image": 2,  # 图片描述最后
    }

    @classmethod
    def deduplicate(cls, chunks: list[LayoutChunk]) -> list[LayoutChunk]:
        """
        对 chunks 去重，返回去重后的列表。

        流程：
        1. 按页分组（跨页不会重叠）
        2. 同页内两两比较 IoU
        3. IoU > 阈值 且 内容相似 → 标记重复
        4. 保留高优先级 chunk
        """
        if len(chunks) <= 1:
            return chunks

        # 按页分组
        pages: dict[int, list[LayoutChunk]] = {}
        for chunk in chunks:
            pages.setdefault(chunk.page, []).append(chunk)

        result: list[LayoutChunk] = []
        for page_chunks in pages.values():
            deduped = cls._deduplicate_page(page_chunks)
            result.extend(deduped)

        return result

    @classmethod
    def _deduplicate_page(cls, page_chunks: list[LayoutChunk]) -> list[LayoutChunk]:
        """单页去重：两两比较，标记重复。"""
        n = len(page_chunks)
        if n <= 1:
            return page_chunks

        # 标记数组：False = 保留，True = 删除
        to_remove: set[int] = set()

        for i in range(n):
            if i in to_remove:
                continue
            for j in range(i + 1, n):
                if j in to_remove:
                    continue

                chunk_i = page_chunks[i]
                chunk_j = page_chunks[j]

                # 计算 IoU
                iou = cls._compute_iou(chunk_i.bbox, chunk_j.bbox)
                if iou <= cls.IOU_THRESHOLD:
                    continue

                # 计算内容相似度
                similarity = cls._content_similarity(
                    chunk_i.raw_content, chunk_j.raw_content
                )

                # 高重叠 + 高相似度 → 判定为重复
                if iou > cls.IOU_THRESHOLD and similarity > cls.CONTENT_SIMILARITY_THRESHOLD:
                    # 保留高优先级，删除低优先级
                    keep, drop = (
                        (i, j) if cls.PRIORITY.get(chunk_i.content_type, 99) <= cls.PRIORITY.get(chunk_j.content_type, 99)
                        else (j, i)
                    )
                    to_remove.add(drop)

        return [chunk for idx, chunk in enumerate(page_chunks) if idx not in to_remove]

    @staticmethod
    def _compute_iou(
        bbox1: tuple[float, float, float, float],
        bbox2: tuple[float, float, float, float],
    ) -> float:
        """
        计算两个 bbox 的 IoU (Intersection over Union)。

        bbox 格式: (x0, y0, x1, y1)
        """
        x0_1, y0_1, x1_1, y1_1 = bbox1
        x0_2, y0_2, x1_2, y1_2 = bbox2

        # 计算交集区域
        inter_x0 = max(x0_1, x0_2)
        inter_y0 = max(y0_1, y0_2)
        inter_x1 = min(x1_1, x1_2)
        inter_y1 = min(y1_1, y1_2)

        # 无交集
        if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
            return 0.0

        inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)

        # 计算各自面积
        area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        area2 = (x1_2 - x0_2) * (y1_2 - y0_2)

        # 计算并集面积
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    @staticmethod
    def _content_similarity(text1: str, text2: str) -> float:
        """
        简单的内容相似度计算：基于字符集合的 Jaccard 相似度。

        适用于检测同一内容的轻微变体（如 pdfplumber 输出 vs PyMuPDF 输出）。
        实际生产可用 embedding cosine similarity 替代。
        """
        if not text1 or not text2:
            return 0.0

        # 归一化：转小写，去空格
        norm1 = set(text1.lower().split())
        norm2 = set(text2.lower().split())

        if not norm1 or not norm2:
            return 0.0

        intersection = len(norm1 & norm2)
        union = len(norm1 | norm2)

        return intersection / union if union > 0 else 0.0


# ─── LayoutMerger ─────────────────────────────────────────────────────────────


class LayoutMerger:
    """
    合并三个 Worker 返回的 LayoutChunk 列表，
    按 page → column → y0 → x0 排序，重建原始阅读顺序。
    """

    @staticmethod
    def merge(chunks: list[LayoutChunk]) -> list[LayoutChunk]:
        """
        对所有 chunks 去重 + 排序，返回去重后的列表。
        """
        if not chunks:
            return []

        # 1. 去重
        deduped = LayoutDeduplicator.deduplicate(chunks)

        # 2. 按 page → column → y0 → x0 排序
        sorted_chunks = sorted(
            deduped,
            key=lambda c: (c.page, c.column, c.bbox[1], c.bbox[0]),
        )

        # 3. 重新分配 order_in_page（跨页清零）和全局 global_order
        last_page = -1
        counter = 0
        for g_idx, chunk in enumerate(sorted_chunks):
            if chunk.page != last_page:
                last_page = chunk.page
                counter = 0
            chunk.order_in_page = counter
            chunk.global_order = g_idx
            counter += 1

        return sorted_chunks

    @staticmethod
    def build_chunk_id(chunk: LayoutChunk) -> str:
        """生成稳定 chunk ID：跨重试时同位置的 chunk 共享同一 ID。"""
        return chunk.chunk_id
