"""
统一分块数据结构：LayoutChunk
每个 chunk 携带内容类型、位置信息（bbox）、页码、栏位，
保证跨 Worker 合并后仍能重建原始阅读顺序。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from src.parsers.config import LAYOUT as CFG


# ─── 内容类型枚举 ──────────────────────────────────────────────────────────────


ContentType = Literal["text", "table", "image"]


# ─── 统一栏位推断 ──────────────────────────────────────────────────────────────


def infer_column(page_width: float, x0: float, x1: float | None = None) -> int:
    """推断 bbox 属于左栏(0) / 右栏(1) / 通栏(2)。

    - x1 provided 且 (x1-x0) > 60% page_width：跨栏元素，返回通栏
    - 否则按 x0 与中线比较（左栏 < 中线，右栏 >= 中线）
    - 单栏版式：所有元素宽度都会 > 60% page_width，自动被归为通栏(2)
    """
    if x1 is not None and (x1 - x0) > CFG.FULL_WIDTH_RATIO * page_width:
        return 2  # 跨栏元素
    x_mid = page_width * 0.5
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
                f"col{self.column}_y{y0:.1f}_x{x0:.1f}"
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
    IOU_THRESHOLD = CFG.IOU_THRESHOLD
    # 内容相似度阈值：超过此值认为是同一内容
    CONTENT_SIMILARITY_THRESHOLD = CFG.CONTENT_SIMILARITY_THRESHOLD

    # 优先级：数字越小优先级越高
    PRIORITY: dict[str, int] = {"table": 0, "text": 1, "image": 2}

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
    def _is_contained_in(
        cls,
        inner: tuple[float, float, float, float],
        outer: tuple[float, float, float, float],
        threshold: float = CFG.CONTAINED_RATIO,
    ) -> bool:
        """判断 inner bbox 有多少比例落在 outer bbox 内（用于 text 被表格包含的检测）。"""
        ix0, iy0, ix1, iy1 = inner
        ox0, oy0, ox1, oy1 = outer

        inter_x0 = max(ix0, ox0)
        inter_y0 = max(iy0, oy0)
        inter_x1 = min(ix1, ox1)
        inter_y1 = min(iy1, oy1)

        if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
            return False

        inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
        inner_area = (ix1 - ix0) * (iy1 - iy0)

        return (inter_area / inner_area) >= threshold if inner_area > 0 else False

    @staticmethod
    def _text_starts_in_table(
        text_bbox: tuple[float, float, float, float],
        table_bbox: tuple[float, float, float, float],
    ) -> bool:
        """判断 text bbox 的顶部（y0）是否在 table bbox 的 y 范围内。
        用于检测整列 text chunk 始于表格区域但延伸到下方的情况。
        不受 +/- tolerance 影响，避免误吞表格下方的独立正文。"""
        tx0, ty0, tx1, ty1 = text_bbox
        ox0, oy0, ox1, oy1 = table_bbox

        # text 顶部在表格 y 范围内
        if not (oy0 <= ty0 <= oy1):
            return False
        # x 方向有重叠
        if tx1 <= ox0 or tx0 >= ox1:
            return False
        return True

    @staticmethod

    @staticmethod
    def _center_in_bbox(
        inner: tuple[float, float, float, float],
        outer: tuple[float, float, float, float],
        tolerance: float = 20.0,
    ) -> bool:
        """判断 inner 的中心点是否在 outer（含容差扩展）内。
        用于处理旋转标签（如 "LNT II"）等位于表格 bbox 外侧但仍属表格内容的场景。"""
        cx = (inner[0] + inner[2]) / 2
        cy = (inner[1] + inner[3]) / 2
        return (
            outer[0] - tolerance <= cx <= outer[2] + tolerance
            and outer[1] - tolerance <= cy <= outer[3] + tolerance
        )

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

                # text vs table：用包含检测（表格内文本行 IoU 低，但被表格 bbox 包含）
                if {chunk_i.content_type, chunk_j.content_type} == {"text", "table"}:
                    table_c = chunk_i if chunk_i.content_type == "table" else chunk_j
                    text_idx = i if chunk_i.content_type == "text" else j
                    if cls._is_contained_in(page_chunks[text_idx].bbox, table_c.bbox, threshold=CFG.CONTAINED_RATIO) \
                            or cls._center_in_bbox(page_chunks[text_idx].bbox, table_c.bbox):
                        to_remove.add(text_idx)
                    continue

                # 其他情况：用 IoU + 内容相似度
                iou = cls._compute_iou(chunk_i.bbox, chunk_j.bbox)
                if iou <= cls.IOU_THRESHOLD:
                    continue

                similarity = cls._content_similarity(
                    chunk_i.raw_content, chunk_j.raw_content
                )

                if iou > cls.IOU_THRESHOLD and similarity > cls.CONTENT_SIMILARITY_THRESHOLD:
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
        内容相似度计算：根据文本长度自适应选择方法。

        - 短文本（词数 < 10）：使用字符级 Jaccard，对短文本更鲁棒
        - 长文本（词数 >= 10）：使用词级 Jaccard
        """
        if not text1 or not text2:
            return 0.0

        # 归一化：转小写
        norm1 = text1.lower()
        norm2 = text2.lower()

        words1 = norm1.split()
        words2 = norm2.split()

        # 短文本：使用字符级 Jaccard（更鲁棒）
        if len(words1) < 10 or len(words2) < 10:
            def _char_set(s: str, n: int = 3) -> set[str]:
                """生成字符 n-gram 集合"""
                s = f" {s} "  # 填充以捕获边界
                if len(s) < n + 2:
                    return set(s.strip())  # 极短文本直接用字符集合
                return set(s[i:i + n] for i in range(len(s) - n))

            set1 = _char_set(norm1)
            set2 = _char_set(norm2)
        else:
            # 长文本：使用词级 Jaccard
            set1 = set(words1)
            set2 = set(words2)

        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0


# 章节标题关键词集合（防止合并时吞掉章节头）
_SECTION_STARTS: set[str] = {
    "ABSTRACT", "ABSTRACT.", "ABSTRACT:",
    "KEYWORDS", "KEYWORDS.", "KEYWORDS:",
    "INTRODUCTION", "INTRODUCTION.", "INTRODUCTION:",
    "METHODS", "METHODS.", "METHODS:",
    "RESULTS", "RESULTS.", "RESULTS:",
    "DISCUSSION", "DISCUSSION.", "DISCUSSION:",
    "CONCLUSION", "CONCLUSIONS", "CONCLUSION.", "CONCLUSIONS.",
    "REFERENCES", "REFERENCES.", "REFERENCES:",
    "BACKGROUND", "BACKGROUND.", "BACKGROUND",
    "ACKNOWLEDGEMENTS", "ACKNOWLEDGEMENTS:",
    "ACKNOWLEDGMENTS", "ACKNOWLEDGMENTS:",
    "FUNDING", "DISCLOSURE",
    "摘要", "关键词", "参考文献", "致谢", "基金",
}


# ─── LayoutMerger ─────────────────────────────────────────────────────────────


class LayoutMerger:
    """
    合并三个 Worker 返回的 LayoutChunk 列表，
    按 page → y0 → 栏优先级 → x0 排序，重建原始阅读顺序。

    栏优先级：通栏(2)在页面顶部时排最前，在页面底部时排最后；
              左栏(0)和右栏(1)按正常顺序排序。
    """

    _SECTION_STARTS: set[str] = _SECTION_STARTS


    @staticmethod
    def _is_section_start(text: str) -> bool:
        """判断文本是否像章节标题（如 ABSTRACT、INTRODUCTION 等）。"""
        first_token = text.strip().split()[0].rstrip(":.").strip().upper()
        return first_token in LayoutMerger._SECTION_STARTS

    @staticmethod
    def _merge_header_blocks(
        chunks: list[LayoutChunk],
        y_gap_threshold: float = CFG.HEADER_MERGE_Y_GAP,
        header_zone_ratio: float = CFG.HEADER_ZONE_RATIO,
    ) -> list[LayoutChunk]:
        """
        合并同页顶部区域内、y 间距小于阈值的相邻文字块。
        支持通栏及同栏（左栏/右栏）内的合并。
        """
        if not chunks:
            return chunks

        result: list[LayoutChunk] = []
        i = 0
        while i < len(chunks):
            current = chunks[i]

            page_height = current.metadata.get("page_height", 0)
            in_header_zone = (
                page_height > 0
                and current.bbox[1] < page_height * header_zone_ratio
            )

            # 标题通常词数少，不触发合并
            is_title_like = len(current.raw_content.split()) < CFG.TITLE_WORD_LIMIT

            if (
                current.content_type == "text"
                and in_header_zone
                and not is_title_like
                and i + 1 < len(chunks)
            ):
                next_chunk = chunks[i + 1]
                same_page = next_chunk.page == current.page
                next_in_zone = (
                    page_height > 0
                    and next_chunk.bbox[1] < page_height * header_zone_ratio
                )
                y_gap = next_chunk.bbox[1] - current.bbox[3]

                # 允许合并：同栏，或至少一个是通栏
                same_column = current.column == next_chunk.column
                one_is_full = current.column == 2 or next_chunk.column == 2
                can_merge = same_column or one_is_full

                if (
                    same_page
                    and can_merge
                    and next_chunk.content_type == "text"
                    and next_in_zone
                    and 0 <= y_gap <= y_gap_threshold
                    and not LayoutMerger._is_section_start(next_chunk.raw_content)
                ):
                    merged_content = current.raw_content + "\n" + next_chunk.raw_content
                    merged_bbox = (
                        min(current.bbox[0], next_chunk.bbox[0]),
                        current.bbox[1],
                        max(current.bbox[2], next_chunk.bbox[2]),
                        next_chunk.bbox[3],
                    )
                    # 合并后的 column 取最大值（通栏=2 优先级最高）
                    merged_column = max(current.column, next_chunk.column)
                    current = LayoutChunk(
                        content_type=current.content_type,
                        raw_content=merged_content,
                        page=current.page,
                        bbox=merged_bbox,
                        column=merged_column,
                        order_in_page=current.order_in_page,
                        metadata=current.metadata,
                    )
                    i += 2
                    result.append(current)
                    continue

            result.append(current)
            i += 1

        return result

    @staticmethod
    def merge(chunks: list[LayoutChunk]) -> list[LayoutChunk]:
        """
        对所有 chunks 去重 + 排序，返回去重后的列表。
        """
        if not chunks:
            return []

        # 1. 去重
        deduped = LayoutDeduplicator.deduplicate(chunks)

        # 2. 按页面分组处理
        by_page: dict[int, list[LayoutChunk]] = {}
        for c in deduped:
            by_page.setdefault(c.page, []).append(c)

        result: list[LayoutChunk] = []
        for page in sorted(by_page.keys()):
            page_chunks = by_page[page]

            # 区分左栏、右栏、双栏元素
            left_chunks = [c for c in page_chunks if c.column == 0]
            right_chunks = [c for c in page_chunks if c.column == 1]
            twin_chunks = left_chunks + right_chunks

            if twin_chunks:
                min_y = min(c.bbox[1] for c in twin_chunks)
                max_y = max(c.bbox[3] for c in twin_chunks)
            else:
                min_y = max_y = None

            # footer 判定：基于 page_height 比例，不依赖 shorter_end
            page_height = page_chunks[0].metadata.get("page_height", 0)
            footer_threshold = page_height * CFG.FOOTER_THRESHOLD_RATIO  # 页面下 15% 视为 footer

            # 建立左栏 y 区间列表，用于判断右栏是否为 header 侧边栏
            left_y_ranges = [(c.bbox[1], c.bbox[3]) for c in left_chunks]

            def _has_left_content_near(y0: float, tolerance: float = CFG.LEFT_CONTENT_TOLERANCE) -> bool:
                """判断 y0 附近是否有左栏内容（用于识别 header 侧边栏）"""
                for ly0, ly1 in left_y_ranges:
                    if abs(ly0 - y0) < tolerance or ly0 <= y0 <= ly1:
                        return True
                return False

            # 排序键：(bucket, y0, sub_priority, x0)
            def _sort_key(c: LayoutChunk):
                y0 = c.bbox[1]

                # footer 判定：页面下 15% 区域
                if page_height > 0 and y0 >= footer_threshold:
                    return (3, y0, 0, c.bbox[0])

                if c.column == 2:  # 通栏
                    if min_y is None or y0 < min_y:
                        return (0, y0, 0, c.bbox[0])  # 顶部通栏
                    elif c.bbox[3] > max_y:
                        return (3, y0, 0, c.bbox[0])  # 底部通栏
                    else:
                        return (1, y0, 1, c.bbox[0])  # 中间通栏
                elif c.column == 0:
                    return (1, y0, 0, c.bbox[0])  # 左栏
                else:  # col1
                    if not left_y_ranges or not _has_left_content_near(y0):
                        # 附近无左栏内容 → header 侧边栏，按 y 插入 bucket 1
                        return (1, y0, 1, c.bbox[0])
                    else:
                        # 正文右栏
                        return (2, y0, 0, c.bbox[0])

            page_chunks.sort(key=_sort_key)
            result.extend(page_chunks)

        # 合并页面顶部相邻的通栏文字块（修复作者+机构被拆块的问题）
        result = LayoutMerger._merge_header_blocks(result)

        # 重新分配 order_in_page（跨页清零）和全局 global_order
        last_page = -1
        counter = 0
        for g_idx, chunk in enumerate(result):
            if chunk.page != last_page:
                last_page = chunk.page
                counter = 0
            chunk.order_in_page = counter
            chunk.global_order = g_idx
            counter += 1

        return result
