"""
单元测试：LayoutDeduplicator 去重逻辑
"""

import pytest

from src.parsers.layout_chunk import LayoutChunk, LayoutDeduplicator


class TestIoU:
    """测试 IoU 计算"""

    def test_no_overlap(self):
        """两个 bbox 完全不重叠"""
        bbox1 = (0, 0, 10, 10)  # 左上角 10x10
        bbox2 = (20, 20, 30, 30)  # 右下角 10x10
        iou = LayoutDeduplicator._compute_iou(bbox1, bbox2)
        assert iou == 0.0

    def test_full_overlap(self):
        """两个 bbox 完全重叠"""
        bbox1 = (0, 0, 10, 10)
        bbox2 = (0, 0, 10, 10)
        iou = LayoutDeduplicator._compute_iou(bbox1, bbox2)
        assert iou == 1.0

    def test_partial_overlap(self):
        """两个 bbox 部分重叠"""
        # bbox1: (0, 0) -> (10, 10)，面积=100
        # bbox2: (5, 5) -> (15, 15)，面积=100
        # 交集: (5, 5) -> (10, 10)，面积=25
        # 并集: 100 + 100 - 25 = 175
        # IoU = 25 / 175 ≈ 0.143
        bbox1 = (0, 0, 10, 10)
        bbox2 = (5, 5, 15, 15)
        iou = LayoutDeduplicator._compute_iou(bbox1, bbox2)
        assert abs(iou - 25 / 175) < 0.01

    def test_one_contains_other(self):
        """一个 bbox 完全包含另一个"""
        # bbox1: (0, 0) -> (20, 20)，面积=400
        # bbox2: (5, 5) -> (15, 15)，面积=100
        # 交集 = bbox2 = 100
        # 并集 = 400
        # IoU = 100 / 400 = 0.25
        bbox1 = (0, 0, 20, 20)
        bbox2 = (5, 5, 15, 15)
        iou = LayoutDeduplicator._compute_iou(bbox1, bbox2)
        assert abs(iou - 0.25) < 0.01

    def test_edge_touching(self):
        """两个 bbox 边缘相接但不重叠"""
        bbox1 = (0, 0, 10, 10)
        bbox2 = (10, 0, 20, 10)
        iou = LayoutDeduplicator._compute_iou(bbox1, bbox2)
        assert iou == 0.0


class TestContentSimilarity:
    """测试内容相似度计算"""

    def test_identical_text(self):
        """完全相同的文本"""
        text1 = "Hello World"
        text2 = "Hello World"
        sim = LayoutDeduplicator._content_similarity(text1, text2)
        assert sim == 1.0

    def test_case_difference(self):
        """只有大小写差异"""
        text1 = "Hello World"
        text2 = "hello world"
        sim = LayoutDeduplicator._content_similarity(text1, text2)
        assert sim == 1.0

    def test_partial_overlap(self):
        """部分词语重叠"""
        text1 = "Hello World Python"
        text2 = "Hello World Java"
        # norm1 = {"hello", "world", "python"}
        # norm2 = {"hello", "world", "java"}
        # intersection = 2, union = 4
        # sim = 0.5
        sim = LayoutDeduplicator._content_similarity(text1, text2)
        assert abs(sim - 0.5) < 0.01

    def test_no_overlap(self):
        """完全没有重叠"""
        text1 = "Hello World"
        text2 = "Python Java"
        sim = LayoutDeduplicator._content_similarity(text1, text2)
        assert sim == 0.0

    def test_empty_text(self):
        """空文本"""
        sim = LayoutDeduplicator._content_similarity("", "Hello")
        assert sim == 0.0


class TestDeduplication:
    """测试去重功能"""

    def _make_chunk(
        self,
        content_type: str,
        raw_content: str,
        page: int = 1,
        bbox: tuple[float, float, float, float] = (0, 0, 100, 100),
    ) -> LayoutChunk:
        return LayoutChunk(
            content_type=content_type,  # type: ignore
            raw_content=raw_content,
            page=page,
            bbox=bbox,
            column=0,
            order_in_page=0,
        )

    def test_no_duplicates(self):
        """无重复 chunks"""
        chunks = [
            self._make_chunk("text", "Hello", bbox=(0, 0, 10, 10)),
            self._make_chunk("text", "World", bbox=(20, 0, 30, 10)),
            self._make_chunk("table", "| A | B |\n|---|---|\n| 1 | 2 |", bbox=(50, 0, 100, 50)),
        ]
        result = LayoutDeduplicator.deduplicate(chunks)
        assert len(result) == 3

    def test_exact_duplicate_removed(self):
        """同一区域 + 相同内容 → 去重"""
        chunks = [
            # 两个完全相同的 text chunk
            self._make_chunk("text", "Hello World", bbox=(0, 0, 50, 20)),
            self._make_chunk("text", "Hello World", bbox=(0, 0, 50, 20)),
        ]
        result = LayoutDeduplicator.deduplicate(chunks)
        assert len(result) == 1

    def test_different_content_same_bbox(self):
        """相同 bbox 但内容不同 → 保留两者"""
        chunks = [
            self._make_chunk("text", "Figure 1", bbox=(0, 0, 50, 20)),
            self._make_chunk("image", "This is a flowchart showing the process", bbox=(0, 0, 50, 20)),
        ]
        result = LayoutDeduplicator.deduplicate(chunks)
        # 内容相似度低，即使 bbox 重叠也保留
        assert len(result) == 2

    def test_same_content_different_bbox(self):
        """相同内容且高度重叠 bbox → 去重，table 优先"""
        chunks = [
            self._make_chunk("text", "Table Data", bbox=(0, 0, 100, 50)),
            self._make_chunk("table", "Table Data", bbox=(10, 10, 90, 40)),
        ]
        result = LayoutDeduplicator.deduplicate(chunks)
        # bbox 重叠(IoU = 0.36 > 0.5/2? No) 且内容完全相同
        # bbox1: 100x50=5000, bbox2: 80x30=2400
        # 交集: (10,10,90,40)=80x30=2400
        # 并集: 5000+2400-2400=5000, IoU=2400/5000=0.48
        # 低于 0.5 阈值，所以保留两者（实际场景这是合理的）
        # 修改为更大重叠
        chunks = [
            self._make_chunk("text", "Table Data", bbox=(0, 0, 100, 50)),
            self._make_chunk("table", "Table Data", bbox=(5, 5, 95, 45)),
        ]
        result = LayoutDeduplicator.deduplicate(chunks)
        # IoU = (90*40)/(5000+3600-3600)=3600/5000=0.72 > 0.5
        # 内容完全相同 → 去重，保留 table
        assert len(result) == 1
        assert result[0].content_type == "table"

    def test_priority_text_vs_table(self):
        """text 和 table 重叠时，table 优先（表格数据更重要）"""
        chunks = [
            self._make_chunk("text", "header data row1 row2", bbox=(0, 0, 100, 50)),
            self._make_chunk("table", "| header |\n|---|\n| data |", bbox=(5, 5, 95, 45)),
        ]
        result = LayoutDeduplicator.deduplicate(chunks)
        # 内容相似度较低，不去重（保留两者）
        assert len(result) == 2

    def test_overlapping_table_with_caption(self):
        """表格和周围描述文字 → 保留两者（内容不相似）"""
        chunks = [
            self._make_chunk("text", "Table 1 shows the results. The data is presented below.", bbox=(0, 0, 200, 100)),
            self._make_chunk("table", "| header |\n|---|\n| data |", bbox=(10, 60, 190, 95)),
        ]
        result = LayoutDeduplicator.deduplicate(chunks)
        # 内容相似度低，bbox 重叠但不去重
        assert len(result) == 2

    def test_cross_page_no_dedup(self):
        """跨页的相同内容不会去重（bbox 不会跨页重叠）"""
        chunks = [
            self._make_chunk("text", "Same content", page=1, bbox=(0, 0, 50, 20)),
            self._make_chunk("text", "Same content", page=2, bbox=(0, 0, 50, 20)),
        ]
        result = LayoutDeduplicator.deduplicate(chunks)
        assert len(result) == 2

    def test_iou_below_threshold(self):
        """IoU 低于阈值 → 不去重"""
        chunks = [
            self._make_chunk("text", "Left side text", bbox=(0, 0, 40, 20)),
            self._make_chunk("text", "Right side text", bbox=(60, 0, 100, 20)),
        ]
        result = LayoutDeduplicator.deduplicate(chunks)
        assert len(result) == 2

    def test_empty_list(self):
        """空列表"""
        result = LayoutDeduplicator.deduplicate([])
        assert result == []

    def test_single_chunk(self):
        """单个 chunk"""
        chunks = [self._make_chunk("text", "Only one", bbox=(0, 0, 50, 20))]
        result = LayoutDeduplicator.deduplicate(chunks)
        assert len(result) == 1


class TestLayoutMergerWithDedup:
    """测试 LayoutMerger 集成去重"""

    def _make_chunk(
        self,
        content_type: str,
        raw_content: str,
        page: int = 1,
        column: int = 0,
        bbox: tuple[float, float, float, float] = (0, 0, 100, 100),
    ) -> LayoutChunk:
        return LayoutChunk(
            content_type=content_type,  # type: ignore
            raw_content=raw_content,
            page=page,
            bbox=bbox,
            column=column,
            order_in_page=0,
        )

    def test_merge_deduplicates(self):
        """merge 时自动去重"""
        from src.parsers.layout_chunk import LayoutMerger

        chunks = [
            self._make_chunk("text", "Hello", bbox=(0, 0, 50, 20)),
            self._make_chunk("text", "Hello", bbox=(0, 0, 50, 20)),  # 重复
            self._make_chunk("table", "| A | B |", bbox=(0, 50, 100, 100)),
        ]
        result = LayoutMerger.merge(chunks)
        # 去重后剩 2 个
        assert len(result) == 2

    def test_merge_sorts_correctly(self):
        """merge 时按阅读顺序排序"""
        from src.parsers.layout_chunk import LayoutMerger

        chunks = [
            self._make_chunk("text", "Third", page=1, bbox=(0, 200, 50, 220)),
            self._make_chunk("text", "First", page=1, bbox=(0, 0, 50, 20)),
            self._make_chunk("text", "Second", page=1, bbox=(0, 100, 50, 120)),
        ]
        result = LayoutMerger.merge(chunks)
        assert result[0].raw_content == "First"
        assert result[1].raw_content == "Second"
        assert result[2].raw_content == "Third"
        assert result[0].order_in_page == 0
        assert result[1].order_in_page == 1
        assert result[2].order_in_page == 2
