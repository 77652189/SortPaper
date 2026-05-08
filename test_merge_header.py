"""
快速测试 _merge_header_blocks 逻辑。
不依赖任何外部包，直接构造 LayoutChunk 验证合并效果。
"""
import sys
sys.path.insert(0, ".")

from src.parsers.layout_chunk import LayoutChunk, LayoutMerger

# ── 模拟一篇论文第一页的 header 区域 ──────────────────────────────────────
# 页面高度假设为 800pt，header_zone_ratio=0.35 → 顶部 280pt 以内
PAGE_HEIGHT = 800.0

def make_chunk(page, y0, y1, content, col=2):
    return LayoutChunk(
        content_type="text",
        raw_content=content,
        page=page,
        bbox=(50.0, y0, 500.0, y1),
        column=col,
        order_in_page=0,
        metadata={"page_height": PAGE_HEIGHT},
    )

# 场景1：作者行和机构行 y_gap=20，应该被合并
chunks_1 = [
    make_chunk(1, 60.0, 78.0, "Alice Author1, Bob Author2"),
    make_chunk(1, 98.0, 116.0, "Dept. of Computer Science, University X"),
    make_chunk(1, 300.0, 320.0, "Abstract: This paper proposes..."),
]

# 场景2：y_gap=50，超过阈值，不应该合并
chunks_2 = [
    make_chunk(1, 60.0, 78.0, "Alice Author1"),
    make_chunk(1, 128.0, 146.0, "University of Somewhere"),
]

# 场景3：不在 header 区域（y0 > 280），不应该合并
chunks_3 = [
    make_chunk(1, 300.0, 320.0, "Some body text line 1"),
    make_chunk(1, 325.0, 345.0, "Some body text line 2"),
]

for idx, (chunks, desc) in enumerate([
    (chunks_1, "场景1: y_gap=20, 应该合并"),
    (chunks_2, "场景2: y_gap=50, 不应该合并"),
    (chunks_3, "场景3: 不在header区, 不应该合并"),
], 1):
    result = LayoutMerger._merge_header_blocks(chunks, y_gap_threshold=30.0)
    print(f"\n=== {desc} ===")
    print(f"输入 chunks: {len(chunks)}  →  输出 chunks: {len(result)}")
    for c in result:
        print(f"  page={c.page} y0={c.bbox[1]:.1f} y1={c.bbox[3]:.1f} | {c.raw_content!r}")
