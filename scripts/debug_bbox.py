"""调试脚本：打印 table bbox 和该页上所有 text chunk 的中心点坐标，用于判断容差是否足够。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.parsers import PyMuPDFParser, TableParser, LayoutMerger

PDF_PATH = "data/2021-AAAA-LNT and LNnT综述.pdf"

# 1. 运行三个解析器
text_parser = PyMuPDFParser(PDF_PATH)
table_parser = TableParser(PDF_PATH)

raw_text_chunks = text_parser.parse()         # 去重前
raw_table_chunks = table_parser.parse(strategy="all")

# 2. 合并（含去重）
all_chunks = raw_text_chunks + raw_table_chunks
merged = LayoutMerger.merge(all_chunks)

# 3. 找到所有有表格的页面
table_pages = sorted(set(c.page for c in raw_table_chunks))

print(f"表格所在页: {table_pages}")
print()

for page in table_pages:
    # --- 表格 bbox ---
    page_tables = [c for c in raw_table_chunks if c.page == page]

    # --- 去重前的 text chunks ---
    pre_texts = [c for c in raw_text_chunks if c.page == page]
    # --- 去重后的 text chunks ---
    post_texts = [c for c in merged if c.content_type in ("text",) and c.page == page]
    post_text_ids = {c.chunk_id for c in post_texts}
    removed_texts = [c for c in pre_texts if c.chunk_id not in post_text_ids]

    print(f"{'='*80}")
    print(f"===== 第 {page} 页 =====")

    print(f"\n--- 表格 bbox ({len(page_tables)} 个) ---")
    for i, t in enumerate(page_tables):
        print(f"  table[{i}]: bbox={t.bbox}  content_preview={t.raw_content[:60]}")

    print(f"\n--- 被去重移除的 text chunks ({len(removed_texts)} 个) ---")
    for c in removed_texts:
        cx = (c.bbox[0] + c.bbox[2]) / 2
        cy = (c.bbox[1] + c.bbox[3]) / 2
        print(f"  REMOVED: bbox={c.bbox}  center=({cx:.1f},{cy:.1f})  content={c.raw_content[:60]}")

    print(f"\n--- 残留的 text chunks ({len(post_texts)} 个) ---")
    for c in post_texts:
        cx = (c.bbox[0] + c.bbox[2]) / 2
        cy = (c.bbox[1] + c.bbox[3]) / 2

        # 找最近的 table 计算距离
        for t in page_tables:
            tx0, ty0, tx1, ty1 = t.bbox
            dl = cx - tx0 if cx < tx0 else 0
            dr = tx1 - cx if cx > tx1 else 0
            dt = cy - ty0 if cy < ty0 else 0
            db = ty1 - cy if cy > ty1 else 0

            # 检查是否在容差范围内
            tol = 20.0
            in_tol = (
                tx0 - tol <= cx <= tx1 + tol
                and ty0 - tol <= cy <= ty1 + tol
            )

            inside_str = "✓" if all(d == 0 for d in [dl, dr, dt, db]) else "✗"
            tol_str = "✓tol" if in_tol else "✗tol"

            print(f"  {inside_str} {tol_str} bbox={c.bbox}  center=({cx:.1f},{cy:.1f})  "
                  f"dist=(L{dl:.0f},R{dr:.0f},T{dt:.0f},B{db:.0f})  "
                  f"content={c.raw_content[:60]}")
            break

    print()
