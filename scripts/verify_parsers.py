"""
Layer 1: Parser 层验证（不调用 LLM）
验证三个 Parser 能否正常解析 PDF，LayoutMerger 能否正确合并。
修正了 API 调用方式匹配项目实际代码。
"""
from pathlib import Path
from collections import Counter

# 添加项目根目录到 sys.path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows 终端默认 GBK，强制 UTF-8 输出，遇到无法显示的字符用 ? 替代
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from src.parsers.pymupdf_parser import PyMuPDFParser
from src.parsers.table_parser import TableParser
from src.parsers.vision_parser import VisionParser
from src.parsers.layout_chunk import LayoutMerger


PDF_DIR = Path("data/sample_papers")
PDF_PATHS = sorted(PDF_DIR.glob("*.pdf"))

for pdf_path in PDF_PATHS:
    print(f"\n{'='*60}")
    print(f"PDF: {pdf_path.name}")

    # 修正1: Parser 构造函数需要 pdf_path，parse() 不传参
    text_chunks  = PyMuPDFParser(pdf_path).parse()
    table_chunks = TableParser(pdf_path).parse()

    # [跳过] VisionParser 会调用 Qwen-VL，成本较高，Layer 1 先只验证 text+table
    image_chunks = []

    # 修正2: LayoutMerger.merge 是静态方法，只接受一个列表参数
    all_chunks = text_chunks + table_chunks + image_chunks
    merged = LayoutMerger.merge(all_chunks)

    print(f"  text chunks : {len(text_chunks)}")
    print(f"  table chunks: {len(table_chunks)}")
    print(f"  image chunks: {len(image_chunks)}")
    print(f"  after merge : {len(merged)}")

    # 检查 chunk_id 唯一性
    ids = [c.chunk_id for c in merged]
    dupes = {i for i in ids if ids.count(i) > 1}
    print(f"  duplicate chunk_ids: {len(dupes)}")
    if dupes:
        for d in dupes:
            print(f"    DUPLICATE: {d}")

    # 检查 column 分布
    col_dist = Counter(c.column for c in merged)
    print(f"  column distribution: {dict(col_dist)}")

    # 检查 page 分布
    page_dist = Counter(c.page for c in merged)
    print(f"  pages: {sorted(page_dist.keys())} (total pages: {len(page_dist)})")

    # 打印前5个 chunk 供人工确认
    print(f"  --- top 5 chunks ---")
    for c in merged[:5]:
        preview = c.raw_content[:80].replace('\n', ' ')
        print(f"  [{c.chunk_id}] col={c.column} order={c.order_in_page} | {preview}")
