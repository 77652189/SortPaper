"""
快速诊断：统计两个 PDF 的 text / table / image chunk 数量（不调 LLM）
"""
import sys
import fitz
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parsers.pymupdf_parser import PyMuPDFParser
from src.parsers.table_parser import TableParser
from src.parsers.layout_chunk import LayoutMerger

_MIN_AREA_PT2 = 5000
_MIN_PIXEL    = 10000

PDFS = [
    "2022-AAA-Engineering Escherichia coli for the High-Titer Biosynthesis of Lacto-N-tetraose.pdf",
    "2021-AAAA-LNT and LNnT综述.pdf",
]

for pdf_name in PDFS:
    pdf = Path("data/sample_papers") / pdf_name
    text_chunks  = PyMuPDFParser(pdf).parse()
    table_chunks = TableParser(pdf).parse()

    # 统计有效图片（不调 VL，仅用 PyMuPDF 计数）
    img_count = 0
    with fitz.open(str(pdf)) as doc:
        for page in doc:
            for info in page.get_image_info(xrefs=True):
                x0, y0, x1, y1 = info.get("bbox", (0, 0, 0, 0))
                if (x1 - x0) * (y1 - y0) < _MIN_AREA_PT2:
                    continue
                xref = info.get("xref")
                if not xref:
                    continue
                ei = doc.extract_image(xref)
                if ei.get("width", 0) * ei.get("height", 0) >= _MIN_PIXEL:
                    img_count += 1

    merged = LayoutMerger.merge(text_chunks + table_chunks)

    print(f"\n{'='*62}")
    print(f"PDF : {pdf_name}")
    print(f"  Text  chunks : {len(text_chunks):>4}")
    print(f"  Table chunks : {len(table_chunks):>4}  (after false-positive filter)")
    print(f"  Image (PDF)  : {img_count:>4}  (will be sent to VisionParser → qwen-vl-max)")
    print(f"  Merged total : {len(merged):>4}  (text + table, before image)")

    if table_chunks:
        print(f"\n  Tables:")
        for c in table_chunks:
            rows_md = c.raw_content.count("\n") + 1
            data_rows = rows_md - 2          # 减去 header + separator
            preview = c.raw_content.split("\n")[0][:72]
            print(f"    [p{c.page}] {data_rows} data rows | {preview}")
    else:
        print(f"\n  Tables: none (all filtered as false positives)")
