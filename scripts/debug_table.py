"""调试 TableParser：追踪表格在哪一步被过滤掉"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pdfplumber
from src.parsers.table_parser import TableParser

pdf_path = "data/sample_papers/2022-AAA-Engineering Escherichia coli for the High-Titer Biosynthesis of Lacto-N-tetraose.pdf"
settings = TableParser._table_settings(None)

with pdfplumber.open(pdf_path) as pdf:
    for page_index, page in enumerate(pdf.pages[:5], start=1):
        found = page.find_tables(table_settings=settings)
        print(f"=== Page {page_index}: found {len(found)} tables ===")
        for t_idx, t in enumerate(found):
            rows = t.extract()
            print(f"  [{t_idx}] rows={len(rows) if rows else 0}", end="")
            if rows:
                norm = TableParser._normalize_table(rows)
                if norm:
                    print(f"  normalized={len(norm)}x{len(norm[0])}", end="")
                else:
                    print(f"  normalized=FILTERED", end="")
            print()
