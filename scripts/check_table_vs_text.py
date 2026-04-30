"""Check whether table content is covered by text chunks on the same page."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.parsers.pymupdf_parser import PyMuPDFParser
from src.parsers.table_parser import TableParser

pdf = Path("data/sample_papers/2021-AAAA-LNT and LNnT综述.pdf")

text_chunks  = PyMuPDFParser(pdf).parse()
table_chunks = TableParser(pdf).parse()

print(f"Total text chunks : {len(text_chunks)}")
print(f"Total table chunks: {len(table_chunks)}")

for t in table_chunks:
    pg = t.page
    print(f"\n{'='*60}")
    print(f"TABLE on page {pg}  ({t.chunk_id})")
    print("  First 400 chars (raw_content):")
    print("  " + t.raw_content[:400].replace("\n", " ↵ "))

    print(f"\n  Text chunks on page {pg} ({sum(1 for c in text_chunks if c.page == pg)} total):")
    for c in [c for c in text_chunks if c.page == pg]:
        snippet = c.raw_content[:100].replace("\n", " ↵ ")
        print(f"    [{c.chunk_id}] col={c.column} | {snippet}")
