"""
Layer 2: Judge 层验证（少量 chunk，控制 API 消耗）
只取前 3 个 text chunk + 1 个 table chunk 试水。
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from src.parsers.pymupdf_parser import PyMuPDFParser
from src.parsers.table_parser import TableParser
from src.parsers.layout_chunk import LayoutMerger
from src.judge.llm_judge import LLMJudge

PDF_PATH = str(Path("data/sample_papers/2021-AAAA-LNT and LNnT综述.pdf"))

# 解析（不调 Vision，节省成本）
text_chunks  = PyMuPDFParser(PDF_PATH).parse()
table_chunks = TableParser(PDF_PATH).parse()
all_chunks = text_chunks + table_chunks
merged = LayoutMerger.merge(all_chunks)

print(f"Total merged chunks: {len(merged)}")
print(f"  text:  {len([c for c in merged if c.content_type == 'text'])}")
print(f"  table: {len([c for c in merged if c.content_type == 'table'])}")

judge = LLMJudge()

# 只取少量 chunk 验证
text_samples = [c for c in merged if c.content_type == "text"][:3]
table_samples = [c for c in merged if c.content_type == "table"][:1]
samples = text_samples + table_samples

print(f"\n--- Judge 验证 (共 {len(samples)} 个 chunk) ---")
for chunk in samples:
    print(f"\n  [{chunk.chunk_id}] type={chunk.content_type}")
    print(f"      preview: {chunk.raw_content[:100].replace(chr(10), ' ')}")
    verdict = judge.judge(chunk.content_type, chunk.raw_content, PDF_PATH)
    print(f"      -> passed={verdict.passed} score={verdict.score:.2f}")
    print(f"      -> feedback: {verdict.feedback[:120]}")
