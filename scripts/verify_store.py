"""
Layer 3: Store 层验证（不依赖 LLM Judge 结果）
直接构造假 chunk 验证 FAISS 写入/查询。
"""
from pathlib import Path
import sys
import tempfile
import shutil
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.parsers.layout_chunk import LayoutChunk
from src.store.faiss_store import FAISSStore
import numpy as np

# 修正1: FAISSStore.add(paper_id, worker_type, content, metadata)
#       而不是 add(content, metadata)
# 修正2: FAISSStore 没有 search() 方法，需用 FAISS 直接查询

# ── 写入验证 ───────────
store = FAISSStore()

fake_chunks = [
    LayoutChunk(
        content_type="text",
        raw_content="This paper proposes a novel method for document parsing.",
        bbox=(0, 0, 100, 20), page=1, column=0,
        order_in_page=0, chunk_id="text_1_0_0_0"
    ),
    LayoutChunk(
        content_type="text",
        raw_content="We evaluate on three benchmark datasets including PubMed and arXiv.",
        bbox=(0, 20, 100, 40), page=1, column=0,
        order_in_page=1, chunk_id="text_1_0_20_0"
    ),
    LayoutChunk(
        content_type="table",
        raw_content="| Method | F1 |\n|---|---|\n| Ours | 0.95 |\n| Baseline | 0.87 |",
        bbox=(0, 100, 200, 180), page=1, column=0,
        order_in_page=2, chunk_id="table_1_0_100_0"
    ),
]

for chunk in fake_chunks:
    store.add(
        paper_id="test_paper_001",
        worker_type=chunk.content_type,
        content=chunk.raw_content,
        metadata={
            "chunk_id": chunk.chunk_id,
            "page": chunk.page,
            "column": chunk.column,
            "bbox": chunk.bbox,
        }
    )

print(f"Index size after add: {store.index.ntotal}")
print(f"Records count: {len(store.records)}")

# ── 保存/重新加载验证 ───────────
test_dir = Path(tempfile.mkdtemp(prefix="test_faiss_"))
try:
    store.save(str(test_dir))
    print(f"\nSaved to: {test_dir}")
    print(f"  index.faiss exists: {(test_dir / 'index.faiss').exists()}")
    print(f"  records.json exists: {(test_dir / 'records.json').exists()}")

    # ── 搜索验证（FAISSStore 没有 search()，用 FAISS 直接查） ──
    import faiss
    query = "document parsing method"
    query_embedding = np.array([store.embed(query)], dtype="float32")
    distances, indices = store.index.search(query_embedding, k=2)

    print(f"\nSearch query: '{query}'")
    print(f"Distances: {distances}")
    print(f"Indices  : {indices}")
    for idx in indices[0]:
        if idx >= 0 and idx < len(store.records):
            r = store.records[idx]
            print(f"  -> [{idx}] {r['content'][:80]}")

    # ── 加载验证 ───────────
    store2 = FAISSStore()
    store2.load(str(test_dir))
    print(f"\nLoaded index size: {store2.index.ntotal}")
    print(f"Loaded records count: {len(store2.records)}")
    assert store2.index.ntotal == 3, "加载后 index 数量不匹配！"
    print("-> 加载验证通过！")

finally:
    shutil.rmtree(test_dir)
    print(f"\nCleaned up: {test_dir}")
