"""精简测试：Judge + 分类 + Qdrant 入库（跳过 Map-Reduce）"""
import sys, hashlib, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv; load_dotenv()

from src.graph.pipeline_graph import build_graph

t0 = time.time()
pdf_path = "data/2021-AAAA-LNT and LNnT综述.pdf"
paper_id = hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()

graph = build_graph()
state = graph.invoke({
    "pdf_path": pdf_path,
    "paper_id": paper_id,
    "text_retries": 0,
    "table_retries": 0,
    "image_retries": 0,
    "status": "processing",
    "output_dir": "faiss_index",
})

elapsed = time.time() - t0
print(f"\n耗时: {elapsed:.1f}s")
print(f"状态: {state.get('status', '?')}")

q = state.get("quality_result", {})
print(f"\n=== 质量评估 ===")
print(f"标题: {q.get('paper_title', '?')}")
print(f"分类: {q.get('category', '?')}")
print(f"状态: {q.get('classify_status', '?')}")
if q.get("classify_result"):
    cr = q["classify_result"]
    print(f"理由: {cr.get('reason', '')}")
    print(f"证据: {cr.get('key_evidence', [])}")
    print(f"产物: {cr.get('target_products', [])}")
    print(f"菌株: {cr.get('organisms', [])}")

# Qdrant 验证
from src.store.qdrant_store import QdrantStore
store = QdrantStore()
print(f"\n=== Qdrant ===")
print(f"入库: {store.count} 个 chunk")

if store.count > 0:
    # 查一条 payload 看字段
    r = store.search("HMO", limit=1)
    if r:
        p = r[0]["payload"]
        print(f"category: {p.get('category')}")
        print(f"credibility: {p.get('credibility')}")
        print(f"classify_status: {p.get('classify_status')}")
        print(f"content_type: {p.get('content_type')}")
        print(f"content 前60字: {p.get('content', '')[:60]}")
