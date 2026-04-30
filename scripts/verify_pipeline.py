"""
Layer 4: 完整 Pipeline 验证（最后跑）
只跑一个 PDF，观察日志输出和最终状态。
修正了 build_graph、state 字段名等 API 不匹配。
"""
from pathlib import Path
import sys
import logging
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

# 打开日志，观察 pipeline 流程
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

from src.graph.pipeline_graph import build_graph

pipeline = build_graph()

initial_state = {
    "pdf_path": str(Path("data/sample_papers/2022-AAA-Engineering Escherichia coli for the High-Titer Biosynthesis of Lacto-N-tetraose.pdf")),
    "paper_id": "test_paper_001",
    "text_retries": 0,
    "table_retries": 0,
    "image_retries": 0,
    "status": "processing",
    "output_dir": "faiss_index_test",
}

try:
    final_state = pipeline.invoke(initial_state, config={"recursion_limit": 100})

    print(f"\n{'='*60}")
    print(f"Final status: {final_state.get('status', 'N/A')}")
    print(f"Error       : {final_state.get('error', 'None')}")

    # 修正: 使用 text_verdicts/table_verdicts/image_verdicts 而不是 verdicts
    all_verdicts = []
    all_verdicts.extend(final_state.get("text_verdicts", []))
    all_verdicts.extend(final_state.get("table_verdicts", []))
    all_verdicts.extend(final_state.get("image_verdicts", []))

    passed = [v for v in all_verdicts if v.get("passed")]
    failed = [v for v in all_verdicts if not v.get("passed")]
    print(f"Verdicts: {len(passed)} passed, {len(failed)} failed")

    # retry 次数
    print(f"Retries: text={final_state.get('text_retries', 0)}, "
          f"table={final_state.get('table_retries', 0)}, "
          f"image={final_state.get('image_retries', 0)}")

    # routes_done
    print(f"Routes done: {final_state.get('routes_done', set())}")

    # merged_chunks
    merged = final_state.get("merged_chunks", [])
    print(f"Merged chunks: {len(merged)}")

    # store 结果
    from pathlib import Path
    index_path = Path("faiss_index_test")
    if index_path.exists():
        from src.store.faiss_store import FAISSStore
        s = FAISSStore()
        s.load(str(index_path))
        print(f"FAISS index: {s.index.ntotal} vectors, {len(s.records)} records")

        # 清理测试索引
        import shutil
        shutil.rmtree(index_path)
        print(f"Cleaned up test index: {index_path}")
    else:
        print("FAISS index: not created (store was skipped or path mismatch)")

except Exception as e:
    import traceback
    print(f"\n{'='*60}")
    print(f"Pipeline 执行失败: {e}")
    traceback.print_exc()
