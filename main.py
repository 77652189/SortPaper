from __future__ import annotations

import hashlib
from pathlib import Path

from dotenv import load_dotenv

from src.graph.pipeline_graph import build_graph

# 加载env
load_dotenv()

# 数据目录
DATA_DIR = Path("data")
# Faiss索引目录
FAISS_INDEX_PATH = Path("faiss_index")


def build_paper_id(pdf_path: Path) -> str:
    """
    构建论文ID
    使用SHA-256哈希算法构建论文ID，确保每个论文ID都是唯一的
    """
    hasher = hashlib.sha256()
    with pdf_path.open("rb") as file_obj:
        while True:
            file_block = file_obj.read(1024 * 1024)
            if not file_block:
                break
            hasher.update(file_block)
    return hasher.hexdigest()


def run() -> None:
    graph = build_graph()
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    results = {"done": 0, "failed": 0}

    for pdf_path in pdf_files:
        paper_id = build_paper_id(pdf_path)
        print(f"Processing {paper_id}...")
        state = graph.invoke(
            {
                "pdf_path": str(pdf_path),
                "paper_id": paper_id,
                "text_retries": 0,
                "table_retries": 0,
                "image_retries": 0,
                "status": "processing",
                "output_dir": str(FAISS_INDEX_PATH),
            }
        )
        results[state["status"]] = results.get(state["status"], 0) + 1
        # 合并三个 verdicts
        all_verdicts = []
        all_verdicts.extend(state.get("text_verdicts", []))
        all_verdicts.extend(state.get("table_verdicts", []))
        all_verdicts.extend(state.get("image_verdicts", []))
        passed = sum(1 for v in all_verdicts if v["passed"])
        failed = sum(1 for v in all_verdicts if not v["passed"])
        print(f"  -> {state['status']} | passed: {passed} | failed: {failed}")

    print(f"\nDone: {results}")


if __name__ == "__main__":
    run()
