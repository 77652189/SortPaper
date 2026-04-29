from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class FAISSStore:
    def __init__(self, dimension: int = 1024, model: str = "text-embedding-v3") -> None:
        import faiss

        self.dimension = dimension
        self.model = model
        self.index = faiss.IndexFlatL2(dimension)
        self.records: list[dict[str, Any]] = []

    def embed(self, text: str) -> list[float]:
        from dashscope import TextEmbedding

        response = TextEmbedding.call(model=self.model, input=text)
        embeddings = getattr(response.output, "embeddings", None) or response.output.get("embeddings", [])
        if not embeddings:
            raise ValueError("No embedding returned from DashScope")
        vector = embeddings[0].get("embedding")
        if not vector:
            raise ValueError("Embedding payload missing vector")
        return [float(value) for value in vector]

    def add(self, paper_id: str, worker_type: str, content: str, metadata: dict[str, Any]) -> None:
        if not content.strip():
            return
        vector = np.array([self.embed(content)], dtype="float32")
        self.index.add(vector)
        self.records.append(
            {
                "paper_id": paper_id,
                "worker_type": worker_type,
                "content": content,
                "metadata": metadata,
            }
        )

    def save(self, path: str | Path) -> None:
        import faiss

        base_path = Path(path)
        base_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(base_path / "index.faiss"))
        (base_path / "records.json").write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, path: str | Path) -> None:
        import faiss

        base_path = Path(path)
        self.index = faiss.read_index(str(base_path / "index.faiss"))
        self.records = json.loads((base_path / "records.json").read_text(encoding="utf-8"))
