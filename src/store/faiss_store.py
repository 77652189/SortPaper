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
        self.index: faiss.IndexFlatL2 | None = None  # 懒创建，实际维度由首次 embed 确定
        self.records: list[dict[str, Any]] = []

    def _ensure_index(self, actual_dim: int) -> None:
        """确保 index 已创建，且维度匹配。"""
        import faiss

        if self.index is not None:
            if self.dimension != actual_dim:
                raise ValueError(
                    f"Embedding 维度不匹配：索引维度={self.dimension}, "
                    f"当前向量维度={actual_dim}。请检查 model 配置。"
                )
            return
        self.dimension = actual_dim
        self.index = faiss.IndexFlatL2(actual_dim)

    def embed(self, text: str) -> list[float]:
        from dashscope import TextEmbedding

        response = TextEmbedding.call(model=self.model, input=text)
        # 统一解析 DashScope 响应，兼容 object 和 dict 两种格式
        output = response.output
        if isinstance(output, dict):
            embeddings = output.get("embeddings", [])
        else:
            embeddings = getattr(output, "embeddings", None) or []
        if not embeddings:
            raise ValueError("No embedding returned from DashScope")
        first = embeddings[0]
        if isinstance(first, dict):
            vector = first.get("embedding")
        else:
            vector = getattr(first, "embedding", None)
        if not vector:
            raise ValueError("Embedding payload missing vector")

        # 确保 index 已创建，且维度匹配
        actual_dim = len(vector)
        self._ensure_index(actual_dim)

        return [float(value) for value in vector]

    def add(self, paper_id: str, worker_type: str, content: str, metadata: dict[str, Any]) -> None:
        if not content.strip():
            return
        embedding = self.embed(content)  # embed() calls _ensure_index(), guarantees self.index is set
        vector = np.array([embedding], dtype="float32")
        record = {
            "paper_id": paper_id,
            "worker_type": worker_type,
            "content": content,
            "metadata": metadata,
        }
        # 先 add index，成功后再 append records，保证位置对齐
        # 如果 index.add 失败，record 不会被追加，两者不会错位
        self.index.add(vector)
        self.records.append(record)

    def add_chunks(
        self,
        paper_id: str,
        worker_type: str,
        content: str,
        metadata: dict[str, Any],
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
    ) -> None:
        for chunk in self._chunk_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            chunk_metadata = {
                **metadata,
                "chunk_index": chunk["chunk_index"],
                "chunk_count": chunk["chunk_count"],
                "char_start": chunk["char_start"],
                "char_end": chunk["char_end"],
            }
            self.add(
                paper_id=paper_id,
                worker_type=worker_type,
                content=chunk["content"],
                metadata=chunk_metadata,
            )

    def _chunk_text(self, content: str, chunk_size: int, chunk_overlap: int) -> list[dict[str, Any]]:
        text = content.strip()
        if not text:
            return []
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        text_len = len(text)
        # 直接按 \n\n 切段落，并在原文中定位每个段落的起始偏移
        paragraphs_with_pos = self._split_paragraphs_with_pos(text)

        # 贪心合并段落，直到达到 chunk_size，再开新 chunk
        chunks: list[dict[str, Any]] = []
        current = ""
        current_start = 0

        for para, para_start in paragraphs_with_pos:
            candidate = para if not current else f"{current}\n\n{para}"
            if len(candidate) <= chunk_size:
                if not current:
                    current_start = para_start
                current = candidate
                continue

            # candidate 超长，先保存已积累的 current
            if current:
                chunks.append(
                    {
                        "content": current,
                        "char_start": current_start,
                        "char_end": current_start + len(current),
                    }
                )
                current = ""

            # 单个段落超长，用滑动窗口切
            if len(para) <= chunk_size:
                current = para
                current_start = para_start
                continue

            for item in self._chunk_long_text(para, chunk_size, chunk_overlap):
                chunks.append(
                    {
                        "content": item["content"],
                        "char_start": para_start + item["char_start"],
                        "char_end": para_start + item["char_end"],
                    }
                )

        if current:
            chunks.append(
                {
                    "content": current,
                    "char_start": current_start,
                    "char_end": current_start + len(current),
                }
            )

        # 补充索引和相对位置字段
        chunk_count = len(chunks)
        for index, chunk in enumerate(chunks):
            chunk["chunk_index"] = index
            chunk["chunk_count"] = chunk_count
            chunk["relative_position"] = round(index / max(chunk_count - 1, 1), 6)
            chunk["relative_char_start"] = round(chunk["char_start"] / max(text_len, 1), 6)
            chunk["relative_char_end"] = round(chunk["char_end"] / max(text_len, 1), 6)
            chunk["page"] = None
            chunk["section_title"] = None
        return chunks

    def _split_paragraphs_with_pos(self, text: str) -> list[tuple[str, int]]:
        """将文本按 \\n\\n 切分为段落列表，并返回每个段落的正文及其在原文中的起始偏移。"""
        results: list[tuple[str, int]] = []
        cursor = 0
        for part in text.split("\n\n"):
            stripped = part.strip()
            if not stripped:
                cursor += len(part) + 2
                continue
            start = text.find(stripped, cursor)
            if start < 0:
                start = cursor
            results.append((stripped, start))
            cursor = start + len(stripped)
        if not results:
            cleaned = text.strip()
            return [(cleaned, text.find(cleaned))] if cleaned else []
        return results

    def _chunk_long_text(self, text: str, chunk_size: int, chunk_overlap: int) -> list[dict[str, Any]]:
        step = chunk_size - chunk_overlap
        chunks: list[dict[str, Any]] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            target_end = min(start + chunk_size, text_len)
            end = target_end
            if end < text_len:
                boundary = text.rfind("\n", start, target_end)
                if boundary < 0:
                    boundary = max(
                        text.rfind("。", start, target_end),
                        text.rfind(".", start, target_end),
                    )
                if boundary > start + int(chunk_size * 0.6):
                    end = boundary + 1

            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append({"content": chunk_content, "char_start": start, "char_end": end})
            if end <= start:
                break
            start = end - chunk_overlap
            if start < 0:
                start = 0
        return chunks

    def save(self, path: str | Path) -> None:
        import faiss

        if self.index is None:
            return  # 空库，无需保存
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
