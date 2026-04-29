from __future__ import annotations

import json
import re
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

        chunks: list[dict[str, Any]] = []
        text_len = len(text)
        sections = self._split_structured_sections(text)

        for section in sections:
            section_chunks = self._chunk_section_text(
                section["body"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            for piece in section_chunks:
                global_start = section["body_start"] + piece["char_start"]
                global_end = section["body_start"] + piece["char_end"]
                chunks.append(
                    {
                        "content": piece["content"],
                        "char_start": global_start,
                        "char_end": global_end,
                        "page": section.get("page"),
                        "section_title": section.get("title"),
                    }
                )

        chunk_count = len(chunks)
        for index, chunk in enumerate(chunks):
            chunk["chunk_index"] = index
            chunk["chunk_count"] = chunk_count
            chunk["relative_position"] = round(index / max(chunk_count - 1, 1), 6)
            chunk["relative_char_start"] = round(chunk["char_start"] / max(text_len, 1), 6)
            chunk["relative_char_end"] = round(chunk["char_end"] / max(text_len, 1), 6)
        return chunks

    def _split_structured_sections(self, text: str) -> list[dict[str, Any]]:
        header_regex = re.compile(r"^##\s+(.+)$", re.MULTILINE)
        matches = list(header_regex.finditer(text))
        if not matches:
            return [{"title": None, "page": None, "body": text, "body_start": 0}]

        sections: list[dict[str, Any]] = []
        for index, match in enumerate(matches):
            title = match.group(1).strip()
            raw_start = match.end()
            body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            raw_body = text[raw_start:body_end]
            left_trim = len(raw_body) - len(raw_body.lstrip())
            body = raw_body.strip()
            if not body:
                continue
            page_match = re.search(r"Page\s+(\d+)", title, flags=re.IGNORECASE)
            page = int(page_match.group(1)) if page_match else None
            sections.append(
                {
                    "title": title,
                    "page": page,
                    "body": body,
                    "body_start": raw_start + left_trim,
                }
            )

        if not sections:
            return [{"title": None, "page": None, "body": text, "body_start": 0}]
        return sections

    def _chunk_section_text(self, body: str, chunk_size: int, chunk_overlap: int) -> list[dict[str, Any]]:
        paragraphs = [part.strip() for part in body.split("\n\n") if part.strip()]
        if not paragraphs:
            paragraphs = [body.strip()]

        chunks: list[dict[str, Any]] = []
        current = ""
        current_start = 0
        cursor = 0

        for paragraph in paragraphs:
            paragraph_start = body.find(paragraph, cursor)
            if paragraph_start < 0:
                paragraph_start = cursor
            cursor = paragraph_start + len(paragraph)
            candidate = paragraph if not current else f"{current}\n\n{paragraph}"

            if len(candidate) <= chunk_size:
                if not current:
                    current_start = paragraph_start
                current = candidate
                continue

            if current:
                chunks.append(
                    {
                        "content": current,
                        "char_start": current_start,
                        "char_end": current_start + len(current),
                    }
                )
                current = ""

            if len(paragraph) <= chunk_size:
                current = paragraph
                current_start = paragraph_start
                continue

            sentence_chunks = self._chunk_long_text(paragraph, chunk_size, chunk_overlap)
            for item in sentence_chunks:
                chunks.append(
                    {
                        "content": item["content"],
                        "char_start": paragraph_start + item["char_start"],
                        "char_end": paragraph_start + item["char_end"],
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
        return chunks

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
