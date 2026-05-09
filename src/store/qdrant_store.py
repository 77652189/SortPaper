"""
Qdrant 向量存储：Hybrid Search（Dense + Sparse + RRF 融合）。

每个 chunk 作为 Qdrant point, 同时写入 dense 向量 (DashScope text-embedding-v3)
和 sparse 向量 (同模型 output_type="dense&sparse" 返回的离散向量),
支持 category/credibility/content_type 等字段的原生过滤。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, SparseVectorParams,
    SparseIndexParams, SparseVector,
    FusionQuery, Fusion, Prefetch,
)

logger = logging.getLogger(__name__)

COLLECTION_NAME = "papers"
EMBEDDING_MODEL = "text-embedding-v3"
EMBEDDING_DIM = 1024                     # text-embedding-v3 默认维度
DENSE_VECTOR_NAME = "dense"              # 密集向量名
SPARSE_VECTOR_NAME = "sparse"            # 稀疏向量名


class QdrantStore:
    """Qdrant 向量存储，支持 Dense + Sparse 混合检索。"""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = COLLECTION_NAME,
    ) -> None:
        self.client = QdrantClient(url=url)
        self.collection = collection
        self._ensure_collection()

    # ── 集合管理 ────────────────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        """确保 collection 存在，同时配置 dense 和 sparse 向量索引。

        如果已存在但不含 sparse 配置（旧版），自动重建。
        """
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection in collections:
            info = self.client.get_collection(self.collection)
            has_sparse = bool(info.config.params.sparse_vectors)
            if not has_sparse:
                logger.warning(
                    "旧版 collection 不含 sparse 索引，正在重建… | %s", self.collection,
                )
                self.client.delete_collection(collection_name=self.collection)
            else:
                return  # 已有 sparse，无需重建

        # 新建或重建
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=EMBEDDING_DIM, distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )
        logger.info(
            "Created collection '%s' (dense=%dd + sparse)", self.collection, EMBEDDING_DIM,
        )

    def _recreate_collection(self, actual_dim: int) -> None:
        """删除并重建 collection（维度不匹配时）。"""
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=actual_dim, distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )
        logger.info(
            "Recreated collection '%s' (dense=%dd + sparse)", self.collection, actual_dim,
        )

    # ── Embedding ──────────────────────────────────────────────────────────

    def embed(self, text: str) -> tuple[list[float], dict[int, float]]:
        """调用 DashScope TextEmbedding，同时返回 dense + sparse 向量。

        Returns:
            (dense_vector, sparse_dict): sparse_dict 为 {index: value} 格式。
        """
        from dashscope import TextEmbedding

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = TextEmbedding.call(
                    model=EMBEDDING_MODEL,
                    input=text,
                    output_type="dense&sparse",
                )
                break
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Embedding failed after 3 attempts: {exc}") from exc

        output = response.output
        if isinstance(output, dict):
            embeddings = output.get("embeddings", [])
        else:
            embeddings = getattr(output, "embeddings", None) or []
        if not embeddings:
            raise ValueError("No embedding returned from DashScope")

        first = embeddings[0]

        # dense 向量
        if isinstance(first, dict):
            dense = first.get("embedding")
        else:
            dense = getattr(first, "embedding", None)
        if not dense:
            raise ValueError("Embedding payload missing dense vector")

        # sparse 向量 — DashScope 返回 [{"index": N, "value": F, "token": T}, ...]
        sparse_raw = (
            first.get("sparse_embedding")
            if isinstance(first, dict)
            else getattr(first, "sparse_embedding", None)
        ) or []
        sparse_dict: dict[int, float] = {
            int(item["index"]): float(item["value"]) for item in sparse_raw
        }

        return ([float(v) for v in dense], sparse_dict)

    # ── 写入 ────────────────────────────────────────────────────────────────

    def add(
        self,
        paper_id: str,
        worker_type: str,
        content: str,
        metadata: dict[str, Any],
        embed_content: str | None = None,
    ) -> None:
        """添加一个 chunk 到 Qdrant（同时写入 dense + sparse 向量）。

        Args:
            content: 存入 payload 的完整文本（可含 context prefix）
            embed_content: 用于向量化的文本。若为 None，用 content
        """
        text_to_embed = embed_content if embed_content is not None else content
        if not text_to_embed.strip():
            return

        dense_vec, sparse_dict = self.embed(text_to_embed)
        actual_dim = len(dense_vec)

        # 检查 collection 维度是否匹配
        collection_info = self.client.get_collection(self.collection)
        current_dim = collection_info.config.params.vectors.get(
            DENSE_VECTOR_NAME
        ).size if DENSE_VECTOR_NAME in collection_info.config.params.vectors else 0
        if current_dim and current_dim != actual_dim:
            self._recreate_collection(actual_dim)

        chunk_id = metadata.get("chunk_id", f"{paper_id}_{worker_type}_{hash(content)}")

        # payload：合并所有 metadata + 原始内容
        payload: dict[str, Any] = {
            "paper_id": paper_id,
            "worker_type": worker_type,
            "content": content,
            **metadata,
        }
        for k, v in payload.items():
            if isinstance(v, tuple):
                payload[k] = list(v)

        # 构建 sparse vector（Qdrant 格式）
        indices = sorted(sparse_dict.keys())
        values = [sparse_dict[i] for i in indices]

        self.client.upsert(
            collection_name=self.collection,
            points=[
                models.PointStruct(
                    id=hash(chunk_id) % (2 ** 63),
                    vector={
                        DENSE_VECTOR_NAME: dense_vec,
                        SPARSE_VECTOR_NAME: SparseVector(
                            indices=indices, values=values,
                        ),
                    },
                    payload=payload,
                )
            ],
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
        """将长文本切分为多个 chunks 后逐一写入。"""
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

    # ── 查询 ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float | None = None,
        filter_kwargs: dict[str, Any] | None = None,
        rerank: bool = False,
    ) -> list[dict[str, Any]]:
        """Hybrid Search + 可选 Rerank。

        Pipeline: Dense + Sparse 召回 → (可选 Rerank) → 返回 top-k。

        Args:
            query: 检索查询
            limit: 返回条数
            score_threshold: Qdrant 打分阈值
            filter_kwargs: {"category": "fermentation_experiment", "credibility_gte": 0.7}
            rerank: 是否启用 qwen3-rerank 二次排序（有额外 API 开销）
        """
        dense_vec, sparse_dict = self.embed(query)

        qdrant_filter = self._build_filter(filter_kwargs)

        # 稀疏向量 → Qdrant 格式
        sparse_indices = sorted(sparse_dict.keys())
        sparse_values = [sparse_dict[i] for i in sparse_indices]
        sparse_vec = SparseVector(indices=sparse_indices, values=sparse_values)

        # Rerank 模式多召回一些候选
        fetch_multiplier = 3 if rerank else 2

        prefetch_dense = Prefetch(
            query=dense_vec,
            using=DENSE_VECTOR_NAME,
            limit=limit * fetch_multiplier,
            filter=qdrant_filter,
        )
        prefetch_sparse = Prefetch(
            query=sparse_vec,
            using=SPARSE_VECTOR_NAME,
            limit=limit * fetch_multiplier,
            filter=qdrant_filter,
        )

        candidates = self.client.query_points(
            collection_name=self.collection,
            prefetch=[prefetch_dense, prefetch_sparse],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit * fetch_multiplier,
            score_threshold=score_threshold,
        )

        results = [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in candidates.points
        ]

        # Rerank: 用 qwen3-rerank 对候选集重新打分排序
        if rerank and results:
            results = self._rerank(query, results, top_n=limit)

        return results[:limit]

    def _rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """qwen3-rerank 二次排序。仅对同次请求内的候选文档做相对排序。"""
        import dashscope
        from http import HTTPStatus

        documents = [c["payload"].get("content", "") for c in candidates]
        if not documents:
            return candidates

        resp = dashscope.TextReRank.call(
            model="qwen3-rerank",
            query=query,
            documents=documents,
            top_n=min(top_n, len(documents)),
            return_documents=False,
            instruct="Given a scientific paper search query, retrieve relevant passages that answer the query.",
        )

        if resp.status_code != HTTPStatus.OK:
            logger.warning("Rerank 失败，回退到原始排序 | code=%s msg=%s", resp.code, resp.message)
            return candidates

        # 按 rerank 结果重排
        rank_map = {
            r.get("index", -1): r.get("relevance_score", 0.0)
            for r in (resp.output.get("results") if isinstance(resp.output, dict)
                      else getattr(resp.output, "results", []))
        }
        reranked = sorted(
            candidates,
            key=lambda c, i=iter(range(len(candidates))): rank_map.get(next(i), 0.0),
            reverse=True,
        )
        # 更新 score 为 rerank 分数
        for i, item in enumerate(reranked):
            new_score = rank_map.get(i, item["score"])
            item["score"] = new_score
            item["reranked"] = True

        logger.info(
            "Rerank %d → %d | tokens=%s",
            len(candidates), len(reranked),
            resp.usage.get("total_tokens", "?") if resp.usage else "?",
        )
        return reranked

    @staticmethod
    def _build_filter(filter_kwargs: dict[str, Any] | None) -> models.Filter | None:
        """将 dict 条件转为 Qdrant Filter。"""
        if not filter_kwargs:
            return None
        conditions = []
        for k, v in filter_kwargs.items():
            if k.endswith("_gte"):
                conditions.append(
                    models.FieldCondition(
                        key=k[:-4], range=models.Range(gte=v),
                    )
                )
            elif k.endswith("_lte"):
                conditions.append(
                    models.FieldCondition(
                        key=k[:-4], range=models.Range(lte=v),
                    )
                )
            else:
                conditions.append(
                    models.FieldCondition(
                        key=k, match=models.MatchValue(value=v),
                    )
                )
        return models.Filter(must=conditions) if conditions else None

    # ── 计数 ────────────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        """当前 collection 中的 point 总数。"""
        try:
            return self.client.count(collection_name=self.collection).count
        except Exception:
            return 0

    # ── Save / Load（Qdrant 自动持久化，no-op）────────────────────────────

    def save(self, path: str | Path) -> None:
        """Qdrant 自动持久化，此方法为 no-op。"""
        pass

    def load(self, path: str | Path) -> None:
        """Qdrant 自动加载，此方法为 no-op。"""
        pass

    # ── 清空 ────────────────────────────────────────────────────────────────

    def clear(self) -> None:
        """删除并重建 collection，清空所有数据。"""
        self.client.delete_collection(collection_name=self.collection)
        self._ensure_collection()
        logger.info("Cleared collection '%s'", self.collection)

    def list_papers(self) -> list[dict[str, Any]]:
        """列出已入库的论文（去重 paper_id），每项含 paper_title / paper_id / chunk_count。
        通过 scroll 遍历所有 points，按 paper_id 聚合。"""
        papers: dict[str, dict[str, Any]] = {}
        offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection, limit=500,
                with_payload=True, with_vectors=False, offset=offset,
            )
            for pt in points:
                payload = pt.payload or {}
                pid = payload.get("paper_id", "?")
                if pid not in papers:
                    papers[pid] = {
                        "paper_id": pid,
                        "paper_title": payload.get("paper_title", "?"),
                        "chunk_count": 0,
                    }
                papers[pid]["chunk_count"] += 1
            if next_offset is None:
                break
            offset = next_offset
        return sorted(papers.values(), key=lambda x: x["paper_title"])

    def delete_by_paper_id(self, paper_id: str) -> int:
        """删除指定 paper_id 的所有 points，返回删除数量。"""
        from qdrant_client.models import FilterSelector

        # 先计数
        count_result = self.client.count(
            collection_name=self.collection,
            count_filter=models.Filter(
                must=[models.FieldCondition(
                    key="paper_id", match=models.MatchValue(value=paper_id),
                )],
            ),
        )
        before = count_result.count
        if before == 0:
            return 0

        self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(
                        key="paper_id", match=models.MatchValue(value=paper_id),
                    )],
                ),
            ),
        )
        logger.info("Deleted %d points for paper_id='%s'", before, paper_id)
        return before

    # ── 文本切分 ────────────────────────────────────────────────────────────

    def _chunk_text(self, content: str, chunk_size: int, chunk_overlap: int) -> list[dict[str, Any]]:
        """将文本按段落切分为 chunks。"""
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
        paragraphs_with_pos = self._split_paragraphs_with_pos(text)

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

            if current:
                chunks.append({
                    "content": current,
                    "char_start": current_start,
                    "char_end": current_start + len(current),
                })
                current = ""

            if len(para) <= chunk_size:
                current = para
                current_start = para_start
                continue

            for item in self._chunk_long_text(para, chunk_size, chunk_overlap):
                chunks.append({
                    "content": item["content"],
                    "char_start": para_start + item["char_start"],
                    "char_end": para_start + item["char_end"],
                })

        if current:
            chunks.append({
                "content": current,
                "char_start": current_start,
                "char_end": current_start + len(current),
            })

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

    @staticmethod
    def _split_paragraphs_with_pos(text: str) -> list[tuple[str, int]]:
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

    @staticmethod
    def _chunk_long_text(text: str, chunk_size: int, chunk_overlap: int) -> list[dict[str, Any]]:
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
