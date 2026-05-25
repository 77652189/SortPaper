"""
Qdrant 向量存储：Hybrid Search（Dense + Sparse + RRF 融合）。

每个 chunk 作为 Qdrant point 写入 dense 向量；DashScope provider 可额外写入 sparse
向量并启用 Hybrid Search，OpenAI provider 默认使用 dense-only 检索。
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

def _stable_hash(s: str) -> int:
    """确定性哈希，替代 Python hash()（跨进程不稳定）。"""
    return int(hashlib.md5(s.encode()).hexdigest()[:16], 16)


def qdrant_point_id(paper_id: str, chunk_id: str) -> int:
    return _stable_hash(f"{paper_id}:{chunk_id}") % (2 ** 63)


def openai_embedding_client_kwargs(api_key: str) -> dict[str, Any]:
    """Build OpenAI embedding client kwargs without inheriting OPENAI_BASE_URL."""
    kwargs: dict[str, Any] = {"api_key": api_key, "timeout": 60.0}
    if OPENAI_EMBEDDING_BASE_URL:
        kwargs["base_url"] = OPENAI_EMBEDDING_BASE_URL
    return kwargs


def dashscope_rerank_api_key() -> str:
    return _env_value("DASHSCOPE_API_KEY")


def _env_value(name: str, default: str = "") -> str:
    return (os.getenv(name) or os.getenv(f"\ufeff{name}") or default).strip()


def _payload_has_quality_enrichment(payload: dict[str, Any]) -> bool:
    """Infer whether legacy payload already contains Map-Reduce quality metadata."""
    quality_keys = (
        "category",
        "classify_status",
        "paper_summary",
        "classify_reason",
        "target_products",
        "organisms",
        "credibility",
        "fermentation_relevance",
        "is_actionable",
    )
    return any(payload.get(key) not in (None, "", [], {}) for key in quality_keys)

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, SparseVectorParams,
    SparseIndexParams, SparseVector,
    FusionQuery, Fusion, Prefetch,
)

from app_config import (
    EMBEDDING_API_KEY_ENV,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    OPENAI_EMBEDDING_BASE_URL,
)

logger = logging.getLogger(__name__)

QUERY_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "what", "how",
    "are", "was", "were", "into", "over", "high", "problem",
}

COLLECTION_NAME = "papers"
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
        """确保 collection 存在，并与当前 embedding provider 的向量配置一致。"""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection in collections:
            info = self.client.get_collection(self.collection)
            has_sparse = bool(info.config.params.sparse_vectors)
            current_dim = info.config.params.vectors.get(
                DENSE_VECTOR_NAME
            ).size if DENSE_VECTOR_NAME in info.config.params.vectors else 0
            if current_dim == EMBEDDING_DIM and has_sparse == self._uses_sparse_vectors:
                return
            logger.warning(
                "collection vector config mismatch; recreating | collection=%s current_dim=%s target_dim=%s current_sparse=%s target_sparse=%s",
                self.collection,
                current_dim,
                EMBEDDING_DIM,
                has_sparse,
                self._uses_sparse_vectors,
            )
            self.client.delete_collection(collection_name=self.collection)

        self._create_collection(EMBEDDING_DIM)

    @property
    def _uses_sparse_vectors(self) -> bool:
        return EMBEDDING_PROVIDER == "dashscope"

    def _create_collection(self, dense_dim: int) -> None:
        kwargs: dict[str, Any] = {
            "collection_name": self.collection,
            "vectors_config": {
                DENSE_VECTOR_NAME: VectorParams(
                    size=dense_dim, distance=Distance.COSINE,
                ),
            },
        }
        if self._uses_sparse_vectors:
            kwargs["sparse_vectors_config"] = {
                SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            }
        self.client.create_collection(**kwargs)
        logger.info(
            "Created collection '%s' (provider=%s dense=%dd sparse=%s)",
            self.collection,
            EMBEDDING_PROVIDER,
            dense_dim,
            self._uses_sparse_vectors,
        )

    def _recreate_collection(self, actual_dim: int) -> None:
        """删除并重建 collection（维度不匹配时）。"""
        self.client.delete_collection(collection_name=self.collection)
        self._create_collection(actual_dim)
        logger.info(
            "Recreated collection '%s' (dense=%dd)", self.collection, actual_dim,
        )

    # ── Embedding ──────────────────────────────────────────────────────────

    def embed(self, text: str) -> tuple[list[float], dict[int, float]]:
        """Return (dense_vector, sparse_dict) for the configured embedding provider."""
        if EMBEDDING_PROVIDER == "openai":
            return self._embed_openai(text), {}
        return self._embed_dashscope(text)

    def _embed_openai(self, text: str) -> list[float]:
        from openai import OpenAI

        api_key = _env_value(EMBEDDING_API_KEY_ENV)
        if not api_key:
            raise ValueError(f"{EMBEDDING_API_KEY_ENV} 未设置")

        from src.judge.llm_runtime import run_llm_call

        client = OpenAI(**openai_embedding_client_kwargs(api_key))
        response = run_llm_call(
            lambda: client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text,
                dimensions=EMBEDDING_DIM,
            ),
            label="openai-embedding",
        )
        if not response.data:
            raise ValueError("No embedding returned from OpenAI")
        return [float(v) for v in response.data[0].embedding]

    def _embed_dashscope(self, text: str) -> tuple[list[float], dict[int, float]]:
        from dashscope import TextEmbedding

        api_key = _env_value(EMBEDDING_API_KEY_ENV)
        if not api_key:
            raise ValueError(f"{EMBEDDING_API_KEY_ENV} 未设置")
        os.environ.setdefault(EMBEDDING_API_KEY_ENV, api_key)

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                from src.judge.llm_runtime import run_llm_call

                response = run_llm_call(
                    lambda: TextEmbedding.call(
                        model=EMBEDDING_MODEL,
                        input=text,
                        output_type="dense&sparse",
                        api_key=api_key,
                    ),
                    label="dashscope-embedding",
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

        chunk_id = metadata.get("chunk_id", f"{worker_type}_{_stable_hash(content)}")

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

        vector_payload: dict[str, Any] = {DENSE_VECTOR_NAME: dense_vec}
        if self._uses_sparse_vectors:
            indices = sorted(sparse_dict.keys())
            values = [sparse_dict[i] for i in indices]
            vector_payload[SPARSE_VECTOR_NAME] = SparseVector(
                indices=indices, values=values,
            )

        self.client.upsert(
            collection_name=self.collection,
            points=[
                models.PointStruct(
                    id=qdrant_point_id(paper_id, chunk_id),
                    vector=vector_payload,
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

        # Rerank 模式多召回一些候选
        fetch_multiplier = 8 if rerank else 2
        if self._uses_sparse_vectors and sparse_dict:
            sparse_indices = sorted(sparse_dict.keys())
            sparse_values = [sparse_dict[i] for i in sparse_indices]
            sparse_vec = SparseVector(indices=sparse_indices, values=sparse_values)
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
        else:
            candidates = self.client.query_points(
                collection_name=self.collection,
                query=dense_vec,
                using=DENSE_VECTOR_NAME,
                query_filter=qdrant_filter,
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
            results = self._merge_lexical_candidates(
                query=query,
                results=results,
                qdrant_filter=qdrant_filter,
                limit=limit * 4,
            )
            results = self._rerank(query, results, top_n=limit)
            results = self._apply_query_relevance_boost(query, results)

        return results[:limit]

    def _merge_lexical_candidates(
        self,
        *,
        query: str,
        results: list[dict[str, Any]],
        qdrant_filter: models.Filter | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        lexical = self._lexical_candidates(query, qdrant_filter=qdrant_filter, limit=limit)
        seen = {item["id"] for item in results}
        for item in lexical:
            if item["id"] in seen:
                continue
            seen.add(item["id"])
            results.append(item)
        return results

    def _lexical_candidates(
        self,
        query: str,
        *,
        qdrant_filter: models.Filter | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        if not self._query_relevance_terms(query):
            return []

        scored: list[dict[str, Any]] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=qdrant_filter,
                limit=500,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            for point in points:
                payload = point.payload or {}
                lexical_score = self._lexical_match_score(
                    query,
                    self._rerank_document_text(payload),
                )
                if lexical_score <= 0:
                    continue
                scored.append({
                    "id": point.id,
                    "score": lexical_score,
                    "payload": payload,
                    "lexical_score": lexical_score,
                })
            if offset is None:
                break

        scored.sort(key=lambda item: item["lexical_score"], reverse=True)
        return scored[:limit]

    def _rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """qwen3-rerank 二次排序。仅对同次请求内的候选文档做相对排序。"""
        import dashscope
        from http import HTTPStatus

        documents = [self._rerank_document_text(c["payload"]) for c in candidates]
        if not documents:
            return candidates
        api_key = dashscope_rerank_api_key()
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY 未设置，无法使用 qwen3-rerank")

        resp = dashscope.TextReRank.call(
            model="qwen3-rerank",
            query=query,
            documents=documents,
            top_n=min(top_n, len(documents)),
            return_documents=False,
            instruct="Given a scientific paper search query, retrieve relevant passages that answer the query.",
            api_key=api_key,
        )

        if resp.status_code != HTTPStatus.OK:
            logger.warning("Rerank 失败，回退到原始排序 | code=%s msg=%s", resp.code, resp.message)
            return candidates

        # 按 rerank 结果重排，rank_map 的 key 是 rerank 返回的原始候选下标。
        rank_map = {
            r.get("index", -1): r.get("relevance_score", 0.0)
            for r in (resp.output.get("results") if isinstance(resp.output, dict)
                      else getattr(resp.output, "results", []))
        }
        ranked_pairs = sorted(
            enumerate(candidates),
            key=lambda item: rank_map.get(item[0], 0.0),
            reverse=True,
        )
        reranked = []
        for original_index, item in ranked_pairs:
            new_score = rank_map.get(original_index, item["score"])
            item["score"] = new_score
            item["reranked"] = True
            reranked.append(item)

        logger.info(
            "Rerank %d → %d | tokens=%s",
            len(candidates), len(reranked),
            resp.usage.get("total_tokens", "?") if resp.usage else "?",
        )
        return reranked

    @classmethod
    def _apply_query_relevance_boost(
        cls,
        query: str,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        for item in results:
            lexical_score = cls._lexical_match_score(
                query,
                cls._rerank_document_text(item["payload"]),
            )
            if lexical_score <= 0:
                continue
            item["lexical_score"] = lexical_score
            item["rerank_score"] = item.get("score", 0.0)
            boost_cap = 1.0 if cls._query_requires_strict_anchor(query) else 0.35
            item["score"] = float(item.get("score", 0.0) or 0.0) + min(lexical_score, boost_cap)
        return sorted(results, key=lambda item: item.get("score", 0.0), reverse=True)

    @staticmethod
    def _rerank_document_text(payload: dict[str, Any]) -> str:
        fields = [
            payload.get("paper_title", ""),
            payload.get("category", ""),
            ", ".join(map(str, payload.get("target_products") or [])),
            ", ".join(map(str, payload.get("organisms") or [])),
            payload.get("paper_summary", ""),
            payload.get("content", ""),
        ]
        return "\n".join(str(field) for field in fields if field)

    @classmethod
    def _lexical_match_score(cls, query: str, text: str) -> float:
        normalized = cls._normalize_search_text(text)
        if not cls._has_required_query_anchor(query, normalized):
            return 0.0
        score = 0.0
        for term, weight in cls._query_relevance_terms(query).items():
            if term == "lnt":
                continue
            if term in normalized:
                score += weight
        if "fermentation_experiment" in normalized or "fermentation experiment" in normalized:
            score += 0.12
        return score

    @classmethod
    def _has_required_query_anchor(cls, query: str, normalized_text: str) -> bool:
        if cls._query_requires_strict_anchor(query):
            return (
                "lnt ii" in normalized_text
                or "lacto n triose ii" in normalized_text
                or "lacto-n-triose ii" in normalized_text
            )
        return True

    @classmethod
    def _query_requires_strict_anchor(cls, query: str) -> bool:
        normalized_query = cls._normalize_search_text(query)
        return "lnt ii" in normalized_query or "lnt-ii" in normalized_query

    @classmethod
    def _query_relevance_terms(cls, query: str) -> dict[str, float]:
        normalized = cls._normalize_search_text(query)
        terms: dict[str, float] = {}

        def add(term: str, weight: float) -> None:
            key = cls._normalize_search_text(term)
            if key:
                terms[key] = max(terms.get(key, 0.0), weight)

        for token in re.findall(r"[a-z0-9][a-z0-9'′-]{2,}", normalized):
            if token not in QUERY_STOPWORDS:
                add(token, 0.04)
        for phrase in re.findall(
            r"\b[a-z0-9][a-z0-9'′-]*\s+(?:i|ii|iii|iv|v|vi|vii|viii|ix|x)\b",
            normalized,
        ):
            add(phrase, 0.18)

        if "lnt ii" in normalized or "lnt-ii" in normalized:
            add("lnt ii", 0.30)
            add("lacto-n-triose ii", 0.30)
            add("lacto n triose ii", 0.30)
            add("lgtA", 0.12)
            add("RBS", 0.10)
        if "大肠杆菌" in query:
            add("escherichia coli", 0.16)
            add("e. coli", 0.16)
        if "代谢" in query:
            add("metabolic", 0.08)
            add("carbon flux", 0.12)
        if "路径" in query or "途径" in query:
            add("pathway", 0.08)
            add("biosynthesis pathway", 0.10)
        if "中间产物" in query:
            add("intermediate", 0.10)
        if "残留" in query or "过高" in query or "积累" in query:
            add("accumulation", 0.10)
            add("accumulated", 0.10)
        if "改造" in query or "逻辑" in query:
            add("engineering", 0.08)
            add("strategy", 0.08)
        return terms

    @staticmethod
    def _normalize_search_text(value: str) -> str:
        text = str(value).lower()
        text = text.replace("\u00ad", "")
        text = re.sub(r"[\u2010-\u2015−]", "-", text)
        text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

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
                        "enrichment_status": "pending",
                        "enrichment_status_note": "",
                        "_has_pending_status": False,
                        "_has_done_status": False,
                        "_has_missing_status": False,
                        "_has_quality_payload": False,
                    }
                papers[pid]["chunk_count"] += 1
                status = payload.get("enrichment_status")
                if status == "pending":
                    papers[pid]["_has_pending_status"] = True
                elif status == "done":
                    papers[pid]["_has_done_status"] = True
                else:
                    papers[pid]["_has_missing_status"] = True
                if _payload_has_quality_enrichment(payload):
                    papers[pid]["_has_quality_payload"] = True
            if next_offset is None:
                break
            offset = next_offset
        for paper in papers.values():
            if paper.pop("_has_pending_status"):
                paper["enrichment_status"] = "pending"
                paper["enrichment_status_note"] = ""
            elif paper.pop("_has_done_status") or paper.pop("_has_quality_payload"):
                paper["enrichment_status"] = "done"
                paper["enrichment_status_note"] = ""
            elif paper.pop("_has_missing_status"):
                paper["enrichment_status"] = "pending"
                paper["enrichment_status_note"] = "旧数据缺少质量分析标记"
            else:
                paper["enrichment_status"] = "pending"
                paper["enrichment_status_note"] = ""
        return sorted(papers.values(), key=lambda x: x["paper_title"])

    def scroll_by_paper_id(self, paper_id: str, page_size: int = 500) -> list[Any]:
        """Return all Qdrant points for one paper_id, following scroll pagination."""
        points: list[Any] = []
        offset = None
        q_filter = models.Filter(
            must=[models.FieldCondition(
                key="paper_id", match=models.MatchValue(value=paper_id),
            )],
        )
        while True:
            batch, next_offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=q_filter,
                limit=page_size,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            points.extend(batch)
            if next_offset is None:
                break
            offset = next_offset
        return points

    def update_paper_payload(
        self,
        paper_id: str,
        payload: dict[str, Any],
        point_ids: list[int | str] | None = None,
    ) -> int:
        """Update payload for a paper. Use point_ids when already known."""
        for key, value in list(payload.items()):
            if isinstance(value, tuple):
                payload[key] = list(value)
        if point_ids:
            self.client.set_payload(
                collection_name=self.collection,
                payload=payload,
                points=point_ids,
            )
            return len(point_ids)

        q_filter = models.Filter(
            must=[models.FieldCondition(
                key="paper_id", match=models.MatchValue(value=paper_id),
            )],
        )
        self.client.set_payload(
            collection_name=self.collection,
            payload=payload,
            points=q_filter,
        )
        return self.client.count(
            collection_name=self.collection,
            count_filter=q_filter,
        ).count

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
