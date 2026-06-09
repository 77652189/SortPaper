"""
Qdrant 向量存储：Hybrid Search（Dense + Sparse + RRF 融合）。

每个 chunk 作为 Qdrant point 写入 dense 向量；DashScope provider 可额外写入 sparse
向量并启用 Hybrid Search，OpenAI provider 默认使用 dense-only 检索。
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.runtime_env import load_project_env

load_project_env()

def _stable_hash(s: str) -> int:
    """确定性哈希，替代 Python hash()（跨进程不稳定）。"""
    return point_identity.stable_hash(s)


def qdrant_point_id(paper_id: str, chunk_id: str) -> int:
    return point_identity.qdrant_point_id(paper_id, chunk_id)


def openai_embedding_client_kwargs(api_key: str) -> dict[str, Any]:
    """Build OpenAI embedding client kwargs without inheriting OPENAI_BASE_URL."""
    return _openai_embedding_client_kwargs(
        api_key,
        base_url=OPENAI_EMBEDDING_BASE_URL,
    )


def dashscope_rerank_api_key() -> str:
    return _dashscope_rerank_api_key()


def _env_value(name: str, default: str = "") -> str:
    return (os.getenv(name) or os.getenv(f"\ufeff{name}") or default).strip()


def _response_debug_summary(response: Any) -> str:
    return response_debug_summary(response)


def _payload_has_quality_enrichment(payload: dict[str, Any]) -> bool:
    """Infer whether legacy payload already contains Map-Reduce quality metadata."""
    return paper_catalog.payload_has_quality_enrichment(payload)

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, SparseVectorParams,
    SparseIndexParams, SparseVector,
)

from src.application.settings import (
    DEFAULT_QDRANT_COLLECTION,
    EMBEDDING_API_KEY_ENV,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    OPENAI_EMBEDDING_BASE_URL,
    QWEN3_LOCAL_EMBEDDING_MAX_LENGTH,
    QWEN3_LOCAL_EMBEDDING_QUERY_INSTRUCT,
)
from src.adapters.embedding.providers import (
    DashScopeEmbeddingProvider,
    LocalQwen3EmbeddingProvider,
    OpenAIEmbeddingProvider,
    openai_embedding_client_kwargs as _openai_embedding_client_kwargs,
    response_debug_summary,
)
from src.adapters.rerank.dashscope import dashscope_rerank_api_key as _dashscope_rerank_api_key
from src.adapters.store import (
    qdrant_collection,
    qdrant_lexical,
    qdrant_library,
    qdrant_payload,
    qdrant_rerank,
    qdrant_search,
    qdrant_search_pipeline,
)
from src.domain import (
    evidence_policy,
    neighbor_policy,
    paper_catalog,
    point_identity,
    result_policy,
    text_chunking,
)
from src.store.search_text import build_search_text

logger = logging.getLogger(__name__)

QUERY_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "what", "how",
    "are", "was", "were", "into", "over", "high", "problem",
    "pdf", "text", "chunk",
}

COLLECTION_NAME = DEFAULT_QDRANT_COLLECTION
DENSE_VECTOR_NAME = "dense"              # 密集向量名
SPARSE_VECTOR_NAME = "sparse"            # 稀疏向量名
_QWEN3_LOCAL_MODEL: Any | None = None
_QWEN3_LOCAL_TOKENIZER: Any | None = None


def _load_qwen3_local_embedding_model() -> tuple[Any, Any]:
    global _QWEN3_LOCAL_MODEL, _QWEN3_LOCAL_TOKENIZER
    if _QWEN3_LOCAL_MODEL is not None and _QWEN3_LOCAL_TOKENIZER is not None:
        return _QWEN3_LOCAL_TOKENIZER, _QWEN3_LOCAL_MODEL

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL,
        padding_side="left",
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(EMBEDDING_MODEL, local_files_only=True)
    model.eval()
    _QWEN3_LOCAL_TOKENIZER = tokenizer
    _QWEN3_LOCAL_MODEL = model
    logger.info("Loaded local Qwen3 embedding model: %s", EMBEDDING_MODEL)
    return tokenizer, model


class QdrantStore:
    """Qdrant 向量存储，支持 Dense + Sparse 混合检索。"""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = COLLECTION_NAME,
    ) -> None:
        self.client = QdrantClient(url=url)
        self.collection = collection
        self._lexical_document_cache: list[dict[str, Any]] | None = None
        self._lexical_index_cache: dict[str, Any] | None = None
        self._ensure_collection()

    # ── 集合管理 ────────────────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        """确保 collection 存在，并与当前 embedding provider 的向量配置一致。"""
        return qdrant_collection.ensure_collection(
            client=self.client,
            collection=self.collection,
            dense_dim=EMBEDDING_DIM,
            embedding_provider=EMBEDDING_PROVIDER,
            uses_sparse_vectors=self._uses_sparse_vectors,
            dense_vector_name=DENSE_VECTOR_NAME,
            sparse_vector_name=SPARSE_VECTOR_NAME,
        )
    @property
    def _uses_sparse_vectors(self) -> bool:
        return EMBEDDING_PROVIDER == "dashscope"

    def _create_collection(self, dense_dim: int) -> None:
        return qdrant_collection.create_collection(
            client=self.client,
            collection=self.collection,
            dense_dim=dense_dim,
            embedding_provider=EMBEDDING_PROVIDER,
            uses_sparse_vectors=self._uses_sparse_vectors,
            dense_vector_name=DENSE_VECTOR_NAME,
            sparse_vector_name=SPARSE_VECTOR_NAME,
        )

    def _recreate_collection(self, actual_dim: int) -> None:
        """删除并重建 collection（维度不匹配时）。"""
        return qdrant_collection.recreate_collection(
            client=self.client,
            collection=self.collection,
            dense_dim=actual_dim,
            embedding_provider=EMBEDDING_PROVIDER,
            uses_sparse_vectors=self._uses_sparse_vectors,
            dense_vector_name=DENSE_VECTOR_NAME,
            sparse_vector_name=SPARSE_VECTOR_NAME,
        )

    # ── Embedding ──────────────────────────────────────────────────────────

    def embed(self, text: str, *, is_query: bool = False) -> tuple[list[float], dict[int, float]]:
        """Return (dense_vector, sparse_dict) for the configured embedding provider."""
        if EMBEDDING_PROVIDER == "openai":
            return self._embed_openai(text), {}
        if EMBEDDING_PROVIDER == "qwen3_local":
            return self._embed_qwen3_local(text, is_query=is_query), {}
        return self._embed_dashscope(text)

    def _embed_openai(self, text: str) -> list[float]:
        provider = OpenAIEmbeddingProvider(
            api_key_env=EMBEDDING_API_KEY_ENV,
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIM,
            base_url=OPENAI_EMBEDDING_BASE_URL,
        )
        return provider.embed(text)

    def _embed_qwen3_local(self, text: str, *, is_query: bool = False) -> list[float]:
        provider = LocalQwen3EmbeddingProvider(
            model_path=EMBEDDING_MODEL,
            max_length=QWEN3_LOCAL_EMBEDDING_MAX_LENGTH,
            query_instruct=QWEN3_LOCAL_EMBEDDING_QUERY_INSTRUCT,
        )
        return provider.embed(text, is_query=is_query)

    def _embed_dashscope(self, text: str) -> tuple[list[float], dict[int, float]]:
        provider = DashScopeEmbeddingProvider(
            api_key_env=EMBEDDING_API_KEY_ENV,
            model=EMBEDDING_MODEL,
        )
        return provider.embed(text)

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
        payload["search_text"] = build_search_text(payload)
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
        self._invalidate_lexical_cache()

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
        selective_rerank: bool = False,
        rerank_top_n: int | None = None,
        lexical_backfill: bool = False,
        neighbor_backfill: bool = False,
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

        # Rerank/backfill modes fetch a larger candidate pool before local post-processing.
        fetch_multiplier = qdrant_search.fetch_multiplier(
            rerank=rerank,
            selective_rerank=selective_rerank,
            lexical_backfill=lexical_backfill,
            neighbor_backfill=neighbor_backfill,
        )
        results = qdrant_search.run_vector_query(
            client=self.client,
            collection=self.collection,
            dense_vector_name=DENSE_VECTOR_NAME,
            sparse_vector_name=SPARSE_VECTOR_NAME,
            dense_vec=dense_vec,
            sparse_dict=sparse_dict,
            qdrant_filter=qdrant_filter,
            uses_sparse_vectors=self._uses_sparse_vectors,
            limit=limit,
            multiplier=fetch_multiplier,
            score_threshold=score_threshold,
        )
        # Rerank: 用 qwen3-rerank 对候选集重新打分排序
        return qdrant_search_pipeline.postprocess_results(
            query=query,
            results=results,
            qdrant_filter=qdrant_filter,
            limit=limit,
            rerank=rerank,
            selective_rerank=selective_rerank,
            rerank_top_n=rerank_top_n,
            lexical_backfill=lexical_backfill,
            neighbor_backfill=neighbor_backfill,
            backfill_anchor_count=self._backfill_anchor_count,
            selective_rerank_reason=lambda q, r: self._selective_rerank_reason(q, r, limit=limit),
            merge_lexical_candidates=self._merge_lexical_candidates,
            merge_neighbor_candidates=self._merge_neighbor_candidates,
            rerank_candidates=lambda q, r, top_n: self._rerank(q, r, top_n=top_n),
            apply_query_relevance_boost=self._apply_query_relevance_boost,
            restore_anchor_candidates=lambda r, a, lim: self._restore_anchor_candidates(r, a, limit=lim),
        )

    def search_evidence(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float | None = None,
        filter_kwargs: dict[str, Any] | None = None,
        rerank: bool = False,
        lexical_backfill: bool = False,
        neighbor_backfill: bool = False,
        paper_limit: int = 10,
        per_paper_limit: int = 4,
        lead_paper_limit: int = 12,
    ) -> list[dict[str, Any]]:
        """Two-stage retrieval: find papers first, then surface evidence chunks.

        The normal chunk search can find the right paper but miss the best evidence chunk.
        This method collects a larger candidate pool once, ranks papers from that pool,
        and then reorders candidates so paper representatives and deeper top-paper
        evidence are both visible. It avoids repeated embedding calls for the same query.
        """
        if limit <= 0:
            return []

        filters = dict(filter_kwargs or {})
        if filters.get("paper_id"):
            return self.search(
                query=query,
                limit=limit,
                score_threshold=score_threshold,
                filter_kwargs=filters,
                rerank=rerank,
                lexical_backfill=lexical_backfill,
                neighbor_backfill=neighbor_backfill,
            )

        probe_limit = max(limit * 8, paper_limit * max(per_paper_limit, lead_paper_limit))
        paper_candidates = self.search(
            query=query,
            limit=probe_limit,
            score_threshold=score_threshold,
            filter_kwargs=filters or None,
            rerank=rerank,
            lexical_backfill=lexical_backfill,
            neighbor_backfill=neighbor_backfill,
        )
        ranked_papers = self._rank_papers_from_results(paper_candidates, limit=paper_limit)
        if not ranked_papers:
            return paper_candidates[:limit]

        candidates_by_paper: dict[str, list[dict[str, Any]]] = {paper_id: [] for paper_id in ranked_papers}
        for result in paper_candidates:
            paper_id = str((result.get("payload") or {}).get("paper_id") or "")
            if paper_id in candidates_by_paper:
                result["paper_rank"] = ranked_papers.index(paper_id) + 1
                result["evidence_stage"] = "pooled"
                candidates_by_paper[paper_id].append(result)

        evidence_by_paper: list[list[dict[str, Any]]] = []
        for paper_index, paper_id in enumerate(ranked_papers):
            group_limit = (
                max(per_paper_limit, min(limit, lead_paper_limit))
                if paper_index == 0
                else per_paper_limit
            )
            evidence_by_paper.append(candidates_by_paper.get(paper_id, [])[:group_limit])
        prioritized = self._prioritize_evidence_groups(evidence_by_paper)
        return self._deduplicate_results(prioritized + paper_candidates)[:limit]

    def search_paper_local_evidence(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float | None = None,
        filter_kwargs: dict[str, Any] | None = None,
        rerank: bool = False,
        lexical_backfill: bool = False,
        neighbor_backfill: bool = False,
        paper_limit: int = 8,
        per_paper_limit: int = 4,
        lead_paper_limit: int = 12,
    ) -> list[dict[str, Any]]:
        """Find likely papers first, then rescore evidence inside those papers."""
        if limit <= 0:
            return []

        filters = dict(filter_kwargs or {})
        if filters.get("paper_id"):
            return self.search(
                query=query,
                limit=limit,
                score_threshold=score_threshold,
                filter_kwargs=filters,
                rerank=rerank,
                lexical_backfill=lexical_backfill,
                neighbor_backfill=neighbor_backfill,
            )

        probe_limit = max(limit * 8, paper_limit * max(per_paper_limit, lead_paper_limit))
        paper_candidates = self.search(
            query=query,
            limit=probe_limit,
            score_threshold=score_threshold,
            filter_kwargs=filters or None,
            rerank=rerank,
            lexical_backfill=lexical_backfill,
            neighbor_backfill=neighbor_backfill,
        )
        ranked_papers = self._rank_papers_from_results(paper_candidates, limit=paper_limit)
        if not ranked_papers:
            return paper_candidates[:limit]

        paper_rank = {paper_id: index + 1 for index, paper_id in enumerate(ranked_papers)}
        query_terms = self._query_relevance_terms(query)
        if not query_terms:
            return paper_candidates[:limit]

        qdrant_filter = self._build_filter(filters or None)
        requires_strict_anchor = self._query_requires_strict_anchor(query)
        lexical_index = self._lexical_document_index()
        evidence_by_paper: dict[str, list[dict[str, Any]]] = {paper_id: [] for paper_id in ranked_papers}
        for entry in self._lexical_document_entries(qdrant_filter=qdrant_filter):
            payload = entry.get("payload") or {}
            paper_id = str(payload.get("paper_id") or "")
            rank = paper_rank.get(paper_id)
            if rank is None:
                continue
            lexical_score = self._lexical_idf_match_score_from_terms(
                entry["normalized_text"],
                query_terms=query_terms,
                requires_strict_anchor=requires_strict_anchor,
                lexical_index=lexical_index,
            )
            if lexical_score <= 0:
                continue
            score = lexical_score + (0.03 / rank)
            evidence_by_paper[paper_id].append({
                "id": entry["id"],
                "score": score,
                "payload": payload,
                "lexical_score": lexical_score,
                "paper_rank": rank,
                "evidence_stage": "paper_local",
            })

        evidence_groups: list[list[dict[str, Any]]] = []
        for index, paper_id in enumerate(ranked_papers):
            group_limit = (
                max(per_paper_limit, min(limit, lead_paper_limit))
                if index == 0
                else per_paper_limit
            )
            group = sorted(
                evidence_by_paper.get(paper_id, []),
                key=lambda item: item.get("score", 0.0),
                reverse=True,
            )
            evidence_groups.append(group[:group_limit])

        prioritized = self._prioritize_evidence_groups(evidence_groups)
        return self._deduplicate_results(prioritized + paper_candidates)[:limit]

    @staticmethod
    def _rank_papers_from_results(
        results: list[dict[str, Any]],
        *,
        limit: int,
    ) -> list[str]:
        return result_policy.rank_papers_from_results(results, limit=limit)

    @staticmethod
    def _interleave_ranked_groups(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        return result_policy.interleave_ranked_groups(groups)

    @classmethod
    def _prioritize_evidence_groups(cls, groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        return result_policy.prioritize_evidence_groups(groups)

    @staticmethod
    def _score_rank_evidence_groups(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        return result_policy.score_rank_evidence_groups(groups)

    @staticmethod
    def _deduplicate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return result_policy.deduplicate_results(results)

    @staticmethod
    def _result_key(result: dict[str, Any]) -> tuple[str, str]:
        return result_policy.result_key(result)

    @staticmethod
    def _backfill_anchor_count(limit: int) -> int:
        return result_policy.backfill_anchor_count(limit)

    @classmethod
    def _restore_anchor_candidates(
        cls,
        results: list[dict[str, Any]],
        anchor_results: list[dict[str, Any]],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        return result_policy.restore_anchor_candidates(
            results,
            anchor_results,
            limit=limit,
        )

    @classmethod
    def _protect_prefix_candidates(
        cls,
        results: list[dict[str, Any]],
        protected_results: list[dict[str, Any]],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        return result_policy.protect_prefix_candidates(
            results,
            protected_results,
            limit=limit,
        )

    def expand_neighbor_context(
        self,
        results: list[dict[str, Any]],
        *,
        filter_kwargs: dict[str, Any] | None = None,
        per_result_limit: int = 2,
        total_limit: int = 10,
        window: int = 2,
    ) -> list[dict[str, Any]]:
        """Return nearby chunks for already-ranked evidence without changing ranking."""
        if not results or per_result_limit <= 0 or total_limit <= 0:
            return []

        qdrant_filter = self._build_filter(filter_kwargs)
        candidate_limit = max(len(results) * per_result_limit, total_limit)
        neighbors = self._neighbor_candidates(
            results,
            qdrant_filter=qdrant_filter,
            limit=candidate_limit,
            window=window,
        )
        return neighbor_policy.context_chunks_from_neighbors(
            results,
            neighbors,
            total_limit=total_limit,
        )

    def expand_paper_local_context(
        self,
        query: str,
        results: list[dict[str, Any]],
        *,
        filter_kwargs: dict[str, Any] | None = None,
        paper_limit: int = 3,
        per_paper_limit: int = 2,
        total_limit: int = 5,
        content_type_preference: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return deeper evidence chunks from already-hit papers for answer synthesis."""
        if not query.strip() or not results or paper_limit <= 0 or per_paper_limit <= 0 or total_limit <= 0:
            return []

        ranked_papers = self._rank_papers_from_results(results, limit=paper_limit)
        if not ranked_papers:
            return []

        query_terms = self._query_relevance_terms(query)
        if not query_terms:
            return []

        qdrant_filter = self._build_filter(filter_kwargs)
        return evidence_policy.paper_local_context_from_entries(
            self._lexical_document_entries(qdrant_filter=qdrant_filter),
            ranked_papers=ranked_papers,
            existing_results=results,
            query_terms=query_terms,
            requires_strict_anchor=self._query_requires_strict_anchor(query),
            lexical_index=self._lexical_document_index(),
            per_paper_limit=per_paper_limit,
            total_limit=total_limit,
            content_type_preference=content_type_preference,
        )

    @staticmethod
    def _content_type_preference_bonus(
        payload: dict[str, Any],
        *,
        content_type_preference: str | None,
    ) -> float:
        return evidence_policy.content_type_preference_bonus(
            payload,
            content_type_preference=content_type_preference,
        )

    def _merge_lexical_candidates(
        self,
        *,
        query: str,
        results: list[dict[str, Any]],
        qdrant_filter: models.Filter | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        lexical = self._lexical_candidates(query, qdrant_filter=qdrant_filter, limit=limit)
        return result_policy.append_unique_by_id(results, lexical)

    def _merge_neighbor_candidates(
        self,
        *,
        results: list[dict[str, Any]],
        qdrant_filter: models.Filter | None,
        limit: int,
        window: int = 2,
    ) -> list[dict[str, Any]]:
        neighbors = self._neighbor_candidates(
            results,
            qdrant_filter=qdrant_filter,
            limit=limit,
            window=window,
        )
        return result_policy.append_unique_by_id(results, neighbors)

    def _neighbor_candidates(
        self,
        results: list[dict[str, Any]],
        *,
        qdrant_filter: models.Filter | None,
        limit: int,
        window: int,
    ) -> list[dict[str, Any]]:
        if not results:
            return []
        return neighbor_policy.neighbor_candidates_from_entries(
            results,
            self._lexical_document_entries(qdrant_filter=qdrant_filter),
            limit=limit,
            window=window,
        )

    @classmethod
    def _neighbor_distance(
        cls,
        source: dict[str, Any],
        candidate: dict[str, Any],
        *,
        window: int,
    ) -> int | None:
        return neighbor_policy.neighbor_distance(source, candidate, window=window)

    @classmethod
    def _chunk_id_neighbor_distance(
        cls,
        source: dict[str, Any],
        candidate: dict[str, Any],
        *,
        window: int,
    ) -> int | None:
        return neighbor_policy.chunk_id_neighbor_distance(source, candidate, window=window)

    @staticmethod
    def _parse_chunk_id_location(value: Any) -> dict[str, Any] | None:
        return neighbor_policy.parse_chunk_id_location(value)

    @staticmethod
    def _as_int(value: Any) -> int | None:
        return neighbor_policy.as_int(value)

    @staticmethod
    def _bbox_centers_are_close(
        source: dict[str, Any],
        candidate: dict[str, Any],
        *,
        center_ratio: float = 0.12,
    ) -> bool:
        return neighbor_policy.bbox_centers_are_close(
            source,
            candidate,
            center_ratio=center_ratio,
        )

    @staticmethod
    def _as_float(value: Any) -> float | None:
        return neighbor_policy.as_float(value)

    @staticmethod
    def _normalize_bbox(value: Any) -> tuple[float, float, float, float] | None:
        return neighbor_policy.normalize_bbox(value)

    @staticmethod
    def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        return neighbor_policy.bbox_center(bbox)

    def _lexical_candidates(
        self,
        query: str,
        *,
        qdrant_filter: models.Filter | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        query_terms = self._query_relevance_terms(query)
        if not query_terms:
            return []
        requires_strict_anchor = self._query_requires_strict_anchor(query)
        entries = self._lexical_candidate_entries(
            query_terms=query_terms,
            qdrant_filter=qdrant_filter,
            limit=limit,
        )
        return qdrant_lexical.lexical_candidates(
            entries=entries,
            query_terms=query_terms,
            requires_strict_anchor=requires_strict_anchor,
            score_fn=self._lexical_match_score_from_terms,
            limit=limit,
        )

    def _lexical_candidate_entries(
        self,
        *,
        query_terms: dict[str, float],
        qdrant_filter: models.Filter | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        if qdrant_filter is not None:
            filtered_entries = self._lexical_document_entries(qdrant_filter=qdrant_filter)
        else:
            filtered_entries = []
        return qdrant_lexical.candidate_entries(
            query_terms=query_terms,
            qdrant_filter=qdrant_filter,
            filtered_entries=filtered_entries,
            lexical_index=self._lexical_document_index(),
            limit=limit,
        )

    def _lexical_document_index(self) -> dict[str, Any]:
        cache = getattr(self, "_lexical_index_cache", None)
        if cache is not None:
            return cache

        entries = self._lexical_document_entries(qdrant_filter=None)
        cache = qdrant_lexical.document_index(entries)
        self._lexical_index_cache = cache
        return cache

    @classmethod
    def _lexical_index_terms(cls, value: str) -> set[str]:
        return evidence_policy.lexical_index_terms(value)

    def _lexical_document_entries(
        self,
        *,
        qdrant_filter: models.Filter | None,
    ) -> list[dict[str, Any]]:
        if qdrant_filter is not None:
            return self._scroll_lexical_document_entries(qdrant_filter=qdrant_filter)

        cache = getattr(self, "_lexical_document_cache", None)
        if cache is None:
            cache = self._scroll_lexical_document_entries(qdrant_filter=None)
            self._lexical_document_cache = cache
        return cache

    def _scroll_lexical_document_entries(
        self,
        *,
        qdrant_filter: models.Filter | None,
    ) -> list[dict[str, Any]]:
        return qdrant_lexical.scroll_document_entries(
            self.client,
            collection=self.collection,
            qdrant_filter=qdrant_filter,
            document_text_builder=self._rerank_document_text,
            text_normalizer=self._normalize_search_text,
        )

    def _rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """qwen3-rerank 二次排序。仅对同次请求内的候选文档做相对排序。"""
        return qdrant_rerank.rerank_candidates(
            query,
            candidates,
            top_n=top_n,
            api_key_getter=dashscope_rerank_api_key,
            document_text_builder=self._rerank_document_text,
        )
    @classmethod
    def _selective_rerank_reason(
        cls,
        query: str,
        results: list[dict[str, Any]],
        *,
        limit: int,
    ) -> str:
        return evidence_policy.selective_rerank_reason(query, results, limit=limit)
    @classmethod
    def _apply_query_relevance_boost(
        cls,
        query: str,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return evidence_policy.apply_query_relevance_boost(query, results)
    def _apply_fielded_lexical_boost(
        self,
        query: str,
        results: list[dict[str, Any]],
        *,
        weight: float,
    ) -> list[dict[str, Any]]:
        return evidence_policy.apply_fielded_lexical_boost(
            query,
            results,
            weight=weight,
            lexical_index=self._lexical_document_index(),
        )
    @classmethod
    def _fielded_lexical_score(
        cls,
        payload: dict[str, Any],
        *,
        query_terms: dict[str, float],
        requires_strict_anchor: bool,
        lexical_index: dict[str, Any],
    ) -> float:
        return evidence_policy.fielded_lexical_score(
            payload,
            query_terms=query_terms,
            requires_strict_anchor=requires_strict_anchor,
            lexical_index=lexical_index,
        )
    @classmethod
    def _fielded_lexical_texts(cls, payload: dict[str, Any]) -> dict[str, str]:
        return evidence_policy.fielded_lexical_texts(payload)
    @classmethod
    def _normalize_fielded_value(cls, value: Any) -> str:
        return evidence_policy.normalize_field_value(value)
    @classmethod
    def _rerank_document_text(cls, payload: dict[str, Any]) -> str:
        return evidence_policy.rerank_document_text(payload)
    @classmethod
    def _clean_rerank_title(cls, value: str) -> str:
        return evidence_policy.clean_rerank_title(value)
    @classmethod
    def _clean_rerank_list(cls, value: Any) -> str:
        return evidence_policy.clean_rerank_list(value)
    @classmethod
    def _clean_rerank_field(cls, value: Any) -> str:
        return evidence_policy.clean_rerank_field(value)

    @staticmethod
    def _expand_scientific_aliases(value: str) -> str:
        return evidence_policy.expand_scientific_aliases(value)
    @classmethod
    def _lexical_match_score(cls, query: str, text: str) -> float:
        return cls._lexical_match_score_normalized(
            query,
            cls._normalize_search_text(text),
        )

    @classmethod
    def _lexical_match_score_normalized(cls, query: str, normalized_text: str) -> float:
        return cls._lexical_match_score_from_terms(
            normalized_text,
            query_terms=cls._query_relevance_terms(query),
            requires_strict_anchor=cls._query_requires_strict_anchor(query),
        )

    @classmethod
    def _lexical_match_score_from_terms(
        cls,
        normalized_text: str,
        *,
        query_terms: dict[str, float],
        requires_strict_anchor: bool,
    ) -> float:
        return evidence_policy.lexical_match_score_from_terms(
            normalized_text,
            query_terms=query_terms,
            requires_strict_anchor=requires_strict_anchor,
        )
    @classmethod
    def _lexical_idf_match_score_from_terms(
        cls,
        normalized_text: str,
        *,
        query_terms: dict[str, float],
        requires_strict_anchor: bool,
        lexical_index: dict[str, Any],
    ) -> float:
        return evidence_policy.lexical_idf_match_score_from_terms(
            normalized_text,
            query_terms=query_terms,
            requires_strict_anchor=requires_strict_anchor,
            lexical_index=lexical_index,
        )
    @classmethod
    def _lexical_term_idf(cls, term: str, lexical_index: dict[str, Any]) -> float:
        return evidence_policy.lexical_term_idf(term, lexical_index)
    @classmethod
    def _has_required_query_anchor(cls, query: str, normalized_text: str) -> bool:
        return evidence_policy.has_required_query_anchor(query, normalized_text)
    @classmethod
    def _has_lnt_ii_anchor(cls, normalized_text: str) -> bool:
        return evidence_policy.has_lnt_ii_anchor(normalized_text)
    @classmethod
    def _query_requires_strict_anchor(cls, query: str) -> bool:
        return evidence_policy.query_requires_strict_anchor(query)
    @classmethod
    def _query_relevance_terms(cls, query: str) -> dict[str, float]:
        return evidence_policy.query_relevance_terms(query)

    @staticmethod
    def _normalize_search_text(value: str) -> str:
        return evidence_policy.normalize_search_text(value)

    @staticmethod
    def _build_filter(filter_kwargs: dict[str, Any] | None) -> models.Filter | None:
        """将 dict 条件转为 Qdrant Filter。"""
        return qdrant_search.build_filter(filter_kwargs)

    # ── 计数 ────────────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        """当前 collection 中的 point 总数。"""
        return qdrant_library.count_points(self.client, collection=self.collection)

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
        qdrant_library.clear_collection(
            self.client,
            collection=self.collection,
            ensure_collection=self._ensure_collection,
            invalidate_cache=self._invalidate_lexical_cache,
        )
        logger.info("Cleared collection '%s'", self.collection)

    def _invalidate_lexical_cache(self) -> None:
        self._lexical_document_cache = None
        self._lexical_index_cache = None

    def list_papers(self) -> list[dict[str, Any]]:
        """列出已入库的论文（去重 paper_id），每项含 paper_title / paper_id / chunk_count。
        通过 scroll 遍历所有 points，按 paper_id 聚合。"""
        return qdrant_library.list_papers(self.client, collection=self.collection, page_size=500)

    def scroll_by_paper_id(self, paper_id: str, page_size: int = 500) -> list[Any]:
        """Return all Qdrant points for one paper_id, following scroll pagination."""
        return qdrant_payload.scroll_by_paper_id(
            client=self.client,
            collection=self.collection,
            paper_id=paper_id,
            page_size=page_size,
        )

    def update_paper_payload(
        self,
        paper_id: str,
        payload: dict[str, Any],
        point_ids: list[int | str] | None = None,
    ) -> int:
        """Update payload for a paper. Use point_ids when already known."""
        updated = qdrant_payload.update_paper_payload(
            client=self.client,
            collection=self.collection,
            paper_id=paper_id,
            payload=payload,
            point_ids=point_ids,
        )
        self._invalidate_lexical_cache()
        return updated

    def delete_by_paper_id(self, paper_id: str) -> int:
        """删除指定 paper_id 的所有 points，返回删除数量。"""
        deleted = qdrant_payload.delete_by_paper_id(
            client=self.client,
            collection=self.collection,
            paper_id=paper_id,
        )
        if deleted:
            self._invalidate_lexical_cache()
        return deleted

    def _chunk_text(self, content: str, chunk_size: int, chunk_overlap: int) -> list[dict[str, Any]]:
        """将文本按段落切分为 chunks。"""
        return text_chunking.chunk_text(content, chunk_size, chunk_overlap)

    @staticmethod
    def _split_paragraphs_with_pos(text: str) -> list[tuple[str, int]]:
        return text_chunking.split_paragraphs_with_pos(text)

    @staticmethod
    def _chunk_long_text(text: str, chunk_size: int, chunk_overlap: int) -> list[dict[str, Any]]:
        return text_chunking.chunk_long_text(text, chunk_size, chunk_overlap)
