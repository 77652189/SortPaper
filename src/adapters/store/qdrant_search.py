from __future__ import annotations

from typing import Any

from qdrant_client.http import models
from qdrant_client.http.models import Fusion, FusionQuery, Prefetch, SparseVector


def build_filter(filter_kwargs: dict[str, Any] | None) -> models.Filter | None:
    if not filter_kwargs:
        return None
    conditions = []
    for key, value in filter_kwargs.items():
        if key.endswith("_gte"):
            conditions.append(
                models.FieldCondition(
                    key=key[:-4],
                    range=models.Range(gte=value),
                )
            )
        elif key.endswith("_lte"):
            conditions.append(
                models.FieldCondition(
                    key=key[:-4],
                    range=models.Range(lte=value),
                )
            )
        else:
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
            )
    return models.Filter(must=conditions) if conditions else None


def fetch_multiplier(
    *,
    rerank: bool,
    selective_rerank: bool,
    lexical_backfill: bool,
    neighbor_backfill: bool,
) -> int:
    return 8 if rerank or selective_rerank or lexical_backfill or neighbor_backfill else 2


def run_vector_query(
    *,
    client: Any,
    collection: str,
    dense_vector_name: str,
    sparse_vector_name: str,
    dense_vec: list[float],
    sparse_dict: dict[int, float],
    qdrant_filter: models.Filter | None,
    uses_sparse_vectors: bool,
    limit: int,
    multiplier: int,
    score_threshold: float | None,
) -> list[dict[str, Any]]:
    query_limit = limit * multiplier
    if uses_sparse_vectors and sparse_dict:
        sparse_indices = sorted(sparse_dict.keys())
        sparse_values = [sparse_dict[index] for index in sparse_indices]
        sparse_vec = SparseVector(indices=sparse_indices, values=sparse_values)
        prefetch_dense = Prefetch(
            query=dense_vec,
            using=dense_vector_name,
            limit=query_limit,
            filter=qdrant_filter,
        )
        prefetch_sparse = Prefetch(
            query=sparse_vec,
            using=sparse_vector_name,
            limit=query_limit,
            filter=qdrant_filter,
        )
        candidates = client.query_points(
            collection_name=collection,
            prefetch=[prefetch_dense, prefetch_sparse],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=query_limit,
            score_threshold=score_threshold,
        )
    else:
        candidates = client.query_points(
            collection_name=collection,
            query=dense_vec,
            using=dense_vector_name,
            query_filter=qdrant_filter,
            limit=query_limit,
            score_threshold=score_threshold,
        )

    return [
        {
            "id": hit.id,
            "score": hit.score,
            "payload": hit.payload,
        }
        for hit in candidates.points
    ]
