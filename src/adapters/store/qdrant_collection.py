from __future__ import annotations

import logging
from typing import Any

from qdrant_client.http.models import (
    Distance,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)


logger = logging.getLogger(__name__)


def ensure_collection(
    *,
    client: Any,
    collection: str,
    dense_dim: int,
    embedding_provider: str,
    uses_sparse_vectors: bool,
    dense_vector_name: str,
    sparse_vector_name: str,
) -> None:
    collections = [c.name for c in client.get_collections().collections]
    if collection in collections:
        info = client.get_collection(collection)
        has_sparse = bool(info.config.params.sparse_vectors)
        current_dim = (
            info.config.params.vectors.get(dense_vector_name).size
            if dense_vector_name in info.config.params.vectors
            else 0
        )
        if current_dim == dense_dim and has_sparse == uses_sparse_vectors:
            return
        if embedding_provider == "qwen3_local" and current_dim == dense_dim:
            logger.warning(
                "collection has sparse vectors from another provider; reusing dense vectors without recreating | collection=%s",
                collection,
            )
            return
        logger.warning(
            "collection vector config mismatch; recreating | collection=%s current_dim=%s target_dim=%s current_sparse=%s target_sparse=%s",
            collection,
            current_dim,
            dense_dim,
            has_sparse,
            uses_sparse_vectors,
        )
        client.delete_collection(collection_name=collection)

    create_collection(
        client=client,
        collection=collection,
        dense_dim=dense_dim,
        embedding_provider=embedding_provider,
        uses_sparse_vectors=uses_sparse_vectors,
        dense_vector_name=dense_vector_name,
        sparse_vector_name=sparse_vector_name,
    )


def create_collection(
    *,
    client: Any,
    collection: str,
    dense_dim: int,
    embedding_provider: str,
    uses_sparse_vectors: bool,
    dense_vector_name: str,
    sparse_vector_name: str,
) -> None:
    kwargs: dict[str, Any] = {
        "collection_name": collection,
        "vectors_config": {
            dense_vector_name: VectorParams(
                size=dense_dim,
                distance=Distance.COSINE,
            ),
        },
    }
    if uses_sparse_vectors:
        kwargs["sparse_vectors_config"] = {
            sparse_vector_name: SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
            ),
        }
    client.create_collection(**kwargs)
    logger.info(
        "Created collection '%s' (provider=%s dense=%dd sparse=%s)",
        collection,
        embedding_provider,
        dense_dim,
        uses_sparse_vectors,
    )


def recreate_collection(
    *,
    client: Any,
    collection: str,
    dense_dim: int,
    embedding_provider: str,
    uses_sparse_vectors: bool,
    dense_vector_name: str,
    sparse_vector_name: str,
) -> None:
    client.delete_collection(collection_name=collection)
    create_collection(
        client=client,
        collection=collection,
        dense_dim=dense_dim,
        embedding_provider=embedding_provider,
        uses_sparse_vectors=uses_sparse_vectors,
        dense_vector_name=dense_vector_name,
        sparse_vector_name=sparse_vector_name,
    )
    logger.info("Recreated collection '%s' (dense=%dd)", collection, dense_dim)
