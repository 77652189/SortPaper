from __future__ import annotations

from src.ports.repositories import SearchRepository, VectorLibraryRepository


def default_search_repository() -> SearchRepository:
    from src.adapters.store.qdrant import QdrantSearchRepository

    return QdrantSearchRepository()


def default_vector_library_repository() -> VectorLibraryRepository:
    from src.adapters.store.qdrant import QdrantVectorLibraryRepository

    return QdrantVectorLibraryRepository()
