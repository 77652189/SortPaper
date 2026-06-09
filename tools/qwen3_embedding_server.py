"""OpenAI-compatible local embedding server for Qwen3-Embedding.

Run:
    python tools/qwen3_embedding_server.py

Then use:
    OPENAI_EMBEDDING_BASE_URL=http://127.0.0.1:8001/v1
    OPENAI_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
    SORTPAPER_EMBEDDING_DIM=1024

Do not point the existing `papers` collection at this model until its chunks
have been re-embedded into a separate collection or the old collection is
intentionally rebuilt.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_ID = os.getenv("QWEN3_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
DEVICE = os.getenv("QWEN3_EMBEDDING_DEVICE", "cpu")
BATCH_SIZE = int(os.getenv("QWEN3_EMBEDDING_BATCH_SIZE", "8"))
NORMALIZE = os.getenv("QWEN3_EMBEDDING_NORMALIZE", "true").lower() != "false"
PROMPT_NAME = os.getenv("QWEN3_EMBEDDING_PROMPT_NAME", "").strip() or None

app = FastAPI(title="PaperSort local Qwen3 embedding server")
_model: Any | None = None
_model_dim: int | None = None
_loaded_at: float | None = None


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str | None = None
    dimensions: int | None = Field(default=None, ge=1)


def _load_model() -> Any:
    global _model, _model_dim, _loaded_at
    if _model is not None:
        return _model

    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    _model = SentenceTransformer(MODEL_ID, device=DEVICE, trust_remote_code=True)
    if hasattr(_model, "get_embedding_dimension"):
        _model_dim = int(_model.get_embedding_dimension() or 0)
    else:
        _model_dim = int(_model.get_sentence_embedding_dimension() or 0)
    _loaded_at = time.perf_counter() - started
    return _model


def _texts(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _encode(texts: list[str]) -> np.ndarray:
    model = _load_model()
    kwargs: dict[str, Any] = {
        "batch_size": BATCH_SIZE,
        "convert_to_numpy": True,
        "normalize_embeddings": NORMALIZE,
        "show_progress_bar": False,
    }
    if PROMPT_NAME:
        kwargs["prompt_name"] = PROMPT_NAME
    try:
        vectors = model.encode(texts, **kwargs)
    except TypeError:
        kwargs.pop("prompt_name", None)
        vectors = model.encode(texts, **kwargs)
    return np.asarray(vectors, dtype=np.float32)


def _resize(vectors: np.ndarray, dimensions: int | None) -> np.ndarray:
    if dimensions is None or vectors.shape[1] == dimensions:
        return vectors
    if dimensions < vectors.shape[1]:
        return vectors[:, :dimensions]
    raise HTTPException(
        status_code=400,
        detail=f"Requested dimensions={dimensions}, but model dimension is {vectors.shape[1]}",
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "loaded": _model is not None,
        "dimension": _model_dim,
        "load_seconds": round(_loaded_at, 3) if _loaded_at is not None else None,
        "prompt_name": PROMPT_NAME,
        "normalize": NORMALIZE,
    }


@app.post("/v1/embeddings")
def embeddings(request: EmbeddingRequest) -> dict[str, Any]:
    texts = _texts(request.input)
    if not texts:
        raise HTTPException(status_code=400, detail="input must not be empty")

    vectors = _resize(_encode(texts), request.dimensions)
    return {
        "object": "list",
        "model": request.model or MODEL_ID,
        "data": [
            {
                "object": "embedding",
                "index": index,
                "embedding": vector.astype(float).tolist(),
            }
            for index, vector in enumerate(vectors)
        ],
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0,
        },
    }


if __name__ == "__main__":
    host = os.getenv("QWEN3_EMBEDDING_HOST", "127.0.0.1")
    port = int(os.getenv("QWEN3_EMBEDDING_PORT", "8001"))
    uvicorn.run(app, host=host, port=port)
