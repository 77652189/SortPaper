from __future__ import annotations

import hashlib


def stable_hash(value: str) -> int:
    return int(hashlib.md5(value.encode()).hexdigest()[:16], 16)


def qdrant_point_id(paper_id: str, chunk_id: str) -> int:
    return stable_hash(f"{paper_id}:{chunk_id}") % (2 ** 63)
