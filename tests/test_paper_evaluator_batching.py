from __future__ import annotations

import json
from types import SimpleNamespace

from src.judge.paper_evaluator import MAX_MAP_BATCH_CHUNKS, PaperQualityEvaluator


def test_map_batches_keep_every_chunk_index() -> None:
    chunks = [SimpleNamespace(raw_content=f"chunk {idx}") for idx in range(10)]

    batches = PaperQualityEvaluator._build_map_batches(chunks)
    indices = [idx for batch in batches for idx, _chunk in batch]

    assert indices == list(range(len(chunks)))
    assert all(len(batch) <= MAX_MAP_BATCH_CHUNKS for batch in batches)


def test_map_chunk_batch_restores_result_per_chunk() -> None:
    class FakeEvaluator(PaperQualityEvaluator):
        def _call_deepseek(self, prompt: str, model: str | None = None) -> str:
            return json.dumps([
                {"chunk_index": 1, "key_points": ["a"], "entities": ["A"]},
                {"chunk_index": 2, "key_points": ["b"], "entities": ["B"]},
            ])

    chunks = [
        SimpleNamespace(raw_content="first"),
        SimpleNamespace(raw_content="second"),
    ]

    results = FakeEvaluator()._map_chunk_batch(
        list(enumerate(chunks)),
        total=len(chunks),
    )

    assert [idx for idx, _result in results] == [0, 1]
    assert [result["key_points"] for _idx, result in results] == [["a"], ["b"]]
