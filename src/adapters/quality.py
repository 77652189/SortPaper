from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.ports.quality import QualityEnrichmentStore, QualityEvaluator


@dataclass(frozen=True)
class QualityEnrichmentDependencies:
    store: QualityEnrichmentStore
    evaluator: QualityEvaluator
    search_text_builder: Any


class PaperQualityEvaluatorAdapter:
    """Quality evaluator adapter backed by the existing PaperQualityEvaluator."""

    def __init__(self, evaluator=None) -> None:
        if evaluator is None:
            from src.judge.paper_evaluator import PaperQualityEvaluator

            evaluator = PaperQualityEvaluator()
        self.evaluator = evaluator

    def evaluate(self, chunks: list, title: str = "") -> dict:
        return self.evaluator.evaluate(chunks, title=title)


def default_quality_enrichment_dependencies() -> QualityEnrichmentDependencies:
    from src.adapters.store.qdrant import QdrantQualityEnrichmentStore
    from src.domain.search_text_policy import build_search_text

    return QualityEnrichmentDependencies(
        store=QdrantQualityEnrichmentStore(),
        evaluator=PaperQualityEvaluatorAdapter(),
        search_text_builder=build_search_text,
    )
