from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class TableEvidence:
    """One deterministic reason why an area may be a table."""

    source: str
    score: float
    reason: str
    bbox: BBox | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "source": self.source,
            "score": round(float(self.score), 3),
            "reason": self.reason,
        }
        if self.bbox is not None:
            data["bbox"] = [float(v) for v in self.bbox]
        if self.details:
            data["details"] = self.details
        return data


@dataclass(frozen=True)
class TableRegion:
    """Recall-oriented table candidate region before structure extraction."""

    page: int
    bbox: BBox
    confidence: float
    evidence: list[TableEvidence]
    source: str = "table_region_detector"
    rejected: bool = False
    rejected_reason: str = ""

    @property
    def region_id(self) -> str:
        x0, y0, x1, y1 = self.bbox
        return f"table_region_p{self.page}_{x0:.1f}_{y0:.1f}_{x1:.1f}_{y1:.1f}"

    @property
    def band(self) -> str:
        if self.confidence >= 0.65:
            return "strong"
        if self.confidence >= 0.40:
            return "gray"
        return "weak"

    def to_metadata(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "region_id": self.region_id,
            "source": self.source,
            "page": self.page,
            "bbox": [float(v) for v in self.bbox],
            "confidence": round(float(self.confidence), 3),
            "band": self.band,
            "evidence": [item.to_metadata() for item in self.evidence],
        }
        if self.rejected:
            data["rejected"] = True
            data["rejected_reason"] = self.rejected_reason
        return data


@dataclass(frozen=True)
class TableExtractionAttempt:
    """Diagnostic record for one parser trying to extract one region."""

    parser: str
    page: int
    bbox: BBox
    succeeded: bool
    region_id: str | None = None
    rows: int = 0
    cols: int = 0
    failure_reason: str = ""

    def to_metadata(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "parser": self.parser,
            "page": self.page,
            "bbox": [float(v) for v in self.bbox],
            "succeeded": self.succeeded,
            "rows": self.rows,
            "cols": self.cols,
        }
        if self.region_id:
            data["region_id"] = self.region_id
        if self.failure_reason:
            data["failure_reason"] = self.failure_reason
        return data
