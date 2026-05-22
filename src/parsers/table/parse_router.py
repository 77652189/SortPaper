from __future__ import annotations

import logging
from collections.abc import Callable

from src.parsers.layout_chunk import LayoutChunk
from src.parsers.table.models import TableRegion

logger = logging.getLogger(__name__)


class TableParseRouter:
    """Route table extraction strategies without knowing parser internals."""

    def __init__(
        self,
        *,
        keyword_guided: Callable[[], list[LayoutChunk]],
        pdfplumber: Callable[[str | None], list[LayoutChunk]],
        deduplicate: Callable[[list[LayoutChunk]], list[LayoutChunk]],
        keyword_regions: Callable[[list[TableRegion]], list[LayoutChunk]] | None = None,
        pdfplumber_regions: Callable[[list[TableRegion], str | None], list[LayoutChunk]] | None = None,
    ) -> None:
        self._keyword_guided = keyword_guided
        self._keyword_regions = keyword_regions
        self._pdfplumber = pdfplumber
        self._pdfplumber_regions = pdfplumber_regions
        self._deduplicate = deduplicate

    def parse(
        self,
        *,
        strategy: str,
        feedback: str | None,
        regions: list[TableRegion] | None = None,
    ) -> list[LayoutChunk]:
        if strategy == "pdfplumber_only":
            return self._dedup(
                self._keyword_guided() + self._pdfplumber(feedback)
            )

        if strategy != "all":
            raise ValueError(f"Unknown table parse strategy: {strategy}")

        chunks: list[LayoutChunk] = []
        if regions and self._keyword_regions:
            chunks.extend(self._keyword_regions(regions))
        if not regions:
            chunks.extend(self._keyword_guided())
        if regions and self._pdfplumber_regions:
            chunks.extend(self._pdfplumber_regions(regions, feedback))
        if regions:
            logger.info("table regions detected; skipping broad full-page table scans")
            return self._dedup(chunks)
        chunks.extend(self._pdfplumber(feedback))
        return self._dedup(chunks)

    def _dedup(self, chunks: list[LayoutChunk]) -> list[LayoutChunk]:
        if len(chunks) <= 1:
            return chunks
        return self._deduplicate(chunks)
