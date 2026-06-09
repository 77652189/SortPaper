from src.parsers.layout_chunk import ContentType, LayoutChunk, LayoutDeduplicator, LayoutMerger, infer_column
from src.parsers.mineru_adapter import MinerUZipParser

__all__ = [
    "LayoutChunk",
    "LayoutDeduplicator",
    "LayoutMerger",
    "ContentType",
    "infer_column",
    "MinerUZipParser",
    "PyMuPDFParser",
    "TableParser",
    "VisionParser",
]


def __getattr__(name: str):
    if name == "PyMuPDFParser":
        from src.parsers.pymupdf_parser import PyMuPDFParser

        return PyMuPDFParser
    if name == "TableParser":
        from src.parsers.table_parser import TableParser

        return TableParser
    if name == "VisionParser":
        from src.parsers.vision_parser import VisionParser

        return VisionParser
    raise AttributeError(name)
