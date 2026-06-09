"""Application mode labels and compatibility helpers."""

from __future__ import annotations


MODE_MINERU = "MinerU \u5feb\u901f\u9884\u89c8\uff08\u65e0\u56fe\u7247\u8f6c\u6587\u5b57\uff09"
MODE_MINERU_FULL = "MinerU \u5b8c\u6574\u89e3\u6790\uff08\u542b\u56fe\u7247\u8f6c\u6587\u5b57\uff09"
MODE_MINERU_AUTO_STORE = "MinerU \u4e00\u952e\u5165\u5e93\uff08\u63a8\u8350\uff09"
MODE_LEGACY_PREVIEW = "\u5386\u53f2\u5feb\u901f\u9884\u89c8\uff08\u5c55\u793a\uff09"
MODE_LEGACY_PIPELINE = "\u5386\u53f2\u5b8c\u6574\u6d41\u6c34\u7ebf\uff08\u5c55\u793a\uff09"
MODE_LEGACY_AUTO_STORE = "\u5386\u53f2\u4e00\u952e\u5165\u5e93\uff08\u517c\u5bb9\uff09"

PRIMARY_MODES = (MODE_MINERU, MODE_MINERU_FULL, MODE_MINERU_AUTO_STORE)
LEGACY_MODES = (MODE_LEGACY_PREVIEW, MODE_LEGACY_PIPELINE, MODE_LEGACY_AUTO_STORE)

LEGACY_MODE_ALIASES = {
    "\u5feb\u901f\u9884\u89c8": MODE_LEGACY_PREVIEW,
    "\u5b8c\u6574\u6d41\u6c34\u7ebf": MODE_LEGACY_PIPELINE,
    "\u4e00\u952e\u5165\u5e93": MODE_LEGACY_AUTO_STORE,
}


def normalize_mode(mode: str) -> str:
    return LEGACY_MODE_ALIASES.get(mode, mode)


def is_mineru_mode(mode: str) -> bool:
    return normalize_mode(mode) == MODE_MINERU


def is_mineru_full_mode(mode: str) -> bool:
    return normalize_mode(mode) == MODE_MINERU_FULL


def is_mineru_auto_store_mode(mode: str) -> bool:
    return normalize_mode(mode) == MODE_MINERU_AUTO_STORE


def is_auto_store_mode(mode: str) -> bool:
    normalized = normalize_mode(mode)
    return normalized in {MODE_MINERU_AUTO_STORE, MODE_LEGACY_AUTO_STORE}


def is_legacy_preview_mode(mode: str) -> bool:
    return normalize_mode(mode) == MODE_LEGACY_PREVIEW


def is_legacy_pipeline_mode(mode: str) -> bool:
    return normalize_mode(mode) == MODE_LEGACY_PIPELINE


def is_legacy_auto_store_mode(mode: str) -> bool:
    return normalize_mode(mode) == MODE_LEGACY_AUTO_STORE
