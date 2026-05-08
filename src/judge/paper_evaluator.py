"""
论文质量评估器：四步流程。

Step 1 分类（含自洽性验证）→ Step 2 Map-Reduce → Step 3 构建 chunk 上下文 → Step 4 入库
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.judge.prompts import (
    CLASSIFY_PROMPT,
    VERIFY_PROMPT,
    MAP_CHUNK_PROMPT,
    MAP_CHUNK_PROMPT_LITE,
    REDUCE_PROMPT,
)

logger = logging.getLogger(__name__)

DEEPSEEK_MODEL = "deepseek-chat"
# 分类用：词数 > 此值的 text chunk 才算有效（避免短噪声干扰）
MIN_CHUNK_WORDS = 20
# 分类用：最多取这么多字符的内容（覆盖长摘要）
MAX_CLASSIFY_TOKENS = 4000
# 分类用：最多取这么多 chunks（覆盖被拆散的多段摘要）
MAX_CLASSIFY_CHUNKS = 25
# 自洽性验证置信度门槛
CONSISTENCY_THRESHOLD = 0.75
# Map 阶段并发数
MAX_MAP_WORKERS = 5


class PaperQualityEvaluator:
    """论文质量评估器。"""

    def __init__(self, model: str = DEEPSEEK_MODEL) -> None:
        self.model = model

    def evaluate(self, merged_chunks: list, pdf_path: str) -> dict[str, Any]:
        """完整四步评估入口。"""

        # ── Step 0: 提取分类用内容 ──────────────────────────────────────
        title = self._extract_title(pdf_path, merged_chunks)
        classify_content = self._extract_classify_content(merged_chunks)

        # ── Step 1: 分类（双 prompt + 自洽性验证）───────────────────────
        t0 = time.time()
        result1 = self._classify(classify_content)
        result2 = self._verify(classify_content)
        t_classify = time.time() - t0

        category = result1.get("category", "other")
        classify_conf = result1.get("confidence", 0.0)
        verify_conf = result2.get("confidence", 0.0)

        # 自洽性判断
        if (
            result1.get("category") == result2.get("category")
            and classify_conf >= CONSISTENCY_THRESHOLD
            and verify_conf >= CONSISTENCY_THRESHOLD
        ):
            classify_status = "confirmed"
        else:
            classify_status = "uncertain"

        # other 类：跳过后续步骤，但保留 uncertain 标记供核查
        if category == "other":
            return {
                "paper_title": title,
                "category": "other",
                "classify_status": classify_status,
                "classify_result": result1,
                "verify_result": result2,
                "chunk_map": {},
                "timing": {"classify": round(t_classify, 1), "map": 0, "reduce": 0},
                "skip": True,
            }

        # ── Step 2: Map-Reduce ─────────────────────────────────────────
        # 过滤：只保留词数 >= MIN_CHUNK_WORDS 的 text chunk
        text_chunks = [
            c for c in merged_chunks
            if c.content_type == "text"
            and len(c.raw_content.split()) >= MIN_CHUNK_WORDS
        ]
        total = len(text_chunks)
        use_lite = category == "biosynthesis_review"

        # 并发 Map（ThreadPoolExecutor，避免串行等待 API）
        t_map_start = time.time()
        map_results: list[dict] = [{}] * total
        with ThreadPoolExecutor(max_workers=min(MAX_MAP_WORKERS, total)) as pool:
            futures = {
                pool.submit(self._map_chunk, text_chunks[i].raw_content, i + 1, total, lite=use_lite): i
                for i in range(total)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    map_results[idx] = future.result()
                except Exception as exc:
                    logger.warning("Map chunk %d/%d failed: %s", idx + 1, total, exc)
                    map_results[idx] = {"key_points": [], "entities": []}
        t_map = time.time() - t_map_start

        t_reduce_start = time.time()

        if category == "fermentation_experiment":
            reduce_result = self._reduce(
                title, category,
                result1.get("target_products", []),
                result1.get("organisms", []),
                map_results,
            )
            paper_summary = reduce_result["summary"]
            credibility = reduce_result["credibility"]
        else:
            # biosynthesis_review: 摘要 + 可信度固定 0.6
            reduce_result = self._reduce(
                title, category,
                result1.get("target_products", []),
                result1.get("organisms", []),
                map_results,
                skip_credibility=True,
            )
            paper_summary = reduce_result["summary"]
            credibility = 0.6

        t_reduce = time.time() - t_reduce_start

        # ── Step 3: 构建 chunk 上下文 ──────────────────────────────────
        fermentation_relevance = result1.get("fermentation_relevance", 0.0)
        key_evidence = result1.get("key_evidence", [])
        reason = result1.get("reason", "")
        is_actionable = result2.get("is_actionable", False)

        context_prefix = (
            f"[标题] {title}"
            f"\n[分类] {category} | 发酵相关性: {fermentation_relevance} | 可信度: {credibility}"
            f"\n[产物] {', '.join(result1.get('target_products', []))}"
            f"\n[菌株] {', '.join(result1.get('organisms', []))}"
            f"\n[摘要] {paper_summary}"
            f"\n[判断依据] {reason}"
        )

        chunk_contexts: dict[str, str] = {}
        for chunk in merged_chunks:
            ctx = (
                f"{context_prefix}"
                f"\n[位置] {chunk.content_type} | chunk {chunk.global_order}/{len(merged_chunks)}"
            )
            chunk_contexts[chunk.chunk_id] = ctx

        # 将 map_results 按 chunk_id 索引（仅词数 >= MIN_CHUNK_WORDS 的 text chunk 有）
        chunk_map: dict[str, dict] = {}
        for i, chunk in enumerate(text_chunks):
            if i < len(map_results):
                chunk_map[chunk.chunk_id] = map_results[i]

        return {
            "paper_title": title,
            "category": category,
            "classify_status": classify_status,
            "classify_result": result1,
            "verify_result": result2,
            "fermentation_relevance": fermentation_relevance,
            "target_products": result1.get("target_products", []),
            "organisms": result1.get("organisms", []),
            "key_evidence": key_evidence,
            "classify_reason": reason,
            "is_actionable": is_actionable,
            "paper_summary": paper_summary,
            "credibility": credibility,
            "chunk_contexts": chunk_contexts,
            "chunk_map": chunk_map,
            "timing": {
                "classify": round(t_classify, 1),
                "map": round(t_map, 1),
                "reduce": round(t_reduce, 1),
            },
            "skip": False,
        }

    # ── 标题提取 ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_title(pdf_path: str, merged_chunks: list | None = None) -> str:
        """提取标题：优先用文件名（有意义且非临时文件），否则取第一页顶部 text chunk 拼接。"""
        filename_title = Path(pdf_path).stem
        # 跳过临时文件名（如 tmpx8r0kccz）
        is_temp = filename_title.startswith("tmp") and len(filename_title) < 20
        # 文件名有意义（含字母/中文且长度>=10）
        has_meaning = (
            not is_temp
            and len(filename_title) >= 10
            and any(c.isalpha() or '\u4e00' <= c <= '\u9fff' for c in filename_title)
        )
        if has_meaning:
            return filename_title
        # 备选：取第一页顶部前 3 个 text chunk，按 y0 排序，拼接前 200 字符
        if merged_chunks:
            top_chunks = sorted(
                [c for c in merged_chunks if c.page == 1 and c.content_type == "text"],
                key=lambda c: c.bbox[1],
            )[:3]
            if top_chunks:
                candidate = " ".join(c.raw_content.strip() for c in top_chunks)[:200]
                if candidate.strip():
                    return candidate
        return filename_title

    # ── 分类用内容提取 ────────────────────────────────────────────────

    @staticmethod
    def _extract_classify_content(merged_chunks: list) -> str:
        """提取用于分类判断的论文开头内容。

        策略：
        1. 优先找 ABSTRACT/Abstract/摘要 关键词，取后续连续 chunks
        2. 找不到时取前 N 个词数 > 30 的 text chunk，跳过短块和噪声
        3. 两种方式都限制在 1500 tokens / 10 个 chunk
        """
        text_chunks = [c for c in merged_chunks if c.content_type == "text"]

        # 方法一：关键词匹配找摘要
        abstract_parts: list[str] = []
        start_idx = -1
        for i, chunk in enumerate(text_chunks):
            content = chunk.raw_content.strip()
            if content.startswith(("ABSTRACT", "Abstract", "摘要")):
                start_idx = i
                break

        if start_idx >= 0:
            for chunk in text_chunks[start_idx:]:
                content = chunk.raw_content.strip()
                upper = content.upper()
                if abstract_parts and (
                    upper.startswith("INTRODUCTION")
                    or upper.startswith("KEYWORDS")
                    or upper.startswith("关键词")
                    or content.startswith("1.")
                ):
                    break
                if len(content.split()) >= MIN_CHUNK_WORDS:
                    abstract_parts.append(content)
                if len(" ".join(abstract_parts)) > MAX_CLASSIFY_TOKENS:
                    break
            result = " ".join(abstract_parts)
            if len(result.split()) >= MIN_CHUNK_WORDS:
                return result[:MAX_CLASSIFY_TOKENS]

        # 方法二（保底）：取前 N 个有意义的 chunk
        logger.info("关键词匹配未找到摘要，使用保底策略")
        parts: list[str] = []
        total_chars = 0
        for chunk in text_chunks:
            content = chunk.raw_content.strip()
            if not content:
                continue
            if len(content.split()) < MIN_CHUNK_WORDS:
                continue
            parts.append(content)
            total_chars += len(content)
            if len(parts) >= MAX_CLASSIFY_CHUNKS or total_chars >= MAX_CLASSIFY_TOKENS:
                break

        return " ".join(parts)[:MAX_CLASSIFY_TOKENS]

    # ── Step 1a: 分类 ─────────────────────────────────────────────────

    def _classify(self, content: str) -> dict[str, Any]:
        prompt = CLASSIFY_PROMPT.format(content=content)
        message = self._call_deepseek(prompt)
        try:
            return json.loads(message)
        except json.JSONDecodeError:
            logger.warning("分类 JSON 解析失败: %s", message[:100])
            return {"category": "other", "confidence": 0.0, "fermentation_relevance": 0.0,
                    "target_products": [], "organisms": [], "key_evidence": [], "reason": ""}

    # ── Step 1b: 自洽性验证 ───────────────────────────────────────────

    def _verify(self, content: str) -> dict[str, Any]:
        prompt = VERIFY_PROMPT.format(content=content)
        message = self._call_deepseek(prompt)
        try:
            return json.loads(message)
        except json.JSONDecodeError:
            logger.warning("验证 JSON 解析失败: %s", message[:100])
            return {"category": "other", "confidence": 0.0, "is_actionable": False, "actionable_reason": ""}

    # ── Step 2 Map: 单 chunk 提取 ─────────────────────────────────────

    def _map_chunk(self, content: str, chunk_index: int, total: int, lite: bool = False) -> dict[str, Any]:
        prompt_template = MAP_CHUNK_PROMPT_LITE if lite else MAP_CHUNK_PROMPT
        prompt = prompt_template.format(
            content=content,
            chunk_index=chunk_index,
            total_chunks=total,
        )
        message = self._call_deepseek(prompt)
        try:
            return json.loads(message)
        except json.JSONDecodeError:
            logger.warning("Map JSON 解析失败: chunk=%d/%d, msg=%s", chunk_index, total, message[:100])
            return {"key_points": [], "entities": []}

    # ── Step 2 Reduce: 聚合 ───────────────────────────────────────────

    def _reduce(
        self,
        title: str,
        category: str,
        target_products: list[str],
        organisms: list[str],
        map_results: list[dict],
        skip_credibility: bool = False,
    ) -> dict[str, Any]:
        lines: list[str] = []
        for i, r in enumerate(map_results):
            kps = "; ".join(r.get("key_points", []))
            entities = "; ".join(r.get("entities", []))
            lines.append(f"[chunk {i+1}] {kps} | entities: {entities}")

        chunk_summaries = "\n".join(lines)

        prompt = REDUCE_PROMPT.format(
            title=title,
            category=category,
            target_products=", ".join(target_products),
            organisms=", ".join(organisms),
            chunk_summaries=chunk_summaries,
        )

        if skip_credibility:
            prompt += '\n注意：只需输出摘要，credibility 字段设为 0.6。'

        message = self._call_deepseek(prompt)
        try:
            return json.loads(message)
        except json.JSONDecodeError:
            logger.warning("Reduce JSON 解析失败: %s", message[:100])
            return {"summary": "", "credibility": 0.0}

    # ── LLM 调用 ──────────────────────────────────────────────────────

    def _call_deepseek(self, prompt: str) -> str:
        from openai import OpenAI

        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY 未设置")

        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1", timeout=60.0)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=4096,
        )
        return response.choices[0].message.content or ""
