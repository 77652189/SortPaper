"""
文献检索 Agent — Qwen Function Calling 自主检索 + 综合建议。

流程：
  用户提问 → Qwen 判断是否需要检索
           → 自动生成关键词 + 过滤条件 → search_literature()
           → 拿到结果 → 判断是否需要再查
           → 够了 → 综合所有结果 → 输出结构化建议

Search tool 对接 QdrantStore.search()，使用 Hybrid Search + Rerank。
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from src.runtime_env import load_project_env
from src.retrieval.multi_query import multi_query_search
from src.store.qdrant_store import QdrantStore

load_project_env()

logger = logging.getLogger(__name__)

DEFAULT_DEEPSEEK_SYNTHESIS_MODEL = "deepseek-v4-flash"


def _env_value(name: str, default: str = "") -> str:
    return (os.getenv(name) or os.getenv(f"\ufeff{name}") or default).strip()


def dashscope_agent_api_key() -> str:
    return _env_value("DASHSCOPE_API_KEY")


def deepseek_api_key() -> str:
    return _env_value("DEEPSEEK_API_KEY")

# ── Tool 定义 ──────────────────────────────────────────────────────────────

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_literature",
        "description": (
            "搜索已入库的学术论文文献库，返回语义相关的 chunk。"
            "支持按论文分类（category）、可信度下限（credibility_gte）过滤。"
            "query 应使用英文专业术语，如 'HMO lacto-N-tetraose biosynthesis yield'。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索查询词，使用英文学术术语",
                },
                "category": {
                    "type": "string",
                    "enum": ["fermentation_experiment", "biosynthesis_review"],
                    "description": "论文分类过滤：实验型或综述型，不传则不限",
                },
                "credibility_gte": {
                    "type": "number",
                    "description": "可信度下限 (0~1)，不传则不限",
                },
            },
            "required": ["query"],
        },
    },
}

SYSTEM_PROMPT = """你是一个发酵工程领域的学术文献助手。你可以使用 search_literature 工具查询已入库的论文 chunk。

## 工作流程
1. 分析用户问题，提取关键概念
2. 调用 search_literature 检索相关文献（每次最多查 5 条）
3. 如果结果不够或角度单一，换个 query 再查一次
4. 综合所有检索结果，输出结构化建议

## 检索策略
- 查询词使用英文学术术语
- 如果用户问题涉及实验参数（pH、温度、补料），优先使用 category="fermentation_experiment"
- 如果涉及机理、通路、瓶颈分析，优先使用 category="biosynthesis_review"
- 不确定时两者都查

## 输出格式
综合建议请用 Markdown 格式，包含：
- 问题分析（一句话）
- 文献方案（表格，含策略/效果/来源）
- 置信度说明
"""

SYNTHESIS_PROMPT = """基于以下文献检索结果，回答用户问题。

用户问题：{query}

检索到的文献 chunk：
{documents}

请综合这些信息，给出结构化建议。如果文献信息不足以回答，请如实说明。
"""


# ── Agent ──────────────────────────────────────────────────────────────────

class LiteratureAgent:
    """文献检索 Agent，封装 Qwen function calling 循环。"""

    def __init__(
        self,
        model: str = "qwen-plus",
        lexical_backfill: bool = True,
        expand_neighbor_context: bool = True,
        expand_paper_local_context: bool = True,
        use_query_rewrite: bool = False,
        multi_query_recall: bool = False,
    ):
        self.model = model
        self.lexical_backfill = lexical_backfill
        self.expand_neighbor_context = expand_neighbor_context
        self.expand_paper_local_context = expand_paper_local_context
        self.use_query_rewrite = use_query_rewrite
        self.multi_query_recall = multi_query_recall
        self.store = QdrantStore()
        self._search_history: list[dict[str, Any]] = []
        self._all_chunks: list[dict[str, Any]] = []
        self._last_search_meta: dict[str, Any] = {}

    def query(self, user_question: str, max_rounds: int = 3) -> dict[str, Any]:
        """执行 Agent 查询，返回最终回答 + 检索历史。"""
        import dashscope
        from http import HTTPStatus

        self._search_history = []
        self._all_chunks = []
        self._last_search_meta = {}

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_question},
        ]

        final_answer = ""
        api_key = dashscope_agent_api_key()
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY 未设置，无法使用 Agent 搜索")

        for round_idx in range(max_rounds):
            response = dashscope.Generation.call(
                model=self.model,
                messages=messages,
                tools=[SEARCH_TOOL],
                result_format="message",
                api_key=api_key,
            )

            if response.status_code != HTTPStatus.OK:
                logger.error("Agent round %d failed: %s", round_idx, response.message)
                break

            choice = response.output.choices[0]
            msg = choice.message

            # 检查是否有 tool_calls（DashScope response 的 __getattr__ 会抛 KeyError，改用 dict.get）
            tool_calls_raw = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
            if tool_calls_raw:
                assistant_msg = {"role": "assistant", "content": msg.content or "",
                                 "tool_calls": tool_calls_raw}
                messages.append(assistant_msg)

                for tc in tool_calls_raw:
                    # DashScope 可能返回对象或 dict，统一处理
                    if isinstance(tc, dict):
                        func_name = tc["function"]["name"]
                        try:
                            args = json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]
                        except (json.JSONDecodeError, TypeError):
                            args = {"query": user_question}
                        tc_id = tc.get("id", "")
                    else:
                        func_name = tc.function.name
                        try:
                            args = json.loads(tc.function.arguments)
                        except (json.JSONDecodeError, TypeError):
                            args = {"query": user_question}
                        tc_id = tc.id

                    logger.info(
                        "Agent round %d | tool=%s | query=%s | category=%s | cred=%s",
                        round_idx, func_name, args.get("query"),
                        args.get("category", "-"), args.get("credibility_gte", "-"),
                    )

                    tool_result = self._execute_tool(func_name, args)
                    history_item = {"round": round_idx, "args": args, "hits": len(tool_result)}
                    if self._last_search_meta:
                        history_item["search_meta"] = self._last_search_meta
                    self._search_history.append(history_item)

                    messages.append({
                        "role": "tool",
                        "content": json.dumps(tool_result, ensure_ascii=False),
                        "tool_call_id": tc_id,
                    })
            else:
                # 无 tool call → 模型认为够了，结束搜索
                break

        # 始终用 DeepSeek 综合检索结果生成最终回答
        if self._all_chunks:
            final_answer = self._synthesize(user_question, self._all_chunks)

        return {
            "answer": final_answer or "无法综合出有效回答，请尝试调整查询条件。",
            "search_history": self._search_history,
            "total_chunks": len(self._all_chunks),
        }

    def _execute_tool(self, name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
        """执行 tool call。"""
        if name != "search_literature":
            return [{"error": f"Unknown tool: {name}"}]

        self._last_search_meta = {}
        query = args.get("query", "")
        if not query:
            return [{"error": "query 不能为空"}]

        filter_kwargs: dict[str, Any] = {}
        if args.get("category"):
            filter_kwargs["category"] = args["category"]
        if args.get("credibility_gte"):
            filter_kwargs["credibility_gte"] = args["credibility_gte"]

        started = time.monotonic()
        search_meta: dict[str, Any] = {}
        if self.use_query_rewrite or self.multi_query_recall:
            multi_result = multi_query_search(
                self.store,
                query,
                limit=5,
                filter_kwargs=filter_kwargs if filter_kwargs else None,
                rerank=True,
                lexical_backfill=self.lexical_backfill,
                use_query_rewrite=self.use_query_rewrite or self.multi_query_recall,
                route_limit=4 if self.multi_query_recall else 2,
            )
            results = multi_result.results
            search_meta = multi_result.metadata()
        else:
            results = self.store.search(
                query=query,
                limit=5,
                filter_kwargs=filter_kwargs if filter_kwargs else None,
                rerank=True,
                lexical_backfill=self.lexical_backfill,
            )
        elapsed_ms = (time.monotonic() - started) * 1000
        context_results: list[dict[str, Any]] = []
        if self.expand_paper_local_context and hasattr(self.store, "expand_paper_local_context"):
            context_results.extend(
                self.store.expand_paper_local_context(
                    query,
                    results,
                    filter_kwargs=filter_kwargs if filter_kwargs else None,
                    paper_limit=5,
                    per_paper_limit=3,
                    total_limit=5,
                )
            )
        if self.expand_neighbor_context:
            existing_context_keys = {
                (
                    str((item.get("payload") or {}).get("paper_id") or ""),
                    str((item.get("payload") or {}).get("chunk_id") or item.get("id") or ""),
                )
                for item in context_results
            }
            neighbor_limit = max(0, 5 - len(context_results))
            neighbor_results = self.store.expand_neighbor_context(
                results + context_results,
                filter_kwargs=filter_kwargs if filter_kwargs else None,
                per_result_limit=2,
                total_limit=neighbor_limit,
            )
            for item in neighbor_results:
                payload = item.get("payload") or {}
                key = (
                    str(payload.get("paper_id") or ""),
                    str(payload.get("chunk_id") or item.get("id") or ""),
                )
                if key not in existing_context_keys:
                    context_results.append(item)
        logger.info(
            "Agent literature search | lexical_backfill=%s query_rewrite=%s multi_query=%s expand_neighbor_context=%s expand_paper_local_context=%s filters=%s hits=%s context_hits=%s elapsed_ms=%.1f",
            self.lexical_backfill,
            self.use_query_rewrite,
            self.multi_query_recall,
            self.expand_neighbor_context,
            self.expand_paper_local_context,
            sorted(filter_kwargs.keys()),
            len(results),
            len(context_results),
            elapsed_ms,
        )

        formatted = []
        for r in results:
            p = r.get("payload", {})
            formatted.append({
                "score": round(r.get("score", 0), 4),
                "content": p.get("content", ""),
                "paper_title": p.get("paper_title", ""),
                "page": p.get("page", "?"),
                "content_type": p.get("content_type", "text"),
                "matched_routes": r.get("matched_routes", []),
            })
            self._all_chunks.append(p)
        self._last_search_meta = search_meta
        for r in context_results:
            p = dict(r.get("payload", {}) or {})
            p["_context_expansion"] = True
            p["_paper_local_context"] = bool(r.get("paper_local_context"))
            p["_neighbor_source_chunk_id"] = r.get("neighbor_source_chunk_id")
            p["_neighbor_distance"] = r.get("neighbor_distance")
            self._all_chunks.append(p)

        return formatted

    def _synthesize(self, query: str, chunks: list[dict[str, Any]]) -> str:
        """强制综合：当 Agent 循环结束但未生成回答时调用。使用 DeepSeek。"""
        from openai import OpenAI

        docs_text = "\n---\n".join(
            f"[来源: {c.get('paper_title', '?')}, p{c.get('page', '?')}]\n{c.get('content', '')}"
            for c in chunks[:10]
        )
        prompt = SYNTHESIS_PROMPT.format(query=query, documents=docs_text)

        try:
            api_key = deepseek_api_key()
            if not api_key:
                return f"检索到 {len(chunks)} 个相关 chunk，但 DEEPSEEK_API_KEY 未设置，无法生成综合回答。"
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1",
                timeout=60.0,
            )
            response = client.chat.completions.create(
                model=_env_value("SORTPAPER_AGENT_SYNTHESIS_MODEL", DEFAULT_DEEPSEEK_SYNTHESIS_MODEL),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2048,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("synthesize via DeepSeek failed: %s", e)
            return f"检索到 {len(chunks)} 个相关 chunk，但综合生成失败：{e}"
