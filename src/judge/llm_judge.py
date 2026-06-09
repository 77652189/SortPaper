from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from src.adapters.llm.deepseek import DeepSeekChatClient
from src.judge.prompts import IMAGE_JUDGE_PROMPT, SUMMARY_PROMPT, TABLE_JUDGE_PROMPT, TEXT_JUDGE_PROMPT
from src.ports.llm import ChatCompletionClient

logger = logging.getLogger(__name__)


@dataclass
class JudgeVerdict:
    passed: bool
    score: float
    feedback: str
    issue_type: str = "none"


@dataclass
class ParseQualityReport:
    """解析质量汇总报告（非论文摘要）。"""
    status: str
    modules: dict[str, Any]
    stored_types: list[str]
    notes: str


PROMPTS = {
    "text": TEXT_JUDGE_PROMPT,
    "table": TABLE_JUDGE_PROMPT,
    "image": IMAGE_JUDGE_PROMPT,
}


class LLMJudge:
    def __init__(
        self,
        deepseek_model: str = "deepseek-chat",
        threshold: float = 0.7,
        llm_client: ChatCompletionClient | None = None,
    ) -> None:
        self.deepseek_model = deepseek_model
        self.threshold = threshold
        self.llm_client = llm_client

    def _call_deepseek(self, prompt: str) -> str:
        """Call DeepSeek through the LLM port while preserving the legacy method."""
        client = self.llm_client or DeepSeekChatClient(timeout=60.0)
        return client.complete(
            system_prompt="",
            user_prompt=prompt,
            model=self.deepseek_model,
            temperature=0.2,
            max_tokens=2048,
            label="deepseek-judge",
        )

    def judge(self, worker_type: str, content: str, pdf_path: str) -> JudgeVerdict:
        """对所有类型（text/table/image）统一使用 DeepSeek V4 Pro 评判。"""
        prompt_template = PROMPTS[worker_type]
        prompt = prompt_template.format(pdf_path=pdf_path, content=content or "")
        return self._judge_with_deepseek(prompt)

    def _judge_with_deepseek(self, prompt: str) -> JudgeVerdict:
        try:
            message = self._call_deepseek(prompt)
            payload = self._extract_payload(message)
            score = float(payload.get("score", 0.0))
            passed = bool(payload.get("passed", False)) and score >= self.threshold
            feedback = str(payload.get("feedback", payload.get("description", "")))
            issue_type = str(payload.get("issue_type", "none"))
            return JudgeVerdict(
                passed=passed, score=score, feedback=feedback, issue_type=issue_type,
            )
        except Exception as e:
            logger.exception("DeepSeek judge call failed")
            return JudgeVerdict(passed=False, score=0.0, feedback=f"DeepSeek call failed: {e}")

    def report_quality(
        self,
        pdf_path: str,
        text_verdict: JudgeVerdict | None,
        table_verdict: JudgeVerdict | None,
        image_verdict: JudgeVerdict | None,
        text_retries: int,
        table_retries: int,
        image_retries: int,
        status: str,
    ) -> ParseQualityReport:

        def _v(v: JudgeVerdict | None, key: str, default: Any) -> Any:
            if v is None:
                return default
            return getattr(v, key, default)

        prompt = SUMMARY_PROMPT.format(
            pdf_path=pdf_path,
            text_score=_v(text_verdict, "score", 0.0),
            text_passed=_v(text_verdict, "passed", False),
            text_retries=text_retries,
            table_score=_v(table_verdict, "score", 0.0),
            table_passed=_v(table_verdict, "passed", False),
            table_retries=table_retries,
            image_score=_v(image_verdict, "score", 0.0),
            image_passed=_v(image_verdict, "passed", False),
            image_retries=image_retries,
            status=status,
        )

        try:
            message = self._call_deepseek(prompt)
            payload = self._extract_payload(message)
            return ParseQualityReport(
                status=str(payload.get("status", status)),
                modules=payload.get("modules", {}),
                stored_types=payload.get("stored_types", []),
                notes=str(payload.get("notes", "")),
            )
        except Exception as e:
            return ParseQualityReport(
                status=status,
                modules={},
                stored_types=[],
                notes=f"quality report via DeepSeek failed: {e}",
            )

    @staticmethod
    def _extract_payload(message: Any) -> dict[str, Any]:
        if isinstance(message, list):
            text = "".join(part.get("text", "") for part in message if isinstance(part, dict))
        else:
            text = str(message)

        # 剥离 markdown 代码块
        text = text.strip()
        for prefix in ("```json", "```"):
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

        # 先尝试直接解析（LLM 严格返回 JSON 的情况）
        try:
            return json.loads(text, strict=False)
        except json.JSONDecodeError:
            pass

        # fallback：提取第一个完整 JSON 对象
        start = text.find("{")
        end = text.rfind("}")
        if start == -1:
            raise ValueError(f"No JSON object found in LLM response: {text[:200]!r}")

        # 截断兜底：JSON 没写完 → 尝试补 } 后解析
        if end == -1 or end <= start:
            text = text + "}"
            end = text.rfind("}")

        candidate = text[start : end + 1]
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON from LLM response: {exc}") from exc
