from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from src.judge.prompts import IMAGE_JUDGE_PROMPT, SUMMARY_PROMPT, TABLE_JUDGE_PROMPT, TEXT_JUDGE_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class JudgeVerdict:
    passed: bool
    score: float
    feedback: str


@dataclass
class SummaryReport:
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
    def __init__(self, model: str = "qwen-max", threshold: float = 0.7) -> None:
        self.model = model
        self.threshold = threshold

    def judge(self, worker_type: str, content: str, pdf_path: str) -> JudgeVerdict:
        from dashscope import Generation

        prompt_template = PROMPTS[worker_type]
        prompt = prompt_template.format(pdf_path=pdf_path, content=content or "")

        try:
            response = Generation.call(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                result_format="message",
            )
        except Exception as e:
            logger.exception("LLMJudge API call failed | type=%s", worker_type)
            return JudgeVerdict(passed=False, score=0.0, feedback=f"API call failed: {e}")

        # 检查 API 错误响应
        # DashScope 成功时 response.code 为空字符串，真正的 HTTP 状态在 response.status_code
        status = getattr(response, "status_code", None)
        if status is not None and status != 200:
            error_msg = getattr(response, "message", "Unknown API error")
            code = getattr(response, "code", "")
            return JudgeVerdict(passed=False, score=0.0, feedback=f"API error {code}: {error_msg}")

        if not hasattr(response, "output") or not response.output:
            return JudgeVerdict(passed=False, score=0.0, feedback="Empty API response (no output)")

        try:
            message = response.output.choices[0].message.content
            payload = self._extract_payload(message)
            score = float(payload.get("score", 0.0))
            passed = bool(payload.get("passed", False)) and score >= self.threshold
            feedback = str(payload.get("feedback", ""))
            return JudgeVerdict(passed=passed, score=score, feedback=feedback)
        except Exception:
            logger.exception("LLMJudge response parse failed | type=%s", worker_type)
            return JudgeVerdict(passed=False, score=0.0, feedback="judge response parse error")

    def summarize(
        self,
        pdf_path: str,
        text_verdict: JudgeVerdict | None,
        table_verdict: JudgeVerdict | None,
        image_verdict: JudgeVerdict | None,
        text_retries: int,
        table_retries: int,
        image_retries: int,
        status: str,
    ) -> SummaryReport:
        from dashscope import Generation

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
            response = Generation.call(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                result_format="message",
            )
        except Exception as e:
            return SummaryReport(
                status=status,
                modules={},
                stored_types=[],
                notes=f"summary API call failed: {e}",
            )

        # 检查 API 错误响应
        # DashScope 成功时 response.code 为空字符串，真正的 HTTP 状态在 response.status_code
        http_status = getattr(response, "status_code", None)
        if http_status is not None and http_status != 200:
            error_msg = getattr(response, "message", "Unknown API error")
            code = getattr(response, "code", "")
            return SummaryReport(
                status=status,
                modules={},
                stored_types=[],
                notes=f"summary API error {code}: {error_msg}",
            )

        if not hasattr(response, "output") or not response.output:
            return SummaryReport(
                status=status,
                modules={},
                stored_types=[],
                notes="summary API returned empty response",
            )

        try:
            message = response.output.choices[0].message.content
            payload = self._extract_payload(message)
            return SummaryReport(
                status=str(payload.get("status", status)),
                modules=payload.get("modules", {}),
                stored_types=payload.get("stored_types", []),
                notes=str(payload.get("notes", "")),
            )
        except Exception:
            return SummaryReport(
                status=status,
                modules={},
                stored_types=[],
                notes="summary response parse failed",
            )

    @staticmethod
    def _extract_payload(message: Any) -> dict[str, Any]:
        if isinstance(message, list):
            text = "".join(part.get("text", "") for part in message if isinstance(part, dict))
        else:
            text = str(message)

        # 先尝试直接解析（LLM 严格返回 JSON 的情况）
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # fallback：提取第一个完整 JSON 对象
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"No JSON object found in LLM response: {text[:200]!r}")
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON from LLM response: {exc}") from exc
