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
    issue_type: str = "none"


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
    def __init__(
        self,
        dashscope_model: str = "qwen-max",
        deepseek_model: str = "deepseek-chat",
        threshold: float = 0.7,
    ) -> None:
        self.dashscope_model = dashscope_model
        self.deepseek_model = deepseek_model
        self.threshold = threshold

    def _call_deepseek(self, prompt: str) -> str:
        """调用 DeepSeek V4-Flash（OpenAI 兼容接口）。"""
        import os

        from openai import OpenAI

        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY 未设置")

        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1", timeout=60.0)
        response = client.chat.completions.create(
            model=self.deepseek_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=4096,
        )
        return response.choices[0].message.content or ""

    def judge(self, worker_type: str, content: str, pdf_path: str) -> JudgeVerdict:
        prompt_template = PROMPTS[worker_type]
        prompt = prompt_template.format(pdf_path=pdf_path, content=content or "")

        if worker_type == "image":
            # 图片保持 DashScope（VL 模型兼容性）
            return self._judge_with_dashscope(prompt)
        else:
            # 文字和表格用 DeepSeek
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

    def _judge_with_dashscope(self, prompt: str) -> JudgeVerdict:
        from dashscope import Generation

        try:
            response = Generation.call(
                model=self.dashscope_model,
                messages=[{"role": "user", "content": prompt}],
                result_format="message",
                timeout=60,
            )
        except Exception as e:
            logger.exception("DashScope judge call failed | type=image")
            return JudgeVerdict(passed=False, score=0.0, feedback=f"DashScope API call failed: {e}")

        status = getattr(response, "status_code", None)
        if status is not None and status != 200:
            error_msg = getattr(response, "message", "Unknown API error")
            code = getattr(response, "code", "")
            return JudgeVerdict(passed=False, score=0.0, feedback=f"DashScope API error {code}: {error_msg}")

        if not hasattr(response, "output") or not response.output:
            return JudgeVerdict(passed=False, score=0.0, feedback="Empty DashScope API response")

        try:
            message = response.output.choices[0].message.content
            payload = self._extract_payload(message)
            score = float(payload.get("score", 0.0))
            passed = bool(payload.get("passed", False)) and score >= self.threshold
            feedback = str(payload.get("feedback", ""))
            return JudgeVerdict(passed=passed, score=score, feedback=feedback)
        except Exception:
            logger.exception("DashScope judge response parse failed")
            return JudgeVerdict(passed=False, score=0.0, feedback="DashScope judge response parse error")

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
            return SummaryReport(
                status=str(payload.get("status", status)),
                modules=payload.get("modules", {}),
                stored_types=payload.get("stored_types", []),
                notes=str(payload.get("notes", "")),
            )
        except Exception as e:
            return SummaryReport(
                status=status,
                modules={},
                stored_types=[],
                notes=f"summary via DeepSeek failed: {e}",
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
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"No JSON object found in LLM response: {text[:200]!r}")
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON from LLM response: {exc}") from exc
