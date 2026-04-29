from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.judge.prompts import IMAGE_JUDGE_PROMPT, SUMMARY_PROMPT, TABLE_JUDGE_PROMPT, TEXT_JUDGE_PROMPT


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
        response = Generation.call(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            result_format="message",
        )

        try:
            message = response.output.choices[0].message.content
            payload = self._extract_payload(message)
            score = float(payload.get("score", 0.0))
            passed = bool(payload.get("passed", False)) and score >= self.threshold
            feedback = str(payload.get("feedback", ""))
            return JudgeVerdict(passed=passed, score=score, feedback=feedback)
        except Exception:
            return JudgeVerdict(passed=False, score=0.0, feedback="judge response parse error")

    def summarize(
        self,
        pdf_path: str,
        text_verdict: dict[str, Any] | None,
        table_verdict: dict[str, Any] | None,
        image_verdict: dict[str, Any] | None,
        text_retries: int,
        table_retries: int,
        image_retries: int,
        status: str,
    ) -> SummaryReport:
        from dashscope import Generation

        def _v(verdict: dict[str, Any] | None, key: str, default: Any) -> Any:
            return (verdict or {}).get(key, default)

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
        response = Generation.call(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            result_format="message",
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
                notes="summary generation failed",
            )

    @staticmethod
    def _extract_payload(message: Any) -> dict[str, Any]:
        if isinstance(message, list):
            text = "".join(part.get("text", "") for part in message if isinstance(part, dict))
        else:
            text = str(message)
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object found")
        return json.loads(text[start : end + 1])
