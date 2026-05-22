"""Compatibility wrapper for table LLM judge.

New code should import from ``src.judge.table_judge``.
"""

from src.judge.table_judge import TableLLMJudgeResult, judge_table_failure_with_llm

__all__ = ["TableLLMJudgeResult", "judge_table_failure_with_llm"]
