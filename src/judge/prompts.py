SUMMARY_PROMPT = """你是论文PDF处理流水线的汇报员。请根据以下三个模块的解析结果，生成一份简洁的处理报告。

报告需包含：
1. 整体处理状态（成功/失败）
2. 各模块（文本/表格/图片）的解析质量评分和通过情况
3. 最终入库的内容类型
4. 如有失败或降级，说明原因

返回严格 JSON，不要输出额外文字：
{{
  "status": "done 或 failed",
  "modules": {{
    "text": {{"passed": true/false, "score": 0-1, "retries": 0, "summary": "一句话描述"}},
    "table": {{"passed": true/false, "score": 0-1, "retries": 0, "summary": "一句话描述"}},
    "image": {{"passed": true/false, "score": 0-1, "retries": 0, "summary": "一句话描述"}}
  }},
  "stored_types": ["text", "table", "image"],
  "notes": "整体备注"
}}

PDF路径: {pdf_path}
文本解析评分: {text_score}, 通过: {text_passed}, 重试次数: {text_retries}
表格解析评分: {table_score}, 通过: {table_passed}, 重试次数: {table_retries}
图片解析评分: {image_score}, 通过: {image_passed}, 重试次数: {image_retries}
最终状态: {status}
"""

TEXT_JUDGE_PROMPT = """你是论文PDF解析质检员。请评估下面的文字解析结果是否合格。重点检查：
1. 双栏论文是否按正确阅读顺序重建，没有左右栏交错。
2. 文字是否完整，没有明显缺段、乱码或重复块。
3. 标题、段落、列表是否基本可读。

返回严格 JSON，不要输出额外文字：
{
  \"passed\": true 或 false,
  \"score\": 0 到 1 的浮点数,
  \"feedback\": \"给解析器的简短改进建议\"
}

PDF路径: {pdf_path}
解析内容:
{content}
"""

TABLE_JUDGE_PROMPT = """你是论文PDF表格解析质检员。请评估下面的表格解析结果是否合格。重点检查：
1. Markdown 表格结构是否合法。
2. 表头、行列是否完整，没有明显错位。
3. 关键单元格内容是否丢失。

返回严格 JSON，不要输出额外文字：
{
  \"passed\": true 或 false,
  \"score\": 0 到 1 的浮点数,
  \"feedback\": \"给解析器的简短改进建议\"
}

PDF路径: {pdf_path}
解析内容:
{content}
"""

IMAGE_JUDGE_PROMPT = """你是论文PDF图片解析质检员。请评估下面的图片解析结果是否合格。重点检查：
1. 是否描述了图片中的关键视觉元素。
2. 是否结合了图注或上下文。
3. 描述是否具体，而不是空泛重复。

返回严格 JSON，不要输出额外文字：
{
  \"passed\": true 或 false,
  \"score\": 0 到 1 的浮点数,
  \"feedback\": \"给解析器的简短改进建议\"
}

PDF路径: {pdf_path}
解析内容:
{content}
"""
