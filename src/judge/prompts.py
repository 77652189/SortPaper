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

TEXT_JUDGE_PROMPT = """你是论文PDF解析质检员。下面是一个文本块的解析内容，请只评估这个文本块本身的解析质量是否合格。重点检查：
1. 文字语义是否连贯，句子是否在奇怪位置中断（可能是双栏错位的信号）
2. 是否出现乱码、重复片段、或明显无意义字符串
3. 标题、段落结构是否基本可读

评分参考：
1.0 = 完全正常，无任何问题
0.8 = 有轻微问题但不影响理解
0.6 = 有明显问题，部分内容受影响
0.4 以下 = 严重问题，内容不可用

返回严格 JSON，不要输出额外文字：
{{
  "passed": true 或 false,
  "score": 0 到 1 的浮点数,
  "feedback": "简洁改进建议（不超过200字）"
}}

PDF路径: {pdf_path}
文本块内容:
{content}
"""

TABLE_JUDGE_PROMPT = """你是论文PDF表格解析质检员。请评估下面的表格解析结果是否合格。重点检查：
1. Markdown 表格结构是否合法（列数是否一致）。
2. 表头、行列是否完整，没有明显错位。
3. 关键单元格内容是否丢失。
4. 是否根本不是表格，而是误检。常见误检场景：

   a. 期刊页眉 / running title：首单元格是连续大写无空格的期刊名（如"CRITICALREVIEWSINBIOTECHNOLOGY"），或作者名格式（如"Y.ZHUETAL."）。
   b. 参考文献列表：首单元格是"[1]"、"[12]"等编号格式，内容为文献条目。
   c. 正文段落被切分：内容是小写字母开头的长句，读起来是流式正文而非表格数据。
   d. 双栏论文的正文被误切成2列表格：每格是完整句子（平均长度 > 80 字符），行数很多（> 20 行）。
   e. 图片/图注被误检：内容以"Figure"或"Fig."开头，且行数很少（< 5 行）。
   f. 页码 + 空行伪装：首单元格是单独的数字，后面大量空单元格。

   注意：学术三线表（仅横线无竖线）结构稀疏是正常的，只要内容是真实表格数据（产物/产率/菌株/酶/反应条件等），不应判为 false_positive。

评分参考：
1.0 = 完全正常，无任何问题
0.8 = 有轻微问题但不影响理解
0.6 = 有明显问题，部分内容受影响
0.4 以下 = 严重问题，内容不可用

返回严格 JSON，不要输出额外文字：
{{
  "passed": true 或 false,
  "score": 0 到 1 的浮点数,
  "issue_type": "structure_error 或 missing_content 或 false_positive 或 none",
  "description": "一句话描述问题（passed 为 true 时可为空）"
}}

issue_type 定义：
- structure_error：Markdown 语法错误、列数不一致、表格结构损坏
- missing_content：结构正确但关键单元格为空、行列明显缺失
- false_positive：根本不是表格，是页眉/参考文献/图注/正文段落等误检
- none：无问题

PDF路径: {pdf_path}
解析内容:
{content}
"""


# ── 论文质量评估 prompts ───────────────────────────────────────────────────────

CLASSIFY_PROMPT = """你是生物工程领域的文献分析专家。根据以下论文开头内容，判断该论文的类型和与微生物发酵实验的相关性。

论文内容：
{content}

请严格按以下 JSON 格式返回，不要输出任何其他文字：
{{
  "category": "fermentation_experiment 或 biosynthesis_review 或 other",
  "fermentation_relevance": 0到1的浮点数,
  "confidence": 0到1的浮点数,
  "target_products": ["目标产物列表，没有则为空列表"],
  "organisms": ["相关微生物或宿主，没有则为空列表"],
  "key_evidence": ["支持分类判断的1-3个关键原文片段"],
  "reason": "一句话说明分类理由"
}}

分类定义：
- fermentation_experiment：论文包含具体的微生物发酵实验，有菌株、培养条件、产率等原始实验数据
- biosynthesis_review：综述、机制研究、代谢通路分析，没有原始发酵实验数据
- other：与微生物发酵完全无关

fermentation_relevance 评分标准：
1.0 = 核心发酵实验论文，有完整的菌株/工艺/产率数据
0.8 = 包含发酵实验但不是唯一主题
0.6 = 涉及发酵相关内容但实验数据有限
0.4 = 提及发酵概念但主要研究其他内容
0.2 = 仅间接相关
0.0 = 与发酵完全无关

confidence 评分标准：
1.0 = 内容非常明确，分类毫无疑问
0.8 = 内容较清晰，分类基本确定
0.6 = 内容有一定歧义，分类较有把握
0.4 = 内容模糊，分类存在不确定性
0.2 以下 = 信息严重不足，无法可靠分类
"""

VERIFY_PROMPT = """你是一位对文献持审慎态度的生物工程审稿人。重新审视以下论文内容，独立判断其类型。

论文内容：
{content}

请严格按以下 JSON 格式返回，不要输出任何其他文字：
{{
  "category": "fermentation_experiment 或 biosynthesis_review 或 other",
  "is_actionable": true或false,
  "actionable_reason": "是否包含可直接指导发酵实验的信息，一句话说明",
  "confidence": 0到1的浮点数
}}

分类定义与第一次相同：
- fermentation_experiment：有菌株、培养条件、产率等原始实验数据
- biosynthesis_review：综述或机制研究，无原始发酵数据
- other：与微生物发酵无关

confidence 评分标准：
1.0 = 分类毫无疑问
0.8 = 分类基本确定
0.6 = 有一定把握
0.4 以下 = 存在明显不确定性
"""

MAP_CHUNK_PROMPT = """你是论文关键信息提取员。下面是一个论文的文本片段（chunk），请提取该片段中的核心信息。

返回严格 JSON，不要输出额外文字：
{{
  "key_points": ["关键要点列表，每点一句话"],
  "contains_claim": true 或 false,
  "claim_detail": "如果包含实验或方法声明，简要描述；否则空字符串",
  "entities": ["提及的实体名称，如酶名、菌株名、化合物名"]
}}

文本片段（第{chunk_index}/{total_chunks}个chunk，共{total_chunks}个）:
{content}
"""

MAP_CHUNK_PROMPT_LITE = """你是论文关键信息提取员。下面是一个文本片段，请提取核心信息。

返回严格 JSON，不要输出额外文字：
{{
  "key_points": ["关键要点列表，每点一句话"],
  "entities": ["提及的实体名称，如酶名、菌株名、化合物名"]
}}

文本片段（第{chunk_index}/{total_chunks}个chunk）:
{content}
"""

REDUCE_PROMPT = """你是论文评估员。以下是某论文中每个文本片段提取出的关键信息集合，请聚合生成整体评估。

需要输出：
1. 论文摘要（2-3句话概括论文核心内容）
2. 可信度评分（0-1），基于所有片段的信息一致性和完整性

评分锚点：
1.0 = 所有 chunks 高度一致，信息完整，无矛盾
0.8 = 大部分 chunks 一致，少量信息缺失
0.6 = 信息基本完整但有多处模糊/不完整
0.4 = 大量 chunk 信息缺失或矛盾

返回严格 JSON，不要输出额外文字：
{{
  "summary": "2-3句的论文核心内容摘要",
  "credibility": 0.0 到 1.0 的浮点数
}}

论文标题: {title}
论文分类: {category}
关键产物: {target_products}
相关菌株: {organisms}

各片段摘要:
{chunk_summaries}
"""

IMAGE_JUDGE_PROMPT = """你是论文PDF图片解析质检员。请评估下面的图片解析结果是否合格。重点检查：
1. 是否描述了图片中的关键视觉元素。
2. 是否结合了图注或上下文。
3. 描述是否具体，而不是空泛重复。

评分参考：
1.0 = 完全正常，无任何问题
0.8 = 有轻微问题但不影响理解
0.6 = 有明显问题，部分内容受影响
0.4 以下 = 严重问题，内容不可用

返回严格 JSON，不要输出额外文字：
{{
  "passed": true 或 false,
  "score": 0 到 1 的浮点数,
  "feedback": "简洁改进建议（不超过200字）"
}}

PDF路径: {pdf_path}
解析内容:
{content}
"""
