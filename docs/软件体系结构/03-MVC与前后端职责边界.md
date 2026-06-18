# 03-MVC 与前后端职责边界

很多 AI 项目最开始没有复杂前端，可能只是 Streamlit、Gradio、一个命令行脚本，或者一个简单 Web 页面。于是很容易出现一种情况：按钮点击、参数读取、业务流程、结果展示、错误处理全写在同一个文件里。

MVC 的价值不是让你套一个老派模式，而是提醒你：用户看到的东西、用户触发的动作、系统真正的业务状态，最好不要混在一起。

## 为什么需要它

当 UI 和业务逻辑混在一起时，问题会变得很具体：

- 想换 UI 框架，业务流程也得重写。
- 想测试论文解析，必须启动页面。
- 想让 Codex 改展示样式，它可能顺手改了入库逻辑。
- Agent 结果展示和 Agent 执行状态混在一起，失败原因很难追。

MVC 可以简单理解为：

- Model：业务数据和状态，不关心怎么展示。
- View：展示结果，不决定业务规则。
- Controller：接收用户动作，调用业务服务，组织返回结果。

你不一定要严格创建 `models/ views/ controllers/` 目录，但这个思路可以保护职责边界。

## 什么时候需要

需要：

- 页面上有多个按钮，对应 preview、full pipeline、store、search、agent query 等流程。
- UI 状态和业务状态混在一起，比如 `session_state` 里同时放文件、解析结果、检索配置、错误信息。
- 同一业务流程既被 UI 调用，也被批处理或测试调用。
- 展示格式经常变，但核心处理流程不应该变。

暂时不需要完整 MVC：

- 只有一个一次性命令行入口。
- 没有交互状态，也没有复用业务流程的需求。
- 只是内部验证脚本，后续会重写。

## 例子：Streamlit 论文解析页面

坏设计：

```text
if st.button("一键入库"):
  读取上传文件
  计算 paper_id
  解析 PDF
  判断 chunk
  写入 Qdrant
  更新 session_state
  st.success(...)
  st.dataframe(...)
```

这个写法一开始很快，但它让按钮处理函数承担了太多责任。后面想做批量导入，必须复制一套解析和入库逻辑。

更好的拆法：

```text
View:
  render_upload_panel()
  render_result_tabs()

Controller:
  handle_preview_request()
  handle_store_request()

Application Service:
  run_preview(pdf_bytes)
  store_parsed_chunks(result)

Model:
  ParsedPaper
  Chunk
  StoreStats
```

UI 只负责收集输入和展示输出。业务流程可以被 UI、测试、批处理共同调用。

## 例子：RAG 查询结果展示

坏设计：

```text
search_page():
  读取用户 query
  生成 embedding
  查 Qdrant
  rerank
  拼 prompt
  调 LLM
  把 sources 转成 HTML
```

这里 View 直接知道了向量库、rerank 和 LLM。这会让页面变成系统中心。

更好的拆法：

```text
RagController:
  接收 query 和过滤条件
  调用 RagService
  转成 SearchViewModel

RagService:
  完成检索、rerank、上下文组装、回答生成

View:
  只渲染 answer、sources、confidence、debug traces
```

这样以后你改回答生成逻辑，不需要碰页面；你改页面布局，也不影响检索。

## 例子：Agent 执行状态

Agent 页面通常要展示：

- 当前计划。
- 已调用工具。
- 工具输入输出。
- 中间观察结果。
- 最终回答。
- 失败原因。

坏设计是让页面直接驱动 Agent 循环。更好的方式是 Agent 执行器产出结构化 trace，View 只负责渲染 trace。

```text
AgentRun
  steps:
    - thought
    - tool_call
    - observation
    - retry
  final_answer
  status
```

## 坏设计长什么样

- 页面函数里出现数据库查询、LLM prompt、embedding 调用。
- 业务服务返回一堆 HTML 或 UI 组件。
- 业务逻辑依赖 `st.session_state`、浏览器状态或前端组件。
- Controller 里写了大量领域规则，比如“什么样的 chunk 可以入库”。
- 测试业务流程时必须模拟按钮点击。

## 更好的拆法

- View 只收集输入、展示输出。
- Controller 只做请求编排和错误转换。
- Application Service 执行业务流程。
- Model 保存业务数据和状态。
- UI 状态和业务状态分开：前者服务交互，后者服务逻辑。

## 可执行产物：页面逻辑拆分模板

```markdown
## 页面拆分设计

页面名称：

### View 负责
- 输入控件：
- 展示区域：
- 加载/错误/空状态：

### Controller 负责
- 用户动作：
- 调用哪个应用服务：
- 成功后更新哪些 UI 状态：
- 失败后展示什么错误：

### Model / ViewModel
- 页面需要展示的数据字段：
- 这些字段来自哪个业务结果：
- 哪些字段只是 UI 临时状态：

### 不允许放在 View 的逻辑
- 数据库访问：
- LLM 调用：
- 入库规则：
- Agent 工具执行：
```

## 给 Codex 的提示词

```text
请按 MVC 思路审查这个页面文件。
重点找出 View、Controller、业务服务混在一起的地方。
请输出：
1. 哪些代码只负责展示
2. 哪些代码是用户动作编排
3. 哪些代码是真正业务逻辑
4. 最小拆分方案，要求保持页面行为不变
先不要修改代码。
```

