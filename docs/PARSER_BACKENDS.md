# PaperSort 解析后端治理策略

当前目标是把 MinerU 逐步接入主流程，同时保留历史自研解析代码用于展示、对照和回归排查。不要一次性删除旧代码，也不要继续把旧代码包装成唯一主路径。

## 当前分层

| 层级 | 后端 | 状态 | 入口 | 处理原则 |
|---|---|---|---|---|
| 主路径 | MinerU 快速预览 | 推荐 | `MinerU 快速预览（无图片转文字）` | 最快解析路径：文本、表格、图片占位、页码、bbox、Figure group metadata；不调用图片 VL |
| 主路径 | MinerU 完整解析 | 推荐 | `MinerU 完整解析（含图片转文字）` | 在快速预览基础上，按 Figure group 调用 VL，把图片转成检索友好的文字描述 |
| 主路径 | MinerU 一键入库 | 推荐 | `MinerU 一键入库（推荐）` | 执行完整解析后写入 Qdrant；图片 embedding 优先使用 Figure group VL 描述 |
| 展示路径 | 历史快速预览 | 隐藏但保留 | 打开“显示历史/兼容解析入口”后选择 `历史快速预览（展示）` | 用于快速看旧文本/表格解析效果，不作为新能力扩展重点 |
| 展示路径 | 历史完整流水线 | 隐藏但保留 | 打开“显示历史/兼容解析入口”后选择 `历史完整流水线（展示）` | 用于对照旧 parser + Judge + VisionParser 结果，帮助验证 MinerU 迁移 |
| 兼容路径 | 历史一键入库 | 隐藏但保留 | 打开“显示历史/兼容解析入口”后选择 `历史一键入库（兼容）` | 继续支持旧数据入库，直到真实样本证明 MinerU 入库和检索质量稳定 |

## 保留标准

以下历史代码应继续保留入口：

- 能帮助验证 MinerU 效果是否真的更好，例如旧 parser 的文本、表格、图片结果展示。
- 能作为回归测试基线，例如 table parser 的区域检测、质量诊断、dedup 规则。
- 仍然被现有入库、检索、评测链路依赖，例如 `LayoutChunk`、`chunk_storage`、`qdrant_store`。

## 隐藏但保留入口

以下代码不应继续作为默认主路径，但可以保留 UI 或开发入口。UI 上默认展示 MinerU 三个入口；历史路径需要手动打开“显示历史/兼容解析入口”：

- `run_preview`：历史快速预览，仅用于展示和调试旧解析效果。
- `run_pipeline` / LangGraph 旧 parser pipeline：历史完整流水线，用于对照和回归。
- `VisionParser` 单图解析：旧图片解析路径。MinerU 路径优先按 `figure_group_id` 做组级 VL 复解析。

代码边界上，历史入口已经集中到 `src/legacy/`：

- `src.legacy.parsing.run_legacy_preview`：承接历史快速预览。
- `src.legacy.parsing.run_legacy_pipeline`：承接历史 LangGraph 完整流水线。
- `src.legacy.vision_parser.VisionParser`：显式标记旧单图视觉解析依赖。
- `app_pipeline.run_preview` / `app_pipeline.run_pipeline` 仅保留兼容 wrapper，供现有 UI、批量处理和测试继续调用。

## 封存候选

满足下面条件后，可以把代码移入 `legacy/` 或只保留测试夹具：

- MinerU 主路径已经覆盖同等能力，并完成真实样本验证。
- 旧代码没有被入库、检索、评测或 UI 展示依赖。
- 已有测试证明迁移后结果结构不退化，尤其是页码、bbox、表格、图片 group、chunk metadata。

当前不建议彻底删除历史 table parser。它仍然有价值：很多 table 质量诊断、区域检测和 dedup 规则可以作为 MinerU 结果校验器，而不一定继续作为主解析器。

## 真实样本验证证据

已用 `Engineering Escherichia coli for the High-Titer Biosynthesis of Lacto-N-tetraose` 作为 LNT 真实样本跑过隔离入库验证：

```bash
python evals/mineru_ingest_eval.py "data/mineru_cache/85a593e1722f022b/2022-AAA-Engineering Escherichia coli for the High-Titer Biosynthesis of Lacto-N-tetraose.pdf" --reset-collection --top-k 5 --out reports/mineru_ingest_eval_lnt.json
```

验证结果写入 `reports/mineru_ingest_eval_lnt.json`：

- 使用 MinerU 本地缓存，未重新调用 MinerU API。
- 生成 71 个 `LayoutChunk`：50 text、4 table、17 image。
- 71 个 chunk 全部有页码和 bbox。
- 识别出 6 个 Figure group。
- 隔离 Qdrant collection `papers_mineru_eval` 中 71/71 成功入库，`missing_verdicts=0`。
- 5 个检索用例全部 Top-5 命中，覆盖标题、LNT titer、工程基因、Table 2 和 Figure/UDP-Gal 相关证据。

这说明 MinerU 主路径已经具备“解析 → LayoutChunk → 入库 → 检索”的闭环证据。但这还不是彻底封存旧 parser 的充分条件，因为图片语义质量仍依赖可选的 Figure group 级 VL 复解析，且旧 table parser 仍可作为表格质量校验器。

图片语义复解析也已补充一轮 LNT 样本验证：

```bash
python evals/mineru_figure_vision_eval.py reports/mineru_lnt_smoke.zip --call-vision --max-workers 2 --cache-path data/mineru_cache/85a593e1722f022b/figure_vision.json --out reports/mineru_figure_vision_eval_lnt.json
```

结果：

- 6 个 Figure group，17 个 visual chunk。
- 6/6 group 成功生成 VL 描述。
- 每组描述长度约 693-1943 字符。
- 描述覆盖 `LNT`、`lacto-N-tetraose`、`lgtA`、`wbgO`、`UDP-Gal`、`UDP-GlcNAc`、`galE`、`galT`、`galK`、`ugd` 等关键词。
- 使用 `data/mineru_cache/85a593e1722f022b/figure_vision.json` 后，重复运行同一评测 6/6 group 命中缓存，耗时约 0.2 秒。

这说明 MinerU 图片路径已经可以按 Figure group 生成检索友好的语义描述，不必继续把旧单图 `VisionParser` 当作默认图片主路径。

主流程已为 Figure group VL 增加两个护栏：

- 缓存：描述写入 `data/mineru_cache/<paper_id>/figure_vision.json`，同一模型、同一 Figure group 内容不会重复调用视觉模型。
- 受控并发：`MINERU_FIGURE_VISION_MAX_WORKERS` 控制并发数，默认 `2`，避免 9 个以上图组完全串行等待，同时降低触发 API 限流的风险。

第二个不同主题/版式样本也已验证：`Kinetic Study of Penicillin Acylase Production by Recombinant E. coli in Batch Cultures`。

入库与检索：

```bash
python evals/mineru_ingest_eval.py "data/mineru_cache/fd4719f529e7b241/1994-AA-Kinetic study of penicillin acylase production by recombinant E. coli in batch cultures(科研通-ablesci.com).pdf" --query-set auto --reset-collection --top-k 5 --out reports/mineru_ingest_eval_penicillin.json
```

- 使用 MinerU 本地缓存，未重新调用 MinerU API。
- 生成 40 个 `LayoutChunk`：31 text、9 image。
- 40 个 chunk 全部有页码和 bbox，覆盖 1-10 页。
- 识别出 9 个 Figure group。
- 隔离 Qdrant collection `papers_mineru_eval` 中 40/40 成功入库，`missing_verdicts=0`。
- 自动弱标签生成 6 个检索用例，全部 Top-5 命中。该结果用于回归 smoke，不等同于人工 gold set。

图片语义复解析：

```bash
python evals/mineru_figure_vision_eval.py data/mineru_cache/fd4719f529e7b241/result.zip --call-vision --max-workers 2 --cache-path data/mineru_cache/fd4719f529e7b241/figure_vision.json --out reports/mineru_figure_vision_eval_penicillin.json
```

- 9 个 Figure group，9 个 visual chunk。
- 9/9 group 成功生成 VL 描述。
- 该样本以单图/单图表居多，和 LNT 多 panel group 样本互补。
- 串行调用 9 个 group 明显更慢；已增加并发数配置和缓存复用，后续可根据实际 API 限流情况调低或调高。

## 近期迁移顺序

1. MinerU 快速预览作为默认入口：已在侧边栏设为无图片转文字的最快模式。
2. MinerU zip 转 `LayoutChunk`：已保留文本、表格、图片、页码、bbox、Figure group metadata。
3. MinerU 完整解析：已提供含 Figure group 级 VL 复解析的完整模式。
4. MinerU chunk 入库：已把 `mineru_preview` 结果接入 `store_parsed_chunks`，并为 MinerU chunk 生成默认 storage verdict；图片 embedding 优先使用 Figure group VL 描述、图注和 MinerU visual text。
5. MinerU 一键入库：已提供 `MinerU 一键入库（推荐）`，单篇和批量都走完整解析后入库，即 PDF → MinerU → Figure group VL → LayoutChunk → Qdrant。
6. 历史一键入库降级：已从默认模式列表隐藏到“显示历史/兼容解析入口”后面，作为兼容和对照入口保留。

## 当前封存判断

| 代码区域 | 当前处理 | 原因 |
|---|---|---|
| `src/parsers/mineru_*` | 保留并继续增强 | 新主路径，负责 MinerU API、zip 转 `LayoutChunk`、Figure group VL 复解析 |
| `src/parsers/layout_chunk.py` | 保留 | MinerU 与历史 parser 的共同 chunk 契约，入库和检索仍依赖 |
| `src/store/*`、`app_utils.qdrant_search` | 保留 | 主路径入库和检索仍依赖，不属于解析后端替换范围 |
| `src/parsers/table/*` | 保留但不再作为默认解析主线扩展 | MinerU 真实样本已证明表格可入库和检索，但旧 table 规则仍可作为质量校验器、回归基线和旧路径展示能力 |
| `src/parsers/vision_parser.py` | 隐藏但保留，进入封存观察 | 历史完整流水线仍需要；MinerU 路径已通过 LNT 与 penicillin acylase 两个样本证明 Figure group 级 VL 复解析可用，并已增加缓存和受控并发 |
| `src/legacy/parsing.py` | 隐藏但保留 | 历史快速预览和 LangGraph 完整流水线的兼容包装边界 |
| `src/legacy/vision_parser.py` | 隐藏但保留 | 旧单图 `VisionParser` 的 legacy re-export，避免主路径继续直接依赖 |
| `src/graph/pipeline_graph.py` | 隐藏但保留，legacy 依赖 | 历史完整流水线和历史一键入库仍可运行，用于对照展示；主入口通过 `src.legacy.parsing` 调用 |
| 历史一键入库 UI | 隐藏但保留 | 避免用户误把旧链路当主路径，同时保留旧数据兼容能力 |

彻底封存还不建议立刻做。已有两篇真实样本证据证明 MinerU 可完成入库、检索和 Figure group 级图片语义复解析闭环，且 Figure group VL 已有缓存和受控并发护栏。旧 `VisionParser` 和 LangGraph 旧流水线已经通过 `src/legacy/` 建立了更明确的模块边界；下一步若继续封存，应把旧实现文件逐步迁移到 `src/legacy/` 内部，并保留兼容 import 或迁移测试夹具。

## MinerU 耗时统计

MinerU 三个主模式都在单篇概览和批量结果中展示耗时：MinerU API、Zip 转 chunk、图片转文字、入库和总耗时。快速预览的图片转文字与入库耗时应为 0；完整解析会统计 Figure group 级图片转文字；一键入库会额外统计写入 Qdrant 的耗时。
