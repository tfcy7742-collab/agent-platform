# Agent 平台

> **六个批次全部完成**：API 与网页界面均可运行，318 条测试离线可跑，
> 评测集与对比报告已就绪。架构设计与文件清单见
> `../agent-platform-ARCHITECTURE.md` 与 `../agent-platform-ARCHITECTURE-part2.md`。

---

## 这是什么

一个**可插拔工具协议的企业级 Agent 系统**：以统一的 Tool 接口接入三类形态完全不同的能力：

* `knowledge_search`：RAG 知识检索（文档上传 → 切块 → 向量化 → 混合检索 → 带引用回答）
* `trip_planner`：多智能体任务编排（景点 / 天气 / 酒店 Agent 并行 → 行程规划 Agent 整合）
* `send_email`：有副作用的写操作（演示 human-in-the-loop 人工确认）

由 Planner **自主路由**，并内置流式输出、超时熔断、失败降级、人工确认、
全链路 trace 与离线评测回归。

**差异化不在"能跑"，而在"可度量、可观测、可回归"。**

---

## 当前进度

| 批次 | 内容 | 状态 |
|---|---|---|
| **B1** | 目录骨架、配置系统、SQLite 层、LLM 客户端、健康检查、依赖实装验证 | ✅ 完成 |
| **B2** | 文档管道（加载 / 切块 / Embedding / Chroma）+ 上传与管理接口 | ✅ 完成 |
| **B3** | 混合检索（向量+BM25+RRF）+ 三层拒答 + 引用校验 + 问答接口 | ✅ 完成 |
| **B4** | 工具协议 + Agent 主循环（LLM 自主路由）+ 轨迹与工具接口 | ✅ 完成 |
| **B5** | SSE 流式问答 + 人工确认闭环 + Gradio 界面（挂载 /ui） | ✅ 完成 |
| **B6** | 评测集（34 问答 + 25 路由）+ 评测框架 + 对比报告 + 完整 README | ✅ 完成 |

---

## 快速开始

### 环境要求

| 依赖 | 版本 |
|---|---|
| Python | 3.10 及以上（已在 3.13 验证） |
| DeepSeek API Key | 可选。不配置则自动进入离线模式，功能与界面完全可用 |

### 安装与启动

```bash
# 1) 创建虚拟环境
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# 2) 安装依赖
pip install -r requirements.txt

# 3) 依赖自检（打印每个关键包的实际版本）
python scripts/check_deps.py

# 4) 配置（可选：不填 Key 也能跑）
copy .env.example .env      # Windows
cp .env.example .env        # macOS / Linux
# 编辑 .env，把 DEEPSEEK_API_KEY 换成真实 Key

# 5) 下载 Embedding 模型（首次必做一次，约 92 MB）
python scripts/download_embedding_model.py

# 6) 生成样例文档（3 种格式，用于演示与评测）
python scripts/make_sample_docs.py

# 7) 启动服务
python app.py

# Windows 上如果用的是项目内虚拟环境，直接调它的解释器即可：
#   .\.venv\Scripts\python.exe app.py
```

启动后：

| 地址 | 说明 |
|---|---|
| <http://127.0.0.1:8000/ui> | **网页界面**：上传文档、对话、来源折叠面板、执行轨迹、人工确认 |
| <http://127.0.0.1:8000/> | 服务信息与接口导航 |
| <http://127.0.0.1:8000/docs> | Swagger 交互式文档 |
| <http://127.0.0.1:8000/health> | 健康检查：各项能力状态 + 向量库统计 + 运行指标 |
| <http://127.0.0.1:8000/api/config> | 当前配置（脱敏，便于排查加载了哪份 .env） |
| <http://127.0.0.1:8000/api/metrics> | 进程内指标：请求数 / 工具耗时 / 延迟分位 / token 成本 |

界面也可以独立启动（通过 HTTP 调用 API，行为与挂载一致）：

```bash
python ui.py          # http://127.0.0.1:7860
```

> **为什么默认挂在 /ui 而不是独立 7860**：单进程单端口，部署与演示只需暴露一个端口，
> 也不用处理跨域；`ui.py` 仍保留独立启动能力，两种情况共用同一份界面代码。

### 下载 Embedding 模型（首次必做一次）

```bash
python scripts/download_embedding_model.py
```

脚本会把 `BAAI/bge-small-zh-v1.5` 下载到 `data/models/`（约 92 MB，512 维），
并做一次真实编码自检（含同义改写的语义检索用例）。

**网络说明（重要）**：部分网络环境无法访问 `huggingface.co`。平台已内置兜底：

* **模型已缓存** → 加载时显式传 `local_files_only=True`，**一个网络请求都不发**；
* **模型未缓存** → 先探测官方站，不通则自动切到 `hf-mirror.com` 并打印 WARNING。

> 这里踩过一个坑：`HF_HUB_OFFLINE` 环境变量在 `huggingface_hub` **被 import 之后就失效了**
> （库在导入时把该变量读成常量）。所以"先 import 再设环境变量"完全没用，
> 必须**显式传参**。否则每次启动都会去连 huggingface.co，在不可达网络里要等
> 5 次重试 × 每个文件 20~80 秒，表现为"服务启动后十几分钟不响应任何请求"。

如需手动指定镜像：

```bash
# Windows PowerShell
$env:HF_ENDPOINT="https://hf-mirror.com"

# macOS / Linux
export HF_ENDPOINT=https://hf-mirror.com
```

### 准备知识库文档

```bash
python scripts/make_sample_docs.py
# 输出到 data/sample_docs/
#   员工手册.pdf（PDF，多页，用于验证页码引用）
#   云笔记产品需求文档.docx（DOCX）
#   运维技术FAQ.md（Markdown）
```

上传方式二选一：

```bash
# 方式一：命令行
curl -X POST http://127.0.0.1:8000/api/documents \
  -F "files=@data/sample_docs/员工手册.pdf" \
  -F "files=@data/sample_docs/云笔记产品需求文档.docx"

# 方式二：浏览器打开界面或 /docs 上传
```

### 提问与验收

```bash
# 直接走 RAG 管道（单工具，默认模式；延迟与 token 可预测，适合做回归基线）
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "年假有几天", "session_id": "demo", "mode": "rag"}'

# 走 Agent 主循环：Planner 自主决定用哪个工具
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "帮我规划北京三日游，喜欢历史文化", "mode": "agent"}'

# 工具目录 / 直接调用工具 / 执行轨迹
curl http://127.0.0.1:8000/api/tools
curl http://127.0.0.1:8000/api/traces?limit=5

# 端到端验收（逐项打印 PASS/FAIL）
python scripts/verify_b2.py   # 文档管道 / 幂等 / 检索 / 删除
python scripts/verify_b3.py   # 检索 / 拒答 / 引用溯源 / 阈值扫描
python scripts/verify_b4.py   # Agent 自主路由 / 工具协议 / 可观测性
python scripts/verify_b5.py   # SSE 流式 / 人工确认闭环 / 界面挂载
python scripts/smoke_test.py  # 全链路冒烟（健康→上传→问答→路由→评测→界面）
python scripts/check_llm.py   # LLM 连通性自检（换 Key / 换模型后跑一次）
```

### 三个内置工具

| 工具 | 成本 | 耗时 | 需确认 | 说明 |
|---|---|---|---|---|
| `knowledge_search` | medium | medium | 否 | 知识库检索问答（RAG） |
| `trip_planner` | high | high | 否 | 多智能体旅行规划（4 个智能体协作） |
| `send_email` | low | low | **是** | 有副作用的写操作，用于演示 human-in-the-loop |

---

## 离线模式说明

**没有 DeepSeek API Key 也能完整运行**，平台会自动进入离线模式（`/health` 里 `llm.online=false`）：

| 能力 | 离线模式表现 |
|---|---|
| 文档上传、切块、向量化 | ✅ 完全可用（Embedding 降级为 hash 后端） |
| 混合检索、拒答、引用 | ✅ 完全可用 |
| RAG 问答生成 | ⚠️ 由命中片段按关键词摘取句子（带引用编号），不做自然语言归纳 |
| **Agent 自主路由** | ✅ 走规则路由（意图打分），多工具编排依然完全可见 |
| 旅行规划子系统 | ✅ 走本地规则引擎生成行程（4 个智能体仍会依次执行并产出轨迹） |
| **流式与人工确认** | ✅ 完全可用（SSE 与确认闭环不依赖大模型） |
| 界面 | ✅ 完全可用（来源与轨迹面板照常展示） |
| 评测回归 | ✅ 召回率 / 拒答准确率 / 路由准确率 / 延迟 / 成本五项可跑 |

这条设计让项目在**无 Key、无网络**的环境（例如 CI）里依然可以跑完整测试：

```bash
python -m pytest tests -q
# 318 passed, 1 skipped（B1 ~ B6 全部阶段）
```

---

## 评测（B6）

### 怎么跑

```bash
# 完整评测（真实大模型生成答案；加 --no-judge 跳过 LLM 判分）
python -m eval.run_eval --config hybrid

# 只跑检索指标（零 token、约 20 秒，适合 CI 回归）
python -m eval.run_eval --retrieval-only --compare baseline hybrid no_prefix

# 多配置对比（报告里给出横向对比表与相对差异）
python -m eval.run_eval --compare baseline hybrid

# 阈值扫描（为拒答阈值找拐点）
python -m eval.run_eval --config threshold_0.5 --retrieval-only

# 快速验证（只跑前 N 条）
python -m eval.run_eval --limit 10
```

报告落在 `eval/reports/`（markdown 给人看，json 给程序读）。HTTP 侧有两个只读接口：

```bash
curl http://127.0.0.1:8000/api/eval/datasets        # 评测集规模与题型分布
curl http://127.0.0.1:8000/api/eval/reports         # 历史报告列表 + 核心指标
curl http://127.0.0.1:8000/api/eval/reports/<name>  # 报告全文
```

**为什么评测走 CLI 而不是 HTTP 触发**：一次完整评测要跑几十次检索与大模型调用，
耗时几分钟且占资源，放在 HTTP 里同步等待容易超时、也不方便看过程日志。
评测入口留给 CI，HTTP 侧只负责读结果。

### 评测集

| 集合 | 规模 | 构成 |
|---|---|---|
| `eval/datasets/rag_qa.jsonl` | **34 条** | 22 条可回答（标注金标来源 + 关键信息）+ 12 条应拒答 |
| `eval/datasets/routing.jsonl` | **25 条** | 10 条知识库 / 7 条旅行规划 / 8 条易混淆边界（含 3 条能力边界题） |

数据集是**人工逐条核对样例文档原文**编写的，不是自动生成——金标来源错了，
整份报告就没有意义。`tests/test_b6_eval.py` 里有对数据集本身的质量自检
（ID 唯一、题型分布、金标来源能在样例文档里找到）。

### 指标口径（很容易写错的地方）

| 指标 | 口径 |
|---|---|
| Recall@k / MRR | **只在有金标来源的 22 条可答用例上算** |
| 拒答准确率 | **只在 12 条应拒答用例上算**，且必须逐字匹配标准话术 |
| 关键字覆盖率 | 答案是否包含金标关键信息（数字/专有名词），离线判分，便宜可复现 |
| LLM 判分 | 由大模型判断答案正确性，更接近人工，但消耗 token 且不完全可复现 |
| 引用有效率 | 回答里的引用编号是否都能对应到检索片段 |

> **踩过的坑**：第一版把应拒答用例也算进 Recall 的分母，它们的 recall 恒为 0，
> 把 100% 稀释成 64.7%。指标口径错了，报告会给出完全错误的结论——
> 所以现在口径写在代码注释里，并有专门的单元测试守住。

### 评测结果

**配置：`hybrid`（向量 + BM25 + RRF，真实 bge-small-zh 模型 + 真实 DeepSeek）**

| 指标 | 值 |
|---|---|
| Recall@3（22 条可答） | **100%** |
| MRR | **1.00** |
| 拒答准确率（12 条应拒答） | **100%** |
| 关键字覆盖率 | **69.3%**（保守下界，见下） |
| 引用有效率 | **100%** |
| 路由准确率（25 条） | **92%**（23/25） |
| P50 / P95 延迟（问答） | **824 ms / 1302 ms** |
| 平均 token / 用例 | 981（问答）｜2749（路由，含行程生成的多次模型调用） |
| 估算成本（整个评测） | 约 0.036 元（问答 34 条）+ 0.091 元（路由 25 条） |

**关键结论**：

1. **检索质量**：22 条可答用例的金标文档**全部进入 Top-3，且全部排在第一位**（MRR=1.00）。
   诚实说明：本样例语料只有 4 份文档 / 10 个块，**检索任务过于简单**，
   所以 `--compare` 里纯向量与混合检索的指标相同（Recall 都是 100%）。
   要体现 BM25 的价值需要更大语料（含编号、型号、专有名词类精确匹配需求）。
2. **拒答质量**：12 条应拒答用例**全部逐字返回标准话术**，零漏拒、零误拒。
   这是三层拒答叠加的效果——单靠阈值做不到（实测有相似度 0.465 的越界问题，
   高于部分正常问题）。
3. **关键字覆盖率 69.3% 是保守下界**：部分金标关键字是**同义表述**
   （金标写"百分之八十"，模型答"80%"），字面匹配判为未覆盖。
   这也是报告里同时提供 LLM 判分的原因——**关键字覆盖率不等于准确率**。
4. **引用可溯源**：100% 有效，未出现引用不存在的编号。
5. **延迟**：P50 824ms / P95 1.3s（含大模型生成），主要成本在模型调用而非检索
   （检索约 15ms，BM25 索引首次构建约 0.5s）。

### 拒答阈值是怎么标定的（真实数据）

`REFUSE_THRESHOLD` 不是拍脑袋定的，而是用真实模型跑出分数分布后选的。
运行 `python scripts/verify_embedding_model.py` 可复现下表：

| 分组 | Top-1 分数范围 | 均值 |
|---|---|---|
| 文档内问题（应召回） | 0.374 ~ 0.770 | 0.561 |
| 文档外问题（应拒答） | 0.205 ~ 0.465 | 0.306 |

两组分数**存在天然重叠**（最难的文档内问题 0.374 < 最"像"的文档外问题 0.465），
单靠阈值无法同时做到"零误拒 + 零漏拒"：

| 阈值 | 文档外被正确拒答 | 文档内被误拒 |
|---|---|---|
| 0.35 | 4/5 | 0/5 |
| 0.50 | 5/5 | 1/5 |
| 0.55 | 5/5 | 3/5 |

**这就是采用"阈值短路 + Prompt 约束 + 引用校验"三层拒答的真正原因**——
单一阈值做不到 100% 稳定，必须靠后两层兜底。

> **B3 的改进**：分析发现，最大的漏检来源不是阈值本身，而是**中文功能词污染了
> BM25 词项**。"如何用 Python 写快速排序" 会因为 "如何""用" 命中运维文档里的
> "如何做性能调优""使用压测工具"，拿到 0.469 的虚高分（与真正的文档内问题几乎相同）。
> 在 `rag/retriever.py` 里加入中文停用词表后，基线集合上的拒答率从 4/5 提升到 **5/5**，
> 且**零误拒**——这比调阈值有效得多。

### 路由评测（25 条）

| 用例类型 | 例子 | 期望行为 |
|---|---|---|
| 典型知识库问题 | "年假有几天" | `knowledge_search` |
| 典型旅行需求 | "帮我规划北京三日游" | `trip_planner` |
| 含城市名但问实时天气 | "今天北京的天气怎么样" | **不调工具**（知识库是静态文档，查不了实时天气） |
| 含"旅行"但问制度 | "公司的年假规定和旅行请假冲突怎么办" | `knowledge_search` |
| 含"规划"但非旅行 | "帮我规划一下这个季度的项目排期" | **不调工具**（三个工具都做不了，应说明能力边界） |
| 写操作意图 | "帮我发一封邮件给 hr@example.com" | `send_email`（且触发人工确认） |
| 只给邮箱无动作 | "我的邮箱是 test@example.com" | **不调工具**（应询问用户要做什么） |

> **数据集设计上踩过的坑（值得单独说）**：第一版把"今天北京的天气怎么样"
> 标成了期望 `knowledge_search`，结果大模型**拒绝调用工具并解释了能力边界**——
> 这是比我预期更正确的行为，却被判成了错误。问题不在模型，而在**金标不真实**。
> 修正后引入了 `decline` 这一期望类型（不调工具 + 给出说明性回答）。
> **教训：评测集的金标错了，指标就没有任何意义。**

> **评测暴露的真实 bug**：模型在"北京有哪些景点推荐"上**不调工具、直接用自身知识作答**
> ——这违反了"必须基于资料"的原则。加了规则 7（"只要涉及外部事实，
> 即使你认为自己知道答案也必须先调用工具"）之后修复，路由准确率 21/25 → 23/25。
> 这类问题只有跑真实评测才会暴露，单测和人工点几次都发现不了。

> **方法学说明（诚实的边界）**：路由评测判定的是"**第一个**被调用的工具"。
> 剩下 2 条按此规则记为错误，但**实际行为是合理的**：
>
> - `把这个季度的考勤统计发到 manager@example.com`：模型先调 `knowledge_search` 取数据
>   （多步任务的正常顺序），再走发信——单步判定覆盖不了这种执行顺序；
> - `帮我规划一下这个季度的项目排期`：模型先查知识库，发现资料里的"排期"是云笔记产品的
>   迭代计划而不是本季度项目排期，于是**如实拒答并解释了原因**——这比直接拒绝更好。
>
> 要正确评价这类行为需要引入"**轨迹级**"评测（比较完整工具调用序列）。
> 这是本评测框架的已知局限，也是下一步的改进方向。

---

## 项目结构

```
agent-platform/
├── app.py                  # FastAPI 入口：lifespan（注册工具 + 挂载 UI）、统一异常
├── ui.py                   # Gradio 界面（挂载 /ui，也可独立跑 7860）
├── requirements.txt        # 依赖（区间约束，让 pip 自行求解兼容组合）
├── .env.example            # 全部配置项与说明（约 40 项）
├── api/
│   ├── documents.py        # 文档上传 / 列表 / 详情 / 删除 / 重建索引
│   ├── chat.py             # 问答（rag / agent 两种模式）、检索预览、拒答话术
│   ├── chat_stream.py      # 流式问答（SSE）与人工确认恢复
│   ├── tools.py            # 工具目录、直接调用、启停、执行轨迹
│   └── evaluation.py       # 评测集概况与历史报告（模块名不能叫 eval，见文件注释）
├── config/settings.py      # 配置 + 能力汇总 + 密钥脱敏 + 模型缓存/镜像兜底
├── core/
│   ├── llm.py              # LLM 客户端：provider 切换 / 错误分类 / JSON 容错 / 用量统计
│   ├── prompts.py          # 提示词集中管理（含 {context} {question} 模板与拒答话术）
│   ├── tools/              # 工具层
│   │   ├── base.py         #   BaseTool / ToolResult / JSON Schema 校验 / 超时熔断
│   │   ├── registry.py     #   注册、发现、启停、统一调用、中文别名
│   │   ├── knowledge_tool.py  # knowledge_search：包装 RAG 引擎
│   │   ├── trip_tool.py    #   trip_planner：包装多智能体旅行规划子系统
│   │   └── email_tool.py   #   send_email：需人工确认的写操作
│   └── runtime/            # Agent 运行时
│       ├── events.py       #   事件契约（plan/tool_start/tool_end/refusal/final/done）
│       ├── planner.py      #   决策中枢：LLM 自主路由 + 规则兜底 + 问题改写
│       ├── executor.py     #   执行治理：重试、超时熔断、人工确认挂起
│       ├── resume.py       #   挂起上下文存储（确认后恢复，不从头重跑）
│       └── agent.py        #   主循环：步数预算、观察累积、回答汇总、拒答优先
├── rag/
│   ├── models.py           # Chunk / RetrievedChunk / SourceRef / 入库结果模型
│   ├── loaders.py          # PDF / DOCX / TXT / MD 加载，统一 metadata，内容哈希 doc_id
│   ├── splitter.py         # 500/50 切块 + 中文分隔符 + 块级去重
│   ├── embeddings.py       # bge-small-zh + 查询前缀 + hash 兜底 + 离线加载
│   ├── store.py            # Chroma（cosine 空间、幂等写入、按文档删除、语料版本号）
│   ├── pipeline.py         # 入库编排（加载→切块→向量化→元数据）
│   ├── retriever.py        # 混合检索：向量 + BM25(停用词过滤) + RRF + 可选重排
│   └── answer.py           # 问答引擎：三层拒答 + 引用校验 + 离线兜底
├── trip_planner/           # 旅行规划子系统（从 trip-planner 项目迁移）
│   ├── coordinator.py      #   MultiAgentTripPlanner：4 智能体并行协调
│   ├── agents/             #   景点 / 天气 / 酒店 / 行程规划 4 个智能体
│   ├── tools/travel_tools.py  # 模拟数据（5 个城市）
│   └── models/schemas.py   #   TripRequest / TripPlan 等
├── infra/
│   ├── db.py               # SQLite：文档元数据 / 会话消息 / run+step 轨迹
│   ├── trace.py            # TraceRecorder：一次提问的分步记录
│   ├── metrics.py          # 进程内指标聚合（含 P50/P95）
│   └── logging_setup.py    # 日志（文本 / JSON 双形态）+ 启动配置快照
├── eval/                   # 评测（B6）
│   ├── dataset.py          #   评测集加载、判分、指标聚合
│   ├── judges.py           #   LLM 判分
│   ├── run_eval.py         #   评测 CLI（配置 / 对比 / 阈值扫描）
│   ├── datasets/           #   rag_qa.jsonl（34 条）+ routing.jsonl（25 条）
│   └── reports/            #   生成的 markdown / json 报告
├── scripts/
│   ├── check_deps.py       # 依赖自检
│   ├── check_llm.py        # LLM 连通性自检（换 Key 后跑一次）
│   ├── download_embedding_model.py  # 下载并自检 bge（含镜像兜底）
│   ├── probe_model_sources.py       # 探测模型下载源
│   ├── probe_refusal_gate.py        # 探测拒答闸门方案（含结论）
│   ├── make_sample_docs.py          # 生成 PDF / DOCX / MD 中文样例
│   ├── migrate_trip_planner.py      # 把 trip-planner 迁移为 trip_planner 子包
│   ├── smoke_test.py                # 全链路冒烟测试
│   ├── verify_b2.py ~ verify_b5.py  # 分批端到端验收
│   └── verify_embedding_model.py    # 真实模型验证 + 拒答阈值标定
└── tests/                  # 318 条测试（1 条因缺中文字体跳过），全部离线可跑
```

---

## 六个批次做了什么

| 批次 | 交付 | 核心能力 |
|---|---|---|
| B1 | 配置系统、SQLite、LLM 客户端、健康检查 | 地基：可运行 + 可观测 |
| B2 | 文档管道（加载/切块/Embedding/Chroma）+ 上传接口 | 知识入库 |
| B3 | 混合检索 + 三层拒答 + 引用校验 | **检索质量与不编造** |
| B4 | 工具协议 + Agent 主循环（LLM 自主路由） | **可插拔编排** |
| B5 | SSE 流式 + 人工确认闭环 + Gradio 界面 | **可用性与安全** |
| B6 | 评测集 + 评测框架 + 对比报告 | **可度量、可回归** |

---

## 核心设计说明

以下几点是这个项目真正花力气的地方，每一点都有**可复现的实测数据**支撑。

### 1. 工具抽象与自主编排（不是 if-else 硬编码）

同一套 `BaseTool` 协议下接入三个形态完全不同的能力：一个检索管道、一个 4 智能体子系统、
一个带副作用的写操作。Planner 按**能力说明 + 成本 + 耗时 + 是否需要确认**做决策——
在线模式下这是大模型的自主选择（实测它把"帮我规划一个去成都的四天旅行"解析成
`{destination: 成都, days: 4, preferences: [美食]}` 并选中 `trip_planner`）。

**为什么不用 LangGraph**：编排只有 200 行，自己写可控可调试。
步数预算的作用与取值、工具调用失败的处理方式（错误回灌让 Planner 换策略，而不是直接 500），
都在下一节展开。

### 2. 拒答的工程化保证（不是靠 Prompt 祈祷）

**实测数据**：真实 bge 模型上，文档内问题分数 0.374~0.770，文档外问题 0.205~0.465——
**两组重叠**。所以单阈值必然出错（0.35 时漏放 1 条，0.50 时误拒 1 条）。

因此设计了三层：**阈值短路**（不调用大模型，成本为零）+ **Prompt 约束**（模型自述资料不足）
+ **引用回原文校验**（编号越界即判定不可信）。评测集上 12/12 正确拒答。

**过程中还发现**：最大的漏检来源不是阈值，而是**中文功能词污染了 BM25 词项**，
加停用词表后基线拒答率从 4/5 提升到 5/5，比调阈值有效得多。

### 3. 可观测与可回归（改动的效果能被度量）

每次问答落一条 trace：每步决策、工具耗时、检索命中数与分数、是否降级、token 与成本。
配合评测集，任何改动（换切块参数、调阈值、加重排）都能跑出**前后对比**，
而不是"感觉好像好一点"。

**评测为什么走 CLI 而不是 HTTP**：便于在 CI 里跑、不依赖服务进程。
指标口径上，Recall 只能在有金标来源的用例上计算，否则会被拒答用例稀释——这是实际踩过的坑。

### 4. 降级可见（不伪装）

没有 API Key、没有模型、没有网络，系统**依然完整可用**：
规则路由保证多工具编排可见、hash 兜底保证检索可用、片段摘取保证回答有依据。
关键是**如实标注**：`degraded=true`、`/health` 里 `llm.online=false`、
回答里写明"离线模式"。实测 318 条测试在零网络零 Key 下全绿——
这既是工程习惯，也让 CI 真正可跑。

---

## 已知限制（诚实清单）

| 限制 | 说明 | 计划 |
|---|---|---|
| 样例语料太小 | 4 份文档 / 10 个块，检索任务过于简单，混合检索与纯向量指标相同 | 换更大语料后可体现 BM25 价值 |
| 路由判定只看第一步 | 多步任务的执行顺序会被误判为"路由错误" | 引入轨迹级评测 |
| 边界问题无法靠阈值拒答 | 字面/语义巧合（如 "Python" 命中 FAQ 里的命令行）会被放行，交由 L3 兜底 | 已用三层拒答覆盖；`ENABLE_RERANK=true` 可进一步改善（需下载约 1GB 模型） |
| 规则路由靠意图打分 | 离线模式的意图打分是针对本样例语料调的，换领域需要调整关键词表 | 在线模式由 LLM 决策，不受此限制 |
| 挂起状态存进程内 | 人工确认的挂起记录用进程内字典 + 15 分钟 TTL，多实例部署不共享 | 接口已隔离（save/load/drop），换 Redis 只改实现 |
| 单轮改写 | 追问改写已实现（`enable_query_rewrite`），但只改写一次，不做多轮递归 | 保持 |
| 流式粒度是"事件级" | 事件即时推送，但最终回答是一次性推送（`token` 事件已预留） | 可改为 token 级流式 |
| 无鉴权 | 单机演示定位，未做用户体系 | 不做 |
| 无 OCR | 扫描版 PDF 解析为空会明确报错，不做图片识别 | 不做 |

---

## 依赖版本（实测通过）

`pip install -r requirements.txt` 在 Python 3.13 下解析并安装成功（21/21 关键包导入通过）：

| 包 | 版本 | 包 | 版本 |
|---|---|---|---|
| langchain | 0.3.30 | fastapi | 0.141.1 |
| langchain-core | 0.3.86 | uvicorn | 0.53.0 |
| langchain-community | 0.3.31 | gradio | 5.50.0 |
| langchain-chroma | 0.2.6 | pydantic | 2.12.3 |
| langchain-openai | 0.3.35 | pydantic-settings | 2.15.0 |
| chromadb | 1.0.21 | pypdf | 6.18.1 |
| sentence-transformers | 5.7.0 | python-docx | 1.2.0 |
| rank-bm25 | 0.2.2 | python-multipart | 0.0.32 |
| jieba | 0.42.1 | httpx | 0.28.1 |
| reportlab | 5.0.1 | — | — |

> 注意：`langchain-chroma` 会约束 `chromadb` 的版本区间，因此 requirements 中**不单独钉死 chromadb**，
> 否则极易出现"装上了但导入报错"的版本冲突。
