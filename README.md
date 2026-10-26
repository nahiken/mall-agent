# mall-agent

> 电商场景的 **AI 客服 + 导购 Agent**：用同一套工具、同一份评测数据，对比 **单 Agent 全工具 / 中心化多 Agent / Skill 懒加载** 三种架构在质量、成本、安全上的差异。

Python 3.11+ · LangGraph · OpenAI 兼容 LLM · FastAPI · MCP

---

## 目录

- [这是什么](#这是什么)
- [三个方案](#三个方案)
- [核心能力](#核心能力)
- [架构](#架构)
- [评测结果](#评测结果)
- [快速开始](#快速开始)
- [配置项](#配置项)
- [运行评测与实验](#运行评测与实验)
- [HTTP 服务](#http-服务)
- [目录结构](#目录结构)
- [指标框架](#指标框架)
- [安全与隐私](#安全与隐私)
- [已知限制](#已知限制)

---

## 这是什么

电商宿主（mall）需要两类 Agent 能力：

- **客服（cs）**：查订单、退款/取消、查政策、确认收货；
- **导购（guide）**：搜商品、比价、加购、领券、推荐。

本项目把两类能力做成 **23 个工具**，再用三种 Agent 架构去调用它们，回答同一个问题：

> 多 Agent 的隔离性，值不值得它的调度开销？Skill 懒加载能不能用更少的上下文拿到同样的效果？

**项目定位**：可复现的 Agent 工程实验台——不是聊天 Demo，重点在 **权限、上下文、成本、安全** 四件事上的可测量对比。

## 三个方案

| 方案 | 形态 | LangGraph 图 | 典型特征 |
|------|------|--------------|----------|
| **1** | 单 Agent 全工具 | `ctx → agent → tools`（ReAct 循环） | 步数最少、零绕路；上下文最大 |
| **2** | 中心化多 Agent | `supervisor →(路由)→ cs_worker / guide_worker` | 职责隔离强；调度与交接开销最大 |
| **3** | 单 Agent + Skill 懒加载 | `router(意图) → cs_loop / guide_loop` | 按意图只装对应工具集；输入 token 最省 |

> 客服与导购是**每个方案都要覆盖的两个业务域**，不是三个独立方案。

## 核心能力

- **LangGraph 三套图**：同一套节点/工具，仅装配方式不同，便于横向对比。
- **三层权限模型**：角色 → 域 → 权限（`read < operate < write`），叠加数据级规则（只能查自己的订单/地址），敏感操作需显式确认。
- **上下文策略**：`full` / `sliding` / `summarize` / `hier`，以及服务端的自动压缩。
- **Skill 懒加载**：按意图路由后只注册相关工具，减少 schema 占用与误调用。
- **RAG 接入**：通过 MCP stdio 调外部 `RAG-MCP-SERVER`（客服/导购/政策三个知识库）。
- **双 harness 评测**：同一批任务在 custom 与 τ-Bench 风格环境交叉验证。
- **安全护栏**：幻觉防护（空结果注入约束）、越权拦截、间接提示注入检测、误拒率。
- **HTTP 服务层**：SSE 流式对话、会话持久化、实时监控、并发压测。

## 架构

```
                 ┌──────────────────────────────┐
   CLI / 评测 ───▶│  FastAPI  :8002（可选）       │
                 │  SSE 对话 / 会话 / 监控 / 压测 │
                 └──────────────┬───────────────┘
                                ▼
                 ┌──────────────────────────────┐
                 │  LangGraph Agent             │
                 │  scheme 1 / 2 / 3            │
                 └──────────────┬───────────────┘
                                ▼
                 ┌──────────────────────────────┐
                 │  ToolExecutor                │
                 │  权限 guard · 幻觉 guard      │
                 └───────┬──────────────┬───────┘
                         ▼              ▼
              mall API :8085       RAG-MCP-SERVER
              或 Mock（mall.sql）   （MCP stdio 子进程）
```

- **Mock 优先**：没有 Java 后端也能跑——`MockMallBackend` 直接解析 `mall.sql`；没有 sql 文件时注入示例数据。
- **真实后端**：配好 `MALL_API_BASE` 与账号后自动切 HTTP + JWT。

## 评测结果

以下数据来自真实 LLM（DeepSeek），每个方案跑 4 个任务。

### 三方案主结果（3 方案 × 双 harness × 4 任务）

| 指标 | 方案 1 单 Agent | 方案 2 多 Agent | 方案 3 Skill 懒加载 |
|------|----------------|----------------|---------------------|
| 成功率 | 1.000 | 1.000 | 1.000 |
| 平均步数 | **2.88** | 7.50 | 6.50 |
| 平均绕路 | **0** | 3.00 | 1.50 |
| 平均输入 token | 5952 | 8493 | **4549** |
| 每任务成本（¥） | 0.006 | 0.010 | **0.005** |

**结论**：方案 1 最直接（步数少、零绕路），方案 3 最省 token（比方案 2 省 46%），方案 2 隔离强但调度开销最大。跨 harness 一致率 **12/12 = 1.00**。

### τ-Bench 官方零售环境

```
Average reward: 1.0（3/3 任务满分，DB 哈希判定）
```

### 上下文策略（4 个客服任务）

| 策略 | Pass@1 | 输入 token | 成本（¥） |
|------|--------|-----------|----------|
| full | 1.00 | 5672 | 0.0245 |
| sliding | 1.00 | 6241 | 0.0271 |

短对话下 `full` 更划算；`sliding` 的优势要更长对话才体现。

### 工具 schema 静态审计（23 个工具，tiktoken cl100k）

| 角色工具集 | 工具数 | schema token |
|-----------|--------|--------------|
| 全量 | 23 | 2214 |
| 客服 cs | 12 | 1233（省 44%） |
| 导购 guide | 13 | 1193（省 46%） |

复现：`python scripts/audit_context_cost.py`

### 服务层容量（Mock 模式）

- 最大并发 **256 不崩**，错误率 0；64 并发内 P95 < 5s；延迟敏感建议 ≤ 16 并发（P95 ≈ 1.4s）。
- 瓶颈是**单进程 CPU**（QPS 稳定 14–19，与并发档位无关），更高吞吐需多实例（见 `deploy/`）。
- 真实 LLM 下并发主要受供应商限速影响，实际约 10–30 并发。

## 快速开始

### 0. 前置

- Python **3.11+**
- 一个 OpenAI 兼容的 LLM Key（默认 DeepSeek）
- 可选：Java 版 mall（真实后端）、`RAG-MCP-SERVER`（RAG 能力）

没有上面两个外部项目也能跑：Mock 模式 + 不启用 RAG。

### 1. 安装

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1
# Windows CMD: .venv\Scripts\activate.bat
# macOS / Linux: source .venv/bin/activate

pip install -e .
# 需要测试依赖：pip install -e ".[dev]"
# 需要 τ-Bench：pip install -e ".[tau]"
```

> PowerShell 若提示禁止运行脚本，可先执行：
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### 2. 配置

```bash
copy .env.example .env      # macOS/Linux: cp
```

至少填 `LLM_API_KEY`；其余可留空。

### 3. 自检

```bash
# 检查 settings / mock / LLM / monitor（会真实调用一次 LLM）
python -X utf8 -m mall_agent.cli.p0_check

# 只验证 mock 链路，不花 token
python -X utf8 -m mall_agent.cli.run_agent --mock-check
```

### 4. 跑一个任务

```bash
# 方案 1：单 Agent
python -X utf8 -m mall_agent.cli.run_agent "帮我查一下我的订单"

# 方案 3：Skill 懒加载
python -X utf8 -m mall_agent.cli.run_agent --scheme 3 "推荐一款 3000 元以内的手机"
```

## 配置项

优先级：**环境变量 / `.env` > `config/settings.local.yaml` > `config/settings.yaml` > 代码默认值**。

| 变量 | 必填 | 说明 |
|------|------|------|
| `LLM_API_KEY` | ✅ | 主 LLM Key |
| `LLM_BASE_URL` | | 默认 `https://api.deepseek.com` |
| `LLM_MODEL` | | 默认 `deepseek-chat` |
| `USER_SIM_API_KEY` / `USER_SIM_BASE_URL` / `USER_SIM_MODEL` | | 用户模拟器可单独配更便宜的模型 |
| `MALL_API_BASE` | | 真实 mall 地址，默认 `http://127.0.0.1:8085` |
| `MALL_USE_MOCK` | | 默认 `true` |
| `MALL_SQL_PATH` | | Mock 用的 `mall.sql` 路径；不填则用内置示例数据 |
| `MALL_TEST_USER` / `MALL_TEST_PASSWORD` | | 测试账号（Mock 自动创建） |
| `MALL_MYSQL_HOST` / `PORT` / `USER` / `PASSWORD` / `DB` | | 归属校验直连 MySQL 时使用 |
| `RAG_MCP_PYTHON` | | RAG-MCP-SERVER 的 Python 解释器；不填则 RAG 降级 |
| `RAG_MCP_CWD` / `RAG_MCP_SRC` / `RAG_MCP_MODULE` | | 同上 |

- `.env` **不会进 git**，仓库里只有 `.env.example` 模板。
- 非密钥的本机差异可写进 `config/settings.local.yaml`（从 `config/settings.local.yaml.example` 复制，同样不入库）。
- 参考：`config/settings.yaml`、`config/permissions.yaml`。

## 运行评测与实验

```bash
# 三方案评测（custom harness）
python -X utf8 -m mall_agent.cli.eval --scheme 1,2,3 --policy sliding --limit 6

# 并发 + 关键任务门槛 + 用户模拟器
python -X utf8 -m mall_agent.cli.eval --scheme 1 --limit 10 --concurrency 4
python -X utf8 -m mall_agent.cli.eval --scheme 1 --limit 10 --critical-ids CP-001,CF-002A,CO-001
python -X utf8 -m mall_agent.cli.eval --scheme 1 --limit 6 --simulator hybrid

# 全矩阵（3 方案 × 上下文策略 × 双 harness）
python -X utf8 -m mall_agent.experiments.exp_matrix --scheme 1,2,3 --policy full,sliding --harness custom,tau

# 三组专项实验
python -X utf8 -m mall_agent.cli.exp --context --limit 6
python -X utf8 -m mall_agent.cli.exp --permission
python -X utf8 -m mall_agent.cli.exp --skill

# 安全测试
python -X utf8 -m mall_agent.cli.safety_test --scheme 1,2,3

# τ-Bench（需 pip install -e ".[tau]"）
python -X utf8 -m mall_agent.cli.tau_bench_test --start 0 --end 3
```

报告输出到 `data/reports/`（不入库）。

### RAG 知识库（可选）

需要先准备外部 `RAG-MCP-SERVER`，并在 `.env` 配好 `RAG_MCP_*`：

```bash
python -X utf8 -m mall_agent.cli.ingest_kb          # 摄取 cs / guide / policy 知识库
python -X utf8 -m mall_agent.cli.eval_rag --ingest  # 检索门禁：hit_rate / MRR
```

## HTTP 服务

```bash
python -X utf8 -m mall_agent.server --port 8002
# 可选：--scheme 1|2|3  --policy auto|full|sliding|summarize|hier  --mock-agent  --rag-off
```

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/mall/chat/stream` | SSE 流式对话（meta / text / products / done） |
| GET | `/api/mall/conversations` | 会话列表 |
| GET | `/api/mall/conversations/{id}/messages` | 会话消息 |
| POST | `/api/mall/conversations/hide` | 隐藏会话 |
| GET | `/api/mall/health` | 健康检查 |
| GET | `/api/monitor/stream` | 实时监控 SSE（QPS / 延迟 / token / 成本） |
| GET | `/api/monitor/snapshot` | 监控快照 |
| GET | `/api/monitor/loadtests` | 压测记录 |

服务层压测：

```bash
# 终端 1：mock agent，只测服务层吞吐
python -X utf8 -m mall_agent.server --mock-agent --port 8002

# 终端 2
python -X utf8 -m mall_agent.cli.load_test --max-concurrency 128 --duration 20
```

## 目录结构

```text
mall-agent/
├── src/mall_agent/
│   ├── agents/          # LangGraph 三方案图 + Skill
│   ├── tools/           # 23 个工具（cs / guide / admin）
│   ├── permission/      # 角色-域-权限 + 数据级规则
│   ├── context/         # full / sliding / summarize / hier
│   ├── mall_api/        # 真实 client + Mock（读 mall.sql）
│   ├── rag/             # RAG-MCP stdio 客户端
│   ├── eval/            # harness、指标、报告
│   ├── server/          # FastAPI + SSE + 监控
│   └── cli/             # 命令行入口
├── config/              # settings / permissions / 本机配置模板
├── data/golden/         # 评测任务数据（提交）
├── scripts/             # 审计、生成、发布脚本
├── tests/               # 混沌测试
└── deploy/              # systemd / nginx 多实例模板
```

更细的调用关系与模块职责见 [README-project-struct.md](README-project-struct.md)。

## 指标框架

按上线前关心的三类指标组织：

| 类别 | 关键指标 |
|------|----------|
| **质量 / 正确性** | 端到端成功率、Pass@1、Pass^k（关键任务）、rubric 加权分、功能正确性、事实一致性、参数准确度、步数超限率 |
| **成本 / 效率 / 延迟** | 平均步数、有效绕路、输入/输出 token、P95/P99、TTFT/TPOT、单任务 token 预算、工具耗时 TopN、并发压测 |
| **安全 / 合规** | 越权拦截率、guardrail 漏拦率、幻觉率、误拒率、间接提示注入、输出毒性、审计字段完整性 |
| **业务可用性**（代理） | 有用性（rubric × 事实一致性）、投诉风险、人工兜底率 |

实现位置：`src/mall_agent/eval/metrics/` 与 `src/mall_agent/monitoring/`。

## 安全与隐私

**Agent 侧**

- 权限三层 guard + 数据级规则（跨用户/越权直接拒绝）。
- 空结果防护：检索类工具返回空时注入"不得编造"约束（`hallucination_guard.py`）。
- 幻觉检测：对最终回复扫描价格/库存/物流等编造信号并标记。
- 安全评测覆盖：不存在商品、要求降价、查询他人信息、间接提示注入、合法基线误拒。

**仓库侧**

- `.env`、`docs/`、`plan_exec/`、进展记录、运行产物（`data/sessions`、`data/reports` 等）均被 `.gitignore` 排除。
- 仓库只保留 `.env.example` 与 `config/settings.local.yaml.example` 两个**无真实值**的模板。
- 提交前可自查：

```bash
git grep -n -I -E "sk-[A-Za-z0-9]{20,}"
git status --short
```

## 已知限制

- **外部依赖**：真实客服链路依赖 Java 版 mall；RAG 依赖外部 `RAG-MCP-SERVER`。仓库内只保证 Mock 链路可独立运行。
- **评测规模**：主表基于 4 个任务、单次试验，适合看相对趋势，不宜当绝对性能结论。
- **幻觉判定**：`HallucinationDetector` 是规则 + 信号启发式，不是严格事实核查。
- **并发结论**：容量数据来自 Mock 模式单机，真实 LLM 下主要受供应商限速约束。
- **平台**：脚本与示例以 Windows 为主，Linux/macOS 需自行替换路径写法。

---

实验报告、阶段复盘等内部文档仅在本地保留，不随仓库分发。
