# mall-agent — 项目结构说明

> AI 客服 + 导购学习项目（Python / LangGraph）。为 mall 电商宿主构建 AI 客服与导购，
> 对比「单 Agent 全工具 / 中心化多 Agent / 单 Agent + Skill 懒加载」三方案，
> 用 τ-Bench 风格 + 黄金数据集评测。知识问答走 RAG（调用 RAG-MCP-SERVER）。
> 位置：`mall-agent`（独立 git 仓库，分支 `main` 基准）

---

## 1. 调用关系（依赖方向）

```
mall(Java 商城)                      RAG-MCP-SERVER            （本项目依赖两者）
  │ mall.sql 只读(mock)                   │ MCP stdio 子进程
  │ HTTP :8085 /product/search ...        ▼
  └─────────────▶  mall-agent ★     ──▶  知识库检索(cs_kb/product_kb/policy_kb)
```
- **调用 mall**：只读 `mall-master/document/sql/mall.sql`（mock 真实数据：33 商品/5 券/51 订单/12 品牌），
  可选 HTTP `:8085`（真实 portal API + JWT）。
- **调用 RAG-MCP-SERVER**：子进程 MCP stdio，配置见 `config/settings.yaml` `rag:` 段。
- 数据流：Mock 先跑通 → 切真实。

---

## 2. 目录结构地图

```
mall-agent\
├── src\mall_agent\                # ★ 核心包
│   ├── agents\                    # LangGraph 三方案图
│   │   ├── graph_factory.py       #   图工厂（方案 1/2/3 装配）
│   │   ├── nodes.py               #   图节点
│   │   └── skills\skill_base.py   #   Skill 懒加载机制
│   ├── agents\skills\             # （技能注册）
│   ├── tools\                     # 23 个工具（按场景拆分）
│   │   ├── base.py / registry.py
│   │   ├── cs_tools.py            #   客服工具
│   │   ├── guide_tools.py         #   导购工具
│   │   └── admin_tools.py         #   后台/越权类（含幻觉防护注入点）
│   ├── permission\                # 三层权限模型 / guard
│   ├── context\                   # 上下文策略（full / sliding / summary 自动压缩）
│   ├── rag\mcp_client.py          # ★ RAG-MCP-SERVER 的 MCP 客户端封装
│   ├── mall_api\                  # mall 对接（真实 client + MockMallBackend 直读 sql）
│   │   ├── mock_backend.py        #   解析 mall.sql 的本地 mock
│   │   ├── client.py product.py cart.py order.py member.py
│   ├── server\                    # ★ HTTP 服务层 :8002
│   │   ├── main.py / __main__.py  #   启动入口
│   │   ├── app.py / agent_api.py  #   SSE 流式对话接口
│   │   ├── conversation.py        #   会话历史（data/sessions/）
│   │   ├── monitor_api.py         #   实时监控接口
│   │   └── sse.py
│   ├── eval\                      # ★ 评测 harness
│   │   ├── golden\generator_v3.py #   黄金数据集生成器（参数化模板）
│   │   ├── harness\               #   custom_runner / tau_adapter / extract_zh / golden_to_tau
│   │   │                          #   hallucination_guard（空结果注入约束）/ safety_judge / probe
│   │   ├── metrics\               #   aggregate / scoring / safety / detection
│   │   └── report.py
│   ├── cli\                       # 命令行入口
│   │   ├── eval.py                #   评测
│   │   ├── exp.py                 #   三实验（context/permission/skill）
│   │   ├── safety_test.py         #   安全测试
│   │   ├── tau_bench_test.py      #   τ-Bench 官方 retail 接入
│   │   ├── ingest_kb.py           # ★ 生成知识源并摄取到 RAG-MCP-SERVER
│   │   ├── run_agent.py / load_test.py / p0_check / p1_check
│   ├── monitoring\ loadtest\ sessions\ core\  # 监控/压测/会话持久化/LLM 与设置
├── config\
│   ├── settings.yaml              # ★ 运行配置（只含可公开的默认值）
│   ├── settings.local.yaml.example# 本机私有覆盖模板（复制为 settings.local.yaml，已 gitignore）
│   └── permissions.yaml           # 工具权限策略
├── data\                          # ★ 评测/运行数据（reports/sessions/loadtest 等已 gitignore）
├── tests\chaos\                   # 混沌测试（chaos_test.py / faulty_llm.py）
├── deploy\                        # 部署（nginx 多实例模板等）
├── README.md                      # 项目说明（根目录唯一提交的文档）
├── docs\  plan_exec\              # 实验报告/过程记录（仅本地，已 gitignore）
├── progress.md  task_plan.md  findings.md  # 进展记录（仅本地，已 gitignore）
└── .env                           # LLM_API_KEY / 密码（仅本地，已 gitignore）
```

---

## 3. 核心入口与运行命令

### 服务层（真实对话）
```bash
cd mall-agent
# 启动 Agent 会话服务 :8002（SSE；真实 LLM）
PYTHONPATH=src .venv\Scripts\python.exe -X utf8 -m mall_agent.server --port 8002
# 端点：POST /api/mall/chat/stream（SSE）· GET /api/mall/conversations · GET /api/mall/health
#       GET /api/monitor/stream|snapshot（监控）
```

### 评测/实验
```bash
# 1. RAG 知识库摄取（先用：cs_kb / product_kb / policy_kb）
.venv\Scripts\python.exe -X utf8 -m mall_agent.cli.ingest_kb
# 2. 三方案评测（custom harness）
.venv\Scripts\python.exe -X utf8 -m mall_agent.cli.eval --scheme 1,2,3 --policy sliding --limit 6
# 3. 三组实验（context / permission / skill）
.venv\Scripts\python.exe -X utf8 -m mall_agent.cli.exp --context --limit 6
.venv\Scripts\python.exe -X utf8 -m mall_agent.cli.exp --permission
.venv\Scripts\python.exe -X utf8 -m mall_agent.cli.exp --skill
# 4. 安全测试（12 边界 + 间接注入 + 3 合法基线）
.venv\Scripts\python.exe -X utf8 -m mall_agent.cli.safety_test --scheme 1,2,3
# 5. τ-Bench 官方接入（retail）
.venv\Scripts\python.exe -X utf8 -m mall_agent.cli.tau_bench_test --start 0 --end 3
```
报告统一输出到 `data/reports/`。

### 关键评测结论（已在仓库验证）
| 维度 | 方案1 单Agent | 方案2 多Agent | 方案3 Skill懒加载 |
|---|---|---|---|
| 成功率(全矩阵) | 1.000 | 1.000 | 1.000 |
| 平均步数 | **2.88** | 7.50 | 6.50 |
| 每任务输入 token | 5952 | 8493 | **4549** |
| 越权拒绝（安全6/6基线） | 6/6 | 6/6 | 6/6 |
| 隐私不泄露（3 条） | 1/3 | 0/3 | **3/3** |

---

## 4. 测试 / 评测数据位置清单（★ 重点）

> 关系：**mock 真实数据**（mall.sql 只读）→ **黄金任务**（golden_*.jsonl）→
> 喂给 harness 跑三方案 → 产物落 `data/`。RAG 政策问答靠 `ingest_kb` 摄取到上游。

### 4.1 评测核心数据（入库，data/golden/）
```
data\golden\
├── golden_v1.jsonl      # 14 条基础黄金任务
├── golden_v2.jsonl      # 23 条难分辨任务（7 难度维度，τ-Bench 风格）
├── golden_v3.jsonl      # 103 条参数化任务（12 模板：CP/PC/MX/CF/ID/EX/TR/SV/OF/PQ/CO/BD）
├── golden_v4_rag.jsonl  # 111 条政策/RAG 扩展任务
├── golden_v5.jsonl      # 22 条全可判定任务
├── golden_safety.jsonl  # 16 条安全任务（12 边界 + 间接注入 + 3 合法基线）
└── manifest*.json       # 版本清单/索引
```
- 生成器：`src/mall_agent/eval/golden/generator_v3.py`（用 mall.sql 真实数据参数化展开）。

### 4.2 mock 真实数据（mall.sql，只读——测试的「商品/券/订单」来源）
```
mall-master\document\sql\mall.sql     # 33 商品 / 5 券 / 51 订单 / 12 品牌 等
                                      # settings.mall.sql_path 指向（路径在 config/settings.yaml）
```

### 4.3 RAG 知识源（供 `ingest_kb` 摄取到上游；对应 12 个 PQ 政策主题）
```
（运行时生成于 data\kb_src\，再子进程摄取到 RAG-MCP-SERVER）
售后/退款/配送/会员/优惠券（cs_kb） · 商品知识(product_kb) · 政策(policy_kb)
```
对应上游已摄取的样例文档见 `RAG-MCP-SERVER\data\uploads\returns_policy.txt` 等。

### 4.4 评测/实验/运行产物（data/，多被 gitignore —— 本机已实际生成）
```
data\reports\            # ★ 报告：full_matrix_report.md + raw_results.json、
                         #   exp_{context,permission,skill}.json、report_*.md、safety_test.json
data\traces\             # 评测轨迹（推理过程）
data\sessions\           # ★ 服务层会话历史（1577+ 个 json，服务器对话持久化）
data\loadtest\           # 并发压测结果（loadtest_*.json）
data\chaos\              # 混沌/容错测试输入
data\server_lt.log       # 压测日志
```
> 注意：`data/sessions`、`data/reports` 等默认 gitignore。运行产物不建议用 `git add -f` 强加进仓库，确需版本化请放独立数据仓库。

### 4.5 测试代码
- 评测/测试的**数据驱动源**在 `data/golden`；**可执行代码**主要经 `cli/*.py` 与 `src/mall_agent/eval/*`。
- 传统 pytest 用例较少（`tests/chaos/`），核心验证方式为上述 CLI 评测 + 本地实验报告。
- τ-Bench 官方 retail 环境已装入 `.venv`（`tau_bench`），测试走 `cli/tau_bench_test.py`。

---

## 5. 文档地图

| 文件 | 内容 |
|---|---|
| `README.md` | 主说明（三方案 / 评测结果 / 命令速查） |
| `README-project-struct.md` | 本文：目录结构、调用关系、命令 |
| `deploy/README.md` | 多实例部署（systemd / nginx） |
| `docs/`（仅本地，不入库） | 测试集指南与各实验报告 |
| `progress.md / task_plan.md / findings.md`（仅本地） | 任务规划与进度复盘 |
| `.env.example` / `config/settings.local.yaml.example` | 配置模板（不含真实密钥） |

## 6. 关键事实速查
| 项 | 说明 |
|---|---|
| 端口 | HTTP 会话服务 **:8002**（SSE） |
| LLM 默认 | DeepSeek（`config/settings.yaml`，`LLM_API_KEY`） |
| 三方案 | 1=单 Agent 全工具 · 2=中心化多 Agent · 3=单 Agent+Skill 懒加载 |
| RAG 对接 | 子进程 MCP stdio → RAG-MCP-SERVER（cs_kb/product_kb/policy_kb） |
| 护幻觉 | `empty_result_guard`（12 检索类工具空结果注入系统约束） |
