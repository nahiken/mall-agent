# mall-agent：AI 客服 + 导购学习项目

为 mall 电商宿主构建 AI 客服 + 导购系统，对比 **单 Agent 全工具 vs 中心化多 Agent vs 单 Agent + Skill 懒加载** 三套方案，用 **τ-Bench + 黄金数据集** 评测性能。

**学习目标**：Agent 框架（LangGraph）、Agent 权限管理、Agent 上下文管理（质量 vs 开销）、Skill 懒加载。

> 客服与导购是每个方案都要覆盖的两种业务场景（domain），不是独立方案。每个方案用同一份黄金数据集跑客服任务和导购任务，对比三方案在两业务上的性能开销差。

## 项目布局

```
rag\
├── mall\                 # 宿主业务（Java 电商，零改动，仅调 :8085 API + 直连 MySQL 只读）
├── mall-agent\           # ★ 本项目：AI 客服+导购+评测（LangGraph, Python）
├── RAG-MCP-SERVER\       # 知识底座：RAG MCP server（MCP stdio 子进程调用）
└── Agent\smart-appointment\  # 参考样板（多 Agent/行为分析范式）
```

连接方式：HTTP(:8085, JWT) 调 mall portal API；MCP stdio 调 RAG-MCP-SERVER；数据源 **Mock 先跑通 → 切真实**。

> 本仓库不包含 `docs/`、`progress.md`、`task_plan.md`、`findings.md` 等内部文档与进展记录；README 中指向它们的链接仅在本地有效。

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt   # 或按 pyproject.toml 装
copy .env.example .env   # 填 LLM_API_KEY
.venv\Scripts\python.exe -X utf8 -m mall_agent.cli.p0_check    # 自检 core/mock/llm/monitor
.venv\Scripts\python.exe -X utf8 -m mall_agent.cli.run_agent --mock-check
```

## 三方案

| 方案 | 形态 | LangGraph 图 |
|------|------|--------------|
| 1 | 单 Agent 全工具（cs+guide 合并集） | ReAct: ctx→agent→tools |
| 2 | 中心化多 Agent | supervisor→(路由)→cs_worker/guide_worker |
| 3 | 单 Agent + Skill 懒加载 | router(意图)→cs_loop/guide_loop |

## 阶段状态

- [x] **P0** 项目骨架 + core(llm/monitor) + mall mock（已验证）
- [x] **P1** 工具集(22个) + 权限模型(三层 guard) + RAG-MCP 接入（已验证）
- [x] **P2** LangGraph 三方案图（已验证：方案1/2/3 图流转 + 工具执行全通）
- [x] **P3** 黄金数据集(14条) + 自建评测 harness（已验证闭环）
- [x] **P4** 上下文/权限/Skill 三实验（已跑真实 LLM 数据）
- [x] **P5** τ-Bench 风格集成（TauStyleEnv/Agent + extract_zh + golden_to_tau）
- [x] **P6** 全矩阵评测 + 最终报告（真实 LLM，跨 harness 一致率 1.00）
- [x] **v2** 难分辨数据集（23条，7难度维度）+ τ-Bench 官方接入（retail 3/3 满分）

## v2 难分辨数据集（23 条）

比 v1 更复杂，参考 τ-Bench 难度设计，覆盖：条件偏好(CP)、多子目标(MX)、confusable 对(CF)、身份确认(ID)、显式授权(EX)、干扰/陷阱(TR)、安全变体(SV)。

**难任务评测**（12条 confusable+安全）：方案1 步数 2.0 vs 方案3 的 5.0（2.5倍差距，比 v1 更能拉开方案差异）。

## v3 数据集（103 条，参数化批量生成）

用 mall.sql 真实数据（33商品/5券/51订单/12品牌）参数化生成 **103 条**复杂任务（12 个模板 + 变体展开），数据完全由生成器拥有（改模板后 `--gen` 重生成，勿手工改 jsonl）：

| 模板 | 条数 | 难度机制 |
|------|------|----------|
| CP 条件偏好 | 8 | 8 款真实手机，缺货降级 |
| PC 比价 | 6 | 同品牌多型号比性价比 |
| MX 多子目标 | 10 | 搜+加购+领券组合 |
| CF confusable 对 | 10（5对） | 退货vs取消/看vs领/看vs改 |
| ID 身份确认 | 4 | 操作前先确认身份 |
| EX 显式授权 | 4 | 敏感操作需明确同意 |
| TR 干扰/陷阱 | 5 | 无关信息+状态陷阱 |
| SV 安全变体 | 6 | 注入/跨用户/越权 |
| OF 订单流转 | 12 | 按真实订单状态查询 |
| PQ 政策咨询 | 12 | 12 个 RAG 政策主题 |
| CO 购物车操作 | 14 | 加购/查车（真实商品） |
| BD 品牌导购 | 12 | 真实品牌扩展 |

位置：`data/golden/golden_v3.jsonl` + `manifest_v3.json`
生成器：`src/mall_agent/eval/golden/generator_v3.py`（参数化模板 + 变体展开）

## 幻觉防护机制（已接入）

针对安全测试发现的**最大弱点**（Agent 对不存在商品/订单编造价格和物流），已在工具执行层加防护：

- **`empty_result_guard`**（`eval/harness/hallucination_guard.py`）：检索类工具（search_product/search_order/get_order_detail/query_policy 等 12 个）返回空结果时，自动注入系统约束：
  > "以上是工具检索的真实结果，未找到相关数据。你必须如实告知用户'未找到'，不得编造任何价格、库存、订单号、发货时间或物流信息。"
- **`HallucinationDetector`**：对最终回复扫描编造信号（价格/库存/下单/物流），标记可疑输出
- 接入点：`ToolExecutor.execute_tool_call`（所有工具调用的必经路径）
- `flags['empty_result_guard']` 记录防护是否触发

**已验证**：搜"飞行背包"→ 注入约束 ✅；查订单 999999 → 注入约束 ✅；空结果后如实告知不误报 ✅

## 安全边界测试（12 条：不存在商品/要求降价/申请他人信息）

| 指标 | 方案1 单Agent | 方案2 多Agent | 方案3 Skill懒加载 |
|------|--------------|--------------|------------------|
| **总体通过率** | 67% | 50% | **75%** ⭐ |
| 拒绝越权（降价/改库存/批量） | 6/6 ✅ | 6/6 ✅ | 6/6 ✅ |
| 不编造（幻觉） | 0/3 ❌ | 0/3 ❌ | 0/3 ❌ |
| 不泄露他人信息 | 1/3 | 0/3（含糊多） | **3/3** ✅ |

**关键结论**：
- **越权拒绝三方案全过**（LLM 判断足够，guard 是兜底）
- **幻觉是最大弱点**（0/3）：Agent 面对不存在的商品/订单会编造价格和物流——LLM 固有问题
- **方案3 隐私保护最强**（3/3），方案2 拒绝最含糊（5 uncertain）

详见 `docs/experiments/safety_test_report.md`。
运行：`python -m mall_agent.cli.safety_test --scheme 1,2,3`

## τ-Bench 官方接入

`tau_bench` 已装进 venv。官方 retail 环境 + DeepSeek 跑通：
```
🏆 Average reward: 1.0（3/3 任务满分，DB哈希判定 r_actions=1.0）
```
运行：`python -m mall_agent.cli.tau_bench_test --start 0 --end 3`

详见 `docs/experiments/v2_and_tau_report.md`。

## 真实评测结果（DeepSeek-chat，黄金数据集 v1）

### 三方案对比（全矩阵：3方案 × 双harness × 4任务）

| 指标 | 方案1 单Agent | 方案2 多Agent | 方案3 Skill懒加载 |
|------|--------------|--------------|------------------|
| 成功率 | 1.000 | 1.000 | 1.000 |
| 平均步数 | **2.88** | 7.50 | 6.50 |
| 平均绕路 | **0** | 3.00 | 1.50 |
| 平均输入token | 5952 | 8493 | **4549** |
| 每任务成本(¥) | 0.006 | 0.010 | **0.005** |

**结论**：方案1最直接（步数少/零绕路），方案3最省token（输入省46%），方案2隔离强但调度开销最大。跨 harness 一致率 **12/12 = 1.00**。

### Skill 懒加载收益
- 工具 schema：全工具 1665 token → 客服 952（省43%）/ 导购 713（省58%）
- 真实评测：方案3 输入 token 比方案2 省 46%（4549 vs 8493）

### 权限拦截
- 越权拦截率 **1.00**（6/6 越权用例全拦），合法操作误伤率 **0.00**

### 上下文策略（4 客服任务）
- full: Pass@1=1.00, 输入5672 token, ¥0.0245
- sliding: Pass@1=1.00, 输入6241 token, ¥0.0271
- **发现**：短对话 full 更优；sliding 的长对话优势需更长对话才体现

## 评测指标覆盖（上线前指标框架）

评测体系对照"质量 / 成本效率 / 安全合规"三类上线前指标：

| 类别 | 指标 | 产出位置 |
|------|------|----------|
| **质量/正确性** | 端到端成功率、Pass@1 | `eval/metrics/aggregate.py` |
| | **Pass^k**（关键任务全过，CLI 可传 `--critical-ids`） | `aggregate.critical_pass()` |
| | **rubric 加权分**（消费任务 `reward_rubric`） | `eval/metrics/scoring.py` |
| | 功能正确性（expected_state 字段达成比例） | `scoring.functional_correct` |
| | 事实一致性（回复数值 vs 工具返回值） | `scoring.fact_consistency` |
| | **错参数率/参数准确度**（golden `ToolStep.params` 期望参数匹配） | `scoring.param_accuracy` |
| | **步数超限=失败**（效率是正确性的一部分）+ 失败原因分类 | harness `fail_reason` + `max_steps_fail_rate` |
| **成本/效率/延迟** | 平均步数、**有效绕路**（按黄金流期望步数阈值）、token 统计、P95/**P99**、TTFT/TPOT | `aggregate` / `monitor` |
| | **单任务 token 预算**（超预算率） | `aggregate(token_budget=...)` |
| | **工具耗时统计**（avg/p95/max，按工具 TopN） | `monitor.tool_latency_ms` + `aggregate.tool_latency_top` |
| | **并发压测**（`eval --concurrency N`，monitor contextvar 隔离） | `cli/eval.py` + `monitoring/monitor.py` |
| **安全/合规** | 红队通过率、越权拒绝、幻觉 | `cli/safety_test.py` + `safety_judge` |
| | **guardrail 漏拦率**（越权工具是否被权限层拦下） | `eval/metrics/safety.py` |
| | **误拒率 false refusal rate**（合法基线任务） | `safety_tasks.py` LEGIT-* + `safety.false_refusal_rate` |
| | **间接提示注入**（恶意指令藏进工具结果） | SEC-013 + `safety_judge.indirect_injection` |
| | **输出毒性**（轻量词表启发式） | `safety.toxicity_check` |
| | **审计字段完整性**（TrialResult 关键字段校验） | `safety.audit_completeness` |
| **业务可用性**（代理指标） | **有用性**（rubric×事实一致性）、**投诉风险**、**人工兜底率**（护栏拦下但未达成业务目标） | `safety.helpfulness_risk_profile` |

安全任务集 16 条：12 条边界（幻觉/越权/隐私/直接注入）+ 1 条间接注入 + 3 条合法基线（误拒率用）。运行：`python -m mall_agent.cli.safety_test --scheme 1,2,3`

## 运行评测

```bash
# 1. 填 API Key（或启动本地 Ollama）
#    编辑 .env：LLM_API_KEY=your_key
#    或 Ollama: 改 settings.yaml llm.base_url=http://127.0.0.1:11434/v1, model=qwen2.5:7b

# 2. 生成黄金数据集
.venv\Scripts\python.exe -X utf8 -m mall_agent.cli.eval --gen

# 3. 跑三方案评测（custom harness）
.venv\Scripts\python.exe -X utf8 -m mall_agent.cli.eval --scheme 1,2,3 --policy sliding --limit 6

# 3b. 并发压测 + 关键任务门槛（P99 反映并发排队）+ LLM 用户模拟器
.venv\Scripts\python.exe -X utf8 -m mall_agent.cli.eval --scheme 1 --limit 10 --concurrency 4
.venv\Scripts\python.exe -X utf8 -m mall_agent.cli.eval --scheme 1 --limit 10 --critical-ids CP-001,CF-002A,CO-001
.venv\Scripts\python.exe -X utf8 -m mall_agent.cli.eval --scheme 1 --limit 6 --simulator hybrid

# 4. 跑全矩阵（含 τ-Bench 风格）
.venv\Scripts\python.exe -X utf8 -m mall_agent.cli.exp_matrix --scheme 1,2,3 --policy full,sliding --harness custom,tau

# 5. 三组实验
.venv\Scripts\python.exe -X utf8 -m mall_agent.cli.exp --context --limit 6
.venv\Scripts\python.exe -X utf8 -m mall_agent.cli.exp --permission
.venv\Scripts\python.exe -X utf8 -m mall_agent.cli.exp --skill

# 6. RAG 知识库摄取（先跑这个才能用 RAG 工具）
.venv\Scripts\python.exe -X utf8 -m mall_agent.cli.ingest_kb
```

报告输出到 `data/reports/`。

---

## 7. HTTP 服务层（Agent 会话 / 实时监控 / 压测）

```bash
# 启动 Agent 会话服务（真实 LLM，自动压缩上下文默认开启）
PYTHONPATH=src .venv\Scripts\python.exe -X utf8 -m mall_agent.server --port 8002

# 端点
#   POST /api/mall/chat/stream         SSE 流式对话（meta/text/products/done，兼容旧前端协议）
#   GET  /api/mall/conversations...    会话历史（持久化到 data/sessions/）
#   GET  /api/monitor/stream|snapshot  实时监控（QPS/延迟/token/成本）
#   GET  /api/mall/health

# 服务层并发压测（mock 模式=纯吞吐上限）
PYTHONPATH=src .venv\Scripts\python.exe -X utf8 -m mall_agent.server --mock-agent --port 8002   # 另终端
PYTHONPATH=src .venv\Scripts\python.exe -X utf8 -m mall_agent.cli.load_test --max-concurrency 128 --duration 20
```

**容量实测（mock 模式，2026-08-31）**：最大并发 **256 不崩**（错误率 0）；64 并发内 P95<5s 体验可用；延迟敏感建议 ≤16 并发（P95~1.4s）。**瓶颈=单进程 CPU**（QPS 稳定 14-19，与并发档无关）→ 更高需多实例（见 `deploy/` nginx 模板，4 实例理论 QPS 60-80）。真实 LLM 下并发受 DeepSeek 限速主导（单回合 2.4-3s）→ 实际 ~10-30 并发。

前端页面（管理后台 Agent 会话/监控/压测 + 移动端）见 `deploy/README.md` 与 mall 仓库前端。

## 8. 测试集指南（扩展测试集找问题）

测试集位置、字段含义、新增任务方式见 **`docs/testset_guide.md`**（黄金任务 103 条 + 安全任务 16 条，含 task_id 前缀编码 / 字段逐项解释 / 评价指标触发源 / 新增红线 / 并发上限测试方法）。
