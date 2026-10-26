# mall-agent 多实例部署（Linux）

## 为什么需要多实例 / nginx

- 单 uvicorn worker = 单进程单事件循环，真实请求被 LLM 时延主导（DeepSeek 限速），
  但**服务层吞吐**（会话读写、SSE、工具调度）受单进程阻塞点限制。
- Windows 开发机：`uvicorn.run(app)` 单进程即可（本仓库默认）；**uvicorn 的
  `workers=N` 参数在 Windows 上不可用**（官方限制），Linux 也建议用 systemd/多个
  uvicorn 实例而非 `workers=`（worker 内仍是单进程共享内存，多实例才能真正多进程）。

## 多实例方案（Linux）

### 方案 A：systemd 起 N 个 uvicorn 实例（推荐，进程隔离）

```ini
# /etc/systemd/system/mall-agent@.service
[Unit]
Description=mall-agent instance %i
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/mall-agent
Environment=PYTHONPATH=/opt/mall-agent/src
ExecStart=/opt/mall-agent/.venv/bin/python -m mall_agent.server --port 800%i
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
systemctl enable mall-agent@2 mall-agent@3 mall-agent@4
systemctl start mall-agent@2 mall-agent@3 mall-agent@4
```

### 方案 B：gunicorn（仅 Linux）+ uvicorn worker

```bash
# 需要 gunicorn + uvicorn[standard]（Linux 才有）
pip install "gunicorn" "uvicorn[standard]"
# 注意：多 worker 需用 import string + gunicorn 管理（uvicorn 自带 workers 在 Windows 不可用）
gunicorn -w 4 -k uvicorn.workers.UvicornWorker \
  "mall_agent.server.app:create_app()" -b 0.0.0.0:8002
```

### nginx 负载均衡

见 `nginx.conf.example`：`upstream mall_agent` 配多个端口，轮询分发；
SSE 路径关 `proxy_buffering`、拉长 `proxy_read_timeout`、`proxy_http_version 1.1`。

## 多实例注意事项

1. **会话存储**：`data/sessions/*.json` 是文件存储。多实例共享同一目录时：
   - `save_session` 用 **原子写**（tmp+rename，阶段16已改）→ 不会产生损坏 JSON；
   - 但**同一会话的并发读写**（同一用户两台设备/重试）仍可能相互覆盖——同一会话
     建议由同一实例处理（可用 nginx 按 user_id 哈希），或将来换 MySQL/Redis。
2. **监控**：`tracing`/`realtime` 是进程内单例，`/api/monitor/*` 只反映当前实例事件。
   多实例时监控页看到的是**被 nginx 命中的那台**——如需全量聚合，需把事件汇到
   共享存储（Redis/Pulsar），当前版本为单实例观测。
3. **RAG-MCP**：每实例各自拉起一个 MCP stdio 子进程（N 实例 = N 个子进程）。
4. **LLM API**：多实例并发会放大 DeepSeek 限流触发概率，`llm.py` 的指数退避重试
   （429/5xx）会兜底，但注意各实例独立计费。

## Windows 开发机快速起 2 个实例（验证多进程）

```bash
# 终端1
PYTHONPATH=src .venv\Scripts\python.exe -X utf8 -m mall_agent.server --port 8002
# 终端2
PYTHONPATH=src .venv\Scripts\python.exe -X utf8 -m mall_agent.server --port 8003
```

（无需 nginx；直接分别 curl 两个端口验证。生产负载均衡才用 nginx。）
