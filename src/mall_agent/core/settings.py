"""mall-agent 配置加载：settings.yaml + .env + 代码默认值。

优先级：环境变量 > settings.yaml > 代码默认值（浅合并）。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, PrivateAttr, Field

# 项目根（本文件在 src/mall_agent/core/settings.py，向上 4 级为项目根）
ROOT = Path(__file__).resolve().parents[3]

load_dotenv(ROOT / ".env")

_DEFAULT_YAML = ROOT / "config" / "settings.yaml"

# 环境变量 → 配置键的覆盖表。
# 密钥、本机路径、测试账号只放在 .env / 环境变量里，仓库内只保留空默认值或模板。
_ENV_MALL = {
    "api_base": "MALL_API_BASE",
    "use_mock": "MALL_USE_MOCK",
    "sql_path": "MALL_SQL_PATH",
    "test_user": "MALL_TEST_USER",
    "test_password": "MALL_TEST_PASSWORD",
    "mysql_host": "MALL_MYSQL_HOST",
    "mysql_port": "MALL_MYSQL_PORT",
    "mysql_user": "MALL_MYSQL_USER",
    "mysql_password": "MALL_MYSQL_PASSWORD",
    "mysql_db": "MALL_MYSQL_DB",
}
_ENV_RAG = {
    "python": "RAG_MCP_PYTHON",
    "module": "RAG_MCP_MODULE",
    "cwd": "RAG_MCP_CWD",
    "src": "RAG_MCP_SRC",
}
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _with_env_overrides(cfg: dict, mapping: dict[str, str]) -> dict:
    """把环境变量覆盖到配置字典上；空值视为未设置，保留 yaml 默认值。"""
    out = dict(cfg)
    for key, env_name in mapping.items():
        raw = os.getenv(env_name)
        if raw in (None, ""):
            continue
        if key == "use_mock":
            out[key] = raw.strip().lower() in _TRUE_VALUES
        else:
            out[key] = raw
    return out


def _deep_merge(base: dict, override: dict) -> dict:
    """浅深合并：override 覆盖 base，缺字段保留 base。"""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Settings(BaseModel):
    """应用设置（pydantic 强类型视图，落到具体字段再取值）。"""

    _raw: dict[str, Any] = PrivateAttr(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        raw: dict[str, Any] = {}
        p = path or _DEFAULT_YAML
        if p.exists():
            with open(p, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        # 允许 config/settings.local.yaml 覆盖
        local = _DEFAULT_YAML.with_name("settings.local.yaml")
        if local.exists():
            with open(local, encoding="utf-8") as f:
                raw = _deep_merge(raw, yaml.safe_load(f) or {})
        obj = cls()
        obj._raw = raw
        return obj

    # ---- 取值辅助 ----
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def env(self, name: str, default: str | None = None) -> str | None:
        v = os.getenv(name)
        return v if v not in (None, "") else default

    # ---- 常用字段快捷访问 ----
    @property
    def llm(self) -> dict:
        return self.get("llm", {})

    @property
    def mall(self) -> dict:
        return _with_env_overrides(self.get("mall", {}), _ENV_MALL)

    @property
    def rag(self) -> dict:
        return _with_env_overrides(self.get("rag", {}), _ENV_RAG)

    @property
    def agent(self) -> dict:
        return self.get("agent", {})

    @property
    def context_cfg(self) -> dict:
        return self.get("context", {})

    @property
    def max_steps(self) -> int:
        return int(os.getenv("MAX_STEPS", self.get("agent.max_steps", 15)))

    @property
    def default_scheme(self) -> str:
        return str(os.getenv("DEFAULT_SCHEME", self.get("agent.default_scheme", "1")))

    @property
    def default_context_policy(self) -> str:
        return str(os.getenv("DEFAULT_CONTEXT_POLICY", self.get("agent.default_context_policy", "sliding")))

    @property
    def cost_rates(self) -> tuple[float, float]:
        """(每 1k 输入人民币, 每 1k 输出人民币)"""
        return (
            float(self.get("metrics.cost_per_1k_input", 0.0015)),
            float(self.get("metrics.cost_per_1k_output", 0.0045)),
        )


settings = Settings.load()
