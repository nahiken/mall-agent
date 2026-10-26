"""会话持久化：AgentState 序列化落盘（data/sessions/{session_id}.json）。

支撑三类场景：
- CLI 交互模式 `--resume <id>`：进程重启后续聊。
- HTTP 服务层 conversations 接口：多轮会话/历史加载/hide。
- 断线恢复：把上轮失败前完整 state 存下，重试时带上下文。

存储：单用户多会话，文件即会话。每个会话文件含 user_id/title/messages/
tool_records/created_at/updated_at/hidden。
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from ..core.settings import settings

# 会话根目录（settings.data_dir/sessions）
_SESSIONS_DIR: Path | None = None
_lock = threading.Lock()


def _dir() -> Path:
    global _SESSIONS_DIR
    if _SESSIONS_DIR is None:
        _SESSIONS_DIR = Path(settings.get("data_dir", "data")) / "sessions"
        _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return _SESSIONS_DIR


def _path(session_id: str) -> Path:
    # 防目录穿越：session_id 只允许 uuid/短 hex
    safe = "".join(c for c in session_id if c.isalnum() or c in "-_")
    return _dir() / f"{safe}.json"


# ----------------------------------------------------------------------
# 序列化：AgentState 含 pydantic 对象（ChatMessage/ToolRecord），需转 dict
# ----------------------------------------------------------------------
def _to_dict(value: Any) -> Any:
    """把 ChatMessage/ToolRecord 等 pydantic 对象转成可 JSON 序列化的 dict。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {k: _to_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_dict(v) for v in value]
    # pydantic BaseModel / 其它 dataclass
    if hasattr(value, "model_dump"):
        return _to_dict(value.model_dump())
    if hasattr(value, "dict"):
        return _to_dict(value.dict())
    return str(value)


def _from_dict(value: Any) -> Any:
    """把存盘的 dict 还原为 dict（服务层/CLI 需要时再转成 ChatMessage 等）。"""
    return value  # 通用容器直接还原为 dict/list；具体类型由调用方按需构造


# ----------------------------------------------------------------------
# 会话记录结构
# ----------------------------------------------------------------------
def _new_session(user_id: str, first_text: str = "") -> dict:
    now = time.time()
    title = (first_text or "").strip()[:30] or "新会话"
    return {
        "session_id": uuid.uuid4().hex[:12],
        "user_id": user_id,
        "title": title,
        "messages": [],          # list[dict]（role/content 等，含 tool 消息）
        "tool_records": [],      # list[dict]
        "state": {},             # 完整 AgentState 快照（断线恢复/续聊用）
        "created_at": now,
        "updated_at": now,
        "hidden": False,
    }


# ----------------------------------------------------------------------
# CRUD
# ----------------------------------------------------------------------
def save_session(session_id: str, data: dict[str, Any]) -> None:
    """保存会话（data 含 user_id/messages/tool_records/state 等）。

    并发安全：读-改-写合并后整体**原子写**（tmp 文件 + os.replace），
    避免写一半进程崩溃/多进程竞争产生损坏 JSON。锁只保护同进程内合并。
    """
    with _lock:
        existing = load_session(session_id)
        merged = {**_new_session(data.get("user_id", "1")), **(existing or {}), **data}
        merged["session_id"] = session_id
        merged["updated_at"] = time.time()
        p = _path(session_id)
        payload = json.dumps(_to_dict(merged), ensure_ascii=False, indent=1)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, p)  # 原子替换


def load_session(session_id: str) -> dict[str, Any] | None:
    """读取会话；不存在返回 None。"""
    p = _path(session_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def create_session(user_id: str, first_text: str = "") -> dict:
    """新建会话并落盘，返回会话记录。"""
    rec = _new_session(user_id, first_text)
    save_session(rec["session_id"], rec)
    return rec


def ensure_conversation(user_id: str, conversation_id: str | None, first_text: str = "") -> dict:
    """续用已存在会话（校验归属）或新建。返回会话记录。"""
    if conversation_id:
        rec = load_session(conversation_id)
        if rec and rec.get("user_id") == user_id:
            return rec
    return create_session(user_id, first_text)


def list_sessions(user_id: str, page: int = 1, page_size: int = 10,
                  include_hidden: bool = False) -> dict[str, Any]:
    """按用户列出会话（updated_at 倒序），分页语义：page=1 为最新一页。"""
    all_recs: list[dict] = []
    for p in _dir().glob("*.json"):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if rec.get("user_id") != user_id:
            continue
        if not include_hidden and rec.get("hidden"):
            continue
        all_recs.append(rec)
    all_recs.sort(key=lambda r: r.get("updated_at", 0), reverse=True)

    total = len(all_recs)
    total_pages = max(1, (total + page_size - 1) // page_size) if total else 1
    # page=1 最新一页，最新在前（前端 loadLatest 取 items[0]）
    items = all_recs[(page - 1) * page_size: page * page_size]
    return {
        "items": [{k: v for k, v in r.items() if k in ("session_id", "title", "message_count", "created_at", "updated_at")}
                  for r in items],
        "total": total, "page": page, "page_size": page_size, "total_pages": total_pages,
    }


def list_messages(session_id: str, user_id: str | None = None,
                  page: int = 1, page_size: int = 40) -> dict[str, Any]:
    """列出会话消息（page=1 最新、页内最旧在前，兼容旧前端协议）。"""
    rec = load_session(session_id)
    if rec is None or (user_id and rec.get("user_id") != user_id):
        return {"items": [], "total": 0, "page": page, "page_size": page_size, "total_pages": 1}
    msgs = rec.get("messages", [])
    total = len(msgs)
    total_pages = max(1, (total + page_size - 1) // page_size) if total else 1
    # page=1 最新一页；页内保持时间序（最旧在前，前端按序 push 还原）
    end = max(0, total - (page - 1) * page_size)
    start = max(0, end - page_size)
    items = msgs[start:end]
    # 兼容旧协议：msg_type/text/products 两条消息（assistant 文本 + products）
    out: list[dict] = []
    for m in items:
        role = m.get("role", "")
        if role == "tool":
            continue  # 工具消息不对外
        if role == "assistant" and m.get("products"):
            out.append({"id": m.get("id", ""), "role": role, "content": m.get("content", ""),
                        "msg_type": "text", "created_at": m.get("ts", time.time())})
            out.append({"id": f"{m.get('id', '')}-p", "role": role,
                        "content": json.dumps(m["products"], ensure_ascii=False),
                        "msg_type": "products", "created_at": m.get("ts", time.time())})
        else:
            out.append({"id": m.get("id", ""), "role": role, "content": m.get("content", ""),
                        "msg_type": "text", "created_at": m.get("ts", time.time())})
    return {"items": out, "total": total, "page": page, "page_size": page_size, "total_pages": total_pages}


def hide_sessions(user_id: str) -> None:
    """隐藏该用户所有会话（保留文件，仅标记 hidden）。"""
    for p in _dir().glob("*.json"):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if rec.get("user_id") == user_id and not rec.get("hidden"):
            rec["hidden"] = True
            rec["updated_at"] = time.time()
            p.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")


def delete_session(session_id: str, user_id: str | None = None) -> bool:
    """删除会话（可选校验归属）。"""
    rec = load_session(session_id)
    if rec is None or (user_id and rec.get("user_id") != user_id):
        return False
    try:
        _path(session_id).unlink()
        return True
    except OSError:
        return False
