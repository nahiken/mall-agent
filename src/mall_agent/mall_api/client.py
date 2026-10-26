"""mall HTTP API 客户端。

封装对 mall-portal(:8085) 的调用：
- 登录拿 JWT，自动注入 Authorization 头。
- 统一返回结构（mall 返回 {code, message, data}，code=200 成功）。
- 提供 Mock 模式开关：use_mock=True 时走 mock 后端（读 mall.sql 只读内存态）。
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from ..core.settings import settings

logger = logging.getLogger(__name__)


class MallApiError(Exception):
    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code  # HTTP status；401 时触发重新登录重发


class MockMallError(Exception):
    pass


class MallClient:
    """mall 前台 API 客户端（真实 HTTP 或 Mock）。"""

    def __init__(self, base_url: str | None = None, use_mock: bool | None = None) -> None:
        self.base_url = base_url or settings.mall.get("api_base", "http://127.0.0.1:8085")
        self.use_mock = settings.mall.get("use_mock", True) if use_mock is None else use_mock
        self.test_user = settings.mall.get("test_user", "agent_test_user")
        self.test_password = settings.mall.get("test_password") or ""
        self._token: str | None = None
        self._user_id: int | None = None
        # Mock 后端懒加载
        self._mock: Any | None = None

    # ------------------------------------------------------------------
    # 会话
    # ------------------------------------------------------------------
    async def ensure_session(self, force: bool = False) -> None:
        """确保已登录：先看有没有 token，没有则登录测试账号。

        force=True 强制重新登录（token 失效刷新用）。
        """
        if self._token and not force:
            return
        if self.use_mock:
            await self._get_mock()
            self._user_id = self._mock.ensure_user(self.test_user, self.test_password)
            self._token = "mock-token"
            # 让 mock 的会话用户与 client 一致（_current_uid 依赖 test_username）
            self._mock.test_username = self.test_user
            return
        await self.login(self.test_user, self.test_password)

    def ensure_mock_sync(self) -> Any:
        """同步确保 mock 就绪（无事件循环时用），返回 mock 后端。"""
        if not self.use_mock:
            return None
        if self._mock is None:
            from .mock_backend import MockMallBackend
            self._mock = MockMallBackend(settings.mall.get("sql_path"))
            self._mock.load_sync()
            self._user_id = self._mock.ensure_user(self.test_user, self.test_password)
            self._token = "mock-token"
            # 让 mock 的会话用户与 client 一致（_current_uid 依赖 test_username）
            self._mock.test_username = self.test_user
        return self._mock

    async def login(self, username: str, password: str) -> dict[str, Any]:
        r = await self._post("/sso/login", json={"username": username, "password": password})
        data = r.get("data") or {}
        self._token = data.get("token")
        self._user_id = data.get("id")
        return data

    # ------------------------------------------------------------------
    # 请求原语
    # ------------------------------------------------------------------
    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.use_mock:
            return self._mock.get(path, params or {})
        return await self._request("get", path, params=params)

    async def _post(self, path: str, json: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.use_mock:
            return self._mock.post(path, json or {})
        return await self._request("post", path, json=json)

    async def _request(self, method: str, path: str, params: dict[str, Any] | None = None,
                       json: dict[str, Any] | None = None) -> dict[str, Any]:
        """公共请求：复用共享连接池；401（token 失效）时重新登录重发一次。"""
        client = _shared_client()

        async def _do() -> dict[str, Any]:
            headers = self._headers()
            if method == "get":
                resp = await client.get(f"{self.base_url}{path}", params=params, headers=headers)
            else:
                resp = await client.post(f"{self.base_url}{path}", json=json, headers=headers)
            return self._check(resp)

        try:
            return await _do()
        except MallApiError as e:
            # 仅 401 走重新登录重发；其余错误直接抛
            if e.code != 401:
                raise
            logger.warning("mall token 失效（401），重新登录后重发 %s %s", method.upper(), path)
            self._token = None
            await self.ensure_session(force=True)
            return await _do()

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    @staticmethod
    def _check(resp: httpx.Response) -> dict[str, Any]:
        try:
            body = resp.json()
        except Exception:
            raise MallApiError(f"非 JSON 响应: {resp.status_code}")
        if resp.status_code != 200 or body.get("code") != 200:
            raise MallApiError(f"mall API 错误 [{resp.status_code}]: {body}", code=resp.status_code)
        return body

    async def _get_mock(self):
        if self._mock is None:
            from .mock_backend import MockMallBackend
            self._mock = MockMallBackend(settings.mall.get("sql_path"))
            await self._mock.load()
        return self._mock


# 全局单例（懒初始化，按需 ensure_session）
_client: MallClient | None = None

# 共享 HTTP 连接池（避免每请求新建 AsyncClient/TLS 握手）
_shared_http: "httpx.AsyncClient | None" = None


def _shared_client() -> "httpx.AsyncClient":
    """返回进程级共享的 httpx.AsyncClient（连接池 + keep-alive）。"""
    global _shared_http
    if _shared_http is None or _shared_http.is_closed:
        _shared_http = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        )
    return _shared_http


def get_mall_client() -> MallClient:
    global _client
    if _client is None:
        _client = MallClient()
    return _client
