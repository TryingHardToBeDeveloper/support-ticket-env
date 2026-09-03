"""Small, dependency-free ASGI security layer for the environment API."""

from __future__ import annotations

import hmac
import json
import os
import threading
import time
from collections import OrderedDict, deque


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


class SecurityMiddleware:
    """Enforce API-key authentication, per-client limits, and safe headers."""

    def __init__(self, app):
        self.app = app
        self.api_key = os.getenv("SUPPORT_ENV_API_KEY", "")
        production = os.getenv("SUPPORT_ENV_MODE", "development").lower() == "production"
        self.require_key = _truthy(os.getenv("SUPPORT_ENV_REQUIRE_API_KEY")) or production
        if self.require_key and not self.api_key:
            raise RuntimeError(
                "SUPPORT_ENV_API_KEY is required when API-key enforcement "
                "or production mode is enabled"
            )
        seed_salt = os.getenv("SUPPORT_ENV_SEED_SALT", "")
        if production and len(self.api_key) < 32:
            raise RuntimeError("SUPPORT_ENV_API_KEY must be at least 32 characters in production")
        if production and len(seed_salt) < 32:
            raise RuntimeError("SUPPORT_ENV_SEED_SALT is required in production mode")
        self.limit = max(1, int(os.getenv("SUPPORT_ENV_RATE_LIMIT", "120")))
        self.window_seconds = 60.0
        self.max_clients = max(100, int(os.getenv("SUPPORT_ENV_MAX_RATE_LIMIT_CLIENTS", "10000")))
        default_public_paths = (
            "/health" if production else "/health,/docs,/openapi.json,/playground"
        )
        self.exempt_prefixes = tuple(
            part.strip()
            for part in os.getenv("SUPPORT_ENV_PUBLIC_PATHS", default_public_paths).split(",")
            if part.strip()
        )
        if any(not prefix.startswith("/") or prefix == "/" for prefix in self.exempt_prefixes):
            raise RuntimeError("Public paths must be specific absolute paths, never '/' ")
        self._requests: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def _is_exempt(self, path: str) -> bool:
        return any(
            path == prefix or path.startswith(prefix + "/") for prefix in self.exempt_prefixes
        )

    def _authorized(self, raw_headers: list[tuple[bytes, bytes]]) -> bool:
        if not self.require_key:
            return True
        credentials: list[str] = []
        for name, value in raw_headers:
            name = name.lower()
            if name == b"x-api-key":
                credentials.append(value.decode("utf-8", "ignore").strip())
            elif name == b"authorization":
                authorization = value.decode("utf-8", "ignore")
                if not authorization.lower().startswith("bearer "):
                    return False
                credentials.append(authorization[7:].strip())
        if len(credentials) != 1:
            return False
        supplied = credentials[0]
        return bool(supplied) and hmac.compare_digest(supplied, self.api_key)

    def _allowed(self, client: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            timestamps = self._requests.get(client)
            if timestamps is None:
                while len(self._requests) >= self.max_clients:
                    self._requests.popitem(last=False)
                timestamps = deque()
                self._requests[client] = timestamps
            else:
                self._requests.move_to_end(client)
            while timestamps and timestamps[0] < cutoff:
                timestamps.popleft()
            if len(timestamps) >= self.limit:
                return False
            timestamps.append(now)
            return True

    async def _response(self, send, status: int, detail: str, extra_headers=()):
        body = json.dumps({"detail": detail}).encode("utf-8")
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
            (b"x-content-type-options", b"nosniff"),
            (b"cache-control", b"no-store"),
            (b"referrer-policy", b"no-referrer"),
            *extra_headers,
        ]
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "/")
        if self._is_exempt(path):
            await self.app(scope, receive, send)
            return
        client = (scope.get("client") or ("unknown", 0))[0]
        if not self._allowed(client):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 4429})
                return
            await self._response(
                send,
                429,
                "Rate limit exceeded",
                ((b"retry-after", str(int(self.window_seconds)).encode("ascii")),),
            )
            return
        if not self._authorized(list(scope.get("headers", []))):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 4401})
                return
            await self._response(send, 401, "Missing or invalid API key")
            return

        if scope["type"] == "websocket":
            await self.app(scope, receive, send)
            return

        async def send_with_security_headers(message):
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).extend(
                    [
                        (b"x-content-type-options", b"nosniff"),
                        (b"x-frame-options", b"DENY"),
                        (b"referrer-policy", b"no-referrer"),
                        (b"cache-control", b"no-store"),
                        (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
                    ]
                )
            await send(message)

        await self.app(scope, receive, send_with_security_headers)
