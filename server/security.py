"""Small, dependency-free ASGI security layer for the environment API."""

from __future__ import annotations

import hmac
import json
import os
import threading
import time
from collections import defaultdict, deque


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
        if production and not os.getenv("SUPPORT_ENV_SEED_SALT"):
            raise RuntimeError("SUPPORT_ENV_SEED_SALT is required in production mode")
        self.limit = max(1, int(os.getenv("SUPPORT_ENV_RATE_LIMIT", "120")))
        self.window_seconds = 60.0
        self.exempt_prefixes = tuple(
            part.strip()
            for part in os.getenv(
                "SUPPORT_ENV_PUBLIC_PATHS", "/health,/docs,/openapi.json,/playground"
            ).split(",")
            if part.strip()
        )
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _is_exempt(self, path: str) -> bool:
        return any(
            path == prefix or path.startswith(prefix + "/") for prefix in self.exempt_prefixes
        )

    def _authorized(self, headers: dict[bytes, bytes]) -> bool:
        if not self.require_key:
            return True
        supplied = headers.get(b"x-api-key", b"").decode("utf-8", "ignore")
        authorization = headers.get(b"authorization", b"").decode("utf-8", "ignore")
        if authorization.lower().startswith("bearer "):
            supplied = authorization[7:].strip()
        return bool(supplied) and hmac.compare_digest(supplied, self.api_key)

    def _allowed(self, client: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            timestamps = self._requests[client]
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
        headers = dict(scope.get("headers", []))
        if not self._authorized(headers):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 4401})
                return
            await self._response(send, 401, "Missing or invalid API key")
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
                    ]
                )
            await send(message)

        await self.app(scope, receive, send_with_security_headers)
