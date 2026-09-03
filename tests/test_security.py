"""ASGI-level tests for authentication and rate limiting."""

import asyncio

import pytest

from client_package.support_ticket_env.client import _validate_auth_transport
from server.security import SecurityMiddleware


class OkApp:
    async def __call__(self, scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


def test_client_refuses_api_key_over_remote_plaintext_websocket():
    with pytest.raises(ValueError, match="insecure remote WebSocket"):
        _validate_auth_transport("ws://example.com/ws", "secret", False)


def test_client_allows_secure_or_loopback_websocket_auth():
    _validate_auth_transport("wss://example.com/ws", "secret", False)
    _validate_auth_transport("ws://127.0.0.1:7860/ws", "secret", False)


def test_client_rejects_url_embedded_credentials():
    with pytest.raises(ValueError, match="must not be embedded"):
        _validate_auth_transport("wss://user:password@example.com/ws", "secret", False)


def request(app, path="/reset", headers=(), client="127.0.0.1"):
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": list(headers),
        "client": (client, 1234),
    }
    asyncio.run(app(scope, receive, send))
    return messages[0]["status"], dict(messages[0]["headers"])


def websocket_request(app, headers=()):
    messages = []

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        messages.append(message)

    scope = {
        "type": "websocket",
        "path": "/ws",
        "headers": list(headers),
        "client": ("127.0.0.1", 1234),
    }
    asyncio.run(app(scope, receive, send))
    return messages


def test_api_key_is_enforced(monkeypatch):
    monkeypatch.setenv("SUPPORT_ENV_REQUIRE_API_KEY", "true")
    monkeypatch.setenv("SUPPORT_ENV_API_KEY", "test-secret")
    app = SecurityMiddleware(OkApp())
    assert request(app)[0] == 401
    assert request(app, headers=[(b"x-api-key", b"wrong")])[0] == 401
    assert request(app, headers=[(b"x-api-key", b"test-secret")])[0] == 200


def test_ambiguous_or_duplicate_credentials_are_rejected(monkeypatch):
    monkeypatch.setenv("SUPPORT_ENV_REQUIRE_API_KEY", "true")
    monkeypatch.setenv("SUPPORT_ENV_API_KEY", "test-secret")
    app = SecurityMiddleware(OkApp())
    duplicate = [(b"x-api-key", b"test-secret"), (b"x-api-key", b"test-secret")]
    conflicting = [
        (b"x-api-key", b"test-secret"),
        (b"authorization", b"Bearer test-secret"),
    ]
    assert request(app, headers=duplicate)[0] == 401
    assert request(app, headers=conflicting)[0] == 401


def test_public_health_endpoint_bypasses_auth(monkeypatch):
    monkeypatch.setenv("SUPPORT_ENV_REQUIRE_API_KEY", "true")
    monkeypatch.setenv("SUPPORT_ENV_API_KEY", "test-secret")
    assert request(SecurityMiddleware(OkApp()), path="/health")[0] == 200


def test_websocket_requires_api_key(monkeypatch):
    monkeypatch.setenv("SUPPORT_ENV_REQUIRE_API_KEY", "true")
    monkeypatch.setenv("SUPPORT_ENV_API_KEY", "test-secret")
    messages = websocket_request(SecurityMiddleware(OkApp()))
    assert messages == [{"type": "websocket.close", "code": 4401}]


def test_rate_limit_returns_429(monkeypatch):
    monkeypatch.delenv("SUPPORT_ENV_REQUIRE_API_KEY", raising=False)
    monkeypatch.setenv("SUPPORT_ENV_RATE_LIMIT", "1")
    app = SecurityMiddleware(OkApp())
    assert request(app)[0] == 200
    status, headers = request(app)
    assert status == 429
    assert b"retry-after" in headers


def test_unauthorized_requests_are_rate_limited(monkeypatch):
    monkeypatch.setenv("SUPPORT_ENV_REQUIRE_API_KEY", "true")
    monkeypatch.setenv("SUPPORT_ENV_API_KEY", "test-secret")
    monkeypatch.setenv("SUPPORT_ENV_RATE_LIMIT", "1")
    app = SecurityMiddleware(OkApp())
    assert request(app)[0] == 401
    assert request(app)[0] == 429


def test_rate_limiter_client_storage_is_bounded(monkeypatch):
    monkeypatch.setenv("SUPPORT_ENV_RATE_LIMIT", "1")
    monkeypatch.setenv("SUPPORT_ENV_MAX_RATE_LIMIT_CLIENTS", "100")
    app = SecurityMiddleware(OkApp())
    for index in range(150):
        assert request(app, client=f"192.0.2.{index}")[0] == 200
    assert len(app._requests) == 100


def test_root_cannot_be_configured_as_public(monkeypatch):
    monkeypatch.setenv("SUPPORT_ENV_PUBLIC_PATHS", "/")
    with pytest.raises(RuntimeError, match="never '/' "):
        SecurityMiddleware(OkApp())


def test_production_fails_closed_without_key(monkeypatch):
    monkeypatch.setenv("SUPPORT_ENV_MODE", "production")
    monkeypatch.delenv("SUPPORT_ENV_API_KEY", raising=False)
    try:
        SecurityMiddleware(OkApp())
    except RuntimeError as error:
        assert "SUPPORT_ENV_API_KEY" in str(error)
    else:
        raise AssertionError("production must not start without an API key")


def test_production_fails_closed_without_seed_salt(monkeypatch):
    monkeypatch.setenv("SUPPORT_ENV_MODE", "production")
    monkeypatch.setenv("SUPPORT_ENV_API_KEY", "k" * 32)
    monkeypatch.delenv("SUPPORT_ENV_SEED_SALT", raising=False)
    try:
        SecurityMiddleware(OkApp())
    except RuntimeError as error:
        assert "SUPPORT_ENV_SEED_SALT" in str(error)
    else:
        raise AssertionError("production must not start without a seed salt")


def test_production_rejects_weak_api_key(monkeypatch):
    monkeypatch.setenv("SUPPORT_ENV_MODE", "production")
    monkeypatch.setenv("SUPPORT_ENV_API_KEY", "short")
    monkeypatch.setenv("SUPPORT_ENV_SEED_SALT", "s" * 32)
    with pytest.raises(RuntimeError, match="at least 32 characters"):
        SecurityMiddleware(OkApp())


def test_production_only_exposes_health_by_default(monkeypatch):
    monkeypatch.setenv("SUPPORT_ENV_MODE", "production")
    monkeypatch.setenv("SUPPORT_ENV_API_KEY", "k" * 32)
    monkeypatch.setenv("SUPPORT_ENV_SEED_SALT", "s" * 32)
    app = SecurityMiddleware(OkApp())
    assert request(app, path="/health")[0] == 200
    assert request(app, path="/docs")[0] == 401
    assert request(app, path="/playground")[0] == 401
