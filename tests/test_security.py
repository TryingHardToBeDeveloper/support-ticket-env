"""ASGI-level tests for authentication and rate limiting."""

import asyncio

from server.security import SecurityMiddleware


class OkApp:
    async def __call__(self, scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


def request(app, path="/reset", headers=()):
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
        "client": ("127.0.0.1", 1234),
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
    monkeypatch.setenv("SUPPORT_ENV_API_KEY", "test-secret")
    monkeypatch.delenv("SUPPORT_ENV_SEED_SALT", raising=False)
    try:
        SecurityMiddleware(OkApp())
    except RuntimeError as error:
        assert "SUPPORT_ENV_SEED_SALT" in str(error)
    else:
        raise AssertionError("production must not start without a seed salt")
