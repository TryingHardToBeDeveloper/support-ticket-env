"""Client for the Customer Support Ticket Resolution Environment."""

import os
from urllib.parse import urlsplit

from openenv.core.env_client import EnvClient
from websockets.asyncio.client import ClientConnection
from websockets.asyncio.client import connect as ws_connect

from support_ticket_env.models import SupportAction, SupportObservation, SupportState


def _validate_auth_transport(url: str, api_key: str | None, allow_insecure_auth: bool) -> None:
    parsed = urlsplit(url)
    if parsed.username or parsed.password:
        raise ValueError("Credentials must not be embedded in the WebSocket URL")
    loopback = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if api_key and parsed.scheme != "wss" and not loopback and not allow_insecure_auth:
        raise ValueError(
            "Refusing to send an API key over an insecure remote WebSocket; "
            "use wss:// or explicitly set allow_insecure_auth=True"
        )


class SupportTicketEnv(EnvClient[SupportAction, SupportObservation, SupportState]):
    """
    OpenEnv client for the Support Ticket Resolution environment.

    Usage (async):
        async with SupportTicketEnv(base_url="http://localhost:8000") as env:
            result = await env.reset(task_id=1)
            result = await env.step(SupportAction(action_type="classify", category="billing"))

    Usage (sync):
        with SupportTicketEnv(base_url="http://localhost:8000").sync() as env:
            result = env.reset(task_id=2)
            result = env.step(SupportAction(action_type="classify", category="technical"))
            result = env.step(SupportAction(action_type="escalate"))
    """

    def __init__(
        self,
        *args,
        api_key: str | None = None,
        allow_insecure_auth: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._ws: ClientConnection | None = getattr(self, "_ws", None)
        self._api_key = api_key or os.getenv("SUPPORT_ENV_API_KEY")
        self._allow_insecure_auth = allow_insecure_auth

    def _validate_auth_transport(self) -> None:
        _validate_auth_transport(self._ws_url, self._api_key, self._allow_insecure_auth)

    async def connect(self) -> "SupportTicketEnv":
        """Connect with an optional API key without placing secrets in the URL."""
        if self._ws is not None:
            return self
        self._validate_auth_transport()
        headers = None
        if self._api_key:
            headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            self._ws = await ws_connect(
                self._ws_url,
                open_timeout=self._connect_timeout,
                max_size=self._max_message_size,
                additional_headers=headers,
            )
        except Exception as error:
            raise ConnectionError(f"Failed to connect to {self._ws_url}") from error
        return self

    def _parse_action(self, action: SupportAction) -> dict:
        return action.model_dump()

    def _parse_result(self, data: dict) -> SupportObservation:
        obs_data = data.get("observation", data)
        return SupportObservation(**obs_data)

    def _parse_state(self, data: dict) -> SupportState:
        return SupportState(**data)
