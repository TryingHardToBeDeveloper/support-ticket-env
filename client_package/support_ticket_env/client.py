"""Client for the Customer Support Ticket Resolution Environment."""

import os

from openenv.core.env_client import EnvClient
from websockets.asyncio.client import ClientConnection
from websockets.asyncio.client import connect as ws_connect

from support_ticket_env.models import SupportAction, SupportObservation, SupportState


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

    def __init__(self, *args, api_key: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._ws: ClientConnection | None = getattr(self, "_ws", None)
        self._api_key = api_key or os.getenv("SUPPORT_ENV_API_KEY")

    async def connect(self) -> "SupportTicketEnv":
        """Connect with an optional API key without placing secrets in the URL."""
        if self._ws is not None:
            return self
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
