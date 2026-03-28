"""
Client for the Customer Support Ticket Resolution Environment.
"""

from openenv.core.env_client import EnvClient
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

    def _parse_action(self, action: SupportAction) -> dict:
        return action.model_dump()

    def _parse_result(self, data: dict) -> SupportObservation:
        obs_data = data.get("observation", data)
        return SupportObservation(**obs_data)

    def _parse_state(self, data: dict) -> SupportState:
        return SupportState(**data)
