"""Source-checkout compatibility wrapper for the client package."""

try:
    from .client_package.support_ticket_env import (
        SupportAction,
        SupportObservation,
        SupportState,
        SupportTicketEnv,
    )
except (ImportError, ModuleNotFoundError):
    from client_package.support_ticket_env import (
        SupportAction,
        SupportObservation,
        SupportState,
        SupportTicketEnv,
    )

__all__ = ["SupportAction", "SupportObservation", "SupportState", "SupportTicketEnv"]
