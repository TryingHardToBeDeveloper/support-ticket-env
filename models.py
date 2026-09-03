"""Source-checkout compatibility wrapper."""

try:
    from .client_package.support_ticket_env.models import (
        SupportAction,
        SupportObservation,
        SupportState,
    )
except (ImportError, ModuleNotFoundError):
    from client_package.support_ticket_env.models import (
        SupportAction,
        SupportObservation,
        SupportState,
    )

__all__ = ["SupportAction", "SupportObservation", "SupportState"]
