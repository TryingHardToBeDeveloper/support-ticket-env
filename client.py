"""Source-checkout compatibility wrapper."""

try:
    from .client_package.support_ticket_env.client import SupportTicketEnv
except (ImportError, ModuleNotFoundError):
    from client_package.support_ticket_env.client import SupportTicketEnv

__all__ = ["SupportTicketEnv"]
