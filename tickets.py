"""Source-checkout compatibility wrapper for the public taxonomy."""

try:
    from .client_package.support_ticket_env.tickets import ACTIONS, CATEGORIES, ActionType, Category
except (ImportError, ModuleNotFoundError):
    from client_package.support_ticket_env.tickets import ACTIONS, CATEGORIES, ActionType, Category

__all__ = ["ACTIONS", "CATEGORIES", "ActionType", "Category"]
