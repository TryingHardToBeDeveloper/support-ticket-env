"""Public ticket schema and taxonomy.

Ground-truth labels and grading rubrics intentionally live in
``support_ticket_env.server.ticket_bank`` and are excluded from the client
wheel. This module is safe to import from agent code.
"""

from __future__ import annotations

from typing import Final, Literal

Category = Literal["billing", "technical", "account", "general", "refund"]
ActionType = Literal["classify", "reply", "escalate", "close"]

CATEGORIES: Final[tuple[str, ...]] = (
    "billing",
    "technical",
    "account",
    "general",
    "refund",
)
ACTIONS: Final[tuple[str, ...]] = ("classify", "reply", "escalate", "close")
