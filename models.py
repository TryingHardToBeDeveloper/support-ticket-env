"""
Typed models for the Customer Support Ticket Resolution Environment.
Works with pydantic (production) or stdlib (offline/testing).
"""
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional

try:
    from pydantic import BaseModel, ConfigDict
    _USE_PYDANTIC = True
except ImportError:
    _USE_PYDANTIC = False

# ── import base classes from openenv (or stub) ──────────────────
from openenv.core.env_server.types import Action, Observation, State


# ═══════════════════════════════════════════════════════════════
# Action
# ═══════════════════════════════════════════════════════════════

if _USE_PYDANTIC:
    class SupportAction(Action, BaseModel):  # type: ignore[misc]
        model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
        metadata: Dict[str, Any] = {}
        action_type: Literal["classify", "reply", "escalate", "close"]
        category: Optional[
            Literal["billing", "technical", "account", "general", "refund"]
        ] = None
        reply_text: Optional[str] = None
        reason: Optional[str] = None

        def model_dump(self, **kw):
            return super().model_dump(**kw)
else:
    _VALID_ACTION_TYPES = {"classify", "reply", "escalate", "close"}
    _VALID_CATEGORIES   = {"billing", "technical", "account", "general", "refund"}

    class SupportAction(Action):  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            action_type = kwargs.get("action_type")
            if action_type not in _VALID_ACTION_TYPES:
                raise ValueError(f"Invalid action_type: {action_type!r}")
            category = kwargs.get("category")
            if category is not None and category not in _VALID_CATEGORIES:
                raise ValueError(f"Invalid category: {category!r}")
            self.action_type = action_type
            self.category    = category
            self.reply_text  = kwargs.get("reply_text")
            self.reason      = kwargs.get("reason")
            self.metadata    = kwargs.get("metadata", {})


# ═══════════════════════════════════════════════════════════════
# Observation
# ═══════════════════════════════════════════════════════════════

if _USE_PYDANTIC:
    class SupportObservation(Observation, BaseModel):  # type: ignore[misc]
        model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
        done: bool = False
        reward: Optional[float] = None
        metadata: Dict[str, Any] = {}
        ticket_id: str = ""
        ticket_text: str = ""
        task_id: int = 1
        current_category: Optional[str] = None
        resolved: bool = False
        step_count: int = 0
        feedback: str = ""
        score: float = 0.0
else:
    class SupportObservation(Observation):  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            self.done             = kwargs.pop("done", False)
            self.reward           = kwargs.pop("reward", None)
            self.metadata         = kwargs.pop("metadata", {})
            self.ticket_id        = kwargs.pop("ticket_id", "")
            self.ticket_text      = kwargs.pop("ticket_text", "")
            self.task_id          = kwargs.pop("task_id", 1)
            self.current_category = kwargs.pop("current_category", None)
            self.resolved         = kwargs.pop("resolved", False)
            self.step_count       = kwargs.pop("step_count", 0)
            self.feedback         = kwargs.pop("feedback", "")
            self.score            = kwargs.pop("score", 0.0)


# ═══════════════════════════════════════════════════════════════
# State
# ═══════════════════════════════════════════════════════════════

if _USE_PYDANTIC:
    class SupportState(State, BaseModel):  # type: ignore[misc]
        model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
        episode_id: Optional[str] = None
        step_count: int = 0
        task_id: int = 1
        ticket_id: str = ""
        correct_category: str = ""
        correct_action: str = ""
        classified: bool = False
        resolved: bool = False
        total_reward: float = 0.0
        tickets_resolved: int = 0
        tickets_total: int = 1
else:
    class SupportState(State):  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            self.episode_id       = kwargs.pop("episode_id", None)
            self.step_count       = kwargs.pop("step_count", 0)
            self.task_id          = kwargs.pop("task_id", 1)
            self.ticket_id        = kwargs.pop("ticket_id", "")
            self.correct_category = kwargs.pop("correct_category", "")
            self.correct_action   = kwargs.pop("correct_action", "")
            self.classified       = kwargs.pop("classified", False)
            self.resolved         = kwargs.pop("resolved", False)
            self.total_reward     = kwargs.pop("total_reward", 0.0)
            self.tickets_resolved = kwargs.pop("tickets_resolved", 0)
            self.tickets_total    = kwargs.pop("tickets_total", 1)
