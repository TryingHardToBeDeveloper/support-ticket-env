"""
Typed models for the Customer Support Ticket Resolution Environment.
Works with pydantic (production) or stdlib (offline/testing).
"""

from __future__ import annotations

from typing import Any, Literal

try:
    from pydantic import BaseModel, ConfigDict, Field, model_validator

    _USE_PYDANTIC = True
except ImportError:
    _USE_PYDANTIC = False

# ── import base classes from openenv (or stub) ──────────────────
from openenv.core.env_server.types import Action, Observation, State

# ═══════════════════════════════════════════════════════════════
# Action
# ═══════════════════════════════════════════════════════════════

if _USE_PYDANTIC:

    class SupportAction(BaseModel):  # type: ignore[misc]
        model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
        metadata: dict[str, Any] = Field(default_factory=dict)
        action_type: Literal["classify", "reply", "escalate", "close"]
        category: Literal["billing", "technical", "account", "general", "refund"] | None = None
        reply_text: str | None = None
        reason: str | None = None

        @model_validator(mode="after")
        def validate_action_payload(self):
            if self.action_type == "classify" and self.category is None:
                raise ValueError("category is required for a classify action")
            if self.action_type != "classify" and self.category is not None:
                raise ValueError("category is only valid for a classify action")
            if self.action_type == "reply" and not (self.reply_text or "").strip():
                raise ValueError("reply_text is required for a reply action")
            if self.action_type != "reply" and self.reply_text is not None:
                raise ValueError("reply_text is only valid for a reply action")
            if self.reply_text is not None and len(self.reply_text) > 2_000:
                raise ValueError("reply_text must not exceed 2000 characters")
            if self.reason is not None and len(self.reason) > 500:
                raise ValueError("reason must not exceed 500 characters")
            return self

        def model_dump(self, **kw):
            return super().model_dump(**kw)
else:
    _VALID_ACTION_TYPES = {"classify", "reply", "escalate", "close"}
    _VALID_CATEGORIES = {"billing", "technical", "account", "general", "refund"}

    class SupportAction(Action):  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            action_type = kwargs.get("action_type")
            if action_type not in _VALID_ACTION_TYPES:
                raise ValueError(f"Invalid action_type: {action_type!r}")
            category = kwargs.get("category")
            if category is not None and category not in _VALID_CATEGORIES:
                raise ValueError(f"Invalid category: {category!r}")
            reply_text = kwargs.get("reply_text")
            reason = kwargs.get("reason")
            if action_type == "classify" and category is None:
                raise ValueError("category is required for a classify action")
            if action_type != "classify" and category is not None:
                raise ValueError("category is only valid for a classify action")
            if action_type == "reply" and not (reply_text or "").strip():
                raise ValueError("reply_text is required for a reply action")
            if action_type != "reply" and reply_text is not None:
                raise ValueError("reply_text is only valid for a reply action")
            if reply_text is not None and len(reply_text) > 2_000:
                raise ValueError("reply_text must not exceed 2000 characters")
            if reason is not None and len(reason) > 500:
                raise ValueError("reason must not exceed 500 characters")
            self.action_type = action_type
            self.category = category
            self.reply_text = reply_text
            self.reason = reason
            self.metadata = kwargs.get("metadata", {})


# ═══════════════════════════════════════════════════════════════
# Observation
# ═══════════════════════════════════════════════════════════════

if _USE_PYDANTIC:

    class SupportObservation(BaseModel):  # type: ignore[misc]
        model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
        done: bool = False
        reward: float | None = None
        metadata: dict[str, Any] = Field(default_factory=dict)
        ticket_id: str = ""
        ticket_text: str = ""
        task_id: int = 1
        current_category: str | None = None
        resolved: bool = False
        step_count: int = 0
        feedback: str = ""
        score: float = 0.0
else:

    class SupportObservation(Observation):  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            self.done = kwargs.pop("done", False)
            self.reward = kwargs.pop("reward", None)
            self.metadata = kwargs.pop("metadata", {})
            self.ticket_id = kwargs.pop("ticket_id", "")
            self.ticket_text = kwargs.pop("ticket_text", "")
            self.task_id = kwargs.pop("task_id", 1)
            self.current_category = kwargs.pop("current_category", None)
            self.resolved = kwargs.pop("resolved", False)
            self.step_count = kwargs.pop("step_count", 0)
            self.feedback = kwargs.pop("feedback", "")
            self.score = kwargs.pop("score", 0.0)


# ═══════════════════════════════════════════════════════════════
# State
# ═══════════════════════════════════════════════════════════════

if _USE_PYDANTIC:

    class SupportState(BaseModel):  # type: ignore[misc]
        model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
        episode_id: str | None = None
        step_count: int = 0
        task_id: int = 1
        ticket_id: str = ""
        classified: bool = False
        resolved: bool = False
        total_reward: float = 0.0
        tickets_resolved: int = 0
        tickets_total: int = 1
else:

    class SupportState(State):  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            self.episode_id = kwargs.pop("episode_id", None)
            self.step_count = kwargs.pop("step_count", 0)
            self.task_id = kwargs.pop("task_id", 1)
            self.ticket_id = kwargs.pop("ticket_id", "")
            self.classified = kwargs.pop("classified", False)
            self.resolved = kwargs.pop("resolved", False)
            self.total_reward = kwargs.pop("total_reward", 0.0)
            self.tickets_resolved = kwargs.pop("tickets_resolved", 0)
            self.tickets_total = kwargs.pop("tickets_total", 1)
