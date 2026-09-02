"""Regression tests for previously demonstrated reward and data leaks."""

import pytest

from server.support_environment import SupportTicketEnvironment
from server.ticket_bank import TICKET_LOOKUP
from support_ticket_env.models import SupportAction


def test_public_ticket_module_contains_no_answers():
    from support_ticket_env import tickets

    assert not hasattr(tickets, "TICKETS")
    assert not hasattr(tickets, "TICKET_LOOKUP")


def test_state_never_serializes_oracle_labels():
    env = SupportTicketEnvironment()
    env.reset(task_id=3, seed=42)
    payload = env.state.model_dump() if hasattr(env.state, "model_dump") else vars(env.state)
    assert "correct_category" not in payload
    assert "correct_action" not in payload
    assert "resolution_hint" not in payload


def test_observation_echoes_agent_guess_not_answer():
    env = SupportTicketEnvironment()
    env.reset(task_id=3, seed=42)
    truth = TICKET_LOOKUP[env.state.ticket_id]["category"]
    guess = next(
        category
        for category in ("billing", "technical", "account", "general", "refund")
        if category != truth
    )
    observation = env.step(SupportAction(action_type="classify", category=guess))
    assert observation.current_category == guess
    assert observation.current_category != truth


def test_actions_reject_missing_or_irrelevant_fields():
    with pytest.raises(ValueError):
        SupportAction(action_type="classify")
    with pytest.raises(ValueError):
        SupportAction(action_type="reply")
    with pytest.raises(ValueError):
        SupportAction(action_type="close", category="billing")


def test_completed_episode_rejects_additional_steps():
    env = SupportTicketEnvironment()
    env.reset(task_id=1, seed=42)
    truth = TICKET_LOOKUP[env.state.ticket_id]["category"]
    env.step(SupportAction(action_type="classify", category=truth))
    with pytest.raises(RuntimeError, match="complete"):
        env.step(SupportAction(action_type="classify", category=truth))
