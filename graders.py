"""
Graders for all three tasks.

Each grader returns a float in [0.0, 1.0].

Task 1 – Classification (easy)
    - 1.0  : correct category
    - 0.0  : wrong category

Task 2 – Action Selection (medium)
    - 1.0  : correct action
    - 0.5  : partially correct (e.g., escalate vs reply both defensible)
    - 0.0  : clearly wrong (e.g., close an unsolved ticket)

Task 3 – Full Resolution (hard)
    Combines classification + action + reply quality into a single score.
    Rewards partial progress so the agent gets signal throughout the trajectory.
"""

from __future__ import annotations
from typing import Dict, Any


# ─────────────────────────── helpers ───────────────────────────

# Pairs of actions that are considered "close enough" for partial credit
_PARTIAL_CREDIT_PAIRS = {
    frozenset({"reply", "escalate"}),  # borderline tickets
}

_KEYWORD_REWARDS: Dict[str, list[str]] = {
    "billing":   ["refund", "charge", "invoice", "payment", "billing"],
    "account":   ["password", "login", "account", "cancel", "subscription"],
    "technical": ["engineering", "escalate", "bug", "crash", "error", "fix"],
    "refund":    ["refund", "return", "credit", "process"],
    "general":   ["hours", "contact", "phone", "information", "help"],
}


def _reply_quality(reply_text: str, category: str) -> float:
    """Return 0.0–0.5 based on how relevant the reply text is."""
    if not reply_text:
        return 0.0
    text_lower = reply_text.lower()
    keywords = _KEYWORD_REWARDS.get(category, [])
    hits = sum(1 for kw in keywords if kw in text_lower)
    # cap at 0.5 (the other 0.5 comes from action correctness)
    return min(0.5, hits * 0.1)


# ─────────────────────────── Task 1 ────────────────────────────

def grade_task1(
    predicted_category: str,
    correct_category: str,
) -> float:
    """Binary classification reward."""
    return 1.0 if predicted_category == correct_category else 0.0


# ─────────────────────────── Task 2 ────────────────────────────

def grade_task2(
    action_type: str,
    correct_action: str,
    category: str | None = None,
) -> float:
    """
    Action-selection reward.
    Full credit for exact match, partial credit for defensible alternatives.
    Penalises closing an unresolved ticket.
    """
    if action_type == correct_action:
        return 1.0

    # Partial credit for ambiguous cases
    pair = frozenset({action_type, correct_action})
    if pair in _PARTIAL_CREDIT_PAIRS:
        return 0.5

    # Closing an unresolved ticket is always wrong
    if action_type == "close":
        return 0.0

    return 0.0


# ─────────────────────────── Task 3 ────────────────────────────

def grade_task3(
    classified_correctly: bool,
    action_correct: bool,
    action_partial: bool,
    reply_text: str | None,
    category: str,
    resolved: bool,
    steps_taken: int,
    max_steps: int = 5,
) -> float:
    """
    Multi-step resolution reward with partial progress.

    Breakdown:
      0.20  – classification correct
      0.40  – action correct  (0.20 if partial)
      0.25  – reply quality   (NLP keyword overlap)
      0.15  – efficiency bonus (fewer steps → higher bonus)
    """
    score = 0.0

    if classified_correctly:
        score += 0.20

    if action_correct:
        score += 0.40
    elif action_partial:
        score += 0.20

    if reply_text:
        score += _reply_quality(reply_text, category) * 0.5   # scaled to 0.25 max

    # Efficiency: full 0.15 for 1 step, 0 for max_steps steps
    if resolved and steps_taken <= max_steps:
        efficiency = max(0.0, (max_steps - steps_taken) / (max_steps - 1))
        score += 0.15 * efficiency

    return round(min(1.0, score), 4)


# ─────────────────────────── Penalty ───────────────────────────

def loop_penalty(step_count: int, max_steps: int = 10) -> float:
    """Return a negative reward if agent is stuck in a loop."""
    if step_count > max_steps:
        return -0.05 * (step_count - max_steps)
    return 0.0
