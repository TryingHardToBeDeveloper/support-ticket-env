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

import re

# ─────────────────────────── helpers ───────────────────────────

# Pairs of actions that are considered "close enough" for partial credit.
_PARTIAL_CREDIT_PAIRS = {
    frozenset({"reply", "escalate"}),  # borderline tickets
}

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "to",
    "was",
    "we",
    "will",
    "with",
    "you",
    "your",
}
_CUSTOMER_TONE = {"we", "you", "your", "please", "sorry", "thank", "can"}


def _reply_quality(
    reply_text: str,
    category: str,
    resolution_hint: str = "",
    ticket_text: str = "",
) -> float:
    """Score a customer-facing reply while rejecting keyword stuffing.

    The per-ticket rubric is server-only. Credit requires adequate length,
    lexical diversity, ticket/rubric specificity, and customer-facing language.
    Exact copies of the private rubric are explicitly rejected.
    """
    if not reply_text:
        return 0.0

    tokens = re.findall(r"[a-z0-9]+", reply_text.lower())
    if not 8 <= len(tokens) <= 120:
        return 0.0
    if len(set(tokens)) / len(tokens) < 0.55:
        return 0.0
    if not set(tokens) & _CUSTOMER_TONE:
        return 0.0

    reply_terms = {word for word in tokens if len(word) > 3 and word not in _STOP_WORDS}
    rubric_terms = {
        word
        for word in re.findall(r"[a-z0-9]+", resolution_hint.lower())
        if len(word) > 3 and word not in _STOP_WORDS
    }
    ticket_terms = {
        word
        for word in re.findall(r"[a-z0-9]+", ticket_text.lower())
        if len(word) > 3 and word not in _STOP_WORDS
    }
    if resolution_hint and " ".join(tokens) == " ".join(
        re.findall(r"[a-z0-9]+", resolution_hint.lower())
    ):
        return 0.0

    rubric_coverage = len(reply_terms & rubric_terms) / max(1, min(6, len(rubric_terms)))
    ticket_coverage = len(reply_terms & ticket_terms) / max(1, min(3, len(ticket_terms)))
    score = 0.16 * min(1.0, rubric_coverage)
    score += 0.05 * min(1.0, ticket_coverage)
    score += 0.04
    return round(min(0.25, score), 4)


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
    resolution_hint: str = "",
    ticket_text: str = "",
    requires_reply: bool = True,
) -> float:
    """
    Multi-step resolution reward with partial progress.

    Breakdown:
      0.20  – classification correct
      0.45  – action correct  (0.20 if partial)
      0.25  – response quality (private rubric and anti-stuffing checks)
      0.10  – efficiency bonus (fewer steps → higher bonus)
    """
    score = 0.0

    if classified_correctly:
        score += 0.20

    if action_correct:
        score += 0.45
    elif action_partial:
        score += 0.20

    if requires_reply and reply_text:
        score += _reply_quality(reply_text, category, resolution_hint, ticket_text)
    elif action_correct and not requires_reply:
        score += 0.25

    # Efficiency: full 0.10 for 1 step, 0 for max_steps steps.
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if resolved and steps_taken <= max_steps:
        efficiency = (
            0.10 if max_steps == 1 else max(0.0, 0.10 * (max_steps - steps_taken) / (max_steps - 1))
        )
        score += efficiency

    return round(min(1.0, score), 4)


# ─────────────────────────── Penalty ───────────────────────────


def loop_penalty(step_count: int, max_steps: int = 10) -> float:
    """Return a negative reward if agent is stuck in a loop."""
    if step_count > max_steps:
        return -0.05 * (step_count - max_steps)
    return 0.0
