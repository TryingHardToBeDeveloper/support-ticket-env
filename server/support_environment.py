"""
Customer Support Ticket Resolution — OpenEnv Environment (server side).

Implements the three tasks:
  Task 1 (easy)   – Classify a single ticket
  Task 2 (medium) – Choose the correct action for a classified ticket
  Task 3 (hard)   – Fully resolve a queue of tickets with minimal steps
"""

from __future__ import annotations

import random
from typing import Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from support_ticket_env.models import SupportAction, SupportObservation, SupportState
from support_ticket_env.tickets import TICKETS, TICKET_LOOKUP
from support_ticket_env.graders import (
    grade_task1,
    grade_task2,
    grade_task3,
    loop_penalty,
)


class SupportTicketEnvironment(Environment):
    """
    OpenEnv environment that simulates a customer-support triage desk.

    The task_id (1, 2, or 3) is set when the environment is reset.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._task_id: int = 1
        self._ticket: dict = {}
        self._classified: bool = False
        self._classified_correctly: bool = False  # tracks actual correctness, not just attempt
        self._task2_cls_score: float = 0.0        # accumulated classification partial credit for Task 2
        self._resolved: bool = False
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._episode_id: Optional[str] = None

        # Task 3: queue of tickets
        self._queue: list[dict] = []
        self._tickets_resolved: int = 0
        self._tickets_total: int = 1
	
    def get_metadata(self):
        from openenv.core.env_server.types import EnvironmentMetadata
        return EnvironmentMetadata(
            name="support_ticket_env",
            description="A real-world customer support ticket triage environment where an AI agent classifies tickets, selects actions, and resolves queues.",
            version="1.0.0",
            author="AlgoCore",
            documentation_url="https://github.com/TryingHardToBeDeveloper/support-ticket-env",
        )

    # ──────────────────────── reset ────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: int = 1,
        **kwargs,
    ) -> SupportObservation:
        rng = random.Random(seed)
        self._episode_id = episode_id
        self._task_id = int(task_id)
        self._step_count = 0
        self._total_reward = 0.0
        self._classified = False
        self._classified_correctly = False
        self._task2_cls_score = 0.0
        self._resolved = False

        if self._task_id == 3:
            # Give the agent a queue of 3 tickets
            self._queue = rng.sample(TICKETS, k=3)
            self._tickets_total = len(self._queue)
            self._tickets_resolved = 0
            self._ticket = self._queue[0]
        else:
            self._ticket = rng.choice(TICKETS)
            self._tickets_total = 1
            self._tickets_resolved = 0

        return self._make_obs(
            feedback="New episode started. Read the ticket and take action.",
            score=0.0,
        )

    # ──────────────────────── step ─────────────────────────────

    def step(self, action: SupportAction, **kwargs) -> SupportObservation:  # type: ignore[override]
        self._step_count += 1
        penalty = loop_penalty(self._step_count)

        if self._task_id == 1:
            obs = self._step_task1(action)
        elif self._task_id == 2:
            obs = self._step_task2(action)
        else:
            obs = self._step_task3(action)

        # Apply loop penalty on top of step reward
        obs.reward = (obs.reward or 0.0) + penalty
        obs.reward = round(max(-1.0, min(1.0, obs.reward)), 4)
        self._total_reward += obs.reward
        obs.step_count = self._step_count
        return obs

    # ──────────────────────── Task 1 ───────────────────────────

    def _step_task1(self, action: SupportAction) -> SupportObservation:
        if action.action_type != "classify":
            return self._make_obs(
                feedback="Task 1 requires a 'classify' action.",
                score=0.0,
                done=False,
            )

        score = grade_task1(
            predicted_category=action.category or "",
            correct_category=self._ticket["category"],
        )
        self._classified = score == 1.0
        correct = self._ticket["category"]

        if score == 1.0:
            feedback = f"✅ Correct! Category: '{correct}'."
            done = True
        else:
            feedback = (
                f"❌ Wrong. You said '{action.category}', correct is '{correct}'."
            )
            done = True  # Task 1 is one-shot — agent gets one attempt

        obs = self._make_obs(feedback=feedback, score=score, done=done)
        if done:
            self._resolved = True
        return obs

    # ──────────────────────── Task 2 ───────────────────────────

    def _step_task2(self, action: SupportAction) -> SupportObservation:
        # First step must be classification
        if not self._classified:
            if action.action_type != "classify":
                return self._make_obs(
                    feedback="Please classify the ticket first.",
                    score=0.0,
                )
            cat_score = grade_task1(
                action.category or "", self._ticket["category"]
            )
            self._classified = True
            self._task2_cls_score = cat_score * 0.3  # store — combined with action score at step 2
            # TODO: store self._classified_correctly here too if grade_task2
            # is ever extended to factor in classification correctness
            return self._make_obs(
                feedback=(
                    f"Classified as '{action.category}'. "
                    f"{'Correct ✅' if cat_score == 1.0 else 'Incorrect ❌'} "
                    "Now choose an action."
                ),
                score=self._task2_cls_score,
            )

        # Second step: choose action
        action_score = grade_task2(
            action_type=action.action_type,
            correct_action=self._ticket["correct_action"],
            category=self._ticket["category"],
        )
        # Scale action score to 0.7 max so classification credit (0.0-0.3) has real room.
        # Total max = 0.7 (perfect action) + 0.3 (correct classify) = 1.0
        # Clamp AFTER addition — pre-clamping would silently discard classification credit.
        score = round(min(1.0, action_score * 0.7 + self._task2_cls_score), 4)
        correct = self._ticket["correct_action"]
        if action_score == 1.0:
            feedback = f"✅ Correct action: '{correct}'."
        elif action_score == 0.5:
            feedback = (
                f"⚠️ Partial credit. '{action.action_type}' is defensible "
                f"but '{correct}' is preferred."
            )
        else:
            feedback = f"❌ Wrong action. Correct: '{correct}'."

        self._resolved = True
        return self._make_obs(feedback=feedback, score=score, done=True)

    # ──────────────────────── Task 3 ───────────────────────────

    def _step_task3(self, action: SupportAction) -> SupportObservation:
        MAX_STEPS = 15

        if not self._classified:
            # Must classify first
            if action.action_type != "classify":
                return self._make_obs(
                    feedback="Classify the ticket before taking action.",
                    score=0.0,
                )
            cat_score = grade_task1(
                action.category or "", self._ticket["category"]
            )
            self._classified = True
            self._classified_correctly = (cat_score == 1.0)  # real correctness tracked
            return self._make_obs(
                feedback=(
                    f"Classified '{self._ticket['id']}' as '{action.category}'. "
                    f"{'Correct ✅' if cat_score == 1.0 else 'Incorrect ❌'} "
                    "Now resolve it."
                ),
                score=cat_score * 0.1,
            )

        # Resolve current ticket
        action_correct = action.action_type == self._ticket["correct_action"]
        pair = frozenset({action.action_type, self._ticket["correct_action"]})
        action_partial = (not action_correct) and pair in {
            frozenset({"reply", "escalate"})
        }

        score = grade_task3(
            classified_correctly=self._classified_correctly,  # real score, not just attempt flag
            action_correct=action_correct,
            action_partial=action_partial,
            reply_text=action.reply_text,
            category=self._ticket["category"],       # ground truth category
            resolution_hint=self._ticket.get("resolution_hint", ""),  # per-ticket hint keywords
            resolved=True,
            steps_taken=self._step_count,
            max_steps=MAX_STEPS,
        )

        self._tickets_resolved += 1
        correct_action = self._ticket["correct_action"]

        # Advance to next ticket in queue
        if self._tickets_resolved < self._tickets_total:
            self._ticket = self._queue[self._tickets_resolved]
            self._classified = False
            feedback = (
                f"Ticket resolved (score {score:.2f}). "
                f"Moving to next ticket ({self._tickets_resolved + 1}/{self._tickets_total})."
            )
            done = False
        else:
            feedback = (
                f"All {self._tickets_total} tickets resolved! "
                f"Episode score: {self._total_reward + score:.2f}"
            )
            done = True
            self._resolved = True

        return self._make_obs(feedback=feedback, score=score, done=done)

    # ──────────────────────── helpers ──────────────────────────

    def _make_obs(
        self,
        feedback: str,
        score: float,
        done: bool = False,
    ) -> SupportObservation:
        return SupportObservation(
            ticket_id=self._ticket.get("id", ""),
            ticket_text=self._ticket.get("text", ""),
            task_id=self._task_id,
            current_category=self._ticket.get("category") if self._classified else None,
            resolved=self._resolved,
            step_count=self._step_count,
            feedback=feedback,
            score=score,
            reward=score,
            done=done,
        )

    # ──────────────────────── state ────────────────────────────

    @property
    def state(self) -> SupportState:
        return SupportState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task_id,
            ticket_id=self._ticket.get("id", ""),
            correct_category=self._ticket.get("category", ""),
            correct_action=self._ticket.get("correct_action", ""),
            classified=self._classified,
            resolved=self._resolved,
            total_reward=self._total_reward,
            tickets_resolved=self._tickets_resolved,
            tickets_total=self._tickets_total,
        )
