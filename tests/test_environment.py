"""
Tests for SupportTicketEnvironment — runs the environment directly
(no HTTP server required).
"""

import pytest

from server.support_environment import SupportTicketEnvironment
from server.ticket_bank import TICKET_LOOKUP
from support_ticket_env.models import SupportAction

# ─────────────────────────── fixtures ──────────────────────────


@pytest.fixture
def env():
    return SupportTicketEnvironment()


def answer(env):
    """Trusted test-only access to the server-side oracle."""
    return TICKET_LOOKUP[env.state.ticket_id]


def correct_resolution(env):
    item = answer(env)
    if item["correct_action"] == "reply":
        return SupportAction(
            action_type="reply",
            reply_text=f"We will review your request and help with: {item['text']}",
        )
    return SupportAction(action_type=item["correct_action"])


# ─────────────────────────── Task 1 ────────────────────────────


class TestTask1:
    def test_reset_returns_observation(self, env):
        obs = env.reset(task_id=1, seed=42)
        assert obs.ticket_text
        assert obs.task_id == 1
        assert obs.done is False

    def test_correct_classification(self, env):
        obs = env.reset(task_id=1, seed=42)
        # Find out the correct category via state
        item = answer(env)
        action = SupportAction(
            action_type="classify",
            category=item["category"],
        )
        obs = env.step(action)
        assert obs.reward == 1.0
        assert obs.done is True

    def test_wrong_classification(self, env):
        env.reset(task_id=1, seed=42)
        item = answer(env)
        wrong_cats = [
            c
            for c in ["billing", "technical", "account", "general", "refund"]
            if c != item["category"]
        ]
        action = SupportAction(action_type="classify", category=wrong_cats[0])
        obs = env.step(action)
        assert obs.reward == 0.0
        assert obs.done is True

    def test_non_classify_action_penalised(self, env):
        env.reset(task_id=1, seed=42)
        obs = env.step(SupportAction(action_type="reply", reply_text="hello"))
        # Should not crash; done might be False and reward 0
        assert obs.reward is not None


# ─────────────────────────── Task 2 ────────────────────────────


class TestTask2:
    def test_full_correct_episode(self, env):
        env.reset(task_id=2, seed=42)
        item = answer(env)

        # Step 1: classify
        obs = env.step(
            SupportAction(
                action_type="classify",
                category=item["category"],
            )
        )
        assert obs.done is False
        assert obs.reward > 0

        # Step 2: correct action
        obs = env.step(correct_resolution(env))
        assert obs.done is True
        assert obs.reward >= 0.5

    def test_must_classify_first(self, env):
        env.reset(task_id=2, seed=7)
        obs = env.step(SupportAction(action_type="escalate"))
        assert obs.done is False
        assert "classify" in obs.feedback.lower()

    def test_state_reflects_progress(self, env):
        env.reset(task_id=2, seed=7)
        item = answer(env)
        state = env.state
        assert state.classified is False

        env.step(
            SupportAction(
                action_type="classify",
                category=item["category"],
            )
        )
        state2 = env.state
        assert state2.classified is True
        assert state2.step_count == 1


# ─────────────────────────── Task 3 ────────────────────────────


class TestTask3:
    def test_queue_has_three_tickets(self, env):
        env.reset(task_id=3, seed=42)
        state = env.state
        assert state.tickets_total == 3
        assert state.tickets_resolved == 0

    def test_resolve_all_tickets(self, env):
        env.reset(task_id=3, seed=42)
        done = False
        steps = 0

        while not done and steps < 30:
            state = env.state
            item = answer(env)
            if not state.classified:
                action = SupportAction(
                    action_type="classify",
                    category=item["category"],
                )
            else:
                ca = item["correct_action"]
                if ca == "reply":
                    action = SupportAction(
                        action_type="reply",
                        reply_text=f"We will investigate and resolve your request: {item['text']}",
                    )
                else:
                    action = SupportAction(action_type=ca)
            obs = env.step(action)
            done = obs.done
            steps += 1

        assert done, "Episode should finish after 3 tickets"
        final_state = env.state
        assert final_state.tickets_resolved == 3

    def test_total_reward_positive(self, env):
        env.reset(task_id=3, seed=123)
        total = 0.0
        done = False
        steps = 0

        while not done and steps < 20:
            state = env.state
            item = answer(env)
            if not state.classified:
                action = SupportAction(
                    action_type="classify",
                    category=item["category"],
                )
            else:
                action = correct_resolution(env)
            obs = env.step(action)
            total += obs.reward or 0.0
            done = obs.done
            steps += 1

        assert total > 0.0
        assert total <= 1.0


# ─────────────────────────── State API ─────────────────────────


class TestStateAPI:
    def test_state_after_reset(self, env):
        env.reset(task_id=1, seed=0)
        state = env.state
        assert state.step_count == 0
        assert state.task_id == 1
        assert state.ticket_id != ""
        dumped = state.model_dump() if hasattr(state, "model_dump") else vars(state)
        assert "correct_category" not in dumped
        assert "correct_action" not in dumped

    def test_wrong_guess_does_not_leak_ground_truth(self, env):
        env.reset(task_id=2, seed=42)
        item = answer(env)
        wrong = next(
            c
            for c in ["billing", "technical", "account", "general", "refund"]
            if c != item["category"]
        )
        obs = env.step(SupportAction(action_type="classify", category=wrong))
        assert obs.current_category == wrong
        assert obs.current_category != item["category"]

    @pytest.mark.parametrize("task_id", [0, 4, -1, 99])
    def test_invalid_task_id_is_rejected(self, env, task_id):
        with pytest.raises(ValueError, match="task_id"):
            env.reset(task_id=task_id, seed=0)

    def test_step_count_increments(self, env):
        env.reset(task_id=1, seed=0)
        item = answer(env)
        env.step(SupportAction(action_type="classify", category=item["category"]))
        assert env.state.step_count == 1


# ─────────────────────────── Reward bounds ─────────────────────


class TestRewardBounds:
    def test_reward_in_range(self, env):
        for seed in [0, 1, 2, 3, 42]:
            for task_id in [1, 2, 3]:
                env.reset(task_id=task_id, seed=seed)
                item = answer(env)
                action = SupportAction(
                    action_type="classify",
                    category=item["category"],
                )
                obs = env.step(action)
                assert -1.0 <= (obs.reward or 0.0) <= 1.0, f"Reward out of bounds: {obs.reward}"
