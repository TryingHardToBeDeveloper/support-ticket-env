"""
Tests for SupportTicketEnvironment — runs the environment directly
(no HTTP server required).
"""

import pytest
from support_ticket_env.server.support_environment import SupportTicketEnvironment
from support_ticket_env.models import SupportAction


# ─────────────────────────── fixtures ──────────────────────────

@pytest.fixture
def env():
    return SupportTicketEnvironment()


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
        state = env.state
        action = SupportAction(
            action_type="classify",
            category=state.correct_category,
        )
        obs = env.step(action)
        assert obs.reward == 1.0
        assert obs.done is True

    def test_wrong_classification(self, env):
        env.reset(task_id=1, seed=42)
        state = env.state
        wrong_cats = [
            c for c in ["billing", "technical", "account", "general", "refund"]
            if c != state.correct_category
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
        state = env.state

        # Step 1: classify
        obs = env.step(SupportAction(
            action_type="classify",
            category=state.correct_category,
        ))
        assert obs.done is False
        assert obs.reward > 0

        # Step 2: correct action
        obs = env.step(SupportAction(action_type=state.correct_action))
        assert obs.done is True
        assert obs.reward >= 0.5

    def test_must_classify_first(self, env):
        env.reset(task_id=2, seed=7)
        obs = env.step(SupportAction(action_type="escalate"))
        assert obs.done is False
        assert "classify" in obs.feedback.lower()

    def test_state_reflects_progress(self, env):
        env.reset(task_id=2, seed=7)
        state = env.state
        assert state.classified is False

        env.step(SupportAction(
            action_type="classify",
            category=state.correct_category,
        ))
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
            if not state.classified:
                action = SupportAction(
                    action_type="classify",
                    category=state.correct_category,
                )
            else:
                ca = state.correct_action
                if ca == "reply":
                    action = SupportAction(
                        action_type="reply",
                        reply_text=f"We are handling your {state.correct_category} issue.",
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
            if not state.classified:
                action = SupportAction(
                    action_type="classify",
                    category=state.correct_category,
                )
            else:
                action = SupportAction(action_type=state.correct_action)
            obs = env.step(action)
            total += obs.reward or 0.0
            done = obs.done
            steps += 1

        assert total > 0.0


# ─────────────────────────── State API ─────────────────────────

class TestStateAPI:
    def test_state_after_reset(self, env):
        env.reset(task_id=1, seed=0)
        state = env.state
        assert state.step_count == 0
        assert state.task_id == 1
        assert state.ticket_id != ""

    def test_step_count_increments(self, env):
        env.reset(task_id=1, seed=0)
        state = env.state
        env.step(SupportAction(action_type="classify", category=state.correct_category))
        assert env.state.step_count == 1


# ─────────────────────────── Reward bounds ─────────────────────

class TestRewardBounds:
    def test_reward_in_range(self, env):
        for seed in [0, 1, 2, 3, 42]:
            for task_id in [1, 2, 3]:
                env.reset(task_id=task_id, seed=seed)
                state = env.state
                action = SupportAction(
                    action_type="classify",
                    category=state.correct_category,
                )
                obs = env.step(action)
                assert -1.0 <= (obs.reward or 0.0) <= 1.0, (
                    f"Reward out of bounds: {obs.reward}"
                )
