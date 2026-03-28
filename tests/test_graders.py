"""Unit tests for grader functions."""

import pytest
from support_ticket_env.graders import (
    grade_task1,
    grade_task2,
    grade_task3,
    loop_penalty,
)


class TestTask1Grader:
    def test_correct_category(self):
        assert grade_task1("billing", "billing") == 1.0

    def test_wrong_category(self):
        assert grade_task1("technical", "billing") == 0.0

    def test_all_categories(self):
        for cat in ["billing", "technical", "account", "general", "refund"]:
            assert grade_task1(cat, cat) == 1.0

    def test_empty_prediction(self):
        assert grade_task1("", "billing") == 0.0


class TestTask2Grader:
    def test_exact_match(self):
        assert grade_task2("reply", "reply") == 1.0
        assert grade_task2("escalate", "escalate") == 1.0
        assert grade_task2("close", "close") == 1.0

    def test_partial_credit_reply_escalate(self):
        score = grade_task2("reply", "escalate")
        assert score == 0.5
        score = grade_task2("escalate", "reply")
        assert score == 0.5

    def test_wrong_action_close(self):
        assert grade_task2("close", "reply") == 0.0
        assert grade_task2("close", "escalate") == 0.0

    def test_classify_when_action_expected(self):
        assert grade_task2("classify", "reply") == 0.0


class TestTask3Grader:
    def test_perfect_resolution(self):
        score = grade_task3(
            classified_correctly=True,
            action_correct=True,
            action_partial=False,
            reply_text="we will process your refund billing payment",
            category="billing",
            resolved=True,
            steps_taken=1,
            max_steps=5,
        )
        assert score > 0.9

    def test_no_classification(self):
        score = grade_task3(
            classified_correctly=False,
            action_correct=True,
            action_partial=False,
            reply_text="here is the refund",
            category="billing",
            resolved=True,
            steps_taken=2,
        )
        # Should not get the 0.20 classification bonus
        assert score < 1.0

    def test_partial_action(self):
        score_partial = grade_task3(
            classified_correctly=True,
            action_correct=False,
            action_partial=True,
            reply_text=None,
            category="technical",
            resolved=True,
            steps_taken=2,
        )
        score_correct = grade_task3(
            classified_correctly=True,
            action_correct=True,
            action_partial=False,
            reply_text=None,
            category="technical",
            resolved=True,
            steps_taken=2,
        )
        assert score_partial < score_correct

    def test_score_capped_at_one(self):
        score = grade_task3(
            classified_correctly=True,
            action_correct=True,
            action_partial=False,
            reply_text="refund billing payment account cancel subscription",
            category="billing",
            resolved=True,
            steps_taken=1,
            max_steps=5,
        )
        assert score <= 1.0


class TestLoopPenalty:
    def test_no_penalty_within_limit(self):
        assert loop_penalty(5) == 0.0
        assert loop_penalty(10) == 0.0

    def test_penalty_beyond_limit(self):
        assert loop_penalty(11) < 0.0
        assert loop_penalty(15) < loop_penalty(11)

    def test_penalty_grows(self):
        p1 = loop_penalty(12)
        p2 = loop_penalty(14)
        assert p2 < p1
