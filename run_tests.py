#!/usr/bin/env python3
"""
run_tests.py — Self-contained test runner for support_ticket_env.
Runs all test cases using only the Python standard library.

Usage:
    python run_tests.py
"""

import sys
import os
import traceback
from typing import Callable, List, Tuple

# ─── path setup ────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
STUB = os.path.join(ROOT, "openenv_stub")
sys.path.insert(0, STUB)
sys.path.insert(0, ROOT)

# ─── minimal test framework ────────────────────────────────────
_tests: List[Tuple[str, Callable]] = []
_passed = 0
_failed = 0
_errors = 0

def test(fn: Callable) -> Callable:
    _tests.append((fn.__qualname__, fn))
    return fn

def assert_eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"{msg} | expected {b!r}, got {a!r}")

def assert_true(val, msg=""):
    if not val:
        raise AssertionError(msg or f"Expected truthy, got {val!r}")

def assert_in_range(val, lo, hi, msg=""):
    if not (lo <= val <= hi):
        raise AssertionError(msg or f"Expected {val!r} in [{lo}, {hi}]")

# ─────────────────────────────── imports ───────────────────────
from support_ticket_env.graders import (
    grade_task1, grade_task2, grade_task3, loop_penalty,
)
from support_ticket_env.server.support_environment import SupportTicketEnvironment
from support_ticket_env.models import SupportAction


def make_env():
    return SupportTicketEnvironment()


# ════════════════════════════════════════════════════════════════
# GRADER TESTS
# ════════════════════════════════════════════════════════════════

@test
def test_grade1_correct():
    assert_eq(grade_task1("billing", "billing"), 1.0)

@test
def test_grade1_wrong():
    assert_eq(grade_task1("technical", "billing"), 0.0)

@test
def test_grade1_all_categories():
    for cat in ["billing", "technical", "account", "general", "refund"]:
        assert_eq(grade_task1(cat, cat), 1.0, f"cat={cat}")

@test
def test_grade1_empty():
    assert_eq(grade_task1("", "billing"), 0.0)

@test
def test_grade2_exact_reply():
    assert_eq(grade_task2("reply", "reply"), 1.0)

@test
def test_grade2_exact_escalate():
    assert_eq(grade_task2("escalate", "escalate"), 1.0)

@test
def test_grade2_exact_close():
    assert_eq(grade_task2("close", "close"), 1.0)

@test
def test_grade2_partial_reply_escalate():
    assert_eq(grade_task2("reply", "escalate"), 0.5)
    assert_eq(grade_task2("escalate", "reply"), 0.5)

@test
def test_grade2_close_wrong():
    assert_eq(grade_task2("close", "reply"), 0.0)

@test
def test_grade3_perfect():
    score = grade_task3(True, True, False,
                        "we will process your refund billing payment",
                        "billing", True, 1, 5)
    assert_true(score >= 0.9, f"Expected >=0.9, got {score}")

@test
def test_grade3_capped_at_one():
    score = grade_task3(True, True, False,
                        "refund billing payment account cancel subscription",
                        "billing", True, 1, 5)
    assert_true(score <= 1.0, f"Score exceeds 1.0: {score}")

@test
def test_grade3_partial_action_less_than_full():
    s_partial = grade_task3(True, False, True, None, "technical", True, 2)
    s_full = grade_task3(True, True, False, None, "technical", True, 2)
    assert_true(s_partial < s_full, f"partial={s_partial} should < full={s_full}")

@test
def test_loop_penalty_none_within_limit():
    assert_eq(loop_penalty(5), 0.0)
    assert_eq(loop_penalty(10), 0.0)

@test
def test_loop_penalty_grows():
    assert_true(loop_penalty(12) < loop_penalty(11))
    assert_true(loop_penalty(11) < 0)


# ════════════════════════════════════════════════════════════════
# ENVIRONMENT TESTS
# ════════════════════════════════════════════════════════════════

@test
def test_env_reset_task1():
    env = make_env()
    obs = env.reset(task_id=1, seed=42)
    assert_true(obs.ticket_text != "", "ticket_text should not be empty")
    assert_eq(obs.task_id, 1)
    assert_eq(obs.done, False)

@test
def test_env_task1_correct_classification():
    env = make_env()
    env.reset(task_id=1, seed=42)
    state = env.state
    obs = env.step(SupportAction(action_type="classify", category=state.correct_category))
    assert_eq(obs.reward, 1.0)
    assert_eq(obs.done, True)

@test
def test_env_task1_wrong_classification():
    env = make_env()
    env.reset(task_id=1, seed=42)
    state = env.state
    wrong = next(c for c in ["billing","technical","account","general","refund"]
                 if c != state.correct_category)
    obs = env.step(SupportAction(action_type="classify", category=wrong))
    assert_eq(obs.reward, 0.0)
    assert_eq(obs.done, True)

@test
def test_env_task2_must_classify_first():
    env = make_env()
    env.reset(task_id=2, seed=42)
    obs = env.step(SupportAction(action_type="escalate"))
    assert_eq(obs.done, False)
    assert_true("classify" in obs.feedback.lower())

@test
def test_env_task2_full_correct_episode():
    env = make_env()
    env.reset(task_id=2, seed=42)
    state = env.state
    env.step(SupportAction(action_type="classify", category=state.correct_category))
    obs = env.step(SupportAction(action_type=state.correct_action))
    assert_eq(obs.done, True)
    assert_true(obs.reward >= 0.5, f"reward={obs.reward}")

@test
def test_env_task3_three_tickets():
    env = make_env()
    env.reset(task_id=3, seed=42)
    assert_eq(env.state.tickets_total, 3)

@test
def test_env_task3_resolves_all():
    env = make_env()
    env.reset(task_id=3, seed=42)
    done = False
    steps = 0
    while not done and steps < 30:
        state = env.state
        if not state.classified:
            action = SupportAction(action_type="classify", category=state.correct_category)
        else:
            ca = state.correct_action
            action = (SupportAction(action_type="reply",
                                    reply_text=f"Handling your {state.correct_category} issue.")
                      if ca == "reply" else SupportAction(action_type=ca))
        obs = env.step(action)
        done = obs.done
        steps += 1
    assert_true(done, "Episode did not finish")
    assert_eq(env.state.tickets_resolved, 3)

@test
def test_env_state_step_count():
    env = make_env()
    env.reset(task_id=1, seed=0)
    assert_eq(env.state.step_count, 0)
    state = env.state
    env.step(SupportAction(action_type="classify", category=state.correct_category))
    assert_eq(env.state.step_count, 1)

@test
def test_env_reward_always_in_range():
    for seed in [0, 1, 2, 42, 99]:
        for task_id in [1, 2, 3]:
            env = make_env()
            env.reset(task_id=task_id, seed=seed)
            state = env.state
            obs = env.step(SupportAction(action_type="classify", category=state.correct_category))
            r = obs.reward or 0.0
            assert_in_range(r, -1.0, 1.0, f"task={task_id} seed={seed} reward={r}")

@test
def test_env_task3_total_reward_positive():
    env = make_env()
    env.reset(task_id=3, seed=7)
    total = 0.0
    done = False
    steps = 0
    while not done and steps < 20:
        state = env.state
        action = (SupportAction(action_type="classify", category=state.correct_category)
                  if not state.classified
                  else SupportAction(action_type=state.correct_action))
        obs = env.step(action)
        total += obs.reward or 0.0
        done = obs.done
        steps += 1
    assert_true(total > 0.0, f"total_reward={total}")


# ════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════

def run_all():
    global _passed, _failed, _errors
    width = max(len(name) for name, _ in _tests) + 2
    print(f"\n{'='*60}")
    print(f"  Running {len(_tests)} tests")
    print(f"{'='*60}")
    for name, fn in _tests:
        try:
            fn()
            print(f"  ✅  {name}")
            _passed += 1
        except AssertionError as e:
            print(f"  ❌  {name}")
            print(f"       {e}")
            _failed += 1
        except Exception:
            print(f"  💥  {name}")
            traceback.print_exc(limit=3)
            _errors += 1
    total = _passed + _failed + _errors
    print(f"\n{'='*60}")
    print(f"  Results: {_passed}/{total} passed | {_failed} failed | {_errors} errors")
    print(f"{'='*60}\n")
    return _failed + _errors == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
