"""
inference.py - Support Ticket Resolution Environment
Follows mandatory [START] [STEP] [END] logging format.
"""

import asyncio
import os
import sys
import json
import re
from typing import List, Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
STUB = os.path.join(ROOT, "openenv_stub")
sys.path.insert(0, STUB)
sys.path.insert(0, ROOT)

from openai import OpenAI
from support_ticket_env.server.support_environment import SupportTicketEnvironment
from support_ticket_env.models import SupportAction

# ── Environment variables ────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME    = "support-ticket-resolution"
BENCHMARK    = "support_ticket_env"
MAX_STEPS    = 10
SUCCESS_SCORE_THRESHOLD = 0.5

VALID_CATEGORIES = ["billing", "technical", "account", "general", "refund"]
VALID_ACTIONS    = ["classify", "reply", "escalate", "close"]

SYSTEM_PROMPT = """You are a customer support AI agent handling tickets.
You receive a JSON with ticket_text, task_id, feedback, and current_category.

Respond ONLY with a JSON object (no markdown, no explanation):
{
  "action_type": "classify" | "reply" | "escalate" | "close",
  "category": "billing" | "technical" | "account" | "general" | "refund",
  "reply_text": "...",
  "reason": "..."
}

Rules:
- For task 1: use action_type=classify and pick correct category.
- For task 2: first classify, then reply/escalate/close.
- For task 3: classify each ticket then resolve it.
- category is only needed when action_type=classify.
- reply_text is only needed when action_type=reply.
"""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def parse_response(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def get_model_action(client: OpenAI, obs, history: List[str]) -> dict:
    user_prompt = json.dumps({
        "ticket_id": obs.ticket_id,
        "ticket_text": obs.ticket_text,
        "task_id": obs.task_id,
        "current_category": obs.current_category,
        "step_count": obs.step_count,
        "feedback": obs.feedback,
    })
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=256,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_response(text)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return {"action_type": "classify", "category": "general"}


def run_task(task_id: int, seed: int, client: OpenAI) -> float:
    env = SupportTicketEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=f"{TASK_NAME}-task{task_id}", env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            action_dict = get_model_action(client, obs, history)
            action_str = f"{action_dict.get('action_type','?')}"
            if action_dict.get("category"):
                action_str += f"/{action_dict['category']}"

            error = None
            try:
                action = SupportAction(**action_dict)
                obs = env.step(action)
                reward = obs.reward or 0.0
                done = obs.done
            except Exception as e:
                reward = 0.0
                done = False
                error = str(e)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")

            if done:
                break

        total = sum(rewards)
        score = min(max(round(total / MAX_STEPS, 3), 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main() -> None:
    if not API_KEY:
        print("[DEBUG] HF_TOKEN not set", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores = {}
    for task_id in [1, 2, 3]:
        scores = []
        for seed in [42, 7, 123]:
            score = run_task(task_id, seed, client)
            scores.append(score)
        avg = round(sum(scores) / len(scores), 4)
        all_scores[f"task{task_id}"] = avg
        print(f"[DEBUG] Task {task_id} avg score: {avg}", flush=True)

    overall = round(sum(all_scores.values()) / len(all_scores), 4)
    print(f"[DEBUG] Overall avg score: {overall}", flush=True)


if __name__ == "__main__":
    main()