#!/usr/bin/env python3
"""
baseline.py — Baseline inference script for the Support Ticket Environment.

Runs an OpenAI-compatible model against all 3 tasks and reports scores.

Usage:
    OPENAI_API_KEY=sk-... python baseline.py --base-url http://localhost:7860

Environment variables:
    OPENAI_API_KEY   : required
    OPENAI_BASE_URL  : optional override (default https://api.openai.com/v1)
    OPENAI_MODEL     : optional model name (default gpt-4o-mini)
"""

import argparse
import json
import os
import asyncio
import re

from openai import AsyncOpenAI
from support_ticket_env.client import SupportTicketEnv
from support_ticket_env.models import SupportAction

# ─────────────────────────── Config ────────────────────────────

VALID_CATEGORIES = ["billing", "technical", "account", "general", "refund"]
VALID_ACTIONS = ["classify", "reply", "escalate", "close"]

SYSTEM_PROMPT = """You are a customer support AI agent operating in a ticket triage environment.

On each turn you receive a JSON observation with:
  - ticket_text : the customer's message
  - feedback    : what happened last step
  - task_id     : 1=classify only, 2=classify then act, 3=full resolution

You must respond with a JSON object (no markdown) matching this schema:
{
  "action_type": "classify" | "reply" | "escalate" | "close",
  "category": "billing" | "technical" | "account" | "general" | "refund",  // only for classify
  "reply_text": "...",  // only for reply
  "reason": "..."       // optional
}

Strategy:
- For task 1: only classify (use action_type="classify" with a category).
- For task 2: first classify, then choose the best action.
- For task 3: classify each ticket, then reply/escalate/close as appropriate.

Always produce valid JSON and nothing else.
"""


def parse_llm_response(text: str) -> dict:
    """Extract JSON from LLM response, stripping markdown fences if present."""
    text = text.strip()
    # Strip ```json ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: try to extract first JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


async def run_task(
    env_base_url: str,
    llm: AsyncOpenAI,
    model: str,
    task_id: int,
    seed: int = 42,
    max_steps: int = 10,
) -> float:
    """Run one episode for a given task_id. Returns the total reward."""
    total_reward = 0.0
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    async with SupportTicketEnv(base_url=env_base_url) as env:
        result = await env.reset(task_id=task_id, seed=seed)
        obs = result.observation

        for step in range(max_steps):
            # Build user message from observation
            obs_text = json.dumps({
                "ticket_id": obs.ticket_id,
                "ticket_text": obs.ticket_text,
                "task_id": obs.task_id,
                "current_category": obs.current_category,
                "resolved": obs.resolved,
                "step_count": obs.step_count,
                "feedback": obs.feedback,
            }, indent=2)

            messages.append({"role": "user", "content": obs_text})

            # Call LLM
            response = await llm.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=256,
            )
            assistant_text = response.choices[0].message.content
            messages.append({"role": "assistant", "content": assistant_text})

            # Parse action
            try:
                action_dict = parse_llm_response(assistant_text)
            except Exception as e:
                print(f"  [step {step+1}] Failed to parse LLM response: {e}")
                break

            try:
                action = SupportAction(**action_dict)
            except Exception as e:
                print(f"  [step {step+1}] Invalid action schema: {e}")
                break

            # Step environment
            result = await env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            total_reward += reward

            print(
                f"  [step {step+1}] action={action.action_type}"
                + (f"/{action.category}" if action.category else "")
                + f"  reward={reward:.3f}  feedback={obs.feedback[:60]}"
            )

            if result.done:
                break

    return round(total_reward, 4)


async def main(env_base_url: str, model: str, seeds: list[int]) -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "not-needed")
    openai_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    llm = AsyncOpenAI(api_key=api_key, base_url=openai_base)

    results = {}
    for task_id in [1, 2, 3]:
        task_scores = []
        print(f"\n{'='*60}")
        print(f"  TASK {task_id}  (seed={seeds[0]})")
        print(f"{'='*60}")
        for seed in seeds:
            score = await run_task(env_base_url, llm, model, task_id, seed=seed)
            task_scores.append(score)
            print(f"  → total_reward for seed {seed}: {score}")
        avg = round(sum(task_scores) / len(task_scores), 4)
        results[f"task{task_id}"] = {"scores": task_scores, "avg": avg}
        print(f"  ► Average: {avg}")

    print("\n" + "="*60)
    print("  BASELINE SUMMARY")
    print("="*60)
    for k, v in results.items():
        print(f"  {k}: avg={v['avg']:.4f}  scores={v['scores']}")
    overall = round(
        sum(v["avg"] for v in results.values()) / len(results), 4
    )
    print(f"\n  Overall avg: {overall:.4f}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline inference for support_ticket_env")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("ENV_BASE_URL", "http://localhost:7860"),
        help="Base URL of the running environment server",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model name",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 7, 123],
        help="Random seeds for reproducibility",
    )
    args = parser.parse_args()
    asyncio.run(main(args.base_url, args.model, args.seeds))
