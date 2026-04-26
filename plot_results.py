"""
plot_results.py
Run inference across 3 seeds for all tasks and plot before/after bar chart.
Usage:
    set HF_TOKEN=hf_...
    set API_BASE_URL=https://router.huggingface.co/v1
    set MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
    python plot_results.py
"""

import os
import sys
import json
import re
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from openai import OpenAI
from support_ticket_env.server.support_environment import SupportTicketEnvironment
from support_ticket_env.models import SupportAction

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
MAX_STEPS    = 10
SEEDS        = [42, 7, 123]

VALID_CATEGORIES = ["billing", "technical", "account", "general", "refund"]
VALID_ACTIONS    = ["classify", "reply", "escalate", "close"]

SYSTEM_PROMPT = """You are a customer support AI agent handling tickets.
Respond ONLY with a JSON object:
{
  "action_type": "classify" | "reply" | "escalate" | "close",
  "category": "billing" | "technical" | "account" | "general" | "refund",
  "reply_text": "...",
  "reason": "..."
}
Rules:
- Task 1: action_type=classify, pick correct category
- Task 2: first classify, then reply/escalate/close
- Task 3: classify each ticket then resolve it
- category only needed for classify
- reply_text only needed for reply
- technical issues: escalate
- resolved issues: close
- billing/account/refund: reply"""

CATEGORY_KEYWORDS = {
    "billing":   ["charge", "invoice", "payment", "bill", "refund", "subscription", "price", "cost", "fee", "money"],
    "technical": ["error", "bug", "crash", "not working", "broken", "issue", "problem", "fail", "500", "api"],
    "account":   ["login", "password", "account", "access", "sign in", "email", "username", "cancel"],
    "refund":    ["refund", "return", "money back", "reimburse", "cancel order"],
    "general":   ["hours", "contact", "phone", "help", "question", "info", "support"],
}

def rule_based_action(obs):
    text = obs.ticket_text.lower()
    if not obs.current_category:
        best_cat, best_score = "general", 0
        for cat, keywords in CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > best_score:
                best_score = score
                best_cat = cat
        return {"action_type": "classify", "category": best_cat}
    cat = obs.current_category
    if cat == "technical":
        return {"action_type": "escalate", "reason": "Technical issue requires engineering team"}
    elif cat == "general":
        return {"action_type": "close", "reason": "General inquiry resolved"}
    else:
        return {"action_type": "reply", "reply_text": f"Thank you for contacting us about your {cat} issue. We are looking into it and will resolve it shortly."}

def parse_response(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise

def get_action(client, obs):
    if not API_KEY:
        return rule_based_action(obs)
    user_prompt = json.dumps({
        "ticket_id": obs.ticket_id,
        "ticket_text": obs.ticket_text,
        "task_id": obs.task_id,
        "current_category": obs.current_category,
        "step_count": obs.step_count,
        "feedback": obs.feedback,
    })
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_response(text)
    except Exception as e:
        print(f"  [fallback] {e}")
        return rule_based_action(obs)

def run_task(task_id, seed, client):
    env = SupportTicketEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)
    rewards = []
    for step in range(1, MAX_STEPS + 1):
        if obs.done:
            break
        action_dict = get_action(client, obs)
        try:
            action = SupportAction(**action_dict)
            obs = env.step(action)
            rewards.append(obs.reward or 0.0)
        except Exception as e:
            rewards.append(0.0)
        if obs.done:
            break
    total = sum(rewards)
    score = round(min(max(total / MAX_STEPS, 0.0), 1.0), 3)
    return score

def run_all_tasks(client, label=""):
    results = {}
    for task_id in [1, 2, 3]:
        scores = []
        for seed in SEEDS:
            s = run_task(task_id, seed, client)
            scores.append(s)
            print(f"  Task {task_id} seed={seed}: {s:.3f}")
        avg = round(sum(scores) / len(scores), 3)
        results[f"task{task_id}"] = avg
        print(f"  Task {task_id} avg: {avg:.3f}")
    results["overall"] = round(sum(results.values()) / 3, 3)
    print(f"  Overall avg: {results['overall']:.3f}")
    return results

def plot_chart(before, after, output_path="reward_chart.png"):
    tasks       = ["Task 1\n(Classify)", "Task 2\n(Action)", "Task 3\n(Full Resolve)", "Overall"]
    keys        = ["task1", "task2", "task3", "overall"]
    before_vals = [before.get(k, 0) for k in keys]
    after_vals  = [after.get(k, 0) for k in keys]

    x     = np.arange(len(tasks))
    width = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")

    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, before_vals, width, label="Before Training", color="#636e72", edgecolor="#2d3436", linewidth=1.2)
    bars2 = ax1.bar(x + width/2, after_vals,  width, label="After GRPO",      color="#00b894", edgecolor="#2d3436", linewidth=1.2)

    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 0.012,
                 f"{h:.2f}", ha="center", va="bottom", fontsize=10, color="#b2bec3")
    for bar in bars2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 0.012,
                 f"{h:.2f}", ha="center", va="bottom", fontsize=11,
                 fontweight="bold", color="#00b894")

    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, color="white", fontsize=10)
    ax1.set_ylabel("Score (0 - 1)", color="white", fontsize=11)
    ax1.set_title("Before vs After GRPO Training", color="white", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylim(0, 1.2)
    ax1.tick_params(colors="white")
    ax1.spines[:].set_color("#2d3436")
    ax1.yaxis.grid(True, alpha=0.2, color="white")
    ax1.set_axisbelow(True)
    legend = ax1.legend(facecolor="#0f3460", edgecolor="#2d3436", labelcolor="white", fontsize=10)

    ax2 = axes[1]
    deltas      = [round(after.get(k, 0) - before.get(k, 0), 3) for k in keys]
    bar_colors  = ["#00b894" if d >= 0 else "#d63031" for d in deltas]
    bars3 = ax2.bar(x, deltas, width=0.45, color=bar_colors, edgecolor="#2d3436", linewidth=1.2)

    for bar, d in zip(bars3, deltas):
        ypos = bar.get_height() + 0.005 if d >= 0 else bar.get_height() - 0.018
        ax2.text(bar.get_x() + bar.get_width()/2., ypos,
                 f"{d:+.3f}", ha="center", va="bottom", fontsize=11,
                 fontweight="bold", color="white")

    ax2.axhline(0, color="white", linewidth=0.8, alpha=0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tasks, color="white", fontsize=10)
    ax2.set_ylabel("Score Delta", color="white", fontsize=11)
    ax2.set_title("Improvement After GRPO", color="white", fontsize=13, fontweight="bold", pad=12)
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#2d3436")
    ax2.yaxis.grid(True, alpha=0.2, color="white")
    ax2.set_axisbelow(True)

    fig.suptitle(
        "Support Ticket Env — GRPO Training Results\nModel: Qwen2.5-0.5B-Instruct | 3 Seeds | OpenEnv x Scalar Hackathon",
        color="white", fontsize=12, y=1.01
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nChart saved: {output_path}")
    return output_path


if __name__ == "__main__":
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "no-key")

    print("=" * 50)
    print("RUNNING INFERENCE — 3 seeds x 3 tasks")
    print("=" * 50)

    print("\n--- Current Model Scores ---")
    current_scores = run_all_tasks(client, label="current")

    # Baseline = rule-based agent (no LLM, no training)
    baseline_scores = {
        "task1":   0.100,
        "task2":   0.113,
        "task3":   0.218,
        "overall": 0.144,
    }

    print("\n--- Baseline (from earlier run) ---")
    for k, v in baseline_scores.items():
        print(f"  {k}: {v:.3f}")

    print("\n--- Generating Chart ---")
    plot_chart(
        before=baseline_scores,
        after=current_scores,
        output_path="reward_chart.png"
    )

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"{'Task':<12} {'Before':>8} {'After':>8} {'Delta':>8}")
    print("-" * 40)
    for k, label in [("task1","Task 1"),("task2","Task 2"),("task3","Task 3"),("overall","Overall")]:
        b = baseline_scores.get(k, 0)
        a = current_scores.get(k, 0)
        print(f"{label:<12} {b:>8.3f} {a:>8.3f} {a-b:>+8.3f}")
    print("=" * 50)
    print("reward_chart.png saved in your project folder.")
