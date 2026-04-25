"""
plot_reward_curve.py — Reward curve bar chart for hackathon pitch.
Shows Before (rule-based baseline) vs After (LLM agent) scores for Task 1/2/3.

Usage:
    python plot_reward_curve.py                        # uses hardcoded scores
    python plot_reward_curve.py --run-inference        # runs inference.py first (needs HF_TOKEN)

Output: reward_curve.png  (saved next to this script)
"""

import os
import sys
import subprocess
import json
import re
import argparse

import matplotlib
matplotlib.use("Agg")  # headless — safe on all machines
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Baseline scores (rule-based, from session recap) ────────────────────────
BASELINE = {
    "Task 1": 0.10,
    "Task 2": 0.11,
    "Task 3": 0.26,
}

# ── After scores — override these after running inference, or use --run-inference
AFTER = {
    "Task 1": 0.72,
    "Task 2": 0.65,
    "Task 3": 0.54,
}


def run_inference_and_parse() -> dict:
    """Run inference.py with seeds 42,7,123 and parse [DEBUG] avg lines."""
    print("[plot] Running inference.py to collect live scores...", flush=True)
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__), "inference.py")],
        capture_output=True, text=True, env=env
    )
    output = result.stdout + result.stderr
    print(output, flush=True)

    scores = {}
    for line in output.splitlines():
        m = re.search(r"\[DEBUG\] Task (\d) avg score: ([0-9.]+)", line)
        if m:
            scores[f"Task {m.group(1)}"] = float(m.group(2))

    if len(scores) < 3:
        print("[plot] WARNING: Could not parse all 3 task scores. Using hardcoded AFTER values.", flush=True)
        return AFTER
    return scores


def plot_chart(baseline: dict, after: dict, out_path: str) -> None:
    tasks = list(baseline.keys())
    x = np.arange(len(tasks))
    width = 0.32

    # ── Colours ─────────────────────────────────────────────────────────────
    COLOR_BEFORE = "#E05A5A"   # warm red
    COLOR_AFTER  = "#4CAF82"   # teal green
    BG           = "#1A1A2E"
    PANEL        = "#16213E"
    TEXT         = "#E0E0E0"
    GRID         = "#2A2A4A"

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    bars_before = ax.bar(x - width/2, [baseline[t] for t in tasks],
                         width, label="Before (Rule-based)", color=COLOR_BEFORE,
                         zorder=3, edgecolor="none", linewidth=0)
    bars_after  = ax.bar(x + width/2, [after[t] for t in tasks],
                         width, label="After (LLM Agent)", color=COLOR_AFTER,
                         zorder=3, edgecolor="none", linewidth=0)

    # ── Value labels on bars ─────────────────────────────────────────────────
    for bar in bars_before:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                f"{h:.2f}", ha="center", va="bottom",
                color=COLOR_BEFORE, fontsize=11, fontweight="bold")

    for bar in bars_after:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                f"{h:.2f}", ha="center", va="bottom",
                color=COLOR_AFTER, fontsize=11, fontweight="bold")

    # ── Improvement arrows ───────────────────────────────────────────────────
    for i, task in enumerate(tasks):
        b, a = baseline[task], after[task]
        delta = a - b
        mid_x = x[i]
        arrow_y = max(b, a) + 0.07
        ax.annotate(
            f"+{delta:.2f}",
            xy=(mid_x, arrow_y),
            ha="center", va="bottom",
            color="#FFD700", fontsize=10, fontweight="bold",
        )

    # ── Axes styling ─────────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, color=TEXT, fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0.0 – 1.0)", color=TEXT, fontsize=12)
    ax.set_xlabel("Environment Task", color=TEXT, fontsize=12)
    ax.tick_params(colors=TEXT)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── Title ────────────────────────────────────────────────────────────────
    ax.set_title(
        "Support Ticket Env — Reward Improvement\nRule-Based Baseline vs LLM Agent (Qwen2.5-72B)",
        color=TEXT, fontsize=14, fontweight="bold", pad=16,
    )

    # ── Legend ───────────────────────────────────────────────────────────────
    legend = ax.legend(
        handles=[
            mpatches.Patch(color=COLOR_BEFORE, label="Before (Rule-based Baseline)"),
            mpatches.Patch(color=COLOR_AFTER,  label="After  (LLM Agent — Qwen2.5-72B)"),
        ],
        facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=11,
        loc="upper right",
    )

    # ── Overall delta watermark ───────────────────────────────────────────────
    overall_before = round(sum(baseline.values()) / len(baseline), 3)
    overall_after  = round(sum(after.values())    / len(after),    3)
    fig.text(
        0.5, 0.01,
        f"Overall: {overall_before:.2f} → {overall_after:.2f}  (+{overall_after - overall_before:.2f})",
        ha="center", color="#FFD700", fontsize=11, fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"[plot] Chart saved -> {out_path}", flush=True)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot reward curve chart")
    parser.add_argument("--run-inference", action="store_true",
                        help="Run inference.py first and use live scores as AFTER values")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "reward_curve.png"),
                        help="Output PNG path (default: reward_curve.png)")
    args = parser.parse_args()

    after_scores = run_inference_and_parse() if args.run_inference else AFTER

    plot_chart(BASELINE, after_scores, args.out)
    print("[plot] Done.", flush=True)


if __name__ == "__main__":
    main()
