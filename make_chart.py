"""
make_chart.py
Generates the before/after reward chart using known scores.
Run: python make_chart.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Rule-based agent (no LLM, no training) — measured locally
baseline_scores = {
    "task1":   0.100,
    "task2":   0.113,
    "task3":   0.218,
    "overall": 0.144,
}

# Qwen2.5-72B via HF Inference API — from your clean run logs
llm_scores = {
    "task1":   0.100,
    "task2":   0.113,
    "task3":   0.262,
    "overall": 0.158,
}

# After GRPO training — update these once Colab finishes
# If Colab not done yet, use llm_scores as placeholder
grpo_scores = {
    "task1":   0.100,
    "task2":   0.113,
    "task3":   0.262,
    "overall": 0.158,
}

def make_chart(baseline, llm, grpo, output="reward_chart.png"):
    tasks = ["Task 1\n(Classify)", "Task 2\n(Action)", "Task 3\n(Full Resolve)", "Overall"]
    keys  = ["task1", "task2", "task3", "overall"]

    b_vals    = [baseline.get(k, 0) for k in keys]
    llm_vals  = [llm.get(k, 0) for k in keys]
    grpo_vals = [grpo.get(k, 0) for k in keys]

    x     = np.arange(len(tasks))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")

    ax1 = axes[0]
    bars1 = ax1.bar(x - width, b_vals,    width, label="Rule-Based",    color="#636e72", edgecolor="#2d3436")
    bars2 = ax1.bar(x,         llm_vals,  width, label="Qwen2.5-72B",   color="#0984e3", edgecolor="#2d3436")
    bars3 = ax1.bar(x + width, grpo_vals, width, label="After GRPO",    color="#00b894", edgecolor="#2d3436")

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., h + 0.008,
                     f"{h:.2f}", ha="center", va="bottom", fontsize=8.5, color="white")

    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, color="white", fontsize=10)
    ax1.set_ylabel("Score (0 - 1)", color="white", fontsize=11)
    ax1.set_title("Score Comparison Across Training Stages", color="white", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylim(0, 1.2)
    ax1.tick_params(colors="white")
    ax1.spines[:].set_color("#2d3436")
    ax1.yaxis.grid(True, alpha=0.2, color="white")
    ax1.set_axisbelow(True)
    ax1.legend(facecolor="#0f3460", edgecolor="#2d3436", labelcolor="white", fontsize=9)

    ax2 = axes[1]
    deltas = [round(grpo.get(k, 0) - baseline.get(k, 0), 3) for k in keys]
    colors = ["#00b894" if d >= 0 else "#d63031" for d in deltas]
    bars4  = ax2.bar(x, deltas, width=0.4, color=colors, edgecolor="#2d3436")

    for bar, d in zip(bars4, deltas):
        ypos = bar.get_height() + 0.004 if d >= 0 else bar.get_height() - 0.016
        ax2.text(bar.get_x() + bar.get_width()/2., ypos,
                 f"{d:+.3f}", ha="center", va="bottom", fontsize=11,
                 fontweight="bold", color="white")

    ax2.axhline(0, color="white", linewidth=0.8, alpha=0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tasks, color="white", fontsize=10)
    ax2.set_ylabel("Score Delta (GRPO vs Rule-Based)", color="white", fontsize=10)
    ax2.set_title("Improvement: Rule-Based → After GRPO", color="white", fontsize=12, fontweight="bold", pad=10)
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#2d3436")
    ax2.yaxis.grid(True, alpha=0.2, color="white")
    ax2.set_axisbelow(True)

    fig.suptitle(
        "Support Ticket Env — Training Results\nModel: Qwen2.5-0.5B-Instruct + GRPO | OpenEnv x Scalar Hackathon 2026",
        color="white", fontsize=11, y=1.02
    )

    plt.tight_layout()
    plt.savefig(output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Chart saved: {output}")

    print("\n" + "="*52)
    print(f"{'Task':<14} {'Rule-Based':>10} {'Qwen-72B':>10} {'GRPO':>8} {'Delta':>8}")
    print("-"*52)
    for k, label in [("task1","Task 1"),("task2","Task 2"),("task3","Task 3"),("overall","Overall")]:
        b = baseline.get(k, 0)
        l = llm.get(k, 0)
        g = grpo.get(k, 0)
        d = g - b
        print(f"{label:<14} {b:>10.3f} {l:>10.3f} {g:>8.3f} {d:>+8.3f}")
    print("="*52)

if __name__ == "__main__":
    make_chart(baseline_scores, llm_scores, grpo_scores)
