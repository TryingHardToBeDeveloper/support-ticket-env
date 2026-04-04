"""
gradio_ui.py — Interactive Gradio web interface for the Support Ticket Environment.

Allows human exploration and debugging without writing code.
Launched automatically when ENABLE_WEB_INTERFACE=true or run directly.

Usage:
    python support_ticket_env/gradio_ui.py
"""

import json
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openenv_stub")
sys.path.insert(0, STUB)
sys.path.insert(0, ROOT)

try:
    import gradio as gr
except ImportError:
    print("gradio not installed. Run: pip install gradio")
    sys.exit(1)

from support_ticket_env.server.support_environment import SupportTicketEnvironment
from support_ticket_env.models import SupportAction

# ─── shared env instance ────────────────────────────────────────
_env = SupportTicketEnvironment()
_history: list[dict] = []
_current_obs = None


# ─── helpers ────────────────────────────────────────────────────

def _format_history() -> str:
    if not _history:
        return "_No actions yet._"
    lines = []
    for i, h in enumerate(_history, 1):
        reward_str = f"{h['reward']:+.3f}" if h["reward"] is not None else "—"
        lines.append(
            f"**Step {i}** | `{h['action']}` → reward `{reward_str}`\n"
            f"> {h['feedback']}"
        )
    return "\n\n".join(lines)


def _obs_to_display(obs) -> tuple[str, str, str]:
    """Return (ticket_box, status_box, score_box)."""
    ticket = f"**[{obs.ticket_id}]** {obs.ticket_text}"
    status = (
        f"Task **{obs.task_id}** | Step **{obs.step_count}** | "
        f"Category: `{obs.current_category or 'unknown'}` | "
        f"Resolved: {'✅' if obs.resolved else '⬜'}"
    )
    score = f"Last step score: **{obs.score:.3f}** | reward: **{obs.reward or 0.0:+.3f}**"
    return ticket, status, score


# ─── UI callbacks ────────────────────────────────────────────────

def do_reset(task_id: int, seed: int):
    global _history, _current_obs
    _history = []
    obs = _env.reset(task_id=task_id, seed=seed)
    _current_obs = obs
    ticket, status, score = _obs_to_display(obs)
    return (
        ticket, status, score,
        _format_history(),
        gr.update(interactive=True),
        obs.feedback,
        gr.update(value=False),  # done flag
    )


def do_step(action_type: str, category: str, reply_text: str, reason: str):
    global _current_obs
    if _current_obs is None:
        return (
            "⚠️ Please reset the environment first.",
            "", "", _format_history(), "", gr.update(value=False),
        )

    # Build action
    kwargs = {"action_type": action_type}
    if action_type == "classify" and category:
        kwargs["category"] = category
    if action_type == "reply" and reply_text:
        kwargs["reply_text"] = reply_text
    if reason:
        kwargs["reason"] = reason

    try:
        action = SupportAction(**kwargs)
    except Exception as e:
        return (
            _current_obs.ticket_text,
            f"❌ Invalid action: {e}", "",
            _format_history(), "", gr.update(value=False),
        )

    obs = _env.step(action)
    _current_obs = obs

    _history.append({
        "action": f"{action_type}" + (f"/{category}" if category and action_type == "classify" else ""),
        "reward": obs.reward,
        "feedback": obs.feedback,
    })

    ticket, status, score = _obs_to_display(obs)
    done_msg = "🏁 Episode finished!" if obs.done else ""
    return (
        ticket, status, score,
        _format_history(),
        obs.feedback,
        gr.update(value=obs.done),
    )


def do_state():
    state = _env.state
    return json.dumps({
        "episode_id": state.episode_id,
        "step_count": state.step_count,
        "task_id": state.task_id,
        "ticket_id": state.ticket_id,
        "correct_category": state.correct_category,
        "correct_action": state.correct_action,
        "classified": state.classified,
        "resolved": state.resolved,
        "total_reward": state.total_reward,
        "tickets_resolved": state.tickets_resolved,
        "tickets_total": state.tickets_total,
    }, indent=2)


# ─── UI layout ──────────────────────────────────────────────────

DESCRIPTION = """
# 🎫 Customer Support Ticket Resolution Environment

An **OpenEnv** environment for training AI agents to handle customer support tickets.

**Tasks:** 1 = Classify · 2 = Classify + Action · 3 = Full Queue Resolution
"""

with gr.Blocks(title="Support Ticket Env") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        # ── Left panel: controls ────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Episode Setup")
            task_slider = gr.Slider(1, 3, value=1, step=1, label="Task ID")
            seed_input  = gr.Number(value=42, label="Seed", precision=0)
            reset_btn   = gr.Button("🔄 Reset Episode", variant="primary")

            gr.Markdown("### 🎬 Take Action")
            action_type  = gr.Radio(
                ["classify", "reply", "escalate", "close"],
                value="classify", label="Action Type",
            )
            category_dd = gr.Dropdown(
                choices=["billing", "technical", "account", "general", "refund"],
                label="Category (for classify)",
                value=None,
                allow_custom_value=False,
            )
            reply_box  = gr.Textbox(label="Reply Text (for reply)", lines=3)
            reason_box = gr.Textbox(label="Reason (optional)")
            step_btn   = gr.Button("▶️ Step", variant="secondary")
            state_btn  = gr.Button("🔍 Show State")

        # ── Right panel: observation ────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 📬 Current Ticket")
            ticket_display  = gr.Markdown("_Reset to start._")
            status_display  = gr.Markdown("")
            score_display   = gr.Markdown("")
            feedback_box    = gr.Textbox(label="Last Feedback", interactive=False)
            done_checkbox   = gr.Checkbox(label="Episode Done", interactive=False)

            gr.Markdown("### 📜 Action History")
            history_display = gr.Markdown("_No actions yet._")

            gr.Markdown("### 🗂️ Raw State (JSON)")
            state_output = gr.Code(language="json", label="state()")

    # ── wire up ─────────────────────────────────────────────────
    reset_btn.click(
        do_reset,
        inputs=[task_slider, seed_input],
        outputs=[ticket_display, status_display, score_display,
                 history_display, step_btn, feedback_box, done_checkbox],
    )
    step_btn.click(
        do_step,
        inputs=[action_type, category_dd, reply_box, reason_box],
        outputs=[ticket_display, status_display, score_display,
                 history_display, feedback_box, done_checkbox],
    )
    state_btn.click(
        do_state, inputs=[], outputs=[state_output],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
