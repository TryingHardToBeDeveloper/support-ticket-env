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
    print("gradio not installed.")
    sys.exit(1)

from support_ticket_env.server.support_environment import SupportTicketEnvironment
from support_ticket_env.models import SupportAction

_env = SupportTicketEnvironment()
_current_obs = None
_history = []

def do_reset(task_id, seed):
    global _current_obs, _history
    _history = []
    obs = _env.reset(task_id=int(task_id), seed=int(seed))
    _current_obs = obs
    state = _env.state
    info = f"TICKET: {obs.ticket_text}\n\nFeedback: {obs.feedback}"
    return info, "Episode started! Now classify the ticket.", ""

def do_step(action_type, category, reply_text, reason):
    global _current_obs
    if _current_obs is None:
        return "Please click Reset first!", "", ""
    kwargs = {"action_type": action_type}
    if action_type == "classify" and category:
        kwargs["category"] = category
    if action_type == "reply" and reply_text:
        kwargs["reply_text"] = reply_text
    if reason:
        kwargs["reason"] = reason
    try:
        action = SupportAction(**kwargs)
        obs = _env.step(action)
        _current_obs = obs
        _history.append(f"Step: {action_type}/{category or ''} -> Reward: {obs.reward:.2f}")
        result = f"Feedback: {obs.feedback}\nReward: {obs.reward:.2f}\nDone: {obs.done}"
        history = "\n".join(_history)
        ticket = f"TICKET: {obs.ticket_text}"
        return ticket, result, history
    except Exception as e:
        return _current_obs.ticket_text, f"Error: {e}", ""

def do_state():
    state = _env.state
    return json.dumps({
        "task_id": state.task_id,
        "ticket_id": state.ticket_id,
        "correct_category": state.correct_category,
        "correct_action": state.correct_action,
        "classified": state.classified,
        "resolved": state.resolved,
        "step_count": state.step_count,
        "total_reward": state.total_reward,
    }, indent=2)

with gr.Blocks(title="Support Ticket Environment") as demo:
    gr.Markdown("# Customer Support Ticket Resolution Environment")
    gr.Markdown("Train AI agents to handle customer support tickets.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Setup")
            task_radio = gr.Radio(
                choices=[1, 2, 3],
                value=1,
                label="Task (1=Easy, 2=Medium, 3=Hard)"
            )
            seed_box = gr.Number(value=42, label="Seed")
            reset_btn = gr.Button("RESET", variant="primary")

            gr.Markdown("## Action")
            action_radio = gr.Radio(
                choices=["classify", "reply", "escalate", "close"],
                value="classify",
                label="Action Type"
            )
            category_radio = gr.Radio(
                choices=["billing", "technical", "account", "general", "refund"],
                value=None,
                label="Category (select for classify)"
            )
            reply_box = gr.Textbox(label="Reply Text (for reply action)", lines=2)
            reason_box = gr.Textbox(label="Reason (optional)")
            step_btn = gr.Button("STEP", variant="secondary")
            state_btn = gr.Button("GET STATE")

        with gr.Column(scale=2):
            gr.Markdown("## Current Ticket")
            ticket_box = gr.Textbox(
                label="Ticket",
                value="Click RESET to start",
                lines=3,
                interactive=False
            )
            gr.Markdown("## Result")
            result_box = gr.Textbox(
                label="Last Step Result",
                lines=4,
                interactive=False
            )
            gr.Markdown("## History")
            history_box = gr.Textbox(
                label="Action History",
                lines=5,
                interactive=False
            )
            gr.Markdown("## State")
            state_box = gr.Code(language="json", label="Environment State")

    reset_btn.click(
        do_reset,
        inputs=[task_radio, seed_box],
        outputs=[ticket_box, result_box, history_box]
    )
    step_btn.click(
        do_step,
        inputs=[action_radio, category_radio, reply_box, reason_box],
        outputs=[ticket_box, result_box, history_box]
    )
    state_btn.click(
        do_state,
        inputs=[],
        outputs=[state_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)