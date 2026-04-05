"""
gradio_ui.py - Professional UI for Support Ticket Resolution Environment
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

_env = SupportTicketEnvironment()
_current_obs = None
_history = []
_total_reward = 0.0
_episode_steps = 0

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #111827;
    --bg-card: #1a2234;
    --bg-hover: #1e2940;
    --accent-blue: #3b82f6;
    --accent-green: #10b981;
    --accent-orange: #f59e0b;
    --accent-red: #ef4444;
    --accent-purple: #8b5cf6;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #475569;
    --border: #1e3a5f;
    --border-bright: #2563eb;
    --glow-blue: 0 0 20px rgba(59, 130, 246, 0.3);
    --glow-green: 0 0 20px rgba(16, 185, 129, 0.3);
}

* { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 0 !important;
}

/* Header Banner */
.header-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border-bottom: 1px solid var(--border);
    padding: 32px 40px;
    position: relative;
    overflow: hidden;
}

.header-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center, rgba(59,130,246,0.05) 0%, transparent 70%);
    animation: pulse 4s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 1; }
}

.header-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #10b981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}

.header-subtitle {
    color: var(--text-secondary);
    font-size: 0.9rem;
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
}

.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    margin-right: 6px;
    margin-top: 8px;
}

.badge-blue { background: rgba(59,130,246,0.15); color: #60a5fa; border: 1px solid rgba(59,130,246,0.3); }
.badge-green { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
.badge-purple { background: rgba(139,92,246,0.15); color: #a78bfa; border: 1px solid rgba(139,92,246,0.3); }
.badge-orange { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }

/* Stat Cards */
.stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    padding: 20px 40px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
}

.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    transition: all 0.2s;
}

.stat-card:hover {
    border-color: var(--accent-blue);
    box-shadow: var(--glow-blue);
    transform: translateY(-2px);
}

.stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent-blue);
}

.stat-label {
    font-size: 0.75rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}

/* Panel styles */
.panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 12px;
}

/* Section boxes */
.gr-column {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 20px !important;
}

.gr-row {
    gap: 16px !important;
    padding: 20px 40px !important;
}

.panel-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--accent-blue);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.panel-title::before {
    content: '';
    display: inline-block;
    width: 3px;
    height: 14px;
    background: var(--accent-blue);
    border-radius: 2px;
}

/* Ticket Display */
.ticket-card {
    background: linear-gradient(135deg, #0f1f3d, #162040);
    border: 1px solid var(--border-bright);
    border-radius: 10px;
    padding: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    line-height: 1.7;
    box-shadow: var(--glow-blue);
    position: relative;
    overflow: hidden;
}

.ticket-card::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
}

/* Reward bar */
.reward-bar-container {
    background: var(--bg-primary);
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
    margin-top: 8px;
}

.reward-bar {
    height: 100%;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-green));
    border-radius: 6px;
    transition: width 0.5s ease;
}

/* History log */
.history-log {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-secondary);
    line-height: 1.8;
    max-height: 200px;
    overflow-y: auto;
    padding: 12px;
    background: var(--bg-primary);
    border-radius: 8px;
    border: 1px solid var(--border);
}

.history-log::-webkit-scrollbar { width: 4px; }
.history-log::-webkit-scrollbar-track { background: var(--bg-primary); }
.history-log::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

.log-entry { padding: 3px 0; border-bottom: 1px solid rgba(30,58,95,0.3); }
.log-entry:last-child { border-bottom: none; }
.log-reward-good { color: var(--accent-green); }
.log-reward-bad { color: var(--accent-red); }
.log-step-num { color: var(--accent-blue); }

/* Buttons */
.btn-primary {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    border: 1px solid #3b82f6 !important;
    color: white !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    padding: 12px 20px !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(37,99,235,0.3) !important;
}

.btn-primary:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    box-shadow: 0 4px 25px rgba(59,130,246,0.5) !important;
    transform: translateY(-1px) !important;
}

.btn-secondary {
    background: linear-gradient(135deg, #064e3b, #065f46) !important;
    border: 1px solid #10b981 !important;
    color: #34d399 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 12px 20px !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}

.btn-state {
    background: linear-gradient(135deg, #3b1f6e, #4c1d95) !important;
    border: 1px solid #8b5cf6 !important;
    color: #a78bfa !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 12px 20px !important;
    border-radius: 8px !important;
}

/* Radio buttons */
.gr-radio label {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 8px 14px !important;
    color: var(--text-secondary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    transition: all 0.15s !important;
    cursor: pointer !important;
}

.gr-radio label:hover {
    border-color: var(--accent-blue) !important;
    color: var(--text-primary) !important;
    background: var(--bg-hover) !important;
}

/* Textbox */
.gr-textbox textarea, .gr-textbox input {
    background: var(--bg-primary) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    border-radius: 8px !important;
}

.gr-textbox textarea:focus, .gr-textbox input:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: var(--glow-blue) !important;
}

/* Slider */
.gr-slider input[type=range] {
    accent-color: var(--accent-blue) !important;
}

/* Code block */
.gr-code {
    background: var(--bg-primary) !important;
    border: 1px solid var(--border) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    border-radius: 8px !important;
}

/* Task difficulty indicators */
.task-easy { color: var(--accent-green); }
.task-medium { color: var(--accent-orange); }
.task-hard { color: var(--accent-red); }

/* Status indicator */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent-green);
    box-shadow: 0 0 6px var(--accent-green);
    animation: blink 2s ease-in-out infinite;
    margin-right: 6px;
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* Feedback box colors */
.feedback-success { color: var(--accent-green) !important; }
.feedback-error { color: var(--accent-red) !important; }
.feedback-info { color: var(--accent-blue) !important; }

/* Main layout */
.main-content {
    padding: 20px 40px;
    display: grid;
    grid-template-columns: 380px 1fr;
    gap: 20px;
}

/* Tabs */
.gr-tab-nav button {
    background: var(--bg-card) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    border-radius: 6px 6px 0 0 !important;
}

.gr-tab-nav button.selected {
    background: var(--bg-hover) !important;
    color: var(--accent-blue) !important;
    border-bottom-color: var(--bg-hover) !important;
}

/* Number input */
.gr-number input {
    background: var(--bg-primary) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 8px !important;
}

/* Markdown */
.gr-markdown {
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

.gr-markdown h3 {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--accent-blue) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.15em !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 8px !important;
    margin-bottom: 12px !important;
}

/* Checkbox */
.gr-checkbox label {
    color: var(--text-secondary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}
"""

HEADER_HTML = """
<div class="header-banner">
    <div style="position:relative;z-index:1;">
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:8px;">
            <span style="font-size:2.5rem;">🎫</span>
            <div>
                <h1 class="header-title">Support Ticket Resolution Environment</h1>
                <p class="header-subtitle">OpenEnv · Real-World AI Agent Training · Customer Support Triage</p>
            </div>
        </div>
        <div style="margin-top:12px;">
            <span class="badge badge-blue">🤖 OpenEnv Compatible</span>
            <span class="badge badge-green">✅ 3 Tasks</span>
            <span class="badge badge-purple">🐳 Docker Ready</span>
            <span class="badge badge-orange">⚡ Live API</span>
        </div>
    </div>
</div>
"""

TASK_INFO_HTML = """
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:12px 0;">
    <div class="stat-card" style="border-color:rgba(16,185,129,0.4);">
        <div style="font-size:1.2rem;margin-bottom:4px;">🟢</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#34d399;font-weight:600;">TASK 1 · EASY</div>
        <div style="font-size:0.75rem;color:#94a3b8;margin-top:4px;">Classify Ticket</div>
    </div>
    <div class="stat-card" style="border-color:rgba(245,158,11,0.4);">
        <div style="font-size:1.2rem;margin-bottom:4px;">🟡</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#fbbf24;font-weight:600;">TASK 2 · MEDIUM</div>
        <div style="font-size:0.75rem;color:#94a3b8;margin-top:4px;">Classify + Action</div>
    </div>
    <div class="stat-card" style="border-color:rgba(239,68,68,0.4);">
        <div style="font-size:1.2rem;margin-bottom:4px;">🔴</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#f87171;font-weight:600;">TASK 3 · HARD</div>
        <div style="font-size:0.75rem;color:#94a3b8;margin-top:4px;">Full Queue Resolution</div>
    </div>
</div>
"""

def format_ticket_html(obs):
    if obs is None:
        return "<div class='ticket-card' style='color:#475569;text-align:center;padding:40px;'>Click RESET to load a ticket</div>"
    category_color = {
        "billing": "#fbbf24", "technical": "#60a5fa",
        "account": "#a78bfa", "general": "#34d399", "refund": "#fb923c"
    }
    cat = obs.current_category
    cat_badge = f"<span style='background:rgba(59,130,246,0.2);color:{category_color.get(cat,'#94a3b8')};padding:2px 10px;border-radius:20px;font-size:0.72rem;font-weight:600;border:1px solid rgba(59,130,246,0.3);'>{cat.upper() if cat else 'UNCLASSIFIED'}</span>" if cat else "<span style='background:rgba(71,85,105,0.3);color:#64748b;padding:2px 10px;border-radius:20px;font-size:0.72rem;font-weight:600;'>UNCLASSIFIED</span>"
    resolved_badge = "<span style='color:#34d399;'>✅ RESOLVED</span>" if obs.resolved else "<span style='color:#f59e0b;'>⏳ OPEN</span>"
    return f"""
    <div class='ticket-card'>
        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;'>
            <span style='color:#3b82f6;font-weight:700;font-size:0.9rem;'>#{obs.ticket_id}</span>
            <div style='display:flex;gap:8px;align-items:center;'>{cat_badge} {resolved_badge}</div>
        </div>
        <div style='color:#e2e8f0;font-size:0.88rem;line-height:1.7;font-family:"Inter",sans-serif;'>
            "{obs.ticket_text}"
        </div>
        <div style='margin-top:12px;padding-top:12px;border-top:1px solid rgba(30,58,95,0.5);display:flex;gap:20px;font-size:0.72rem;color:#64748b;'>
            <span>📊 Task <strong style='color:#60a5fa;'>{obs.task_id}</strong></span>
            <span>👣 Step <strong style='color:#60a5fa;'>{obs.step_count}</strong></span>
            <span>🏆 Score <strong style='color:#34d399;'>{obs.score:.3f}</strong></span>
        </div>
    </div>
    """

def format_feedback_html(feedback, reward):
    if not feedback:
        return ""
    icon = "✅" if reward and reward > 0.5 else "⚠️" if reward and reward > 0 else "❌"
    color = "#34d399" if reward and reward > 0.5 else "#fbbf24" if reward and reward > 0 else "#f87171"
    reward_pct = int((reward or 0) * 100)
    return f"""
    <div style='background:var(--bg-card);border:1px solid {color}33;border-radius:8px;padding:14px;margin-top:8px;'>
        <div style='color:{color};font-weight:600;font-size:0.85rem;margin-bottom:6px;'>{icon} {feedback}</div>
        <div style='background:#0a0e1a;border-radius:4px;height:6px;overflow:hidden;'>
            <div style='width:{reward_pct}%;height:100%;background:linear-gradient(90deg,#3b82f6,#10b981);border-radius:4px;transition:width 0.5s;'></div>
        </div>
        <div style='color:#64748b;font-size:0.72rem;margin-top:4px;font-family:"JetBrains Mono",monospace;'>reward: {reward:.3f} / 1.000</div>
    </div>
    """

def format_history_html(history):
    if not history:
        return "<div style='color:#475569;font-family:\"JetBrains Mono\",monospace;font-size:0.78rem;padding:8px;'>No actions yet...</div>"
    entries = ""
    for i, h in enumerate(history, 1):
        r = h.get("reward", 0) or 0
        color = "#34d399" if r > 0.5 else "#fbbf24" if r > 0 else "#f87171"
        entries += f"""
        <div class='log-entry'>
            <span style='color:#3b82f6;'>step {i:02d}</span>
            <span style='color:#94a3b8;'> › </span>
            <span style='color:#e2e8f0;'>{h.get("action","?")}</span>
            <span style='color:#475569;'> → </span>
            <span style='color:{color};font-weight:600;'>{r:+.3f}</span>
        </div>"""
    return f"<div class='history-log'>{entries}</div>"

def do_reset(task_id, seed):
    global _current_obs, _history, _total_reward, _episode_steps
    _history = []
    _total_reward = 0.0
    _episode_steps = 0
    obs = _env.reset(task_id=int(task_id), seed=int(seed))
    _current_obs = obs
    ticket_html = format_ticket_html(obs)
    feedback_html = format_feedback_html(obs.feedback, 0.0)
    history_html = format_history_html(_history)
    stats = f"<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:10px;'><div class='stat-card'><div class='stat-value' style='color:#3b82f6;'>0</div><div class='stat-label'>Steps</div></div><div class='stat-card'><div class='stat-value' style='color:#10b981;'>0.000</div><div class='stat-label'>Total Reward</div></div><div class='stat-card'><div class='stat-value' style='color:#8b5cf6;'>{int(task_id)}</div><div class='stat-label'>Task ID</div></div><div class='stat-card'><div class='stat-value' style='color:#f59e0b;'>OPEN</div><div class='stat-label'>Status</div></div></div>"
    return ticket_html, feedback_html, history_html, stats, gr.update(value=False)

def do_step(action_type, category, reply_text, reason):
    global _current_obs, _history, _total_reward, _episode_steps
    if _current_obs is None:
        return (
            "<div class='ticket-card' style='color:#ef4444;'>⚠️ Please click RESET first!</div>",
            "", format_history_html(_history), "", gr.update(value=False)
        )
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
        reward = obs.reward or 0.0
        _total_reward += reward
        _episode_steps += 1
        action_str = f"{action_type}" + (f"/{category}" if category and action_type == "classify" else "")
        _history.append({"action": action_str, "reward": reward, "feedback": obs.feedback})
        ticket_html = format_ticket_html(obs)
        feedback_html = format_feedback_html(obs.feedback, reward)
        history_html = format_history_html(_history)
        status = "DONE ✅" if obs.done else "OPEN ⏳"
        status_color = "#34d399" if obs.done else "#f59e0b"
        stats = f"<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:10px;'><div class='stat-card'><div class='stat-value' style='color:#3b82f6;'>{_episode_steps}</div><div class='stat-label'>Steps</div></div><div class='stat-card'><div class='stat-value' style='color:#10b981;'>{_total_reward:.3f}</div><div class='stat-label'>Total Reward</div></div><div class='stat-card'><div class='stat-value' style='color:#8b5cf6;'>{obs.task_id}</div><div class='stat-label'>Task ID</div></div><div class='stat-card'><div class='stat-value' style='color:{status_color};font-size:1rem;'>{status}</div><div class='stat-label'>Status</div></div></div>"
        return ticket_html, feedback_html, history_html, stats, gr.update(value=obs.done)
    except Exception as e:
        return (
            format_ticket_html(_current_obs),
            f"<div style='color:#ef4444;padding:10px;background:rgba(239,68,68,0.1);border-radius:8px;font-family:\"JetBrains Mono\",monospace;font-size:0.8rem;'>❌ Error: {str(e)}</div>",
            format_history_html(_history), "", gr.update(value=False)
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

with gr.Blocks(css=CUSTOM_CSS, title="Support Ticket Environment") as demo:

    gr.HTML(HEADER_HTML)

    with gr.Row():
        with gr.Column(scale=1, min_width=360):
            gr.HTML("<div style='padding:16px 0 8px;'><div class='panel-title'>⚙️ Episode Configuration</div></div>")

            gr.HTML(TASK_INFO_HTML)

            task_radio = gr.Radio(
                choices=[1, 2, 3],
                value=1,
                label="Select Task",
            )
            seed_input = gr.Number(value=42, label="Random Seed", precision=0)

            reset_btn = gr.Button("🔄 RESET EPISODE", elem_classes=["btn-primary"])

            gr.HTML("<div style='height:1px;background:var(--border);margin:16px 0;'></div>")
            gr.HTML("<div class='panel-title'>🎬 Take Action</div>")

            action_radio = gr.Radio(
                choices=["classify", "reply", "escalate", "close"],
                value="classify",
                label="Action Type",
            )
            category_radio = gr.Radio(
                choices=["billing", "technical", "account", "general", "refund"],
                value=None,
                label="Category (for classify action)",
            )
            reply_input = gr.Textbox(
                label="Reply Text (for reply action)",
                placeholder="Type your reply to the customer...",
                lines=2,
            )
            reason_input = gr.Textbox(
                label="Reason (optional)",
                placeholder="Justification for your action...",
                lines=1,
            )

            with gr.Row():
                step_btn = gr.Button("▶ STEP", elem_classes=["btn-secondary"])
                state_btn = gr.Button("🔍 STATE", elem_classes=["btn-state"])

            done_check = gr.Checkbox(label="Episode Complete", interactive=False)

        with gr.Column(scale=2):
            gr.HTML("<div style='padding:16px 0 8px;'><div class='panel-title'>📬 Current Ticket</div></div>")
            ticket_display = gr.HTML(format_ticket_html(None))

            gr.HTML("<div class='panel-title' style='margin-top:12px;'>💬 Last Action Result</div>")
            feedback_display = gr.HTML("")

            gr.HTML("<div class='panel-title' style='margin-top:12px;'>📊 Episode Stats</div>")
            stats_display = gr.HTML(
                "<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:10px;'>"
                "<div class='stat-card'><div class='stat-value' style='color:#3b82f6;'>—</div><div class='stat-label'>Steps</div></div>"
                "<div class='stat-card'><div class='stat-value' style='color:#10b981;'>—</div><div class='stat-label'>Total Reward</div></div>"
                "<div class='stat-card'><div class='stat-value' style='color:#8b5cf6;'>—</div><div class='stat-label'>Task ID</div></div>"
                "<div class='stat-card'><div class='stat-value' style='color:#f59e0b;'>—</div><div class='stat-label'>Status</div></div>"
                "</div>"
            )

            gr.HTML("<div class='panel-title' style='margin-top:12px;'>📜 Action History</div>")
            history_display = gr.HTML(format_history_html([]))

            gr.HTML("<div class='panel-title' style='margin-top:12px;'>🗂️ Raw Environment State</div>")
            state_display = gr.Code(language="json", label="state()")

    gr.HTML("""
    <div style='padding:20px 40px;border-top:1px solid var(--border);margin-top:20px;background:var(--bg-secondary);'>
        <div style='display:flex;justify-content:space-between;align-items:center;'>
            <div style='font-family:"JetBrains Mono",monospace;font-size:0.72rem;color:#475569;'>
                🎫 Support Ticket Resolution Environment · OpenEnv Compatible · MIT License
            </div>
            <div style='display:flex;gap:16px;font-size:0.72rem;'>
                <a href='https://huggingface.co/spaces/AlgoCore/support-ticket-env' style='color:#3b82f6;text-decoration:none;font-family:"JetBrains Mono",monospace;'>🤗 HF Space</a>
                <a href='https://github.com/TryingHardToBeDeveloper/support-ticket-env' style='color:#3b82f6;text-decoration:none;font-family:"JetBrains Mono",monospace;'>🐙 GitHub</a>
                <a href='https://github.com/meta-pytorch/OpenEnv' style='color:#3b82f6;text-decoration:none;font-family:"JetBrains Mono",monospace;'>📦 OpenEnv</a>
            </div>
        </div>
    </div>
    """)

    reset_btn.click(
        do_reset,
        inputs=[task_radio, seed_input],
        outputs=[ticket_display, feedback_display, history_display, stats_display, done_check],
    )
    step_btn.click(
        do_step,
        inputs=[action_radio, category_radio, reply_input, reason_input],
        outputs=[ticket_display, feedback_display, history_display, stats_display, done_check],
    )
    state_btn.click(
        do_state,
        inputs=[],
        outputs=[state_display],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
