---
title: Support Ticket Env
emoji: 🎫
colorFrom: blue
colorTo: green
sdk: docker
tags:
  - openenv
pinned: false
---

# Customer Support Ticket Resolution Environment

> 🏆 **OpenEnv x Scalar Hackathon** — Theme **#3.1 Professional Tasks** | Sub-theme: **Scaler AI Labs — Multi-App RL Environment for Enterprise Workflows**

A real-world [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment where an AI agent acts as a customer support executive, triaging and resolving incoming tickets through a remote, partially observable API.

## Overview

Customer support triage is one of the most common real-world tasks for AI agents in enterprise settings. Every company handles thousands of tickets daily. Getting the classification wrong routes the ticket to the wrong team. Choosing the wrong action has direct business impact. This environment trains agents to handle exactly this challenge — with real tool interaction, dynamic state, and a multi-step reward structure that resists shortcuts.

## Quick Start

```python
from support_ticket_env import SupportAction, SupportTicketEnv

with SupportTicketEnv(base_url="https://algocore-support-ticket-env.hf.space").sync() as env:
    # Task 1 - Classify a ticket
    result = env.reset(task_id=1, seed=42)
    print(result.observation.ticket_text)

    result = env.step(SupportAction(action_type="classify", category="billing"))
    print(result.reward)  # 1.0 if correct
```

For protected deployments, pass `api_key="..."` to `SupportTicketEnv` or set
`SUPPORT_ENV_API_KEY` in the client process.

## Tasks

| Task | Difficulty | Description | Score Range |
|------|-----------|-------------|-------------|
| Task 1 | Easy | Classify ticket into correct category | 0.0 - 1.0 |
| Task 2 | Medium | Classify then choose correct action | 0.0 - 1.0 |
| Task 3 | Hard | Resolve a full queue of 3 tickets | 0.0 - 1.0 episode return |

## Action Space

Actions are `SupportAction` Pydantic objects:

| Field | Type | Required | Values |
|-------|------|----------|--------|
| `action_type` | str | always | `classify` / `reply` / `escalate` / `close` |
| `category` | str | for classify | `billing` / `technical` / `account` / `general` / `refund` |
| `reply_text` | str | for reply | free text |
| `reason` | str | optional | free text |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `ticket_id` | str | Unique ticket ID |
| `ticket_text` | str | Customer message |
| `task_id` | int | 1, 2, or 3 |
| `current_category` | str | Category assigned so far |
| `resolved` | bool | Whether ticket is resolved |
| `step_count` | int | Steps taken this episode |
| `feedback` | str | Human-readable feedback |
| `reward` | float | Reward signal |
| `done` | bool | Episode finished |

## Reward Function

Rewards provide partial progress signals throughout the trajectory:

- **Task 1:** 1.0 for correct category, 0.0 for wrong
- **Task 2:** 1.0 correct action, 0.5 defensible alternative, 0.3 classification only
- **Task 3:** Each ticket contributes up to one third of the episode return: 0.20 classification + 0.45 action + 0.25 response quality + 0.10 efficiency
- **Penalty:** -0.05 per step over 10 (loop deterrent)

Reply quality uses a private, per-ticket rubric plus coherence, specificity, length,
and lexical-diversity checks. Exact rubric copies and simple keyword repetition receive
no reply-quality credit.

## Project Structure

```
support_ticket_env/
├── __init__.py               # Package exports
├── models.py                 # SupportAction, SupportObservation, SupportState
├── tickets.py                # Public taxonomy only (no evaluator answers)
├── graders.py                # Reward functions
├── client.py                 # EnvClient subclass
├── baseline.py               # Baseline inference script
├── get_baseline.py           # Fetch & save baseline results
├── gradio_ui.py              # Interactive Gradio playground UI
├── make_chart.py             # Plot training reward curves
├── plot_results.py           # Visualise evaluation results
├── grpo_results.png          # GRPO training results chart
├── reward_chart.png          # Reward curve chart
├── openenv.yaml              # Environment metadata
├── Dockerfile                # Container definition
├── train_sft.ipynb           # Step 1: SFT pre-training notebook
├── train_grpo.ipynb          # Step 2: GRPO fine-tuning notebook
└── server/
    ├── app.py                # FastAPI entry point (+ Gradio UI mounted at /playground)
    ├── security.py           # Authentication, throttling, security headers
    ├── ticket_bank.py        # Server-only tickets, labels, and private rubrics
    ├── support_environment.py # Environment logic
    └── requirements.txt      # Server dependencies
```

## Setup

```bash
# Install the client
pip install -e .

# Install server and development tooling
pip install -e ".[server,dev]"

# Run locally
$env:SUPPORT_ENV_API_KEY="replace-with-a-long-random-secret"  # PowerShell
$env:SUPPORT_ENV_SEED_SALT="replace-with-an-independent-random-secret"
$env:SUPPORT_ENV_MODE="production"
uv run uvicorn server.app:app --host 0.0.0.0 --port 7860

# Or via Docker
docker build -t support-ticket-env .
docker run -p 7860:7860 -e SUPPORT_ENV_API_KEY="..." -e SUPPORT_ENV_SEED_SALT="..." support-ticket-env

# Run tests
python -m pytest
```

> 🎮 **Playground UI** available at `http://localhost:7860/playground` once the server is running.
> It is enabled by default only in development. Production deployments can opt in
> with `SUPPORT_ENV_ENABLE_PLAYGROUND=true` behind an authenticated reverse proxy.

## 📈 Training Results (GRPO) — Evidence of Improvement

Fine-tuned `Qwen2.5-0.5B-Instruct` using **2-stage training** (SFT pre-training → GRPO) via HuggingFace TRL over **700+ steps** on the live environment API:

![GRPO Training Results](https://raw.githubusercontent.com/TryingHardToBeDeveloper/support-ticket-env/main/grpo_results.png)

| Task | Before GRPO | After GRPO | Improvement |
|------|-------------|------------|-------------|
| Task 1 - Classification | 0.67 | **1.00** | +49% 🚀 |
| Task 2 - Action Selection | 0.12 | **0.48** | +300% 🚀 |
| Task 3 - Full Resolution | 0.08 | **0.23** | +187% 🚀 |
| **Overall** | **0.29** | **0.57** | **+96% 🚀** |

## Baseline Scores

Measured with `gpt-4o-mini`, seeds `[42, 7, 123]`:

| Task | Avg Score |
|------|-----------|
| Task 1 - Classification | 0.87 |
| Task 2 - Action Selection | 0.71 |
| Task 3 - Full Resolution | 0.58 |
| **Overall** | **0.72** |

## 🎯 Why This Fits Theme 3.1 — Professional Tasks

> *"Real interaction with tools, APIs, or dynamic systems where the model does real hard work instead of exploiting shortcuts"*

- ✅ **Live FastAPI environment** — agent interacts with a real stateful API, not a simulation
- ✅ **Explicit trust boundary** — the client wheel excludes labels and evaluator rubrics; scored agents must use the remote API
- ✅ **Adversarial regression coverage** — tests prevent label leaks, rubric copying, keyword stuffing, invalid tasks, and post-terminal actions
- ✅ **Persistent world state** — ticket queue, classification state, and resolution state tracked across steps
- ✅ **Multi-step causal reasoning** — classify → choose action → craft reply → resolve, all causally linked
- ✅ **Enterprise workflow complexity** — billing, technical, account, general, refund categories with real business rules
- ✅ **Scaler AI Labs sub-theme** — demonstrates complex enterprise workflows and business rule nuances in an RL environment

## Links

- **HuggingFace Space:** https://huggingface.co/spaces/AlgoCore/support-ticket-env
- **GitHub:** https://github.com/TryingHardToBeDeveloper/support-ticket-env
- **OpenEnv Docs:** https://meta-pytorch.org/OpenEnv/

## Security model

The public repository necessarily exposes the reference dataset to repository
readers. Meaningful scored evaluation therefore requires a held-out private ticket
bank in deployment and an agent process that can access only the remote API and
client wheel. Fixed public seeds are for reproducible demonstrations, not secure
leaderboard evaluation.

Production mode fails closed unless `SUPPORT_ENV_API_KEY` and
`SUPPORT_ENV_SEED_SALT` are set. API routes accept
the key via `X-API-Key` or a Bearer token and are protected by a configurable
per-client rate limit (`SUPPORT_ENV_RATE_LIMIT`, default 120 requests/minute). See
`SECURITY.md` for the deployment boundary and reporting process.

## License

MIT
