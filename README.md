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

A real-world OpenEnv environment where an AI agent learns to handle customer support tickets.

## Tasks
- Task 1 (Easy): Classify ticket into correct category
- Task 2 (Medium): Classify then choose correct action
- Task 3 (Hard): Resolve a full queue of 3 tickets

## Action Space
- action_type: classify / reply / escalate / close
- category: billing / technical / account / general / refund
- reply_text: free text

## Observation Space
- ticket_id, ticket_text, task_id, current_category, resolved, step_count, feedback, score

## Baseline Scores
- Task 1: 0.87
- Task 2: 0.71
- Task 3: 0.58
