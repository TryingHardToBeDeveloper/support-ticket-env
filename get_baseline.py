import os, sys
sys.path.insert(0, r'C:\Users\Admin\OneDrive\Desktop\OpenEnv Hacathon\support_ticket_env')

from support_ticket_env.server.support_environment import SupportTicketEnvironment
from support_ticket_env.models import SupportAction

CATEGORY_KEYWORDS = {
    "billing":   ["charge", "invoice", "payment", "bill", "refund", "subscription", "price", "cost", "fee", "money"],
    "technical": ["error", "bug", "crash", "not working", "broken", "issue", "problem", "fail", "500", "api"],
    "account":   ["login", "password", "account", "access", "sign in", "email", "username", "cancel"],
    "refund":    ["refund", "return", "money back", "reimburse", "cancel order"],
    "general":   ["hours", "contact", "phone", "help", "question", "info", "support"],
}

def rule_based(obs):
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
        return {"action_type": "escalate", "reason": "needs engineering"}
    elif cat == "general":
        return {"action_type": "close", "reason": "resolved"}
    else:
        return {"action_type": "reply", "reply_text": f"Thank you for contacting us about your {cat} issue."}

SEEDS = [42, 7, 123]
MAX_STEPS = 10
results = {}

for task_id in [1, 2, 3]:
    scores = []
    for seed in SEEDS:
        env = SupportTicketEnvironment()
        obs = env.reset(task_id=task_id, seed=seed)
        rewards = []
        for _ in range(MAX_STEPS):
            if obs.done:
                break
            action_dict = rule_based(obs)
            try:
                action = SupportAction(**action_dict)
                obs = env.step(action)
                rewards.append(obs.reward or 0.0)
            except:
                rewards.append(0.0)
            if obs.done:
                break
        score = round(min(max(sum(rewards) / MAX_STEPS, 0.0), 1.0), 3)
        scores.append(score)
        print(f"  Task {task_id} seed={seed}: {score:.3f}")
    avg = round(sum(scores) / len(scores), 3)
    results["task" + str(task_id)] = avg
    print(f"  Task {task_id} avg: {avg:.3f}")

overall = round(sum(results.values()) / 3, 3)
results["overall"] = overall
print(f"Overall rule-based avg: {overall:.3f}")
print("Rule-based scores:", results)
