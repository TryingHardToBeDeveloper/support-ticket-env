"""
Realistic support ticket dataset with ground-truth labels.
Each ticket includes:
  - id
  - text   : customer message
  - category : ground-truth category
  - correct_action : best first action ("reply" | "escalate" | "close")
  - resolution_hint : ideal reply / close reason (used for reward scoring)
"""

TICKETS = [
    {
        "id": "T001",
        "text": "Hi, I was charged twice for my subscription this month. Please help!",
        "category": "billing",
        "correct_action": "reply",
        "resolution_hint": "apologize and initiate refund for duplicate charge",
    },
    {
        "id": "T002",
        "text": "I cannot log into my account. The password reset email never arrives.",
        "category": "account",
        "correct_action": "reply",
        "resolution_hint": "guide user to check spam folder and verify email address",
    },
    {
        "id": "T003",
        "text": "Your app crashes every time I try to upload a file larger than 10 MB.",
        "category": "technical",
        "correct_action": "escalate",
        "resolution_hint": "escalate to engineering team with crash details",
    },
    {
        "id": "T004",
        "text": "I'd like a full refund. I haven't used the service at all this month.",
        "category": "refund",
        "correct_action": "reply",
        "resolution_hint": "verify account activity and process refund per policy",
    },
    {
        "id": "T005",
        "text": "What are your business hours and do you have a phone number I can call?",
        "category": "general",
        "correct_action": "reply",
        "resolution_hint": "provide business hours and contact information",
    },
    {
        "id": "T006",
        "text": "My invoice shows a charge for a plan I never subscribed to.",
        "category": "billing",
        "correct_action": "escalate",
        "resolution_hint": "escalate potential fraudulent charge to billing team",
    },
    {
        "id": "T007",
        "text": "How do I cancel my subscription? I can't find the option anywhere.",
        "category": "account",
        "correct_action": "reply",
        "resolution_hint": "guide user to account settings > subscription > cancel",
    },
    {
        "id": "T008",
        "text": "The API is returning 500 errors intermittently for the past 2 hours.",
        "category": "technical",
        "correct_action": "escalate",
        "resolution_hint": "escalate to on-call engineering with timestamps",
    },
    {
        "id": "T009",
        "text": "Thank you! The issue has been resolved. You guys are awesome.",
        "category": "general",
        "correct_action": "close",
        "resolution_hint": "acknowledge and close the ticket",
    },
    {
        "id": "T010",
        "text": "I need an itemised invoice for my company's accounting department.",
        "category": "billing",
        "correct_action": "reply",
        "resolution_hint": "generate and send itemised invoice to customer email",
    },
]

TICKET_LOOKUP = {t["id"]: t for t in TICKETS}
