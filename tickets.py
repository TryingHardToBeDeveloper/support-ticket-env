"""
Realistic support ticket dataset with ground-truth labels.
Each ticket includes:
  - id
  - text             : customer message
  - category         : ground-truth category
  - correct_action   : best first action ("reply" | "escalate" | "close")
  - resolution_hint  : specific resolution keywords used for reply quality scoring
                       (two-tier: category keywords @0.03, hint keywords @0.05)

Distribution: 10 per category x 5 categories = 50 tickets
Difficulty varies within each category from straightforward to complex/ambiguous.
"""

TICKETS = [
    # ── billing (12) ──────────────────────────────────────────────────────────
    {
        "id": "B001",
        "text": "I was charged twice for my subscription this month. Please help!",
        "category": "billing",
        "correct_action": "reply",
        "resolution_hint": "apologize for duplicate charge and initiate refund to original payment method within 3-5 days",
    },
    {
        "id": "B002",
        "text": "My invoice shows a charge for a plan I never subscribed to.",
        "category": "billing",
        "correct_action": "escalate",
        "resolution_hint": "escalate potential unauthorized plan charge to billing team for investigation and correction",
    },
    {
        "id": "B003",
        "text": "I need an itemised invoice for my company's accounting department.",
        "category": "billing",
        "correct_action": "reply",
        "resolution_hint": "generate itemised invoice with line-item breakdown and email to customer accounting address",
    },
    {
        "id": "B004",
        "text": "Why was I charged before my trial period ended?",
        "category": "billing",
        "correct_action": "reply",
        "resolution_hint": "verify trial end date in billing system and issue refund for premature charge before expiry",
    },
    {
        "id": "B005",
        "text": "I switched plans but was still billed at the old rate.",
        "category": "billing",
        "correct_action": "reply",
        "resolution_hint": "confirm plan switch date in system and issue prorated credit for overcharge at old rate",
    },
    {
        "id": "B006",
        "text": "My payment method was charged three times in one day.",
        "category": "billing",
        "correct_action": "escalate",
        "resolution_hint": "escalate triple charge incident to billing fraud team and freeze further charges pending review",
    },
    {
        "id": "B007",
        "text": "I cancelled my plan but the charge still appeared this month.",
        "category": "billing",
        "correct_action": "reply",
        "resolution_hint": "verify cancellation timestamp confirm post-cancel charge and process refund for final month",
    },
    {
        "id": "B008",
        "text": "Can you send me a receipt for my last payment?",
        "category": "billing",
        "correct_action": "reply",
        "resolution_hint": "locate last successful payment record and email PDF receipt to customer registered address",
    },
    {
        "id": "B009",
        "text": "I was charged in USD but I signed up for GBP billing.",
        "category": "billing",
        "correct_action": "reply",
        "resolution_hint": "identify currency mismatch at signup and issue credit note for exchange rate difference",
    },
    {
        "id": "B010",
        "text": "The discount code I applied is not reflected in my invoice.",
        "category": "billing",
        "correct_action": "reply",
        "resolution_hint": "locate discount code application log verify failure reason and apply credit to next invoice",
    },
    {
        "id": "B011",
        "text": "I need to update my billing address on the invoice.",
        "category": "billing",
        "correct_action": "reply",
        "resolution_hint": "update billing address in account settings and reissue corrected invoice for their records",
    },
    {
        "id": "B012",
        "text": "My credit card was charged even though a payment failed notification was sent.",
        "category": "billing",
        "correct_action": "escalate",
        "resolution_hint": "escalate ghost charge to payments team attach failed payment notification as evidence for review",
    },

    # ── account (10) ──────────────────────────────────────────────────────────
    {
        "id": "A001",
        "text": "I cannot log into my account. The password reset email never arrives.",
        "category": "account",
        "correct_action": "reply",
        "resolution_hint": "check spam folder verify registered email address resend password reset link account locked",
    },
    {
        "id": "A002",
        "text": "How do I cancel my subscription? I can't find the option anywhere.",
        "category": "account",
        "correct_action": "reply",
        "resolution_hint": "navigate account settings subscription tab locate cancel option confirm cancellation effective date",
    },
    {
        "id": "A003",
        "text": "I want to change my email address associated with the account.",
        "category": "account",
        "correct_action": "reply",
        "resolution_hint": "verify identity via security question update email address send confirmation to both old and new",
    },
    {
        "id": "A004",
        "text": "My account was locked after too many failed login attempts.",
        "category": "account",
        "correct_action": "reply",
        "resolution_hint": "unlock account after failed login attempts verify identity via backup code or support email",
    },
    {
        "id": "A005",
        "text": "I accidentally deleted my account. Can it be restored?",
        "category": "account",
        "correct_action": "reply",
        "resolution_hint": "check account deletion grace period restore from backup if within 30 days confirm data intact",
    },
    {
        "id": "A006",
        "text": "I need to transfer my account to a different email.",
        "category": "account",
        "correct_action": "reply",
        "resolution_hint": "verify ownership of both accounts initiate transfer request update billing and login credentials",
    },
    {
        "id": "A007",
        "text": "Two-factor authentication is not working for my account.",
        "category": "account",
        "correct_action": "reply",
        "resolution_hint": "verify 2FA device registration resync authenticator app or issue backup recovery codes immediately",
    },
    {
        "id": "A008",
        "text": "I can't find where to download my data for GDPR purposes.",
        "category": "account",
        "correct_action": "reply",
        "resolution_hint": "provide GDPR data export link in account privacy settings confirm 30-day download window",
    },
    {
        "id": "A009",
        "text": "My username was changed without my permission.",
        "category": "account",
        "correct_action": "escalate",
        "resolution_hint": "escalate unauthorized username change to security team flag for account compromise investigation",
    },
    {
        "id": "A010",
        "text": "I want to upgrade my account from free to premium.",
        "category": "account",
        "correct_action": "reply",
        "resolution_hint": "confirm current free plan limits explain premium features and provide upgrade link with pricing",
    },

    # ── technical (10) ────────────────────────────────────────────────────────
    {
        "id": "T001",
        "text": "Your app crashes every time I try to upload a file larger than 10 MB.",
        "category": "technical",
        "correct_action": "escalate",
        "resolution_hint": "escalate to engineering with file size limit crash reproduction steps and device logs attached",
    },
    {
        "id": "T002",
        "text": "The API is returning 500 errors intermittently for the past 2 hours.",
        "category": "technical",
        "correct_action": "escalate",
        "resolution_hint": "escalate API 500 errors to on-call engineering with timestamps error codes and affected endpoints",
    },
    {
        "id": "T003",
        "text": "The dashboard is completely blank after the latest update.",
        "category": "technical",
        "correct_action": "escalate",
        "resolution_hint": "escalate blank dashboard to engineering with browser version last working date and console errors",
    },
    {
        "id": "T004",
        "text": "Export to CSV is broken — it downloads an empty file.",
        "category": "technical",
        "correct_action": "escalate",
        "resolution_hint": "escalate empty CSV export bug to engineering with sample dataset and export configuration used",
    },
    {
        "id": "T005",
        "text": "Notifications are not being delivered to my email or phone.",
        "category": "technical",
        "correct_action": "escalate",
        "resolution_hint": "escalate notification delivery failure to infrastructure team check email provider and push config",
    },
    {
        "id": "T006",
        "text": "The mobile app freezes on the login screen on iOS 17.",
        "category": "technical",
        "correct_action": "escalate",
        "resolution_hint": "escalate iOS 17 freeze to mobile engineering with device model OS version and crash report",
    },
    {
        "id": "T007",
        "text": "Search functionality returns no results for any query.",
        "category": "technical",
        "correct_action": "escalate",
        "resolution_hint": "escalate search returning no results to engineering with query examples and index rebuild request",
    },
    {
        "id": "T008",
        "text": "Data sync between devices stopped working 3 days ago.",
        "category": "technical",
        "correct_action": "escalate",
        "resolution_hint": "escalate device sync failure to backend team with affected device IDs and last sync timestamp",
    },
    {
        "id": "T009",
        "text": "The webhook integration keeps timing out and losing events.",
        "category": "technical",
        "correct_action": "escalate",
        "resolution_hint": "escalate webhook timeout to integrations team with endpoint URL payload size and retry logs",
    },
    {
        "id": "T010",
        "text": "Browser extension throws a JavaScript error on every page load.",
        "category": "technical",
        "correct_action": "escalate",
        "resolution_hint": "escalate browser extension JavaScript error to frontend team with browser version and error stack",
    },

    # ── refund (8) ────────────────────────────────────────────────────────────
    {
        "id": "R001",
        "text": "I'd like a full refund. I haven't used the service at all this month.",
        "category": "refund",
        "correct_action": "reply",
        "resolution_hint": "confirm zero usage this billing period process full refund within 5-7 business days to original payment method",
    },
    {
        "id": "R002",
        "text": "I was double charged and need a refund for the extra payment.",
        "category": "refund",
        "correct_action": "reply",
        "resolution_hint": "verify double charge in payment gateway logs process refund for duplicate amount to card on file",
    },
    {
        "id": "R003",
        "text": "The product did not work as advertised. I want my money back.",
        "category": "refund",
        "correct_action": "reply",
        "resolution_hint": "review product description versus delivered functionality confirm mismatch and process refund",
    },
    {
        "id": "R004",
        "text": "I cancelled within the 30-day window but have not received my refund.",
        "category": "refund",
        "correct_action": "reply",
        "resolution_hint": "verify cancellation date within refund window locate delayed refund in processor and escalate",
    },
    {
        "id": "R005",
        "text": "I would like a partial refund for the unused months of my annual plan.",
        "category": "refund",
        "correct_action": "reply",
        "resolution_hint": "calculate unused months on annual plan process prorated refund for remaining subscription period",
    },
    {
        "id": "R006",
        "text": "A refund was promised by your support agent 2 weeks ago but never arrived.",
        "category": "refund",
        "correct_action": "escalate",
        "resolution_hint": "escalate undelivered promised refund to billing manager attach original support agent transcript",
    },
    {
        "id": "R007",
        "text": "I need a refund processed urgently as it was a fraudulent charge.",
        "category": "refund",
        "correct_action": "escalate",
        "resolution_hint": "escalate fraudulent charge to payments fraud team freeze account initiate chargeback process",
    },
    {
        "id": "R008",
        "text": "How long does a refund take to appear on my credit card?",
        "category": "refund",
        "correct_action": "reply",
        "resolution_hint": "explain refund timeline 5-7 business days for credit card 1-3 days for original payment method",
    },

    # ── general (10) ──────────────────────────────────────────────────────────
    {
        "id": "G001",
        "text": "What are your business hours and do you have a phone number I can call?",
        "category": "general",
        "correct_action": "reply",
        "resolution_hint": "provide support hours 9am-6pm weekdays toll free number and link to contact page for phone",
    },
    {
        "id": "G002",
        "text": "Thank you! The issue has been resolved. You guys are awesome.",
        "category": "general",
        "correct_action": "close",
        "resolution_hint": "acknowledge resolution thank customer for positive feedback and close ticket with satisfaction note",
    },
    {
        "id": "G003",
        "text": "Do you offer a student discount or non-profit pricing?",
        "category": "general",
        "correct_action": "reply",
        "resolution_hint": "confirm student discount eligibility criteria provide non-profit pricing page and application form",
    },
    {
        "id": "G004",
        "text": "Where can I find your terms of service and privacy policy?",
        "category": "general",
        "correct_action": "reply",
        "resolution_hint": "share direct links to terms of service privacy policy and data processing agreement documents",
    },
    {
        "id": "G005",
        "text": "Is your service available in my country? I am based in Brazil.",
        "category": "general",
        "correct_action": "reply",
        "resolution_hint": "confirm service availability in Brazil note any regional restrictions and provide local pricing",
    },
    {
        "id": "G006",
        "text": "Can I use your product for commercial purposes?",
        "category": "general",
        "correct_action": "reply",
        "resolution_hint": "confirm commercial use rights under current plan outline enterprise licensing for larger usage",
    },
    {
        "id": "G007",
        "text": "Problem resolved, thanks for the quick response!",
        "category": "general",
        "correct_action": "close",
        "resolution_hint": "acknowledge quick resolution compliment note feedback for team performance review close ticket",
    },
    {
        "id": "G008",
        "text": "Do you have an affiliate or referral program?",
        "category": "general",
        "correct_action": "reply",
        "resolution_hint": "provide affiliate program signup link commission structure and referral tracking dashboard access",
    },
    {
        "id": "G009",
        "text": "What integrations do you support with third-party tools?",
        "category": "general",
        "correct_action": "reply",
        "resolution_hint": "list supported third-party integrations provide API docs link and Zapier connector instructions",
    },
    {
        "id": "G010",
        "text": "I just wanted to say your product has been amazing for our team.",
        "category": "general",
        "correct_action": "close",
        "resolution_hint": "acknowledge positive team feedback forward compliment to product team and close with gratitude",
    },
]

TICKET_LOOKUP = {t["id"]: t for t in TICKETS}
