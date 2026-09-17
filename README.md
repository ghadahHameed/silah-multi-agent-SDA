# Agent 1 — Prospect Discovery Agent

Agent 1 is the first stage of the SILAH multi-agent workflow.

It collects the user's product information, sender information, and optional search preferences, then converts the product description into a structured prospect profile for Agent 2.

Agent 1 does **not** search, score, or rank companies.

## Pipeline position

```text
User Input
    ↓
Agent 1: Prospect Discovery   ← this component
    ↓
ICP Industries + Buyer Profile
+ Sender Info + Search Preferences
    ↓
Agent 2: Qualification & Prioritization
    ↓
Agent 3: Outreach & Next Action
Input

Agent 1 receives:

product_name
product_description
sender_name
sender_company_name
sender_email
sender_phone
preferred_city — optional
requested_company_count — optional, default 5

Only the product name and description are analyzed by the LLM.

Sender information and search preferences are passed through unchanged.

Prospect understanding

Agent 1 uses one LLM call per run to generate:

icp_industries
buyer_profile
ICP industries

The industries are selected only from the fixed taxonomy used by the SILAH company dataset.

This prevents the model from generating industry labels that do not exist in the dataset.

Example:

"icp_industries": [
  "Cybersecurity",
  "Information Technology",
  "Software"
]
Buyer profile

The buyer profile describes the type of company most likely to benefit from the product.

Example:

Organizations that operate digital systems and need recurring
employee cybersecurity awareness and phishing-prevention training.

Agent 2 uses this profile when comparing the product need with company descriptions.

Clarification

If the product description is too vague to identify suitable industries, Agent 1 does not guess.

It returns:

{
  "status": "needs_clarification",
  "icp_industries": [],
  "buyer_profile": "",
  "clarification_question": "What business problem does this product solve?"
}
Output

Example successful output:

{
  "status": "ready",
  "icp_industries": [
    "Cybersecurity",
    "Information Technology"
  ],
  "buyer_profile": "Organizations that operate digital systems and need employee cybersecurity training.",
  "sender_info": {
    "name": "Aseel Alsaad",
    "company_name": "Example Company",
    "email": "aseel@example.com",
    "phone": "+966500000000"
  },
  "search_preferences": {
    "preferred_city": "Riyadh",
    "requested_company_count": 5
  },
  "clarification_question": null
}
Handoff to Agent 2

Agent 1 passes:

icp_industries
buyer_profile
sender_info
search_preferences

Agent 2 then handles company filtering, scoring, and ranking.

Files
File	Responsibility
product_agent.py	Agent 1 logic, models, prompt, and structured output
test_agent.py	Agent 1 test cases
requirements.txt	Python dependencies
.gitignore	Excludes local environment files and API keys
Usage
from product_agent import analyze_product

result = analyze_product(
    product_name="Cybersecurity Awareness Platform",
    product_description="A B2B platform that trains employees on cybersecurity awareness and phishing risks.",
    sender_name="Aseel Alsaad",
    sender_company_name="Example Company",
    sender_email="aseel@example.com",
    sender_phone="+966500000000",
    preferred_city="Riyadh",
    requested_company_count=5
)

print(result.model_dump_json(indent=2))
Testing

Run:

python test_agent.py

The tests verify that:

different products generate different prospect profiles
industries stay within the dataset taxonomy
vague descriptions trigger clarification
sender information and search preferences are preserved
Scope

Agent 1 handles:

Product Understanding
→ ICP Industries
→ Buyer Profile

Agent 2 handles company qualification and ranking.

Agent 3 handles contacts and outreach.
