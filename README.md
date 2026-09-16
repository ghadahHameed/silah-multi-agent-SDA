# SILAH - Agent 1: Prospect Discovery

Agent 1 converts the user's product information into a structured prospect profile that can be used by Agent 2 to search and rank companies from the existing company dataset.

## Input

Agent 1 receives:

- Product name
- Product description
- Sender name
- Sender company name
- Sender email
- Sender phone
- Preferred city (optional)
- Number of companies to return (optional, default = 5)

## How It Works

Agent 1 uses one LLM call per run.

The LLM performs two main tasks:

1. **ICP Industry Classification**
   - Identifies industries suitable for the product.
   - Industries must come from the fixed taxonomy used by the company dataset.
   - The model cannot invent new industry labels.

2. **Buyer Profile Generation**
   - Generates a description of the type of company most likely to benefit from the product.
   - This profile is passed to Agent 2 for better comparison with company descriptions.

Agent 1 does not search for companies because the company dataset already exists.

## Clarification

If the product description is too vague to determine suitable industries, Agent 1 does not guess.

Instead, it returns one clarification question to the user.

## Output

```json
{
  "status": "ready",
  "icp_industries": [
    "Banking",
    "Government"
  ],
  "buyer_profile": "Organizations with a large workforce and strong cybersecurity awareness and compliance needs.",
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
