# SILAH - Product Understanding Agent

Agent 1 is responsible for understanding the product and generating an Ideal Customer Profile (ICP).

## Input

The agent receives:

- Product name
- Product description
- Sender name
- Sender company name
- Sender email
- Sender phone number

## Process

1. The product information is researched using Tavily.
2. The LLM analyzes the product and market context.
3. The agent generates:
   - Product summary
   - Target industries
   - Preferred company sizes
4. Sender information is passed through without being analyzed by the LLM.

## Output

```json
{
  "product_summary": "...",
  "ideal_customer_profile": {
    "target_industries": [],
    "preferred_company_size": []
  },
  "sender_info": {
    "name": "...",
    "company_name": "...",
    "email": "...",
    "phone": "..."
  }
}
