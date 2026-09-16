# SILAH - Agent 1: Prospect Discovery

Agent 1 is the first stage of the SILAH multi-agent outbound workflow.

Its role is to understand the user's product and convert it into structured prospect criteria that can be used by Agent 2 to search the existing company dataset.

## What Agent 1 Does

Agent 1 receives:

- Product name
- Product description
- Sender name
- Sender company name
- Sender email
- Sender phone
- Preferred city (optional)
- Number of companies to return (optional)

It then uses one LLM call to generate:

- `icp_industries`
- `buyer_profile`

The selected industries are restricted to the same industry taxonomy used in the company dataset.

This prevents the model from generating labels that do not exist in the dataset.

## Why Buyer Profile?

The product description and the company description represent different types of text.

For example:

```text
Product:
Cybersecurity awareness platform for employee training
