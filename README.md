# SILAH - Product Understanding Agent

This repository contains Agent 1 of the SILAH multi-agent outbound system.

## Overview

Agent 1 is responsible for understanding the product provided by the user and generating a structured Ideal Customer Profile (ICP).

It uses Tavily to research the product and its market context, then uses an LLM to analyze the product and generate the final structured output.

The output is passed to Agent 2, which handles company research, filtering, fit analysis, and ranking.

---

## Agent 1 Responsibilities

Agent 1:

- Receives the product name and description.
- Receives sender information.
- Receives optional company search preferences.
- Researches the product and market context using Tavily.
- Generates a short product summary.
- Identifies suitable target industries.
- Identifies suitable company sizes.
- Returns the results in a structured format.
- Passes sender information and search preferences to Agent 2.

Agent 1 does NOT:

- Search for specific companies.
- Calculate Fit Scores.
- Rank companies.
- Find contacts.
- Generate outreach messages.

---

## Input

Agent 1 receives:

```json
{
  "product_name": "Cybersecurity Awareness Platform",
  "product_description": "A B2B platform that helps organizations train employees on cybersecurity awareness.",
  "sender_name": "Aseel Alsaad",
  "sender_company_name": "Example Company",
  "sender_email": "aseel@example.com",
  "sender_phone": "+966500000000",
  "preferred_city": "Riyadh",
  "requested_company_count": 5
}
