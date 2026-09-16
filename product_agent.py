import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


load_dotenv()


ALLOWED_INDUSTRIES = [
    "Agriculture Investment",
    "Airlines",
    "Artificial Intelligence",
    "Aviation Services",
    "Banking",
    "Chemicals",
    "Cloud Communications",
    "Construction",
    "Consumer Goods",
    "Cybersecurity",
    "Data & AI",
    "Delivery Technology",
    "E-commerce Technology",
    "Economic Development",
    "Education",
    "Energy",
    "Entertainment & Tourism",
    "Financial Services",
    "Fintech",
    "Food & Beverage",
    "Food & Retail",
    "Food Delivery",
    "Food Technology",
    "Gaming",
    "Healthcare",
    "Healthcare & Consumer",
    "Healthcare Supply Chain",
    "Hospitality",
    "Information Technology",
    "Insurance",
    "Investment",
    "Logistics",
    "Logistics & Infrastructure",
    "Manufacturing",
    "Mining",
    "Pharmaceuticals",
    "Real Estate",
    "Real Estate & Tourism",
    "Retail",
    "Software",
    "Technology",
    "Telecommunications",
    "Tourism",
    "Tourism & Hospitality",
    "Travel",
    "Travel Technology",
    "Utilities"
]

class SenderInfo(BaseModel):
    name: str
    company_name: str
    email: str
    phone: str


class SearchPreferences(BaseModel):
    preferred_city: str | None = None

    requested_company_count: int = Field(
        default=5,
        ge=1
    )


class ProspectAnalysis(BaseModel):
    icp_industries: list[str] = Field(
        description=(
            "Industries that best match the product. "
            "Must only contain values from the provided taxonomy."
        )
    )

    buyer_profile: str = Field(
        description=(
            "A description of the type of company most likely "
            "to benefit from the product."
        )
    )

    clarification_question: str | None = Field(
        default=None,
        description=(
            "One clarification question if the product description "
            "is too vague to determine target industries."
        )
    )


class Agent1Output(BaseModel):
    status: str
    icp_industries: list[str]
    buyer_profile: str
    sender_info: SenderInfo
    search_preferences: SearchPreferences
    clarification_question: str | None = None


model = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL") or None,
    temperature=0
).with_structured_output(ProspectAnalysis)


def analyze_product(
    product_name: str,
    product_description: str,
    sender_name: str,
    sender_company_name: str,
    sender_email: str,
    sender_phone: str,
    preferred_city: str | None = None,
    requested_company_count: int | None = None
):

    taxonomy = "\n".join(
        f"- {industry}"
        for industry in ALLOWED_INDUSTRIES
    )

    prompt = f"""
You are Agent 1 — Prospect Discovery in the SILAH multi-agent system.

Your job is to understand the user's product and create a profile
that Agent 2 can use to search the existing company dataset.

Product Name:
{product_name}

Product Description:
{product_description}

Allowed Industry Taxonomy:
{taxonomy}

Your tasks:

1. Select the industries that best fit the product.

2. You MUST select industries only from the provided taxonomy.
   Do not create, rename, merge, or invent industry categories.

3. Generate a buyer_profile describing the type of company
   most likely to benefit from this product.

The buyer_profile should describe the potential buyer company,
not the product itself.

It may include characteristics such as:
- business type
- operational needs
- relevant challenges
- workforce characteristics
- likely reasons for needing the product

Do NOT:
- search for companies
- name specific companies
- calculate fit scores
- rank companies
- find contacts
- generate outreach messages

If the product description is too vague to confidently identify
at least one industry:

- return an empty icp_industries list
- provide one short clarification_question
- do not guess

If the description is clear enough:
- return suitable icp_industries
- generate the buyer_profile
- set clarification_question to null
"""

    analysis = model.invoke(prompt)

    sender_info = SenderInfo(
        name=sender_name,
        company_name=sender_company_name,
        email=sender_email,
        phone=sender_phone
    )

    search_preferences = SearchPreferences(
        preferred_city=preferred_city,
        requested_company_count=(
            requested_company_count
            if requested_company_count is not None
            else 5
        )
    )

    if not analysis.icp_industries:
        return Agent1Output(
            status="needs_clarification",
            icp_industries=[],
            buyer_profile="",
            sender_info=sender_info,
            search_preferences=search_preferences,
            clarification_question=(
                analysis.clarification_question
                or "What business problem does this product solve?"
            )
        )

    return Agent1Output(
        status="ready",
        icp_industries=analysis.icp_industries,
        buyer_profile=analysis.buyer_profile,
        sender_info=sender_info,
        search_preferences=search_preferences,
        clarification_question=None
    )


if __name__ == "__main__":

    product_name = input(
        "Enter product name: "
    )

    product_description = input(
        "Enter product description: "
    )

    sender_name = input(
        "Enter sender name: "
    )

    sender_company_name = input(
        "Enter sender company name: "
    )

    sender_email = input(
        "Enter sender email: "
    )

    sender_phone = input(
        "Enter sender phone: "
    )

    preferred_city_input = input(
        "Enter preferred city (optional): "
    ).strip()

    requested_company_count_input = input(
        "Enter number of companies (optional, default = 5): "
    ).strip()

    preferred_city = (
        preferred_city_input
        if preferred_city_input
        else None
    )

    requested_company_count = (
        int(requested_company_count_input)
        if requested_company_count_input
        else None
    )

    result = analyze_product(
        product_name=product_name,
        product_description=product_description,
        sender_name=sender_name,
        sender_company_name=sender_company_name,
        sender_email=sender_email,
        sender_phone=sender_phone,
        preferred_city=preferred_city,
        requested_company_count=requested_company_count
    )

    print(
        result.model_dump_json(
            indent=2
        )
    )
