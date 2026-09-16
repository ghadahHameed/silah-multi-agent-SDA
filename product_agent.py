import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


load_dotenv()


class SenderInfo(BaseModel):
    name: str = Field(
        description="Name of the person representing the product company."
    )

    company_name: str = Field(
        description="Name of the company offering the product."
    )

    email: str = Field(
        description="Sender email address."
    )

    phone: str = Field(
        description="Sender phone number."
    )


class SearchPreferences(BaseModel):
    preferred_city: str | None = Field(
        default=None,
        description="Optional preferred city for company search."
    )

    requested_company_count: int = Field(
        default=5,
        ge=1,
        description="Number of companies requested by the user."
    )


class IdealCustomerProfile(BaseModel):
    target_industries: list[str] = Field(
        description="Industries most suitable for the product."
    )

    preferred_company_size: list[str] = Field(
        description=(
            "Suitable company sizes. Use only: "
            "Small (1-50 employees), "
            "Medium (51-250 employees), "
            "Large (251+ employees)."
        )
    )


class ProductAnalysis(BaseModel):
    product_summary: str = Field(
        description="Short summary of what the product does."
    )

    ideal_customer_profile: IdealCustomerProfile


class ProductAgentOutput(BaseModel):
    product_summary: str
    ideal_customer_profile: IdealCustomerProfile
    sender_info: SenderInfo
    search_preferences: SearchPreferences


model = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL") or None,
    temperature=0
).with_structured_output(ProductAnalysis)


search_tool = TavilySearch(
    max_results=5,
    search_depth="basic"
)


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

    search_query = f"""
    Research this B2B product and its market context.

    Product Name: {product_name}

    Product Description:
    {product_description}

    Focus on:
    - what the product does
    - business use cases
    - industries that could benefit from it
    - suitable company sizes
    """

    research_results = search_tool.invoke(
        {"query": search_query}
    )

    prompt = f"""
    You are the Product Understanding Agent in SILAH.

    Analyze the product using the provided product information
    and the web research results.

    Product Name:
    {product_name}

    Product Description:
    {product_description}

    Web Research Results:
    {research_results}

    Generate:

    1. A short product summary.

    2. An Ideal Customer Profile containing:
       - target industries
       - preferred company sizes

    For company size, use ONLY:
    - Small (1-50 employees)
    - Medium (51-250 employees)
    - Large (251+ employees)

    Select only the sizes suitable for the product.

    Do not:
    - search for specific companies
    - calculate Fit Scores
    - rank companies
    - find contacts
    - generate outreach messages

    Agent 2 handles company research, scoring, and ranking.
    Agent 3 handles contacts and outreach.
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

    return ProductAgentOutput(
        product_summary=analysis.product_summary,
        ideal_customer_profile=analysis.ideal_customer_profile,
        sender_info=sender_info,
        search_preferences=search_preferences
    )


if __name__ == "__main__":

    product_name = input("Enter product name: ")

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
