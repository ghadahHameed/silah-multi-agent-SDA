import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


load_dotenv()


class IdealCustomerProfile(BaseModel):
    target_industries: list[str] = Field(
        description="Industries most suitable for the product."
    )

    preferred_company_size: list[str] = Field(
        description="Preferred company sizes."
    )


class ProductAgentOutput(BaseModel):
    product_summary: str = Field(
        description="Short summary of what the product does."
    )

    ideal_customer_profile: IdealCustomerProfile


model = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL") or None,
    temperature=0
).with_structured_output(ProductAgentOutput)


search_tool = TavilySearch(
    max_results=5,
    search_depth="basic"
)


def analyze_product(product_name: str, product_description: str):

    search_query = f"""
    Research this B2B product and its market context.

    Product Name: {product_name}
    Product Description: {product_description}

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

    Research Results:
    {research_results}

    Generate:

    1. A short product summary.
    2. An Ideal Customer Profile containing:
       - target industries
       - preferred company sizes

    Do not:
    - search for specific companies
    - calculate Fit Scores
    - rank companies
    - find contacts
    - generate outreach messages

    Agent 2 handles company research, scoring, and ranking.
    Agent 3 handles contacts and outreach.
    """

    return model.invoke(prompt)


if __name__ == "__main__":

    product_name = input("Enter product name: ")

    product_description = input(
        "Enter product description: "
    )

    result = analyze_product(
        product_name,
        product_description
    )

    print(result.model_dump_json(indent=2))
