import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent
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
)


search_tool = TavilySearch(
    max_results=5,
    search_depth="basic"
)


SYSTEM_PROMPT = """
You are the Product Understanding Agent in SILAH.

Your task is to:

1. Understand the product.
2. Research the product and its market context using Tavily.
3. Generate an Ideal Customer Profile.

The final output must include:
- product_summary
- target_industries
- preferred_company_size

Do not:
- search for specific companies
- calculate Fit Scores
- rank companies
- find contacts
- generate outreach messages

Agent 2 handles company research, scoring, and ranking.
Agent 3 handles contacts and outreach.
"""


product_agent = create_agent(
    model=model,
    tools=[search_tool],
    system_prompt=SYSTEM_PROMPT,
    response_format=ProductAgentOutput
)


def analyze_product(product_name: str, product_description: str):

    message = f"""
Product Name: {product_name}

Product Description: {product_description}

Research the product first, then generate the ICP.
"""

    result = product_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": message
                }
            ]
        }
    )

    return result["structured_response"]


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