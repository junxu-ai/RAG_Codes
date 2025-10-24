import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI


# Define a simple calculator tool
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b

def financial_product(product: str) -> float:
    """provide financial information by product"""
    return "financial information by product".format(product)


def financial_region(region: str) -> float:
    """provide financial information by region or location"""
    return "financial information by region".format(region)

# Create an agent workflow with our calculator tool
agent = FunctionAgent(
    name="calculator_agent",  # Add a name
    description="An agent that can multiply two numbers, and provide financial information.",  # Add a description
    tools=[multiply, financial_product, financial_region],  # Add the calculator tool
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="You are a helpful assistant that can multiply two numbers and provide finanical information.",
)

questions = [
    # "What is 1234 * 4567?",
    # "What is the financial information by production?",
    # "What is the financial information by region?",
    "what is the financial situation in the US?",
    "what is the financial situation for the phone?",
]


async def main():
    # Run the agent
    for question in questions:
        print(f"Question: {question}")
        response = await agent.run(question)
        print(f"Response: {response}\n")


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())