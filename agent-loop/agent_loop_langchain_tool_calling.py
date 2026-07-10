import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen2.5:latest"

load_dotenv()

os.environ["LANGSMITH_PROJECT"] = "agent_loop_langchain_tool_calling"

# --- Tools (LangChain @tool decorator) ---


@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f'    >> Exceuting get_product_price(product="{product}")')
    prices = {
        "laptop": 1299.99,
        "headphones": 199.99,
        "smartphone": 899.99,
        "keyboard": 499.99,
    }
    return prices.get(product.lower(), 0.0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price based on the discount tier and return the discounted price.
    Available tiers: bronze, silver, gold"""
    print(
        f'    >> Exceuting apply_discount(price={price}, discount_tier="{discount_tier}")'
    )
    discounts = {"gold": 23, "silver": 12, "bronze": 5}
    discount = discounts.get(discount_tier.lower(), 0.0)
    return round(price * (1 - discount / 100), 2)


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(
        f"ollama:{MODEL}",
        temperature=0,
        base_url="http://localhost:11434",
        # reasoning=True,
    )
    llm_with_tools = llm.bind_tools(tools)

    print(f'Question: "{question}"')
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant."
                "You have access to a product catalog tool and a discount tool."
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product price or discount. You must use the tools provided to get the real price.\n"
                "2. Only call apply_discount ONCE and AFTER you have obtained the product price using get_product_price.\n"
                "3. NEVER calculate discounts yourself using math. You must use the apply_discount tool to get the discounted price.\n"
                "4. If you don't know the answer, you must say 'I don't know' and not make up an answer.\n"
                "5. If the user does not specify a discount tier, you must ask the user to provide one before applying any discount.\n"
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"Final answer: {ai_message.content}")
            return ai_message.content

        # Process only the first tool, force one tool per iteration for simplicity
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"    [Tool Selected]: {tool_name} with args {tool_args}")

        tool_to_call = tools_dict.get(tool_name)
        if not tool_to_call:
            raise ValueError(f"Tool '{tool_name}' not found.")

        observation = tool_to_call.invoke(tool_args)

        print(f"    [Tool Result]: {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("Max iterations reached without a final answer.")
    return None


if __name__ == "__main__":
    print("=== LangChain Agent Loop with Tool Calling ===")
    result = run_agent("What is the price of a laptop with a gold discount?")
