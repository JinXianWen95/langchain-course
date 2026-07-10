import os

import ollama
from dotenv import load_dotenv
from langsmith import traceable
from ollama import Client

MAX_ITERATIONS = 10
MODEL = "qwen2.5:latest"

load_dotenv()

os.environ["LANGSMITH_PROJECT"] = "agent_loop_raw_function_calling"


@traceable(name="tool")
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


@traceable(name="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price based on the discount tier and return the discounted price.
    Available tiers: bronze, silver, gold"""
    print(
        f'    >> Exceuting apply_discount(price={price}, discount_tier="{discount_tier}")'
    )
    discounts = {"gold": 23, "silver": 12, "bronze": 5}
    discount = discounts.get(discount_tier.lower(), 0.0)
    return round(price * (1 - discount / 100), 2)


# Difference 2: Without using the @tool decorator, we need to manually create a dictionary of tools for the agent to use.

tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The name of the product to look up. e.g., 'laptop', 'headphones', 'smartphone', 'keyboard'",
                    }
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount to a price based on the discount tier and return the discounted price. Available tiers: bronze, silver, gold",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "The original price of the product.",
                    },
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier to apply (bronze, silver, gold).",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

# NOTE: Ollama can also generate the tool schema automatically from the function signature if you pass the functions
# directly as tools [get_product_price, apply_discount]. However. this requires the docstrings to follow the Google style format.
# For example, the docstring for get_product_price would need to be:
# """  Look up the price of a product in the catalog.
# Args: product (str): The name of the product to look up. e.g., 'laptop', 'headphones', 'smartphone', 'keyboard'
# Returns: float: The price of the product. If the product is not found, returns 0.0.
# """


# --- Helper: traced Ollama call ---
# Difference 3: Without LangChain, we must manually trace LLM calls for LangSmith.


@traceable(name="ollama chat", run_type="llm")
def ollama_chat_traced(messages, model):
    client = Client(host=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
    return client.chat(
        model=model,
        messages=messages,
        tools=tools_for_llm,
    )


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f'Question: "{question}"')
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant."
                "You have access to a product catalog tool and a discount tool."
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product price or discount. You must use the tools provided to get the real price.\n"
                "2. Only call apply_discount ONCE and AFTER you have obtained the product price using get_product_price.\n"
                "3. NEVER calculate discounts yourself using math. You must use the apply_discount tool to get the discounted price.\n"
                "4. If you don't know the answer, you must say 'I don't know' and not make up an answer.\n"
                "5. If the user does not specify a discount tier, you must ask the user to provide one before applying any discount.\n"
            ),
        },
        {"role": "user", "content": question},
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        response = ollama_chat_traced(messages=messages, model=MODEL)
        ai_message = response.message

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"Final answer: {ai_message.content}")
            return ai_message.content

        # Process only the first tool, force one tool per iteration for simplicity
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"    [Tool Selected]: {tool_name} with args {tool_args}")

        tool_to_call = tools_dict.get(tool_name)
        if not tool_to_call:
            raise ValueError(f"Tool '{tool_name}' not found.")

        observation = tool_to_call(**tool_args)

        print(f"    [Tool Result]: {observation}")

        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "content": str(observation),
            }
        )

    print("Max iterations reached without a final answer.")
    return None


if __name__ == "__main__":
    print("=== LangChain Agent Loop with Tool Calling ===")
    result = run_agent("What is the price of a laptop with a gold discount?")
