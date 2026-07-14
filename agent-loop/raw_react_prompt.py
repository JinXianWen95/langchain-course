import inspect
import json
import os
import re

import ollama
from dotenv import load_dotenv
from langsmith import traceable
from ollama import Client

MAX_ITERATIONS = 10
MODEL = "qwen3.5:latest"

load_dotenv()

os.environ["LANGSMITH_PROJECT"] = "raw_react_prompt"


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
    price = float(price)  # in case price is passed as string
    discounts = {"gold": 23, "silver": 12, "bronze": 5}
    discount = discounts.get(discount_tier.lower(), 0.0)
    return round(price * (1 - discount / 100), 2)


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

# CHANGE: Delete the JSON schemas. Tools now live inside the prompt as plain text.
# We derive descriptions from the function docstrings and signatures using inspect.


def get_tool_descriptions(tools_dict):
    tool_descriptions = []
    for name, func in tools_dict.items():
        original_function = getattr(
            func, "__wrapped__", func
        )  # Unwrap the function if it's decorated
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function) or "No description available."
        tool_descriptions.append(f"{name}{signature} - {docstring}")

    return "\n".join(tool_descriptions)


tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""


# --- Helper: traced Ollama call ---
# Difference 3: Without LangChain, we must manually trace LLM calls for LangSmith.


@traceable(name="ollama chat", run_type="llm")
def ollama_chat_traced(messages, model, options):
    client = Client(host=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
    return client.chat(model=model, messages=messages, options=options)


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):

    print(f'Question: "{question}"')
    print("=" * 60)

    prompt = react_prompt.format(question=question)
    scratchpad = ""  # Initialize an empty scratchpad for observations

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        full_prompt = prompt + scratchpad

        # stop token toprevent the LLM from generating its own Observation
        response = ollama_chat_traced(
            messages=[{"role": "user", "content": full_prompt}],
            model=MODEL,
            options={"stop": ["\nObservation:"], "temperature": 0.0},
        )

        output = response.message.content
        print(f"LLM Output: {output}")

        print(f"    [Parsing] Looking for Final Answer in LLM output")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer

        print(f"    [Parsing] Looking for Action and Action Input in LLM output")
        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(
                f"    [Pasrsing] ERROR: Could not parse Action/Action Input from LLM output"
            )
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        # Split comma-separated args; strip key= prefix if LLM outputs key=value format
        parsed_input = json.loads(tool_input_raw)
        args = list(parsed_input.values())

        print(f"  [Tool Executing] {tool_name}({args})...")
        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))

        print(f"  [Tool Result] {observation}")

        # CHANGE 7: History is one growing string re-sent every iteration (replaces messages.append).
        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("=== LangChain Agent Loop with Tool Calling ===")
    result = run_agent("What is the price of a laptop with a gold discount?")
