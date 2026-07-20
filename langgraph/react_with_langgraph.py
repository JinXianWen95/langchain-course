import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen2.5:latest"

load_dotenv()

os.environ["LANGSMITH_PROJECT"] = "react_with_langgraph"


@tool
def triple(num: float) -> float:
    """
    param num: a number to triple
    returns: the triple of the input number
    """
    return float(num) * 3


tools = [TavilySearch(max_results=1), triple]

llm = init_chat_model(
    f"ollama:{MODEL}",
    temperature=0,
    base_url="http://localhost:11434",
).bind_tools(tools)
