import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openrouter import ChatOpenRouter
from langchain_tavily import TavilySearch
from tavily import TavilyClient

load_dotenv()
os.environ["LANGSMITH_PROJECT"] = "search-agent"
tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search results
    """
    print(f"Searching for: {query}")
    return tavily.search(query)


# llm = ChatOpenRouter(temperature=0, model="openrouter/free")
llm = ChatOllama(
    temperature=0,
    model="qwen2.5:latest",
    base_url="http://127.0.0.1:11434",
    num_ctx=16384,  # increae context window since the result of search could be long
)

agent = create_agent(
    llm, tools=[TavilySearch()]
)  # using langchain_tavily's TavilySearch tool instead of the custom search tool


def main():
    print("Hello from langchain-course!")

    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="Search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details."
            )
        }
    )
    print(result)


if __name__ == "__main__":
    main()
