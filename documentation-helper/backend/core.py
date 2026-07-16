import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore

load_dotenv()

# Initialize embeddings
embeddings = PineconeEmbeddings(
    model="llama-text-embed-v2",
)

# Initialize vector store
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_LANGCHAIN_NAME"], embedding=embeddings
)

# Initialize chat model
MODEL = "qwen2.5:latest"

model = init_chat_model(
    f"ollama:{MODEL}",
    temperature=0,
    base_url="http://localhost:11434",
)


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    # Retrieve top 4 most similar documents
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)

    # Serialize documents for the model
    serialized = "\n\n".join(
        (
            f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
        )
        for doc in retrieved_docs
    )

    # Return both serialized content and raw documents
    return serialized, retrieved_docs


def run_llm(query: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation and chat history.

    Args:
        query: The user's question
        chat_history: The previous conversation history from the UI

    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """
    # Create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    # Build the messages sequence starting with past history
    messages = []

    # Standardize and append previous chat history if present
    if chat_history:
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Append the latest user query at the end
    messages.append({"role": "user", "content": query})

    # Invoke the agent passing the full session dialog
    response = agent.invoke({"messages": messages})

    # Extract the final answer from the last AI message
    answer = response["messages"][-1].content

    # Extract context documents from ToolMessage artifacts
    context_docs = []
    for message in response["messages"]:
        # Check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {"answer": answer, "context": context_docs}


if __name__ == "__main__":
    result = run_llm(query="what are deep agents?")
    print(result)
