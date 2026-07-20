import os
from typing import Any, Dict, List

import gradio as gr
from backend.core import run_llm

os.environ["LANGSMITH_PROJECT"] = "document-helper"


def _format_sources(context_docs: List[Any]) -> List[str]:
    return [
        str((meta.get("source") or "Unknown"))
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]


def _clean_history_for_backend(raw_history: List[Any]) -> List[Dict[str, str]]:
    """
    Standardizes the raw chat history from Gradio into a clean List[Dict[str, str]]
    suitable for the LLM, stripping away UI elements like HTML <details> blocks.
    """
    cleaned = []

    for item in raw_history:
        # Scenario A: Item is a Dict (e.g. {"role": "user", "content": "..."})
        if isinstance(item, dict):
            role = item.get("role", "user")
            content = item.get("content", "")
        # Scenario B: Item is a List/Tuple (e.g. [user_msg, bot_msg])
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, bot_msg = item
            if user_msg:
                cleaned.append({"role": "user", "content": user_msg})
            if bot_msg:
                # Strip out sources toggle UI element so LLM memory isn't polluted
                clean_bot = str(bot_msg).split("\n\n<details>")[0]
                cleaned.append({"role": "assistant", "content": clean_bot})
            continue
        else:
            continue

        # Strip sources toggle UI element from dict content
        if role == "assistant":
            content = str(content).split("\n\n<details>")[0]

        cleaned.append({"role": role, "content": content})

    return cleaned


def respond(message: str, chat_history: List[Any]):
    """
    Handles user input, calls the backend with standardized history,
    formats sources, and updates the chat UI.
    """
    if not message.strip():
        return "", chat_history

    # Standardize history for backend usage
    clean_history = _clean_history_for_backend(chat_history)

    # Filter out the initial welcome message from the LLM's system memory context
    active_history = [
        msg
        for msg in clean_history
        if not (msg["role"] == "assistant" and "Ask me anything" in msg["content"])
    ]

    try:
        # 1. Query the LLM backend
        result: Dict[str, Any] = run_llm(message, chat_history=active_history)
        answer = str(result.get("answer", "")).strip() or "(No answer returned.)"
        sources = _format_sources(result.get("context", []))

        # 2. Format response with expandable HTML sources
        full_response = answer
        if sources:
            source_list = "\n".join([f"- {s}" for s in sources])
            full_response += f"\n\n<details><summary><b>Sources</b></summary>\n\n{source_list}\n</details>"

    except Exception as e:
        full_response = (
            f"⚠️ **Error:** Failed to generate a response.\n\n*Details:* {str(e)}"
        )

    # 3. Append new messages to UI history matching whatever format Gradio sent
    # This guarantees Gradio renders the update properly without format conflicts
    if chat_history and isinstance(chat_history[0], (list, tuple)):
        chat_history.append([message, full_response])
    else:
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": full_response})

    # Return empty string to clear the input textbox, and the updated history
    return "", chat_history


def clear_session():
    return [
        {
            "role": "assistant",
            "content": "Ask me anything about LangChain docs. I’ll retrieve relevant context and cite sources.",
        }
    ]


with gr.Blocks(title="LangChain Documentation Helper") as demo:
    gr.Markdown("# LangChain Documentation Helper")

    chatbot = gr.Chatbot(
        value=[
            {
                "role": "assistant",
                "content": "Ask me anything about LangChain docs. I’ll retrieve relevant context and cite sources.",
            }
        ],
        elem_id="chatbot",
    )

    with gr.Row():
        txt_input = gr.Textbox(
            show_label=False, placeholder="Ask a question about LangChain…", scale=9
        )
        submit_btn = gr.Button("Send", variant="primary", scale=1)

    with gr.Row():
        clear_btn = gr.Button("Clear chat", variant="stop")

    # Bind interactions
    txt_input.submit(
        fn=respond, inputs=[txt_input, chatbot], outputs=[txt_input, chatbot]
    )
    submit_btn.click(
        fn=respond, inputs=[txt_input, chatbot], outputs=[txt_input, chatbot]
    )
    clear_btn.click(fn=clear_session, inputs=None, outputs=chatbot)

if __name__ == "__main__":
    demo.launch()
