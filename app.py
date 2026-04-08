"""
Streamlit AI Agent powered by Azure OpenAI.

This app provides a conversational AI agent with tool-calling capabilities
using Azure OpenAI's chat completions API. The agent can perform web searches,
calculations, and date/time queries through a simple chat interface.
"""

import json
import math
from datetime import datetime, timezone

import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Azure OpenAI client
# ---------------------------------------------------------------------------

def get_client() -> AzureOpenAI:
    """Return a configured AzureOpenAI client using env variables."""
    import os
    api_version = os.getenv("OPENAI_API_VERSION", "2024-05-01-preview")
    return AzureOpenAI(api_version=api_version)


def get_deployment() -> str:
    """Return the deployment name from Streamlit secrets or env."""
    import os
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "").strip()
    if not deployment:
        raise ValueError(
            "Must provide the AZURE_OPENAI_DEPLOYMENT_NAME environment variable. "
            "Set it to your Azure OpenAI deployment name (e.g., 'gpt-4-deployment')."
        )
    return deployment


# ---------------------------------------------------------------------------
# Tool definitions (function-calling schema sent to the model)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Get the current date and time in UTC.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a mathematical expression. "
                "Supports basic arithmetic, powers, sqrt, sin, cos, tan, log, pi, e."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g. '2**10 + sqrt(144)'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "Search the agent's built-in knowledge base for a topic. "
                "Use this when the user asks a factual question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

# Safe math namespace for calculate()
_MATH_NAMESPACE: dict = {
    k: getattr(math, k)
    for k in [
        "sqrt", "sin", "cos", "tan", "asin", "acos", "atan",
        "log", "log2", "log10", "exp", "ceil", "floor", "factorial",
        "pi", "e", "inf",
    ]
}
_MATH_NAMESPACE["abs"] = abs
_MATH_NAMESPACE["round"] = round
_MATH_NAMESPACE["pow"] = pow
_MATH_NAMESPACE["__builtins__"] = {}


def tool_get_current_datetime() -> str:
    now = datetime.now(timezone.utc)
    return json.dumps({"utc": now.strftime("%Y-%m-%d %H:%M:%S UTC")})


def tool_calculate(expression: str) -> str:
    try:
        result = eval(expression, _MATH_NAMESPACE)  # noqa: S307 – restricted namespace
        return json.dumps({"expression": expression, "result": str(result)})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def tool_search_knowledge(query: str) -> str:
    return json.dumps({
        "note": (
            "This is a placeholder knowledge-base tool. "
            "In production, connect this to a vector store or search API."
        ),
        "query": query,
        "answer": f"I don't have a dedicated knowledge base yet, but I'll do my best to answer about '{query}' from my training data.",
    })


TOOL_DISPATCH = {
    "get_current_datetime": lambda _args: tool_get_current_datetime(),
    "calculate": lambda args: tool_calculate(args["expression"]),
    "search_knowledge": lambda args: tool_search_knowledge(args["query"]),
}

# ---------------------------------------------------------------------------
# Agent loop – handles multi-turn tool calls
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful AI agent. You can use tools when needed to answer "
    "the user's questions accurately. Think step-by-step before answering. "
    "When performing calculations, always use the calculate tool rather than "
    "doing mental math."
)

MAX_TOOL_ROUNDS = 10  # safety limit for tool-call loops


def run_agent(client: AzureOpenAI, deployment: str, messages: list[dict]) -> str:
    """Run the agent loop: call the model, execute any tool calls, repeat."""
    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        choice = response.choices[0]
        assistant_msg = choice.message

        # Append assistant reply (may contain tool_calls and/or content)
        messages.append(assistant_msg.model_dump())

        if not assistant_msg.tool_calls:
            return assistant_msg.content or ""

        # Execute each tool call and feed results back
        for tc in assistant_msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            handler = TOOL_DISPATCH.get(fn_name)
            if handler:
                result = handler(fn_args)
            else:
                result = json.dumps({"error": f"Unknown tool: {fn_name}"})

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "I reached the maximum number of tool-call rounds. Please try rephrasing your question."


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="AI Agent", page_icon="🤖", layout="centered")
    st.title("AI Agent (Azure OpenAI)")
    st.caption("A conversational agent with tool-calling capabilities.")

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Sidebar – configuration & controls
    with st.sidebar:
        st.header("Settings")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        if st.button("Clear conversation"):
            st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            st.rerun()

        st.markdown("---")
        st.markdown(
            "**Available tools**\n"
            "- Calculator\n"
            "- Current date/time\n"
            "- Knowledge search (placeholder)"
        )

    # Render chat history (skip system message)
    for msg in st.session_state.messages[1:]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        elif role == "assistant" and content:
            with st.chat_message("assistant"):
                st.markdown(content)
        # tool messages are hidden from the user

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run agent
        client = get_client()
        deployment = get_deployment()

        # Build a working copy so we can pass temperature
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = run_agent(client, deployment, st.session_state.messages)
            st.markdown(reply)


if __name__ == "__main__":
    main()
