"""
Streamlit AI Agent powered by Azure OpenAI.

This app provides a conversational AI agent with tool-calling capabilities
using Azure OpenAI's chat completions API. The agent can perform web searches,
calculations, and date/time queries through a simple chat interface.
"""

import json
import math
import os
from datetime import datetime, timezone

import requests
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
# LSEG Data Platform connector
# ---------------------------------------------------------------------------

LSEG_BASE_URL = os.getenv("LSEG_BASE_URL", "https://api.refinitiv.com")
LSEG_AUTH_URL = os.getenv(
    "LSEG_AUTH_URL",
    "https://api.refinitiv.com/auth/oauth2/v1/token",
)


class LSEGConnector:
    """Connector for LSEG / Refinitiv Data Platform APIs.

    Requires the following environment variables:
      - LSEG_CLIENT_ID      – OAuth2 client ID (app key)
      - LSEG_CLIENT_SECRET  – OAuth2 client secret
      - LSEG_USERNAME        – LSEG platform username
      - LSEG_PASSWORD        – LSEG platform password
    """

    def __init__(self) -> None:
        self._access_token: str | None = None
        self._refresh_token: str | None = None

    # -- authentication ----------------------------------------------------

    def _authenticate(self) -> str:
        """Obtain an access token via the OAuth2 password grant."""
        if self._access_token:
            return self._access_token

        client_id = os.getenv("LSEG_CLIENT_ID", "")
        client_secret = os.getenv("LSEG_CLIENT_SECRET", "")
        username = os.getenv("LSEG_USERNAME", "")
        password = os.getenv("LSEG_PASSWORD", "")

        if not client_id or not username or not password:
            raise RuntimeError(
                "LSEG credentials not configured. "
                "Set LSEG_CLIENT_ID, LSEG_USERNAME, and LSEG_PASSWORD."
            )

        payload = {
            "grant_type": "password",
            "username": username,
            "password": password,
            "client_id": client_id,
            "scope": "trapi",
        }
        if client_secret:
            payload["client_secret"] = client_secret

        resp = requests.post(
            LSEG_AUTH_URL,
            data=payload,
            headers={"Accept": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        self._access_token = data["access_token"]
        self._refresh_token = data.get("refresh_token")
        return self._access_token

    def _headers(self) -> dict[str, str]:
        token = self._authenticate()
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

    # -- public helpers ----------------------------------------------------

    def get_stock_quote(self, ric: str) -> dict:
        """Fetch a real-time or delayed quote for the given RIC."""
        url = f"{LSEG_BASE_URL}/data/pricing/snapshots/v1/"
        params = {"universe": ric}
        try:
            resp = requests.get(
                url, headers=self._headers(), params=params, timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            return {"error": str(exc), "ric": ric}

    def search_instrument(self, query: str) -> dict:
        """Search for instruments (equities, bonds, etc.) by keyword."""
        url = f"{LSEG_BASE_URL}/discovery/search/v1/"
        body = {
            "View": "Entities",
            "Filter": f"SearchAllCategoryv3 eq 'equities'",
            "Terms": query,
            "Select": (
                "DocumentTitle,RIC,ExchangeName,"
                "IssuerCommonName,Currency"
            ),
            "Top": 10,
        }
        try:
            resp = requests.post(
                url, headers=self._headers(), json=body, timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            return {"error": str(exc), "query": query}

    def get_historical_pricing(
        self, ric: str, start: str, end: str, interval: str = "P1D",
    ) -> dict:
        """Fetch historical end-of-day (or intraday) pricing for a RIC.

        Parameters
        ----------
        ric : str
            Reuters Instrument Code, e.g. "AAPL.O".
        start : str
            ISO-8601 start date, e.g. "2025-01-01".
        end : str
            ISO-8601 end date, e.g. "2025-12-31".
        interval : str
            Price interval – "P1D" (daily), "PT1H" (hourly), etc.
        """
        url = (
            f"{LSEG_BASE_URL}/data/historical-pricing/v1"
            f"/views/summaries/{ric}"
        )
        params = {
            "start": start,
            "end": end,
            "interval": interval,
        }
        try:
            resp = requests.get(
                url, headers=self._headers(), params=params, timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            return {"error": str(exc), "ric": ric}

    def get_company_fundamentals(self, ric: str) -> dict:
        """Fetch key fundamental data for a company by RIC."""
        url = f"{LSEG_BASE_URL}/data/environmental-social-governance/v2/views/scores-full"
        params = {"universe": ric}
        try:
            resp = requests.get(
                url, headers=self._headers(), params=params, timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            return {"error": str(exc), "ric": ric}

    def get_news_headlines(self, query: str, count: int = 10) -> dict:
        """Fetch recent news headlines related to a query or RIC."""
        url = f"{LSEG_BASE_URL}/data/news/v1/headlines"
        params = {"query": query, "count": min(count, 50)}
        try:
            resp = requests.get(
                url, headers=self._headers(), params=params, timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            return {"error": str(exc), "query": query}


# Singleton instance (created lazily on first tool invocation)
_lseg: LSEGConnector | None = None


def _get_lseg() -> LSEGConnector:
    global _lseg
    if _lseg is None:
        _lseg = LSEGConnector()
    return _lseg


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
    {
        "type": "function",
        "function": {
            "name": "lseg_stock_quote",
            "description": (
                "Get a real-time or delayed stock quote from LSEG (Refinitiv) "
                "for the given RIC (Reuters Instrument Code). "
                "Examples: 'AAPL.O' (Apple on NASDAQ), 'VOD.L' (Vodafone on LSE)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ric": {
                        "type": "string",
                        "description": "Reuters Instrument Code, e.g. 'AAPL.O', 'MSFT.O', 'VOD.L'.",
                    }
                },
                "required": ["ric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lseg_search_instrument",
            "description": (
                "Search LSEG for financial instruments (equities, bonds, etc.) "
                "by keyword. Returns up to 10 matching instruments with their "
                "RICs, names, and exchange information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keyword, e.g. 'Apple', 'Tesla', 'oil futures'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lseg_historical_pricing",
            "description": (
                "Fetch historical pricing data from LSEG for a given RIC "
                "over a date range. Returns OHLCV (open, high, low, close, volume) data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ric": {
                        "type": "string",
                        "description": "Reuters Instrument Code, e.g. 'AAPL.O'.",
                    },
                    "start": {
                        "type": "string",
                        "description": "Start date in ISO-8601 format, e.g. '2025-01-01'.",
                    },
                    "end": {
                        "type": "string",
                        "description": "End date in ISO-8601 format, e.g. '2025-12-31'.",
                    },
                    "interval": {
                        "type": "string",
                        "description": "Price interval: 'P1D' (daily), 'PT1H' (hourly), 'P1W' (weekly), 'P1M' (monthly). Defaults to 'P1D'.",
                        "default": "P1D",
                    },
                },
                "required": ["ric", "start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lseg_company_fundamentals",
            "description": (
                "Fetch fundamental / ESG data for a company from LSEG, "
                "including ESG scores and key financial metrics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ric": {
                        "type": "string",
                        "description": "Reuters Instrument Code, e.g. 'AAPL.O'.",
                    }
                },
                "required": ["ric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lseg_news_headlines",
            "description": (
                "Fetch recent financial news headlines from LSEG related to "
                "a query or instrument."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query or RIC, e.g. 'AAPL.O' or 'oil prices'.",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of headlines to return (max 50). Defaults to 10.",
                        "default": 10,
                    },
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


def tool_lseg_stock_quote(ric: str) -> str:
    return json.dumps(_get_lseg().get_stock_quote(ric), default=str)


def tool_lseg_search_instrument(query: str) -> str:
    return json.dumps(_get_lseg().search_instrument(query), default=str)


def tool_lseg_historical_pricing(
    ric: str, start: str, end: str, interval: str = "P1D",
) -> str:
    return json.dumps(
        _get_lseg().get_historical_pricing(ric, start, end, interval),
        default=str,
    )


def tool_lseg_company_fundamentals(ric: str) -> str:
    return json.dumps(_get_lseg().get_company_fundamentals(ric), default=str)


def tool_lseg_news_headlines(query: str, count: int = 10) -> str:
    return json.dumps(_get_lseg().get_news_headlines(query, count), default=str)


TOOL_DISPATCH = {
    "get_current_datetime": lambda _args: tool_get_current_datetime(),
    "calculate": lambda args: tool_calculate(args["expression"]),
    "search_knowledge": lambda args: tool_search_knowledge(args["query"]),
    "lseg_stock_quote": lambda args: tool_lseg_stock_quote(args["ric"]),
    "lseg_search_instrument": lambda args: tool_lseg_search_instrument(args["query"]),
    "lseg_historical_pricing": lambda args: tool_lseg_historical_pricing(
        args["ric"], args["start"], args["end"], args.get("interval", "P1D"),
    ),
    "lseg_company_fundamentals": lambda args: tool_lseg_company_fundamentals(args["ric"]),
    "lseg_news_headlines": lambda args: tool_lseg_news_headlines(
        args["query"], args.get("count", 10),
    ),
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
            "- Knowledge search (placeholder)\n"
            "- LSEG Stock Quote\n"
            "- LSEG Instrument Search\n"
            "- LSEG Historical Pricing\n"
            "- LSEG Company Fundamentals\n"
            "- LSEG News Headlines"
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
