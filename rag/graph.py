"""The RAG agent: a LangGraph ReAct agent on OpenRouter with retrieval tools.

Passed straight to ``FastDash(callback_fn=graph, chat=True)`` — Fast Dash's
langstage bridge streams its tokens and tool calls into the chat transcript,
and adds a checkpointer for multi-turn memory.
"""

from __future__ import annotations

from .config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL
from .tools import add_source, search_knowledge

SYSTEM_PROMPT = """You are a knowledge-base assistant. Users add sources (web \
pages, YouTube videos, PDFs, or pasted text) and ask questions about them.

- When the user shares a URL or text to remember, call `add_source`.
- When the user asks a question about their sources, ALWAYS call \
`search_knowledge` first, then answer using ONLY the retrieved passages. Add \
inline numbered citations like [1], [2] referring to the passage numbers, and \
list the cited sources at the end.
- If the retrieved passages do not contain the answer, say so plainly — never \
invent an answer or use outside knowledge.
- Be concise and format answers as markdown."""


def build_graph(model=None):
    """Compile the ReAct RAG agent. ``model`` is injectable for tests."""
    if model is None:
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model=OPENROUTER_MODEL,
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            temperature=0,
            streaming=True,
            default_headers={
                "HTTP-Referer": "https://github.com/dkedar7/embedchain-fastdash",
                "X-Title": "embedchain-fastdash",
            },
        )
    from langgraph.prebuilt import create_react_agent

    return create_react_agent(
        model, [add_source, search_knowledge], prompt=SYSTEM_PROMPT
    )
