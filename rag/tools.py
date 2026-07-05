"""LangGraph tools the RAG agent uses to ingest and retrieve.

Both read the chat session's ``thread_id`` from the injected ``RunnableConfig``
so each conversation operates on its own knowledge base.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from .config import RETRIEVE_K
from .knowledge import format_passages, ingest, retrieve


def _thread_id(config: RunnableConfig) -> str:
    return ((config or {}).get("configurable") or {}).get("thread_id") or "default"


@tool
def add_source(source: str, config: RunnableConfig) -> str:
    """Add a source to the knowledge base so it can be searched.

    ``source`` may be a web page URL, a YouTube link, a PDF URL, or raw text to
    remember. Call this whenever the user shares something to add.
    """
    try:
        n, label = ingest(_thread_id(config), source)
    except Exception as exc:  # noqa: BLE001 — surface a friendly message to the model
        return (
            f"Could not add that source ({type(exc).__name__}). "
            "Try a different URL, or paste the text directly."
        )
    if n == 0:
        return f"Nothing could be extracted from {source!r}."
    return (
        f"Added '{label}' to the knowledge base ({n} chunks). "
        "You can now answer questions about it."
    )


@tool
def search_knowledge(query: str, config: RunnableConfig) -> str:
    """Search the knowledge base for passages relevant to a question.

    Call this before answering any question about the user's sources. Returns
    numbered passages, each tagged with its source, for inline citation.
    """
    docs = retrieve(_thread_id(config), query, k=RETRIEVE_K)
    return format_passages(docs)
