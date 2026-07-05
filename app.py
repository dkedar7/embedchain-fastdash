"""Chat with your knowledge base.

The whole app is Fast Dash's native chat mode wrapped around a LangGraph RAG
agent on OpenRouter. Add web pages, PDFs, YouTube videos, or text by chatting,
then ask questions — the agent retrieves and answers with citations.

A modern rebuild of the original Embedchain demo: no Embedchain, no OpenAI key
input, no form — just a conversational agent.
"""

from __future__ import annotations

from fast_dash import FastDash

from rag.graph import build_graph

graph = build_graph()

app = FastDash(
    callback_fn=graph,
    chat=True,
    title="Chat with your knowledge base",
    subheader=(
        "Share a web page, PDF, YouTube link, or paste text — then ask. "
        "A LangGraph + OpenRouter RAG agent, built on Fast Dash."
    ),
    github_url="https://github.com/dkedar7/embedchain-fastdash",
    accent="indigo",
)
server = app.app.server  # WSGI entry point for gunicorn / deployment

if __name__ == "__main__":
    app.run()
