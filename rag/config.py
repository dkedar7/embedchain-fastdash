"""Configuration: OpenRouter model, local embeddings, chunking, retrieval."""

from __future__ import annotations

import os

# LLM via OpenRouter (chat only — no embeddings). Server-side key, never a UI input.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.environ.get(
    "KNOWLEDGE_CHAT_MODEL", "anthropic/claude-haiku-4.5"
).strip()
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Local, key-free semantic embeddings (FastEmbed / ONNX).
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5").strip()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVE_K = 5


def has_llm() -> bool:
    """True when an OpenRouter key is configured (else the app runs read-only)."""
    return bool(OPENROUTER_API_KEY)
