"""The knowledge base: per-session vector store + source ingestion + retrieval.

Replaces Embedchain. Sources (web pages, YouTube transcripts, PDFs, raw text)
are loaded with LangChain loaders, split, embedded locally with FastEmbed, and
stored in an in-memory vector store keyed by chat session (``thread_id``) so
each conversation has its own knowledge base.
"""

from __future__ import annotations

import os
import re

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL, RETRIEVE_K

# WebBaseLoader warns without a UA; set a polite default.
os.environ.setdefault("USER_AGENT", "embedchain-fastdash-rag/1.0")

_YOUTUBE = re.compile(r"(youtube\.com|youtu\.be)", re.IGNORECASE)
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)

# --- embeddings (lazy: the model downloads on first use) ------------------ #
_embeddings = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_community.embeddings import FastEmbedEmbeddings

        _embeddings = FastEmbedEmbeddings(model_name=EMBED_MODEL)
    return _embeddings


# --- per-session vector stores -------------------------------------------- #
_STORES: dict[str, InMemoryVectorStore] = {}


def get_store(thread_id: str) -> InMemoryVectorStore:
    store = _STORES.get(thread_id)
    if store is None:
        store = InMemoryVectorStore(_get_embeddings())
        _STORES[thread_id] = store
    return store


def reset(thread_id: str) -> None:
    _STORES.pop(thread_id, None)


# --- loading -------------------------------------------------------------- #
def _load(source: str) -> list[Document]:
    """Dispatch a source string to the right loader; raw text is stored as-is."""
    s = source.strip()
    if s.lower().startswith(("http://", "https://")):
        if _YOUTUBE.search(s):
            from langchain_community.document_loaders import YoutubeLoader

            return YoutubeLoader.from_youtube_url(s, add_video_info=False).load()
        if s.lower().split("?")[0].endswith(".pdf"):
            from langchain_community.document_loaders import PyPDFLoader

            return PyPDFLoader(s).load()
        from langchain_community.document_loaders import WebBaseLoader

        return WebBaseLoader(s).load()
    return [Document(page_content=s, metadata={"source": "pasted text"})]


def ingest(thread_id: str, source: str) -> tuple[int, str]:
    """Load, split, embed, and store a source. Returns ``(n_chunks, label)``."""
    docs = _load(source)
    for d in docs:
        d.metadata.setdefault("source", source)
    chunks = _splitter.split_documents(docs)
    if not chunks:
        return 0, source
    get_store(thread_id).add_documents(chunks)
    label = (docs[0].metadata.get("title") or source).strip()
    return len(chunks), label


# --- retrieval ------------------------------------------------------------ #
def retrieve(thread_id: str, query: str, k: int = RETRIEVE_K) -> list[Document]:
    store = _STORES.get(thread_id)
    if store is None:
        return []
    return store.similarity_search(query, k=k)


def format_passages(docs: list[Document]) -> str:
    """Number retrieved passages and tag each with its source for citation."""
    if not docs:
        return (
            "No relevant passages found. The knowledge base may be empty — ask the "
            "user to add a source first."
        )
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        parts.append(f"[{i}] (source: {src})\n{d.page_content.strip()}")
    return "\n\n".join(parts)
