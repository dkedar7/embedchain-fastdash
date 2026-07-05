# Production image. uv installs from the lockfile; the FastEmbed embedding model
# is baked in at build time so the first request isn't slow. gunicorn (gthread,
# 1 worker) serves the Fast Dash app + its Flask-SocketIO chat stream.
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/usr/local \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --extra prod

# Pre-download the FastEmbed model into the image (first-request warmup).
RUN python -c "from langchain_community.embeddings import FastEmbedEmbeddings; \
FastEmbedEmbeddings(model_name='BAAI/bge-small-en-v1.5').embed_query('warmup')"

COPY rag ./rag
COPY app.py README.md ./
RUN uv sync --frozen --no-dev --extra prod

EXPOSE 8080

# Single worker (in-process chat history + per-session knowledge bases live in
# one process); gthread handles concurrent callbacks and socket.io long-polls.
CMD ["gunicorn", "app:server", \
     "--worker-class", "gthread", "--workers", "1", "--threads", "8", \
     "--bind", "0.0.0.0:8080", "--timeout", "120", "--access-logfile", "-"]
