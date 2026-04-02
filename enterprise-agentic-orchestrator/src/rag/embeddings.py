"""OpenAI embedding generation wrapper for the RAG pipeline.

Provides a thin wrapper around the OpenAI Embeddings API that reads
model configuration from config/config.yaml when available, falling
back to ``text-embedding-3-small`` by default.
"""

from __future__ import annotations

from pathlib import Path

import openai


def _default_model() -> str:
    """Read the embedding model name from config.yaml, or return fallback."""
    try:
        from src.config.settings import ConfigLoader

        config = ConfigLoader(config_dir=Path("config"))
        return config.app().providers.embedding_model
    except Exception:
        return "text-embedding-3-small"


def embed_texts(
    texts: list[str],
    model: str | None = None,
) -> list[list[float]]:
    """Generate embeddings for a list of text strings.

    Parameters
    ----------
    texts:
        Text strings to embed.  An empty list returns an empty list.
    model:
        OpenAI embedding model name.  When *None*, the model name is
        read from ``config/config.yaml`` (key ``providers.embedding_model``),
        falling back to ``text-embedding-3-small``.

    Returns
    -------
    list[list[float]]
        One embedding vector per input text (1536 dimensions for
        text-embedding-3-small).
    """
    if not texts:
        return []

    if model is None:
        model = _default_model()

    client = openai.OpenAI()
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]
