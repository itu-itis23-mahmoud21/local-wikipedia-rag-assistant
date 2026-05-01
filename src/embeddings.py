"""Local embedding generation placeholder.

This module will call Ollama's local embedding model, `nomic-embed-text`, to
convert chunks and queries into vectors without using external embedding APIs.
"""


def embed_text(text: str) -> list[float]:
    """Generate an embedding for text using the local Ollama model."""

    raise NotImplementedError("Embedding generation is not implemented yet.")
