"""Local Ollama embedding client.

This module converts text into vectors by calling the local Ollama Python
package. It does not use external embedding APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

from src import config

try:
    import ollama
except ImportError:  # pragma: no cover - exercised through EmbeddingError tests.
    ollama = None


class EmbeddingError(Exception):
    """Raised when local embedding generation fails."""


@dataclass(frozen=True)
class EmbeddingResult:
    """Text, vector, and model metadata for one embedding."""

    text: str
    embedding: list[float]
    model: str


class OllamaEmbeddingClient:
    """Client for generating embeddings with a local Ollama model."""

    def __init__(self, model: str = config.OLLAMA_EMBEDDING_MODEL) -> None:
        """Create an embedding client for a local Ollama model."""

        if not model.strip():
            raise ValueError("model must not be blank.")
        self.model = model

    def embed_text(self, text: str) -> list[float]:
        """Embed one text string using the configured local Ollama model."""

        clean_text = _validate_text(text)
        response = self._call_ollama(clean_text)
        return self._extract_embedding(response)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in order and return vectors in the same order."""

        if not texts:
            raise ValueError("texts must not be empty.")

        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"texts[{index}] must not be blank.")

        return [self.embed_text(text) for text in texts]

    def embed_chunk_records(self, chunks: list[dict]) -> list[dict]:
        """Embed chunk rows from MetadataDB.list_chunks()."""

        records: list[dict] = []
        for index, chunk in enumerate(chunks):
            if "id" not in chunk:
                raise ValueError(f"chunks[{index}] is missing required id.")
            if "text" not in chunk:
                raise ValueError(f"chunks[{index}] is missing required text.")

            text = chunk["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"chunks[{index}] text must not be blank.")

            records.append(
                {
                    "chunk_id": chunk["id"],
                    "text": text,
                    "embedding": self.embed_text(text),
                    "model": self.model,
                }
            )

        return records

    def _call_ollama(self, text: str) -> object:
        """Call the local Ollama embedding API with compatibility fallback."""

        if ollama is None:
            raise EmbeddingError(
                "Ollama Python package is not available for local embedding "
                f"model '{self.model}'."
            )

        try:
            if hasattr(ollama, "embed"):
                return ollama.embed(model=self.model, input=text)
            if hasattr(ollama, "embeddings"):
                return ollama.embeddings(model=self.model, prompt=text)
        except Exception as exc:
            raise EmbeddingError(
                f"Ollama local embedding model '{self.model}' failed: {exc}"
            ) from exc

        raise EmbeddingError(
            "Ollama Python package does not expose an embedding method for "
            f"local model '{self.model}'."
        )

    def _extract_embedding(self, response: object) -> list[float]:
        """Extract and validate an embedding vector from common Ollama shapes."""

        if isinstance(response, dict):
            if "embedding" in response:
                return ensure_embedding_vector(response["embedding"])

            if "embeddings" in response:
                embeddings = response["embeddings"]
                if not isinstance(embeddings, (list, tuple)) or not embeddings:
                    raise EmbeddingError(
                        "Ollama local embedding model "
                        f"'{self.model}' returned empty embeddings."
                    )
                return ensure_embedding_vector(embeddings[0])

            raise EmbeddingError(
                "Ollama local embedding model "
                f"'{self.model}' returned no embedding data."
            )

        object_embedding = getattr(response, "embedding", None)
        if object_embedding is not None:
            return ensure_embedding_vector(object_embedding)

        object_embeddings = getattr(response, "embeddings", None)
        if object_embeddings is not None:
            if not isinstance(object_embeddings, (list, tuple)) or not object_embeddings:
                raise EmbeddingError(
                    "Ollama local embedding model "
                    f"'{self.model}' returned empty embeddings."
                )
            return ensure_embedding_vector(object_embeddings[0])

        raise EmbeddingError(
            "Ollama local embedding model "
            f"'{self.model}' returned malformed output."
        )


def embed_text(
    text: str,
    model: str = config.OLLAMA_EMBEDDING_MODEL,
) -> list[float]:
    """Generate one embedding with the local Ollama model."""

    return OllamaEmbeddingClient(model=model).embed_text(text)


def ensure_embedding_vector(value: object) -> list[float]:
    """Validate and convert an embedding vector to ``list[float]``."""

    if not isinstance(value, (list, tuple)):
        raise EmbeddingError("Ollama/local model returned a malformed vector.")
    if not value:
        raise EmbeddingError("Ollama/local model returned an empty vector.")

    vector: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise EmbeddingError(
                "Ollama/local model returned a non-numeric embedding value "
                f"at index {index}."
            )
        vector.append(float(item))

    return vector


def _validate_text(text: str) -> str:
    """Validate and strip text before sending it to Ollama."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must not be blank.")
    return text.strip()
