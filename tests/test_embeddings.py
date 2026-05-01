"""Tests for the local Ollama embedding client."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src import config
from src.embeddings import (
    EmbeddingError,
    OllamaEmbeddingClient,
    embed_text,
    ensure_embedding_vector,
)


class TestOllamaEmbeddingClient(unittest.TestCase):
    """Tests for embedding generation with mocked Ollama calls."""

    def test_embed_text_returns_float_list_for_embedding_shape(self) -> None:
        """embed_text should parse {'embedding': [...]} responses."""

        fake_embed = Mock(return_value={"embedding": [1, 2.5, 3]})
        fake_ollama = SimpleNamespace(embed=fake_embed)

        with patch("src.embeddings.ollama", fake_ollama):
            result = OllamaEmbeddingClient().embed_text("hello world")

        self.assertEqual(result, [1.0, 2.5, 3.0])
        fake_embed.assert_called_once_with(
            model=config.OLLAMA_EMBEDDING_MODEL,
            input="hello world",
        )

    def test_embed_text_returns_float_list_for_embeddings_shape(self) -> None:
        """embed_text should parse {'embeddings': [[...]]} responses."""

        fake_ollama = SimpleNamespace(
            embed=Mock(return_value={"embeddings": [[0.1, 2, 3.5]]})
        )

        with patch("src.embeddings.ollama", fake_ollama):
            result = OllamaEmbeddingClient().embed_text("hello")

        self.assertEqual(result, [0.1, 2.0, 3.5])

    def test_embed_text_rejects_blank_text(self) -> None:
        """Blank text should be rejected before calling Ollama."""

        with self.assertRaises(ValueError):
            OllamaEmbeddingClient().embed_text("   ")

    def test_embed_text_raises_embedding_error_on_ollama_exception(self) -> None:
        """Ollama exceptions should be wrapped in EmbeddingError."""

        fake_ollama = SimpleNamespace(embed=Mock(side_effect=RuntimeError("offline")))

        with patch("src.embeddings.ollama", fake_ollama):
            with self.assertRaises(EmbeddingError) as context:
                OllamaEmbeddingClient().embed_text("hello")

        message = str(context.exception)
        self.assertIn("Ollama", message)
        self.assertIn(config.OLLAMA_EMBEDDING_MODEL, message)

    def test_embed_text_raises_embedding_error_on_missing_embedding_key(self) -> None:
        """Missing embedding keys should raise EmbeddingError."""

        fake_ollama = SimpleNamespace(embed=Mock(return_value={"model": "test"}))

        with patch("src.embeddings.ollama", fake_ollama):
            with self.assertRaises(EmbeddingError):
                OllamaEmbeddingClient().embed_text("hello")

    def test_embed_text_raises_embedding_error_on_empty_embedding(self) -> None:
        """Empty embedding vectors should raise EmbeddingError."""

        fake_ollama = SimpleNamespace(embed=Mock(return_value={"embedding": []}))

        with patch("src.embeddings.ollama", fake_ollama):
            with self.assertRaises(EmbeddingError):
                OllamaEmbeddingClient().embed_text("hello")

    def test_embed_text_raises_embedding_error_on_non_numeric_values(self) -> None:
        """Non-numeric vector values should raise EmbeddingError."""

        fake_ollama = SimpleNamespace(embed=Mock(return_value={"embedding": [1, "x"]}))

        with patch("src.embeddings.ollama", fake_ollama):
            with self.assertRaises(EmbeddingError):
                OllamaEmbeddingClient().embed_text("hello")

    def test_embed_texts_preserves_order_and_calls_each_text(self) -> None:
        """embed_texts should embed each text in order."""

        calls: list[str] = []

        def fake_embed(model: str, input: str) -> dict:
            calls.append(input)
            return {"embedding": [len(input), 1]}

        fake_ollama = SimpleNamespace(embed=fake_embed)

        with patch("src.embeddings.ollama", fake_ollama):
            results = OllamaEmbeddingClient().embed_texts(["one", "three"])

        self.assertEqual(calls, ["one", "three"])
        self.assertEqual(results, [[3.0, 1.0], [5.0, 1.0]])

    def test_embed_texts_rejects_empty_input_list(self) -> None:
        """Empty text lists should be rejected."""

        with self.assertRaises(ValueError):
            OllamaEmbeddingClient().embed_texts([])

    def test_embed_texts_rejects_blank_item(self) -> None:
        """Blank items in text lists should be rejected."""

        with self.assertRaises(ValueError):
            OllamaEmbeddingClient().embed_texts(["valid", "  "])

    def test_embed_chunk_records_returns_expected_records(self) -> None:
        """Chunk rows should be converted into embedding records."""

        fake_ollama = SimpleNamespace(embed=Mock(return_value={"embedding": [1, 2]}))
        chunks = [{"id": 10, "text": "chunk text"}]

        with patch("src.embeddings.ollama", fake_ollama):
            records = OllamaEmbeddingClient(model="test-model").embed_chunk_records(
                chunks
            )

        self.assertEqual(
            records,
            [
                {
                    "chunk_id": 10,
                    "text": "chunk text",
                    "embedding": [1.0, 2.0],
                    "model": "test-model",
                }
            ],
        )

    def test_embed_chunk_records_rejects_missing_id(self) -> None:
        """Chunk records must include an id."""

        with self.assertRaises(ValueError):
            OllamaEmbeddingClient().embed_chunk_records([{"text": "chunk"}])

    def test_embed_chunk_records_rejects_missing_text(self) -> None:
        """Chunk records must include text."""

        with self.assertRaises(ValueError):
            OllamaEmbeddingClient().embed_chunk_records([{"id": 1}])

    def test_ensure_embedding_vector_converts_ints_and_floats(self) -> None:
        """ensure_embedding_vector should return a list of floats."""

        self.assertEqual(ensure_embedding_vector([1, 2.5, 3]), [1.0, 2.5, 3.0])

    def test_convenience_embed_text_uses_requested_model(self) -> None:
        """Module-level embed_text should create a client with the given model."""

        fake_embed = Mock(return_value={"embedding": [0.25, 0.75]})
        fake_ollama = SimpleNamespace(embed=fake_embed)

        with patch("src.embeddings.ollama", fake_ollama):
            result = embed_text("hello", model="custom-model")

        self.assertEqual(result, [0.25, 0.75])
        fake_embed.assert_called_once_with(model="custom-model", input="hello")
