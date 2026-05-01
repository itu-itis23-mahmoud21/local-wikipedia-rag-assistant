"""Tests for Streamlit app helper functions."""

from pathlib import Path
import unittest

from app import (
    answer_to_chat_message,
    error_to_chat_message,
    get_system_status,
    handle_user_query,
    initialize_session_state,
    normalize_source,
)
from src import config
from src.generator import GeneratedAnswer


class FakeDB:
    """Fake metadata database for app status tests."""

    def __init__(self) -> None:
        self.db_path = Path("fake.sqlite")
        self.initialized = False

    def init_schema(self) -> None:
        """Record schema initialization."""

        self.initialized = True

    def get_summary_counts(self) -> dict:
        """Return deterministic counts."""

        return {
            "entities": 100,
            "documents": 100,
            "chunks": 250,
            "people": 50,
            "places": 50,
            "successful_documents": 98,
            "failed_documents": 2,
        }


class FakeVectorStore:
    """Fake vector store for status tests."""

    def count(self) -> int:
        """Return deterministic vector count."""

        return 250


class FakeGenerator:
    """Fake answer generator for query handling tests."""

    def answer_query(self, query: str, top_k: int | None = None) -> GeneratedAnswer:
        """Return a deterministic generated answer."""

        return GeneratedAnswer(
            query=query,
            answer="Local answer.",
            route="person",
            context="Context used.",
            sources=[{"rank": 1, "entity": "Albert Einstein"}],
            model=config.OLLAMA_GENERATION_MODEL,
        )


class FailingGenerator:
    """Fake generator that simulates a runtime failure."""

    def answer_query(self, query: str, top_k: int | None = None) -> GeneratedAnswer:
        """Raise an Ollama-like runtime error."""

        raise RuntimeError("Ollama connection failed")


class TestAppHelpers(unittest.TestCase):
    """Tests for pure helper behavior in app.py."""

    def test_initialize_session_state_creates_messages_list_when_missing(self) -> None:
        """initialize_session_state should create an empty messages list."""

        state: dict = {}

        initialize_session_state(state)

        self.assertEqual(state["messages"], [])

    def test_get_system_status_handles_database_counts(self) -> None:
        """System status should include database counts and vector count."""

        status = get_system_status(
            db_factory=FakeDB,
            vector_store_factory=FakeVectorStore,
        )

        self.assertEqual(status["counts"]["entities"], 100)
        self.assertEqual(status["counts"]["people"], 50)
        self.assertEqual(status["counts"]["places"], 50)
        self.assertEqual(status["vector_count"], 250)
        self.assertIsNone(status["database_error"])
        self.assertIsNone(status["vector_error"])

    def test_normalize_source_handles_expected_source_fields(self) -> None:
        """Source normalization should return all display fields."""

        source = normalize_source(
            {
                "rank": 1,
                "entity": "Albert Einstein",
                "entity_type": "person",
                "source_url": "https://example.test/einstein",
                "chunk_id": 7,
                "distance": 0.2,
                "preview": "Preview text",
            }
        )

        self.assertEqual(source["rank"], 1)
        self.assertEqual(source["entity"], "Albert Einstein")
        self.assertEqual(source["entity_type"], "person")
        self.assertEqual(source["source_url"], "https://example.test/einstein")
        self.assertEqual(source["chunk_id"], 7)
        self.assertEqual(source["distance"], 0.2)
        self.assertEqual(source["preview"], "Preview text")

    def test_answer_to_chat_message_converts_generated_answer(self) -> None:
        """GeneratedAnswer should become an assistant chat message."""

        answer = GeneratedAnswer(
            query="Who was Albert Einstein?",
            answer="Albert Einstein was a physicist.",
            route="person",
            context="Context.",
            sources=[{"rank": 1}],
            model=config.OLLAMA_GENERATION_MODEL,
        )

        message = answer_to_chat_message(answer)

        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["content"], "Albert Einstein was a physicist.")
        self.assertEqual(message["route"], "person")
        self.assertEqual(message["sources"], [{"rank": 1}])
        self.assertEqual(message["context"], "Context.")
        self.assertFalse(message["error"])

    def test_handle_user_query_returns_assistant_message(self) -> None:
        """handle_user_query should convert generator output to a chat message."""

        message = handle_user_query("Who was Albert Einstein?", generator=FakeGenerator())

        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["content"], "Local answer.")
        self.assertEqual(message["route"], "person")
        self.assertFalse(message["error"])

    def test_runtime_exception_becomes_friendly_error_message(self) -> None:
        """Expected runtime failures should not expose raw stack traces."""

        message = handle_user_query(
            "Who was Albert Einstein?",
            generator=FailingGenerator(),
        )

        self.assertEqual(message["role"], "assistant")
        self.assertTrue(message["error"])
        self.assertIn("local Ollama", message["content"])
        self.assertIn("Details:", message["content"])

    def test_error_to_chat_message_handles_vector_store_error_text(self) -> None:
        """Vector-store failures should get a clear setup-oriented message."""

        message = error_to_chat_message(RuntimeError("Chroma collection not found"))

        self.assertTrue(message["error"])
        self.assertIn("vector store is not ready", message["content"])
