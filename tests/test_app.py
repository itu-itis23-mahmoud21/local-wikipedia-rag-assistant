"""Tests for Streamlit app helper functions."""

from pathlib import Path
import unittest

from app import (
    answer_to_chat_message,
    build_random_prompt,
    build_stopped_message,
    build_chat_export_text,
    error_to_chat_message,
    finish_generation_if_ready,
    format_assistant_caption,
    format_context_for_copy,
    format_entity_list_for_sidebar,
    format_sources_for_copy,
    format_status_metric_value,
    get_export_filename,
    get_system_status,
    handle_user_query,
    initialize_session_state,
    is_generation_active,
    normalize_source,
    set_random_prompt_in_chat_input,
    start_generation,
    stop_generation,
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
            "entities": 50,
            "documents": 100,
            "chunks": 250,
            "people": 25,
            "places": 25,
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


class FakeFuture:
    """Small future-like object for generation state tests."""

    def __init__(
        self,
        result: dict | None = None,
        exception: Exception | None = None,
        done: bool = True,
    ) -> None:
        self._result = result
        self._exception = exception
        self._done = done
        self.cancelled = False

    def done(self) -> bool:
        """Return whether the fake future has completed."""

        return self._done

    def result(self) -> dict:
        """Return the fake result or raise the fake exception."""

        if self._exception is not None:
            raise self._exception
        return self._result or _assistant_message("Finished answer.")

    def cancel(self) -> bool:
        """Record cancellation."""

        self.cancelled = True
        return True


class FakeExecutor:
    """Small executor-like object that returns a prebuilt fake future."""

    def __init__(self, future: FakeFuture) -> None:
        self.future = future
        self.submitted = None

    def submit(self, fn, *args, **kwargs) -> FakeFuture:
        """Record submitted work and return the fake future."""

        self.submitted = (fn, args, kwargs)
        return self.future


class TestAppHelpers(unittest.TestCase):
    """Tests for pure helper behavior in app.py."""

    def test_initialize_session_state_creates_messages_list_when_missing(self) -> None:
        """initialize_session_state should create chat and generation state keys."""

        state: dict = {}

        initialize_session_state(state)

        self.assertEqual(state["messages"], [])
        self.assertIsNone(state["active_request_id"])
        self.assertIsNone(state["active_future_request_id"])
        self.assertIsNone(state["active_future"])
        self.assertIsNone(state["active_prompt"])
        self.assertFalse(state["stop_requested"])
        self.assertFalse(state["is_generating"])
        self.assertIsNone(state["last_random_prompt"])

    def test_get_system_status_handles_database_counts(self) -> None:
        """System status should include database counts and vector count."""

        status = get_system_status(
            db_factory=FakeDB,
            vector_store_factory=FakeVectorStore,
        )

        self.assertEqual(status["counts"]["entities"], 50)
        self.assertEqual(status["counts"]["people"], 25)
        self.assertEqual(status["counts"]["places"], 25)
        self.assertEqual(status["vector_count"], 250)
        self.assertIsNone(status["database_error"])
        self.assertIsNone(status["vector_error"])

    def test_format_status_metric_value_formats_counts(self) -> None:
        """Sidebar metric values should be stable and readable."""

        self.assertEqual(format_status_metric_value(1000), "1,000")
        self.assertEqual(format_status_metric_value(None), "0")
        self.assertEqual(format_status_metric_value("ready"), "ready")

    def test_format_entity_list_for_sidebar_numbers_entities(self) -> None:
        """Sidebar entity lists should be numbered and readable."""

        text = format_entity_list_for_sidebar(["Albert Einstein", "Marie Curie"])

        self.assertIn("1. Albert Einstein", text)
        self.assertIn("2. Marie Curie", text)

    def test_format_entity_list_for_sidebar_handles_empty_list(self) -> None:
        """Empty sidebar entity lists should still render a useful note."""

        self.assertIn("No configured entities", format_entity_list_for_sidebar([]))

    def test_build_random_prompt_for_person(self) -> None:
        """Random prompt helper should format person starter questions."""

        self.assertEqual(
            build_random_prompt(entity_type="person", entity_name="Albert Einstein"),
            "Who is Albert Einstein?",
        )

    def test_build_random_prompt_for_place(self) -> None:
        """Random prompt helper should format place starter questions."""

        self.assertEqual(
            build_random_prompt(entity_type="place", entity_name="Eiffel Tower"),
            "Where is Eiffel Tower located?",
        )

    def test_build_random_prompt_rejects_invalid_entity_type(self) -> None:
        """Random prompt helper should reject invalid entity types."""

        with self.assertRaises(ValueError):
            build_random_prompt(entity_type="animal", entity_name="Example")

    def test_set_random_prompt_in_chat_input_updates_state(self) -> None:
        """Random prompt state helper should populate the keyed chat input."""

        state: dict = {}

        prompt = set_random_prompt_in_chat_input(
            state,
            prompt="Who is Ada Lovelace?",
        )

        self.assertEqual(prompt, "Who is Ada Lovelace?")
        self.assertEqual(state["main_chat_input"], "Who is Ada Lovelace?")
        self.assertEqual(state["last_random_prompt"], "Who is Ada Lovelace?")

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

    def test_format_sources_for_copy_includes_expected_fields(self) -> None:
        """Source copy text should include all displayed source fields."""

        text = format_sources_for_copy(
            [
                {
                    "rank": 1,
                    "entity": "Albert Einstein",
                    "entity_type": "person",
                    "source_url": "https://example.test/einstein",
                    "chunk_id": 7,
                    "distance": 0.2,
                    "preview": "Preview text",
                }
            ]
        )

        self.assertIn("Source 1", text)
        self.assertIn("Albert Einstein", text)
        self.assertIn("person", text)
        self.assertIn("https://example.test/einstein", text)
        self.assertIn("Chunk ID: 7", text)
        self.assertIn("Distance: 0.2", text)
        self.assertIn("Preview text", text)

    def test_format_sources_for_copy_handles_empty_sources(self) -> None:
        """Empty source lists should produce a clear copy string."""

        self.assertEqual(
            format_sources_for_copy([]),
            "No retrieved sources available.",
        )

    def test_format_context_for_copy_strips_context(self) -> None:
        """Context copy text should be stripped safely."""

        self.assertEqual(format_context_for_copy("  Context text. \n"), "Context text.")

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

    def test_format_assistant_caption_uses_middle_dot_separator(self) -> None:
        """Assistant captions should use the polished metadata format."""

        caption = format_assistant_caption(
            {"route": "person", "model": config.OLLAMA_GENERATION_MODEL}
        )

        self.assertEqual(
            caption,
            f"Route: person · Model: {config.OLLAMA_GENERATION_MODEL}",
        )

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

    def test_start_generation_sets_active_request_state(self) -> None:
        """start_generation should submit work and lock generation state."""

        state: dict = {}
        future = FakeFuture(done=False)
        executor = FakeExecutor(future)

        started = start_generation(
            " Who was Albert Einstein? ",
            session_state=state,
            executor=executor,
            query_handler=lambda prompt: _assistant_message(prompt),
        )

        self.assertTrue(started)
        self.assertTrue(state["is_generating"])
        self.assertTrue(is_generation_active(state))
        self.assertIs(state["active_future"], future)
        self.assertIsNotNone(state["active_request_id"])
        self.assertEqual(state["active_future_request_id"], state["active_request_id"])
        self.assertEqual(state["active_prompt"], "Who was Albert Einstein?")
        self.assertFalse(state["stop_requested"])
        self.assertEqual(
            state["messages"],
            [{"role": "user", "content": "Who was Albert Einstein?"}],
        )
        submitted = executor.submitted
        assert submitted is not None
        self.assertEqual(submitted[1], ("Who was Albert Einstein?",))

    def test_start_generation_ignores_prompt_while_generating(self) -> None:
        """A second prompt should not be accepted while generation is active."""

        state: dict = {}
        first_future = FakeFuture(done=False)
        first_executor = FakeExecutor(first_future)
        second_executor = FakeExecutor(FakeFuture(done=False))

        self.assertTrue(
            start_generation(
                "First prompt",
                session_state=state,
                executor=first_executor,
            )
        )
        self.assertFalse(
            start_generation(
                "Second prompt",
                session_state=state,
                executor=second_executor,
            )
        )

        self.assertEqual(len(state["messages"]), 1)
        self.assertIs(state["active_future"], first_future)
        self.assertIsNone(second_executor.submitted)

    def test_stop_generation_marks_stopped_and_appends_message(self) -> None:
        """stop_generation should cancel the active future and add a stopped message."""

        state: dict = {}
        future = FakeFuture(done=False)
        start_generation("Prompt", session_state=state, executor=FakeExecutor(future))

        stopped = stop_generation(state)

        self.assertTrue(stopped)
        self.assertTrue(future.cancelled)
        self.assertFalse(state["is_generating"])
        self.assertIsNone(state["active_future"])
        self.assertTrue(state["stop_requested"])
        self.assertEqual(state["messages"][-1]["content"], "Generation stopped by user.")
        self.assertTrue(state["messages"][-1]["stopped"])

    def test_finish_generation_appends_answer_when_ready(self) -> None:
        """A completed active future should append its assistant message."""

        state: dict = {}
        expected = _assistant_message("Finished answer.")
        start_generation(
            "Prompt",
            session_state=state,
            executor=FakeExecutor(FakeFuture(result=expected, done=True)),
        )

        changed = finish_generation_if_ready(state)

        self.assertTrue(changed)
        self.assertFalse(state["is_generating"])
        self.assertEqual(state["messages"][-1], expected)

    def test_finish_generation_ignores_answer_when_stop_was_requested(self) -> None:
        """A stopped request should not append a normal assistant answer."""

        state: dict = {}
        start_generation(
            "Prompt",
            session_state=state,
            executor=FakeExecutor(FakeFuture(result=_assistant_message("Late answer."))),
        )
        state["stop_requested"] = True

        changed = finish_generation_if_ready(state)

        self.assertTrue(changed)
        self.assertEqual(len(state["messages"]), 1)
        self.assertFalse(state["is_generating"])

    def test_finish_generation_ignores_stale_request_id(self) -> None:
        """A future with a stale request id should not append its result."""

        state: dict = {}
        start_generation(
            "Prompt",
            session_state=state,
            executor=FakeExecutor(FakeFuture(result=_assistant_message("Stale answer."))),
        )
        state["active_request_id"] = "newer-request"

        changed = finish_generation_if_ready(state)

        self.assertTrue(changed)
        self.assertEqual(len(state["messages"]), 1)
        self.assertFalse(state["is_generating"])

    def test_finish_generation_converts_future_exception_to_friendly_error(self) -> None:
        """Future exceptions should become friendly assistant error messages."""

        state: dict = {}
        start_generation(
            "Prompt",
            session_state=state,
            executor=FakeExecutor(
                FakeFuture(exception=RuntimeError("Ollama connection failed"))
            ),
        )

        changed = finish_generation_if_ready(state)

        self.assertTrue(changed)
        self.assertTrue(state["messages"][-1]["error"])
        self.assertIn("local Ollama", state["messages"][-1]["content"])

    def test_finish_generation_does_nothing_for_pending_future(self) -> None:
        """Pending futures should remain active and not append an answer."""

        state: dict = {}
        start_generation(
            "Prompt",
            session_state=state,
            executor=FakeExecutor(FakeFuture(done=False)),
        )

        changed = finish_generation_if_ready(state)

        self.assertFalse(changed)
        self.assertTrue(state["is_generating"])
        self.assertEqual(len(state["messages"]), 1)

    def test_build_stopped_message_shape(self) -> None:
        """Stopped messages should be assistant messages with no sources."""

        message = build_stopped_message()

        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["content"], "Generation stopped by user.")
        self.assertEqual(message["sources"], [])
        self.assertTrue(message["stopped"])

    def test_build_chat_export_text_includes_assistant_answers(self) -> None:
        """Chat export should always include assistant answers."""

        export_text = build_chat_export_text(
            _sample_messages(),
            include_user_questions=False,
            include_answers=False,
        )

        self.assertIn("Assistant answer:", export_text)
        self.assertIn("Albert Einstein was a physicist.", export_text)

    def test_build_chat_export_text_includes_user_questions_when_selected(self) -> None:
        """Chat export should include user messages when requested."""

        export_text = build_chat_export_text(
            _sample_messages(),
            include_user_questions=True,
        )

        self.assertIn("User question:", export_text)
        self.assertIn("Who was Albert Einstein?", export_text)

    def test_build_chat_export_text_excludes_user_questions_when_not_selected(self) -> None:
        """Chat export should omit user messages when not requested."""

        export_text = build_chat_export_text(
            _sample_messages(),
            include_user_questions=False,
        )

        self.assertNotIn("User question:", export_text)
        self.assertNotIn("Who was Albert Einstein?", export_text)
        self.assertIn("Albert Einstein was a physicist.", export_text)

    def test_build_chat_export_text_includes_sources_only_when_selected(self) -> None:
        """Retrieved sources should be controlled by the export option."""

        without_sources = build_chat_export_text(
            _sample_messages(),
            include_sources=False,
        )
        with_sources = build_chat_export_text(
            _sample_messages(),
            include_sources=True,
        )

        self.assertNotIn("Retrieved sources:", without_sources)
        self.assertIn("Retrieved sources:", with_sources)
        self.assertIn("Chunk ID: 7", with_sources)

    def test_build_chat_export_text_includes_context_only_when_selected(self) -> None:
        """Retrieved context should be controlled by the export option."""

        without_context = build_chat_export_text(
            _sample_messages(),
            include_context=False,
        )
        with_context = build_chat_export_text(
            _sample_messages(),
            include_context=True,
        )

        self.assertNotIn("Retrieved context used:", without_context)
        self.assertIn("Retrieved context used:", with_context)
        self.assertIn("Einstein context.", with_context)

    def test_build_chat_export_text_includes_metadata_only_when_selected(self) -> None:
        """Route/model metadata should be controlled by the export option."""

        without_metadata = build_chat_export_text(
            _sample_messages(),
            include_metadata=False,
        )
        with_metadata = build_chat_export_text(
            _sample_messages(),
            include_metadata=True,
        )

        self.assertNotIn("Metadata:", without_metadata)
        self.assertIn("Metadata:", with_metadata)
        self.assertIn("Route: person", with_metadata)
        self.assertIn(config.OLLAMA_GENERATION_MODEL, with_metadata)

    def test_build_chat_export_text_handles_empty_chat_history(self) -> None:
        """Empty chat exports should still produce a readable TXT body."""

        export_text = build_chat_export_text([])

        self.assertIn("Local Wikipedia RAG Assistant Chat Export", export_text)
        self.assertIn("No chat messages are available.", export_text)

    def test_get_export_filename_returns_txt_filename(self) -> None:
        """Export filename should be a .txt file."""

        filename = get_export_filename()

        self.assertTrue(filename.endswith(".txt"))
        self.assertIn("chat_export", filename)


def _assistant_message(content: str) -> dict:
    """Build a representative assistant chat message for app tests."""

    return {
        "role": "assistant",
        "content": content,
        "route": "person",
        "sources": [],
        "context": "",
        "model": config.OLLAMA_GENERATION_MODEL,
        "error": False,
    }


def _sample_messages() -> list[dict]:
    """Return representative chat messages for export tests."""

    return [
        {"role": "user", "content": "Who was Albert Einstein?"},
        {
            "role": "assistant",
            "content": "Albert Einstein was a physicist.",
            "route": "person",
            "sources": [
                {
                    "rank": 1,
                    "entity": "Albert Einstein",
                    "entity_type": "person",
                    "source_url": "https://example.test/einstein",
                    "chunk_id": 7,
                    "distance": 0.2,
                    "preview": "Preview text",
                }
            ],
            "context": "Einstein context.",
            "model": config.OLLAMA_GENERATION_MODEL,
            "error": False,
        },
    ]
