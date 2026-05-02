"""Tests for local grounded answer generation."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src import config
from src.generator import (
    GeneratedAnswer,
    GenerationError,
    OllamaAnswerGenerator,
    generate_answer,
)
from src.query_router import QueryRoute, ROUTE_PERSON
from src.retriever import RetrievedContext
from src.vector_store import VectorSearchResult


class FakeRetriever:
    """Fake retriever for answer-query tests."""

    def __init__(self) -> None:
        self.retrieve_calls: list[dict] = []
        self.results = [
            VectorSearchResult(
                vector_id="chunk-1",
                text="Albert Einstein was a physicist.",
                metadata={
                    "chunk_id": 1,
                    "entity": "Albert Einstein",
                    "entity_type": "person",
                    "source_url": "https://example.test/einstein",
                },
                distance=0.1,
            )
        ]
        self.route = QueryRoute(
            route=ROUTE_PERSON,
            matched_people=["Albert Einstein"],
            matched_places=[],
            reason="known person mentioned",
        )

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievedContext:
        """Return deterministic retrieved context."""

        self.retrieve_calls.append({"query": query, "top_k": top_k})
        return RetrievedContext(query=query, route=self.route, results=self.results)

    def format_context(
        self,
        results: list[VectorSearchResult],
        max_chars: int = 4000,
    ) -> str:
        """Return a simple context block."""

        return (
            "[Source 1] entity=Albert Einstein, type=person, "
            "url=https://example.test/einstein\n"
            "Albert Einstein was a physicist."
        )

    def get_source_summary(self, results: list[VectorSearchResult]) -> list[dict]:
        """Return deterministic source summaries."""

        return [
            {
                "rank": 1,
                "entity": "Albert Einstein",
                "entity_type": "person",
                "source_url": "https://example.test/einstein",
                "chunk_id": 1,
                "distance": 0.1,
                "preview": "Albert Einstein was a physicist.",
            }
        ]


class TestOllamaAnswerGenerator(unittest.TestCase):
    """Tests for prompt construction and local generation behavior."""

    def test_build_prompt_includes_query(self) -> None:
        """Prompt should include the user query."""

        prompt = OllamaAnswerGenerator().build_prompt(
            "Who was Albert Einstein?",
            "Albert Einstein was a physicist.",
        )

        self.assertIn("Who was Albert Einstein?", prompt)

    def test_build_prompt_includes_context(self) -> None:
        """Prompt should include retrieved context."""

        prompt = OllamaAnswerGenerator().build_prompt(
            "Who was Albert Einstein?",
            "Albert Einstein was a physicist.",
        )

        self.assertIn("Albert Einstein was a physicist.", prompt)

    def test_build_prompt_instructs_use_only_retrieved_context(self) -> None:
        """Prompt should tell the model to use only retrieved context."""

        prompt = OllamaAnswerGenerator().build_prompt("Question?", "Context.")

        self.assertIn("Use only the retrieved context", prompt)
        self.assertIn("Do not use outside knowledge", prompt)

    def test_build_prompt_includes_i_do_not_know_instruction(self) -> None:
        """Prompt should include the required unknown-answer instruction."""

        prompt = OllamaAnswerGenerator().build_prompt("Question?", "Context.")

        self.assertIn("I don't know.", prompt)

    def test_build_prompt_guides_which_person_place_questions(self) -> None:
        """Prompt should prefer the main retrieved entity for which questions."""

        prompt = OllamaAnswerGenerator().build_prompt(
            "Which person is associated with computing?",
            "[Source 1] entity=Ada Lovelace, type=person\nAda Lovelace text.",
        )

        self.assertIn("which person", prompt)
        self.assertIn("which place", prompt)
        self.assertIn("main configured entity", prompt)

    def test_build_prompt_says_not_to_list_every_mentioned_name(self) -> None:
        """Prompt should avoid listing names merely mentioned in context."""

        prompt = OllamaAnswerGenerator().build_prompt("Question?", "Context.")

        self.assertIn("Do not list every person or place merely mentioned", prompt)

    def test_build_prompt_keeps_comparison_facts_separated(self) -> None:
        """Prompt should tell the model not to mix comparison facts."""

        prompt = OllamaAnswerGenerator().build_prompt(
            "Compare Albert Einstein and Marie Curie.",
            "Context.",
        )

        self.assertIn("Keep facts separated by entity", prompt)
        self.assertIn("do not transfer facts from one entity to another", prompt)

    def test_build_prompt_prevents_fact_transfer_between_compared_entities(self) -> None:
        """Prompt should say one-entity facts must stay under that entity."""

        prompt = OllamaAnswerGenerator().build_prompt(
            "Compare Lionel Messi and Cristiano Ronaldo.",
            "[Source 1] entity=Lionel Messi, type=person\nMessi won a trophy.",
        )

        self.assertIn("If a fact appears only in one entity's source/context", prompt)
        self.assertIn("do not assign it to the other entity", prompt)
        self.assertIn("Do not invent balanced statistics", prompt)

    def test_build_prompt_prefers_structured_comparison_output(self) -> None:
        """Prompt should guide comparison answers into separated sections."""

        prompt = OllamaAnswerGenerator().build_prompt(
            "Compare the Eiffel Tower and the Statue of Liberty.",
            "Context.",
        )

        self.assertIn("Entity A:", prompt)
        self.assertIn("Entity B:", prompt)
        self.assertIn("Comparison:", prompt)

    def test_build_prompt_discourages_irrelevant_unsupported_context_summary(self) -> None:
        """Unsupported answers should not list random retrieved entities."""

        prompt = OllamaAnswerGenerator().build_prompt(
            "Who is the president of Mars?",
            "[Source 1] entity=Lionel Messi, type=person\nUnrelated context.",
        )

        self.assertIn("do not list unrelated retrieved entities", prompt)
        self.assertIn("summarize irrelevant context", prompt)
        self.assertIn("Do not say \"it only mentions X\"", prompt)

    def test_blank_query_raises_value_error(self) -> None:
        """Blank queries should be rejected."""

        with self.assertRaises(ValueError):
            OllamaAnswerGenerator().build_prompt("  ", "Context.")

    def test_constructor_rejects_blank_model(self) -> None:
        """Model name must not be blank."""

        with self.assertRaises(ValueError):
            OllamaAnswerGenerator(model="  ")

    def test_constructor_rejects_invalid_temperature(self) -> None:
        """Temperature must be between zero and one."""

        with self.assertRaises(ValueError):
            OllamaAnswerGenerator(temperature=-0.1)

        with self.assertRaises(ValueError):
            OllamaAnswerGenerator(temperature=1.1)

    def test_generate_from_context_returns_unknown_without_ollama_for_blank_context(
        self,
    ) -> None:
        """Blank context should return I don't know without calling Ollama."""

        fake_generate = Mock()
        fake_ollama = SimpleNamespace(generate=fake_generate)

        with patch("src.generator.ollama", fake_ollama):
            answer = OllamaAnswerGenerator().generate_from_context("Question?", "  ")

        self.assertEqual(answer, "I don't know.")
        fake_generate.assert_not_called()

    def test_generate_from_context_parses_dict_response_shape(self) -> None:
        """Dict responses with response key should be parsed."""

        fake_generate = Mock(return_value={"response": " Albert Einstein was a physicist. "})
        fake_ollama = SimpleNamespace(generate=fake_generate)

        with patch("src.generator.ollama", fake_ollama):
            answer = OllamaAnswerGenerator().generate_from_context(
                "Who was Albert Einstein?",
                "Albert Einstein was a physicist.",
            )

        self.assertEqual(answer, "Albert Einstein was a physicist.")
        fake_generate.assert_called_once()
        _, kwargs = fake_generate.call_args
        self.assertEqual(kwargs["model"], config.OLLAMA_GENERATION_MODEL)
        self.assertEqual(kwargs["options"]["temperature"], config.DEFAULT_GENERATION_TEMPERATURE)

    def test_generate_from_context_parses_object_response_shape(self) -> None:
        """Object responses with .response should be parsed."""

        fake_ollama = SimpleNamespace(
            generate=Mock(return_value=SimpleNamespace(response="Object answer."))
        )

        with patch("src.generator.ollama", fake_ollama):
            answer = OllamaAnswerGenerator().generate_from_context(
                "Question?",
                "Context.",
            )

        self.assertEqual(answer, "Object answer.")

    def test_generate_from_context_raises_generation_error_on_ollama_exception(
        self,
    ) -> None:
        """Ollama exceptions should be wrapped in GenerationError."""

        fake_ollama = SimpleNamespace(generate=Mock(side_effect=RuntimeError("offline")))

        with patch("src.generator.ollama", fake_ollama):
            with self.assertRaises(GenerationError) as context:
                OllamaAnswerGenerator().generate_from_context("Question?", "Context.")

        message = str(context.exception)
        self.assertIn("Ollama", message)
        self.assertIn(config.OLLAMA_GENERATION_MODEL, message)

    def test_generate_from_context_raises_generation_error_on_blank_response(
        self,
    ) -> None:
        """Blank model output should raise GenerationError."""

        fake_ollama = SimpleNamespace(generate=Mock(return_value={"response": "  "}))

        with patch("src.generator.ollama", fake_ollama):
            with self.assertRaises(GenerationError):
                OllamaAnswerGenerator().generate_from_context("Question?", "Context.")

    def test_answer_query_uses_fake_retriever_and_returns_generated_answer(self) -> None:
        """answer_query should return a GeneratedAnswer from retriever context."""

        fake_ollama = SimpleNamespace(
            generate=Mock(return_value={"response": "Albert Einstein was a physicist."})
        )
        retriever = FakeRetriever()

        with patch("src.generator.ollama", fake_ollama):
            answer = OllamaAnswerGenerator().answer_query(
                "Who was Albert Einstein?",
                retriever=retriever,
                top_k=3,
            )

        self.assertIsInstance(answer, GeneratedAnswer)
        self.assertEqual(answer.query, "Who was Albert Einstein?")
        self.assertEqual(answer.answer, "Albert Einstein was a physicist.")
        self.assertEqual(answer.route, ROUTE_PERSON)
        self.assertIn("Albert Einstein was a physicist.", answer.context)
        self.assertEqual(answer.model, config.OLLAMA_GENERATION_MODEL)
        self.assertEqual(retriever.retrieve_calls, [{"query": "Who was Albert Einstein?", "top_k": 3}])

    def test_answer_query_includes_source_summaries(self) -> None:
        """GeneratedAnswer should include source summaries from retriever."""

        fake_ollama = SimpleNamespace(generate=Mock(return_value={"response": "Answer."}))

        with patch("src.generator.ollama", fake_ollama):
            answer = OllamaAnswerGenerator().answer_query(
                "Who was Albert Einstein?",
                retriever=FakeRetriever(),
            )

        self.assertEqual(len(answer.sources), 1)
        self.assertEqual(answer.sources[0]["entity"], "Albert Einstein")
        self.assertEqual(answer.sources[0]["chunk_id"], 1)

    def test_convenience_generate_answer_calls_generator_flow(self) -> None:
        """Convenience function should delegate to OllamaAnswerGenerator."""

        expected = GeneratedAnswer(
            query="Question?",
            answer="Answer.",
            route=ROUTE_PERSON,
            context="Context.",
            sources=[],
            model=config.OLLAMA_GENERATION_MODEL,
        )

        with patch(
            "src.generator.OllamaAnswerGenerator.answer_query",
            return_value=expected,
        ) as mock_answer_query:
            result = generate_answer("Question?")

        self.assertEqual(result, expected)
        mock_answer_query.assert_called_once_with("Question?")
