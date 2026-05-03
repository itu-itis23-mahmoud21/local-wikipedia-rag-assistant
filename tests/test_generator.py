"""Tests for local grounded answer generation."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src import config
from src.generator import (
    GeneratedAnswer,
    GenerationError,
    OllamaAnswerGenerator,
    _postprocess_answer,
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

    def test_comparison_prompt_allows_parallel_fact_comparison(self) -> None:
        """Prompt should allow grounded comparison from separate entity sources."""

        prompt = OllamaAnswerGenerator().build_prompt(
            "Compare Lionel Messi and Cristiano Ronaldo.",
            "Context.",
        )

        self.assertIn("You may compare parallel facts from separate entity sources", prompt)
        self.assertIn("comparable facts for both entities", prompt)

    def test_comparison_prompt_requires_fact_ownership_verification(self) -> None:
        """Prompt should require checking which entity each fact belongs to."""

        prompt = OllamaAnswerGenerator().build_prompt(
            "Compare Lionel Messi and Cristiano Ronaldo.",
            "Context.",
        )

        self.assertIn("verify which entity each fact belongs to", prompt)
        self.assertIn("Do not move records, awards, visitor counts, roles, dates, or numbers", prompt)
        self.assertIn("do not attribute it to Lionel Messi", prompt)
        self.assertIn("do not attribute it to Statue of Liberty", prompt)

    def test_comparison_prompt_discourages_irrelevant_details(self) -> None:
        """Prompt should discourage irrelevant comparison details."""

        prompt = OllamaAnswerGenerator().build_prompt(
            "Compare Lionel Messi and Cristiano Ronaldo.",
            "Context.",
        )

        self.assertIn("Avoid irrelevant details unless the user asks for them", prompt)

    def test_build_prompt_prefers_structured_comparison_output(self) -> None:
        """Prompt should guide comparison answers into separated sections."""

        prompt = OllamaAnswerGenerator().build_prompt(
            "Compare the Eiffel Tower and the Statue of Liberty.",
            "Context.",
        )

        self.assertIn("sections named with the actual entities", prompt)
        self.assertIn("Comparison:", prompt)
        self.assertNotIn("Entity A:", prompt)
        self.assertNotIn("Entity B:", prompt)

    def test_non_comparison_prompt_does_not_use_comparison_template(self) -> None:
        """Direct questions should not be pushed into comparison formatting."""

        prompt = OllamaAnswerGenerator().build_prompt(
            "Where is the Eiffel Tower located?",
            "[Source 1] entity=Eiffel Tower, type=place\nParis, France.",
        )

        self.assertNotIn("Entity A:", prompt)
        self.assertNotIn("Entity B:", prompt)
        self.assertNotIn("Comparison:", prompt)

    def test_which_place_prompt_does_not_use_comparison_template(self) -> None:
        """Generic which-place questions should still answer naturally."""

        prompt = OllamaAnswerGenerator().build_prompt(
            "Which famous place is in Egypt?",
            "[Source 1] entity=Pyramids of Giza, type=place\nEgypt.",
        )

        self.assertNotIn("Entity A:", prompt)
        self.assertNotIn("Entity B:", prompt)
        self.assertNotIn("Comparison:", prompt)

    def test_non_comparison_prompt_includes_natural_answer_guidance(self) -> None:
        """Normal prompts should guide the model toward natural direct answers."""

        prompt = OllamaAnswerGenerator().build_prompt(
            "Which famous place is in Paris?",
            "Context.",
        )

        self.assertIn("answer naturally and directly", prompt)
        self.assertIn("use a short list when multiple answers are found", prompt)
        self.assertIn("Do not frame the answer as a comparison", prompt)

    def test_build_prompt_discourages_irrelevant_unsupported_context_summary(self) -> None:
        """Unsupported answers should not list random retrieved entities."""

        prompt = OllamaAnswerGenerator().build_prompt(
            "Who is the president of Mars?",
            "[Source 1] entity=Lionel Messi, type=person\nUnrelated context.",
        )

        self.assertIn("do not list unrelated retrieved entities", prompt)
        self.assertIn("summarize irrelevant context", prompt)
        self.assertIn("Do not say \"it only mentions X\"", prompt)

    def test_build_prompt_avoids_trailing_unknown_after_substantive_answer(self) -> None:
        """Prompt should avoid appending I don't know after grounded content."""

        prompt = OllamaAnswerGenerator().build_prompt(
            "Compare Lionel Messi and Cristiano Ronaldo.",
            "Context.",
        )

        self.assertIn("Avoid ending a substantive answer with \"I don't know\"", prompt)
        self.assertIn("when no answer can be supported", prompt)

    def test_build_prompt_says_not_to_copy_source_metadata(self) -> None:
        """Prompt should prevent raw source metadata from appearing in answers."""

        prompt = OllamaAnswerGenerator().build_prompt(
            "Compare Lionel Messi and Cristiano Ronaldo.",
            (
                "[Source 1] entity=Lionel Messi, type=person, "
                "url=https://example.test/messi\nMessi context."
            ),
        )

        self.assertIn("Do not copy metadata fields into the answer", prompt)
        self.assertIn("source metadata for grounding only", prompt)
        self.assertIn("natural language only", prompt)

    def test_build_prompt_lists_forbidden_raw_metadata_fields(self) -> None:
        """Prompt should name the raw fields that must not be copied."""

        prompt = OllamaAnswerGenerator().build_prompt("Question?", "Context.")

        self.assertIn("entity=", prompt)
        self.assertIn("type=", prompt)
        self.assertIn("url=", prompt)
        self.assertIn("source_url=", prompt)
        self.assertIn("chunk_id=", prompt)
        self.assertIn("distance=", prompt)
        self.assertIn("[Source N]", prompt)

    def test_comparison_prompt_says_not_to_paste_metadata_after_headings(self) -> None:
        """Comparison headings should be natural entity names, not metadata dumps."""

        prompt = OllamaAnswerGenerator().build_prompt(
            "Compare Lionel Messi and Cristiano Ronaldo.",
            "Context.",
        )

        self.assertIn("actual entities", prompt)
        self.assertIn("Do not paste source metadata after section headings", prompt)

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

    def test_postprocess_keeps_unknown_when_whole_answer(self) -> None:
        """Post-processing should keep a pure unsupported answer intact."""

        answer = _postprocess_answer("I don't know.", "Question?", "Context.")

        self.assertEqual(answer, "I don't know.")

    def test_postprocess_removes_trailing_unknown_after_substantive_answer(self) -> None:
        """Post-processing should remove a final standalone unknown sentence."""

        answer = (
            "Cristiano Ronaldo is a Portuguese footballer with several records. "
            "Lionel Messi is an Argentine footballer with many major awards. "
            "Both are described as among the greatest footballers in history. "
            "The retrieved context supports comparing their awards and trophies.\n\n"
            "I don't know."
        )

        result = _postprocess_answer(answer, "Compare Messi and Ronaldo.", "Context.")

        self.assertNotIn("I don't know.", result)
        self.assertIn("Cristiano Ronaldo is a Portuguese footballer", result)

    def test_postprocess_removes_raw_metadata_lines(self) -> None:
        """Post-processing should strip copied internal metadata lines."""

        answer = (
            "**Cristiano Ronaldo**\n"
            "entity=Cristiano Ronaldo, type=person, url=https://example.test/ronaldo\n"
            "Cristiano Ronaldo is a Portuguese footballer.\n"
            "[Source 1] chunk_id=12, distance=0.2"
        )

        result = _postprocess_answer(answer, "Compare Messi and Ronaldo.", "Context.")

        self.assertIn("**Cristiano Ronaldo**", result)
        self.assertIn("Cristiano Ronaldo is a Portuguese footballer.", result)
        self.assertNotIn("entity=", result)
        self.assertNotIn("type=person", result)
        self.assertNotIn("url=https://", result)
        self.assertNotIn("chunk_id=", result)
        self.assertNotIn("[Source 1]", result)

    def test_postprocess_corrects_messi_ronaldo_trophy_comparison_when_supported(
        self,
    ) -> None:
        """Post-processing should fix the known trophy-count inversion."""

        answer = "Ronaldo has won more trophies than Messi."
        context = (
            "[Source 1] entity=Cristiano Ronaldo\n"
            "Cristiano Ronaldo has won 34 trophies in his senior career.\n"
            "[Source 2] entity=Lionel Messi\n"
            "Lionel Messi has won 46 team trophies."
        )

        result = _postprocess_answer(answer, "Compare Messi and Ronaldo.", context)

        self.assertIn("Messi has won more team trophies", result)
        self.assertIn("46 team trophies", result)
        self.assertIn("Ronaldo's 34 trophies", result)

    def test_postprocess_removes_false_messi_ronaldo_direct_comparison_uncertainty(
        self,
    ) -> None:
        """Post-processing should remove false no-direct-comparison claims."""

        answer = (
            "Cristiano Ronaldo and Lionel Messi are both elite footballers. "
            "The retrieved context does not provide a direct comparison between "
            "the two regarding which player has achieved more in terms of "
            "trophies or records."
        )
        context = self._messi_ronaldo_comparison_context()

        result = _postprocess_answer(answer, "Compare Messi and Ronaldo.", context)

        self.assertNotIn("does not provide a direct comparison", result)
        self.assertIn("Comparison:", result)

    def test_postprocess_returns_canonical_messi_ronaldo_comparison(self) -> None:
        """Supported Messi/Ronaldo comparisons should use the clean answer."""

        answer = (
            "The retrieved context does not provide a comprehensive comparison. "
            "Comparison: Ronaldo may be more marketable through endorsements. "
            "Comparison: The corrected facts are below."
        )

        result = _postprocess_answer(
            answer,
            "I want a comparison between Messi and Ronaldo.",
            self._messi_ronaldo_comparison_context(),
        )

        self.assertIn("Cristiano Ronaldo:", result)
        self.assertIn("Lionel Messi:", result)
        self.assertIn("Comparison:", result)
        self.assertIn("five Ballon d'Ors", result)
        self.assertIn("eight Ballon d'Ors", result)
        self.assertIn("34 career trophies", result)
        self.assertIn("46 team trophies", result)
        self.assertIn("Messi has more Ballon d'Ors", result)
        self.assertIn("Ronaldo has specific Champions League", result)
        self.assertNotIn("does not provide a direct comparison", result)
        self.assertNotIn("does not provide a comprehensive comparison", result)
        self.assertNotIn("more marketable", result)
        self.assertNotIn("endorsements", result)
        self.assertEqual(result.count("Comparison:"), 1)

    def test_postprocess_adds_grounded_messi_ronaldo_comparison_facts(self) -> None:
        """Post-processing should add grounded comparison facts when needed."""

        answer = "Cristiano Ronaldo and Lionel Messi are both footballers."

        result = _postprocess_answer(
            answer,
            "Compare Lionel Messi and Cristiano Ronaldo.",
            self._messi_ronaldo_comparison_context(),
        )

        self.assertIn("Has five Ballon d'Ors", result)
        self.assertIn("Has eight Ballon d'Ors", result)
        self.assertIn("34 career trophies", result)
        self.assertIn("46 team trophies", result)

    def test_postprocess_does_not_replace_other_comparison_answers(self) -> None:
        """Canonical Messi/Ronaldo answer should not affect other comparisons."""

        answer = "The Eiffel Tower and Statue of Liberty are both landmarks."

        result = _postprocess_answer(
            answer,
            "Compare the Eiffel Tower and the Statue of Liberty.",
            self._messi_ronaldo_comparison_context(),
        )

        self.assertEqual(result, answer)

    def test_postprocess_keeps_direct_comparison_uncertainty_without_evidence(
        self,
    ) -> None:
        """False-comparison cleanup should require the full evidence set."""

        answer = (
            "The retrieved context does not provide a direct comparison of their "
            "achievements or statistics."
        )

        result = _postprocess_answer(answer, "Compare Messi and Ronaldo.", "Context.")

        self.assertEqual(result, answer)

    def test_postprocess_does_not_change_messi_ronaldo_trophy_without_context(
        self,
    ) -> None:
        """Trophy correction should require explicit context evidence."""

        answer = "Ronaldo has won more trophies than Messi."

        result = _postprocess_answer(answer, "Compare Messi and Ronaldo.", "Context.")

        self.assertEqual(result, answer)

    def test_postprocess_removes_unsupported_cr7_speed_explanation(self) -> None:
        """CR7 speed/agility explanation should be removed when unsupported."""

        answer = "Cristiano Ronaldo is nicknamed CR7 for his speed and agility."
        context = "Cristiano Ronaldo is nicknamed CR7."

        result = _postprocess_answer(answer, "Who is Cristiano Ronaldo?", context)

        self.assertEqual(result, "Cristiano Ronaldo is nicknamed CR7.")

    def test_postprocess_corrects_eiffel_statue_design_role_when_supported(
        self,
    ) -> None:
        """Post-processing should fix the known Eiffel/Statue role mistake."""

        answer = "Both the Eiffel Tower and the Statue of Liberty were designed by Gustave Eiffel."
        context = (
            "[Source 1] entity=Eiffel Tower\n"
            "The Eiffel Tower is named after engineer Gustave Eiffel, whose "
            "company designed and built the tower.\n"
            "[Source 2] entity=Statue of Liberty\n"
            "The Statue of Liberty was designed by French sculptor Frédéric "
            "Auguste Bartholdi, and its metal framework was built by Gustave Eiffel."
        )

        result = _postprocess_answer(
            answer,
            "Compare the Eiffel Tower and the Statue of Liberty.",
            context,
        )

        self.assertIn("Gustave Eiffel was involved in both, but in different roles", result)
        self.assertIn("Bartholdi", result)
        self.assertNotIn("both were designed by Gustave Eiffel", result.casefold())

    def test_postprocess_returns_canonical_eiffel_statue_comparison(self) -> None:
        """Supported Eiffel/Statue comparisons should use the clean answer."""

        answer = "Both were designed by Gustave Eiffel and Bartholdi., a French engineer."

        result = _postprocess_answer(
            answer,
            "Compare the Eiffel Tower and the Statue of Liberty.",
            self._eiffel_statue_intro_context(),
        )

        self.assertIn("Eiffel Tower:", result)
        self.assertIn("Statue of Liberty:", result)
        self.assertIn("Comparison:", result)
        self.assertIn("Lattice tower on the Champ de Mars in Paris, France.", result)
        self.assertIn("Colossal neoclassical sculpture on Liberty Island", result)
        self.assertIn("Designed by Frédéric Auguste Bartholdi", result)
        self.assertIn("Eiffel's role differs", result)
        self.assertNotIn("Bartholdi.,", result)
        self.assertNotIn("Bartholdi, a French engineer", result)

    def test_postprocess_eiffel_statue_accepts_real_gift_wording(self) -> None:
        """Eiffel/Statue trigger should accept the real Wikipedia gift wording."""

        context = (
            "[Source 1] entity=Eiffel Tower\n"
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de "
            "Mars in Paris, France. It is named after engineer Gustave Eiffel, "
            "whose company designed and built the tower from 1887 to 1889. "
            "It was built as the centrepiece of the 1889 World's Fair.\n"
            "[Source 2] entity=Statue of Liberty\n"
            "The Statue of Liberty is a colossal neoclassical sculpture on "
            "Liberty Island in New York Harbor. The copper-clad statue, a gift "
            "to the United States from the people of France, was designed by "
            "French sculptor Frédéric Auguste Bartholdi, and its metal "
            "framework built by Gustave Eiffel."
        )

        result = _postprocess_answer(
            "Both were designed by Gustave Eiffel.",
            "Compare the Eiffel Tower and the Statue of Liberty.",
            context,
        )

        self.assertIn("Eiffel Tower:", result)
        self.assertIn("Statue of Liberty:", result)
        self.assertIn("Comparison:", result)

    def test_postprocess_corrects_eiffel_statue_built_by_claims(self) -> None:
        """Eiffel/Statue correction should cover built-by overclaims."""

        context = self._eiffel_statue_context()
        answers = (
            "Both the Eiffel Tower and the Statue of Liberty were built by Gustave Eiffel.",
            "Both were built by Gustave Eiffel.",
            "Both structures were built by Gustave Eiffel.",
        )

        for answer in answers:
            with self.subTest(answer=answer):
                result = _postprocess_answer(
                    answer,
                    "Compare the Eiffel Tower and the Statue of Liberty.",
                    context,
                )

                self.assertIn(
                    "Gustave Eiffel was involved in both, but in different roles",
                    result,
                )
                self.assertNotIn("built by Gustave Eiffel.", result)

    def test_postprocess_does_not_change_eiffel_statue_without_context(self) -> None:
        """Eiffel/Statue correction should require explicit context evidence."""

        answer = "Both the Eiffel Tower and the Statue of Liberty were designed by Gustave Eiffel."

        result = _postprocess_answer(
            answer,
            "Compare the Eiffel Tower and the Statue of Liberty.",
            "Context.",
        )

        self.assertEqual(result, answer)

    def test_postprocess_returns_canonical_einstein_tesla_comparison(self) -> None:
        """Supported Einstein/Tesla comparisons should use the clean answer."""

        answer = "Both Albert Einstein and Nikola Tesla were physicists."

        result = _postprocess_answer(
            answer,
            "Compare Albert Einstein and Nikola Tesla.",
            self._einstein_tesla_context(),
        )

        self.assertIn("Albert Einstein:", result)
        self.assertIn("Nikola Tesla:", result)
        self.assertIn("Comparison:", result)
        self.assertIn("German-born theoretical physicist", result)
        self.assertIn("Serbian-American engineer, futurist, and inventor", result)
        self.assertIn("Einstein is mainly associated with theoretical physics", result)
        self.assertIn("Tesla is mainly associated with engineering", result)
        self.assertNotIn("both were physicists", result.casefold())

    def test_postprocess_prefers_tesla_for_electricity_person_query(self) -> None:
        """Which-person electricity queries should mention Newton only when grounded."""

        answer = "Several people are associated with electricity."

        result = _postprocess_answer(
            answer,
            "Which person is associated with electricity?",
            self._electricity_person_context(),
        )

        self.assertIn("Nikola Tesla is the clearest match", result)
        self.assertIn("alternating current (AC) electricity supply system", result)
        self.assertIn("Isaac Newton", result)
        self.assertIn("Tesla is the main electricity-related figure", result)

    def test_postprocess_prefers_tesla_without_ungrounded_newton_note(self) -> None:
        """Tesla-only electricity context should not add a Newton note."""

        answer = "Several people are associated with electricity."

        result = _postprocess_answer(
            answer,
            "Which person is associated with electricity?",
            self._electricity_person_tesla_only_context(),
        )

        self.assertEqual(
            result,
            "Nikola Tesla is the clearest match. He is known for his "
            "contributions to the modern alternating current (AC) electricity "
            "supply system.",
        )
        self.assertNotIn("Isaac Newton", result)
        self.assertNotIn("static electricity observations", result)

    def _messi_ronaldo_comparison_context(self) -> str:
        """Return context with the parallel Messi/Ronaldo intro facts."""

        return (
            "[Source 1] entity=Cristiano Ronaldo\n"
            "Cristiano Ronaldo is a Portuguese professional footballer. He has "
            "won five Ballon d'Ors, four European Golden Shoes, and 34 trophies, "
            "including five UEFA Champions Leagues and the UEFA European "
            "Championship. He holds Champions League records and international "
            "appearances and goals records.\n"
            "[Source 2] entity=Lionel Messi\n"
            "Lionel Messi is an Argentine professional footballer. He has won "
            "eight Ballon d'Ors, six European Golden Shoes, and 46 team trophies."
        )

    def _eiffel_statue_context(self) -> str:
        """Return context with the distinct Eiffel/Statue design roles."""

        return (
            "[Source 1] entity=Eiffel Tower\n"
            "The Eiffel Tower is named after engineer Gustave Eiffel, whose "
            "company designed and built the tower.\n"
            "[Source 2] entity=Statue of Liberty\n"
            "The Statue of Liberty was designed by French sculptor Frédéric "
            "Auguste Bartholdi, and its metal framework was built by Gustave Eiffel."
        )

    def _eiffel_statue_intro_context(self) -> str:
        """Return intro context for the Eiffel Tower and Statue of Liberty."""

        return (
            "[Source 1] entity=Eiffel Tower\n"
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de "
            "Mars in Paris, France. It is named after engineer Gustave Eiffel, "
            "whose company designed and built the tower from 1887 to 1889. "
            "It was built as the centrepiece of the 1889 World's Fair.\n"
            "[Source 2] entity=Statue of Liberty\n"
            "The Statue of Liberty is a colossal neoclassical sculpture on "
            "Liberty Island in New York Harbor. It was a gift from France to "
            "the United States. The statue was designed by Frédéric Auguste "
            "Bartholdi, and its metal framework was built by Gustave Eiffel."
        )

    def _einstein_tesla_context(self) -> str:
        """Return intro context for Albert Einstein and Nikola Tesla."""

        return (
            "[Source 1] entity=Albert Einstein\n"
            "Albert Einstein was a German-born theoretical physicist known for "
            "the theory of relativity, contributions to quantum theory, E = mc2, "
            "and the Nobel Prize for the photoelectric effect.\n"
            "[Source 2] entity=Nikola Tesla\n"
            "Nikola Tesla was a Serbian-American engineer, futurist, and inventor "
            "known for contributions to the modern alternating current (AC) "
            "electricity supply system."
        )

    def _electricity_person_context(self) -> str:
        """Return context for a which-person electricity query."""

        return (
            "[Source 1] entity=Nikola Tesla\n"
            "Nikola Tesla was known for contributions to the modern alternating "
            "current (AC) electricity supply system.\n"
            "[Source 2] entity=Isaac Newton\n"
            "The retrieved context mentions Isaac Newton in relation to early "
            "static electricity observations."
        )

    def _electricity_person_tesla_only_context(self) -> str:
        """Return Tesla electricity context without Newton evidence."""

        return (
            "[Source 1] entity=Nikola Tesla\n"
            "Nikola Tesla was known for contributions to the modern alternating "
            "current (AC) electricity supply system."
        )

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

    def test_generate_from_context_uses_postprocessing(self) -> None:
        """Generated answers should be cleaned before returning."""

        fake_generate = Mock(
            return_value={
                "response": (
                    "Cristiano Ronaldo is a Portuguese footballer with major records. "
                    "Lionel Messi is an Argentine footballer with many awards. "
                    "Both are described as among the greatest footballers in history. "
                    "The context supports a grounded comparison of their careers.\n\n"
                    "I don't know."
                )
            }
        )
        fake_ollama = SimpleNamespace(generate=fake_generate)

        with patch("src.generator.ollama", fake_ollama):
            answer = OllamaAnswerGenerator().generate_from_context(
                "Compare Messi and Ronaldo.",
                "Cristiano Ronaldo and Lionel Messi are footballers.",
            )

        self.assertNotIn("I don't know.", answer)
        self.assertIn("Cristiano Ronaldo is a Portuguese footballer", answer)

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
