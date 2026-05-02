"""Tests for retrieval orchestration."""

import unittest

from src.query_router import ROUTE_BOTH, ROUTE_PERSON, ROUTE_PLACE, ROUTE_UNKNOWN
from src.retriever import (
    RAGRetriever,
    RetrievedContext,
    deduplicate_retrieved_overlaps,
)
from src.vector_store import VectorSearchResult


class FakeVectorStore:
    """Fake vector store that records search calls."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.intro_calls: list[dict] = []
        self.results = [
            VectorSearchResult(
                vector_id="chunk-1",
                text="Later Albert Einstein context.",
                metadata={
                    "chunk_id": 1,
                    "entity": "Albert Einstein",
                    "entity_type": "person",
                    "source_url": "https://example.test/einstein",
                    "chunk_index": 2,
                },
                distance=0.12,
            ),
            VectorSearchResult(
                vector_id="chunk-2",
                text="Nikola Tesla worked with electricity.",
                metadata={
                    "chunk_id": 2,
                    "entity": "Nikola Tesla",
                    "entity_type": "person",
                    "source_url": "https://example.test/tesla",
                    "chunk_index": 2,
                },
                distance=0.25,
            ),
        ]
        self.intro_results = [
            VectorSearchResult(
                vector_id="chunk-intro-einstein",
                text="Albert Einstein was a German-born theoretical physicist.",
                metadata={
                    "chunk_id": 10,
                    "entity": "Albert Einstein",
                    "entity_type": "person",
                    "source_url": "https://example.test/einstein",
                    "chunk_index": 0,
                },
                distance=None,
            ),
            VectorSearchResult(
                vector_id="chunk-intro-tesla",
                text="Nikola Tesla was an inventor and electrical engineer.",
                metadata={
                    "chunk_id": 20,
                    "entity": "Nikola Tesla",
                    "entity_type": "person",
                    "source_url": "https://example.test/tesla",
                    "chunk_index": 0,
                },
                distance=None,
            ),
        ]

    def search(
        self,
        query: str,
        top_k: int = 5,
        entity_type: str | None = None,
        entity_names: list[str] | None = None,
    ) -> list[VectorSearchResult]:
        """Record search parameters and return fake results."""

        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "entity_type": entity_type,
                "entity_names": entity_names,
            }
        )
        results = self.results
        if entity_type is not None:
            results = [
                result
                for result in results
                if result.metadata.get("entity_type") == entity_type
            ]
        if entity_names is not None:
            allowed_names = {name.casefold() for name in entity_names}
            results = [
                result
                for result in results
                if str(result.metadata.get("entity", "")).casefold() in allowed_names
            ]
        return results[:top_k]

    def get_intro_chunks(
        self,
        entity_names: list[str],
        per_entity: int = 1,
        entity_type: str | None = None,
    ) -> list[VectorSearchResult]:
        """Record intro-chunk parameters and return matching fake intro chunks."""

        self.intro_calls.append(
            {
                "entity_names": entity_names,
                "per_entity": per_entity,
                "entity_type": entity_type,
            }
        )
        allowed_names = {name.casefold() for name in entity_names}
        results = [
            result
            for result in self.intro_results
            if str(result.metadata.get("entity", "")).casefold() in allowed_names
        ]
        return results[: max(0, per_entity * len(allowed_names))]


class TestRAGRetriever(unittest.TestCase):
    """Tests for routed retrieval."""

    OVERLAP_PREVIOUS = (
        "Albert Einstein moved to Switzerland at the age of seventeen, he enrolled "
        "in the mathematics and physics teaching diploma program at the Swiss "
        "federal polytechnic school in Zurich, graduating in 1900."
    )
    OVERLAP_CURRENT = (
        "seventeen, he enrolled in the mathematics and physics teaching diploma "
        "program at the Swiss federal polytechnic school in Zurich, graduating "
        "in 1900.\n\nHe acquired Swiss citizenship a year later and worked at "
        "the patent office in Bern."
    )

    def test_retrieve_person_query_applies_person_filter(self) -> None:
        """Exact person queries should filter by type and entity name."""

        vector_store = FakeVectorStore()
        retriever = RAGRetriever(vector_store=vector_store)

        context = retriever.retrieve("Who was Albert Einstein?")

        self.assertEqual(context.route.route, ROUTE_PERSON)
        self.assertEqual(vector_store.calls[0]["entity_type"], "person")
        self.assertEqual(vector_store.calls[0]["entity_names"], ["Albert Einstein"])
        self.assertEqual(
            vector_store.intro_calls[0],
            {
                "entity_names": ["Albert Einstein"],
                "per_entity": 1,
                "entity_type": "person",
            },
        )

    def test_retrieve_place_query_applies_place_filter(self) -> None:
        """Exact place queries should filter by type and entity name."""

        vector_store = FakeVectorStore()
        vector_store.intro_results = [
            VectorSearchResult(
                vector_id="chunk-intro-eiffel",
                text="The Eiffel Tower is a landmark in Paris.",
                metadata={
                    "chunk_id": 30,
                    "entity": "Eiffel Tower",
                    "entity_type": "place",
                    "source_url": "https://example.test/eiffel",
                    "chunk_index": 0,
                },
                distance=None,
            )
        ]
        retriever = RAGRetriever(vector_store=vector_store)

        context = retriever.retrieve("Where is the Eiffel Tower located?")

        self.assertEqual(context.route.route, ROUTE_PLACE)
        self.assertEqual(vector_store.calls[0]["entity_type"], "place")
        self.assertEqual(vector_store.calls[0]["entity_names"], ["Eiffel Tower"])
        self.assertEqual(vector_store.intro_calls[0]["entity_type"], "place")

    def test_turkey_place_query_uses_location_entity_hints(self) -> None:
        """Location-only Turkey questions should retrieve configured Turkish places."""

        vector_store = FakeVectorStore()
        vector_store.intro_results = [
            self._result("chunk-hagia-intro", "Hagia Sophia is in Istanbul.", "Hagia Sophia"),
            self._result("chunk-blue-intro", "The Blue Mosque is in Istanbul.", "Blue Mosque"),
            self._result(
                "chunk-topkapi-intro",
                "Topkapı Palace is in Istanbul.",
                "Topkapı Palace",
            ),
        ]
        vector_store.results = [
            self._result("chunk-hagia", "Hagia Sophia Turkey context.", "Hagia Sophia"),
            self._result("chunk-blue", "Blue Mosque Turkey context.", "Blue Mosque"),
            self._result("chunk-topkapi", "Topkapı Palace Turkey context.", "Topkapı Palace"),
        ]
        retriever = RAGRetriever(vector_store=vector_store)

        context = retriever.retrieve("Which famous place is located in Turkey?")

        hinted_entities = ["Hagia Sophia", "Blue Mosque", "Topkapı Palace"]
        self.assertEqual(context.route.route, ROUTE_PLACE)
        self.assertEqual(vector_store.calls[0]["entity_type"], "place")
        self.assertEqual(vector_store.calls[0]["entity_names"], hinted_entities)
        self.assertEqual(vector_store.intro_calls[0]["entity_names"], hinted_entities)
        self.assertEqual(vector_store.intro_calls[0]["entity_type"], "place")
        self.assertTrue(
            any(result.metadata["entity"] in hinted_entities for result in context.results)
        )

    def test_egypt_place_query_uses_location_entity_hints(self) -> None:
        """Location-only Egypt questions should retrieve Pyramids of Giza."""

        vector_store = FakeVectorStore()
        vector_store.intro_results = [
            self._result(
                "chunk-pyramids-intro",
                "The Pyramids of Giza are in Egypt.",
                "Pyramids of Giza",
            )
        ]
        vector_store.results = [
            self._result(
                "chunk-pyramids",
                "Pyramids of Giza Egypt context.",
                "Pyramids of Giza",
            )
        ]
        retriever = RAGRetriever(vector_store=vector_store)

        context = retriever.retrieve("Which famous place is in Egypt?")

        self.assertEqual(context.route.route, ROUTE_PLACE)
        self.assertEqual(vector_store.calls[0]["entity_type"], "place")
        self.assertEqual(vector_store.calls[0]["entity_names"], ["Pyramids of Giza"])
        self.assertEqual(vector_store.intro_calls[0]["entity_names"], ["Pyramids of Giza"])
        self.assertEqual(context.results[0].metadata["entity"], "Pyramids of Giza")

    def test_retrieve_both_query_applies_no_filter(self) -> None:
        """Comparison queries should search each mentioned entity separately."""

        vector_store = FakeVectorStore()
        retriever = RAGRetriever(vector_store=vector_store)

        context = retriever.retrieve("Compare Albert Einstein and Nikola Tesla")

        self.assertEqual(context.route.route, ROUTE_BOTH)
        self.assertIsNone(vector_store.calls[0]["entity_type"])
        self.assertEqual(
            vector_store.calls[0]["entity_names"],
            ["Albert Einstein"],
        )
        self.assertIsNone(vector_store.calls[1]["entity_type"])
        self.assertEqual(
            vector_store.calls[1]["entity_names"],
            ["Nikola Tesla"],
        )
        self.assertEqual(
            vector_store.intro_calls[0]["entity_names"],
            ["Albert Einstein", "Nikola Tesla"],
        )
        self.assertIsNone(vector_store.intro_calls[0]["entity_type"])

    def test_keyword_only_person_query_keeps_type_filter_only(self) -> None:
        """Person keyword queries without exact names should use type only."""

        vector_store = FakeVectorStore()
        retriever = RAGRetriever(vector_store=vector_store)

        context = retriever.retrieve("Which person is associated with electricity?")

        self.assertEqual(context.route.route, ROUTE_PERSON)
        self.assertEqual(vector_store.calls[0]["entity_type"], "person")
        self.assertIsNone(vector_store.calls[0]["entity_names"])
        self.assertEqual(vector_store.intro_calls, [])

    def test_retrieve_unknown_query_applies_no_filter(self) -> None:
        """Unknown-routed queries should search without entity_type filter."""

        vector_store = FakeVectorStore()
        retriever = RAGRetriever(vector_store=vector_store)

        context = retriever.retrieve("Tell me about a random concept")

        self.assertEqual(context.route.route, ROUTE_UNKNOWN)
        self.assertIsNone(vector_store.calls[0]["entity_type"])
        self.assertIsNone(vector_store.calls[0]["entity_names"])
        self.assertEqual(vector_store.intro_calls, [])

    def test_retrieve_rejects_invalid_top_k(self) -> None:
        """top_k must be positive."""

        retriever = RAGRetriever(vector_store=FakeVectorStore())

        with self.assertRaises(ValueError):
            retriever.retrieve("Albert Einstein", top_k=0)

    def test_format_context_includes_source_labels_metadata_and_text(self) -> None:
        """format_context should include source labels and chunk text."""

        retriever = RAGRetriever(vector_store=FakeVectorStore())
        context = retriever.format_context(FakeVectorStore().results)

        self.assertIn("[Source 1]", context)
        self.assertIn("entity=Albert Einstein", context)
        self.assertIn("type=person", context)
        self.assertIn("https://example.test/einstein", context)
        self.assertIn("Later Albert Einstein context.", context)

    def test_format_context_respects_max_chars(self) -> None:
        """format_context should stop before exceeding max_chars."""

        retriever = RAGRetriever(vector_store=FakeVectorStore())
        results = FakeVectorStore().results

        context = retriever.format_context(results, max_chars=140)

        self.assertIn("[Source 1]", context)
        self.assertNotIn("[Source 2]", context)
        self.assertLessEqual(len(context), 140)

    def test_format_context_returns_empty_string_for_no_results(self) -> None:
        """No results should produce empty context."""

        retriever = RAGRetriever(vector_store=FakeVectorStore())

        self.assertEqual(retriever.format_context([]), "")

    def test_get_source_summary_returns_compact_ranked_metadata(self) -> None:
        """Source summaries should include rank, metadata, distance, and preview."""

        retriever = RAGRetriever(vector_store=FakeVectorStore())
        summary = retriever.get_source_summary(FakeVectorStore().results)

        self.assertEqual(summary[0]["rank"], 1)
        self.assertEqual(summary[0]["entity"], "Albert Einstein")
        self.assertEqual(summary[0]["entity_type"], "person")
        self.assertEqual(summary[0]["source_url"], "https://example.test/einstein")
        self.assertEqual(summary[0]["chunk_id"], 1)
        self.assertEqual(summary[0]["distance"], 0.12)
        self.assertIn("Albert Einstein", summary[0]["preview"])

    def test_retrieve_returns_retrieved_context_with_route_and_results(self) -> None:
        """retrieve should return a RetrievedContext object."""

        retriever = RAGRetriever(vector_store=FakeVectorStore())

        context = retriever.retrieve("What did Albert Einstein discover?", top_k=1)

        self.assertIsInstance(context, RetrievedContext)
        self.assertEqual(context.query, "What did Albert Einstein discover?")
        self.assertEqual(context.route.route, ROUTE_PERSON)
        self.assertEqual(len(context.results), 1)

    def test_exact_entity_query_places_intro_chunk_first(self) -> None:
        """Exact entity retrieval should prepend the intro chunk."""

        retriever = RAGRetriever(vector_store=FakeVectorStore())

        context = retriever.retrieve("Who was Albert Einstein?", top_k=3)

        self.assertEqual(context.results[0].vector_id, "chunk-intro-einstein")
        self.assertIn("German-born theoretical physicist", context.results[0].text)
        self.assertIsNone(context.results[0].distance)

    def test_exact_entity_query_deduplicates_intro_and_semantic_results(self) -> None:
        """Intro chunks should not be duplicated if semantic search returns them."""

        vector_store = FakeVectorStore()
        vector_store.intro_results = [vector_store.results[0]]
        retriever = RAGRetriever(vector_store=vector_store)

        context = retriever.retrieve("Who was Albert Einstein?", top_k=3)

        self.assertEqual(
            [result.vector_id for result in context.results],
            ["chunk-1"],
        )

    def test_comparison_query_includes_intro_chunks_for_both_entities(self) -> None:
        """Comparison retrieval should include one intro chunk per matched entity."""

        retriever = RAGRetriever(vector_store=FakeVectorStore())

        context = retriever.retrieve("Compare Albert Einstein and Nikola Tesla", top_k=4)

        self.assertEqual(
            [result.vector_id for result in context.results[:2]],
            ["chunk-intro-einstein", "chunk-intro-tesla"],
        )

    def test_comparison_query_performs_per_entity_semantic_searches(self) -> None:
        """Comparison retrieval should avoid one broad mixed semantic search."""

        vector_store = FakeVectorStore()
        retriever = RAGRetriever(vector_store=vector_store)

        retriever.retrieve("Compare Albert Einstein and Nikola Tesla", top_k=5)

        self.assertEqual(
            [call["entity_names"] for call in vector_store.calls],
            [["Albert Einstein"], ["Nikola Tesla"]],
        )

    def test_comparison_retrieval_respects_top_k(self) -> None:
        """Balanced comparison retrieval should still honor the public top_k."""

        vector_store = FakeVectorStore()
        vector_store.results = [
            self._result("chunk-e-1", "Einstein semantic context with many useful words.", "Albert Einstein"),
            self._result("chunk-e-2", "Einstein second semantic context with many useful words.", "Albert Einstein"),
            self._result("chunk-t-1", "Tesla semantic context with many useful words.", "Nikola Tesla"),
            self._result("chunk-t-2", "Tesla second semantic context with many useful words.", "Nikola Tesla"),
        ]
        retriever = RAGRetriever(vector_store=vector_store)

        context = retriever.retrieve("Compare Albert Einstein and Nikola Tesla", top_k=3)

        self.assertEqual(len(context.results), 3)
        self.assertEqual(
            [result.vector_id for result in context.results[:2]],
            ["chunk-intro-einstein", "chunk-intro-tesla"],
        )

    def test_low_information_semantic_chunks_are_filtered_when_better_chunks_exist(
        self,
    ) -> None:
        """Very short semantic chunks should be removed when useful chunks remain."""

        vector_store = FakeVectorStore()
        vector_store.results = [
            self._result("chunk-short-e", "Rivalry with Lionel Messi", "Albert Einstein"),
            self._result(
                "chunk-good-e",
                "Albert Einstein useful semantic context has enough words for retrieval.",
                "Albert Einstein",
            ),
            self._result("chunk-short-t", "Rivalry with Albert Einstein", "Nikola Tesla"),
            self._result(
                "chunk-good-t",
                "Nikola Tesla useful semantic context has enough words for retrieval.",
                "Nikola Tesla",
            ),
        ]
        retriever = RAGRetriever(vector_store=vector_store)

        context = retriever.retrieve("Compare Albert Einstein and Nikola Tesla", top_k=5)
        vector_ids = [result.vector_id for result in context.results]

        self.assertIn("chunk-good-e", vector_ids)
        self.assertIn("chunk-good-t", vector_ids)
        self.assertNotIn("chunk-short-e", vector_ids)
        self.assertNotIn("chunk-short-t", vector_ids)

    def test_low_information_filtering_does_not_remove_intro_chunks(self) -> None:
        """Intro chunks should remain even when their text is short."""

        vector_store = FakeVectorStore()
        vector_store.intro_results = [
            self._result("chunk-intro-einstein", "Einstein", "Albert Einstein"),
            self._result("chunk-intro-tesla", "Tesla", "Nikola Tesla"),
        ]
        vector_store.results = [
            self._result(
                "chunk-good-e",
                "Albert Einstein useful semantic context has enough words for retrieval.",
                "Albert Einstein",
            ),
            self._result(
                "chunk-good-t",
                "Nikola Tesla useful semantic context has enough words for retrieval.",
                "Nikola Tesla",
            ),
        ]
        retriever = RAGRetriever(vector_store=vector_store)

        context = retriever.retrieve("Compare Albert Einstein and Nikola Tesla", top_k=4)

        self.assertEqual(
            [result.vector_id for result in context.results[:2]],
            ["chunk-intro-einstein", "chunk-intro-tesla"],
        )

    def test_overlap_cleanup_removes_same_entity_repeated_prefix(self) -> None:
        """Same-entity overlap should be removed from later result text."""

        results = deduplicate_retrieved_overlaps(
            [
                self._result("chunk-1", self.OVERLAP_PREVIOUS, "Albert Einstein"),
                self._result("chunk-2", self.OVERLAP_CURRENT, "Albert Einstein"),
            ]
        )

        self.assertEqual(results[1].vector_id, "chunk-2")
        self.assertEqual(results[1].metadata["entity"], "Albert Einstein")
        self.assertIsNone(results[1].distance)
        self.assertTrue(results[1].text.startswith("He acquired Swiss citizenship"))
        self.assertNotIn("seventeen, he enrolled", results[1].text[:40])

    def test_overlap_cleanup_does_not_cross_entities(self) -> None:
        """Overlap cleanup should not compare chunks from different entities."""

        results = deduplicate_retrieved_overlaps(
            [
                self._result("chunk-1", self.OVERLAP_PREVIOUS, "Albert Einstein"),
                self._result("chunk-2", self.OVERLAP_CURRENT, "Nikola Tesla"),
            ]
        )

        self.assertEqual(results[1].text, self.OVERLAP_CURRENT)

    def test_overlap_cleanup_keeps_text_when_overlap_is_too_small(self) -> None:
        """Small overlaps should not be trimmed."""

        previous = "Albert Einstein studied physics in Zurich."
        current = "physics in Zurich. He later developed theories."

        results = deduplicate_retrieved_overlaps(
            [
                self._result("chunk-1", previous, "Albert Einstein"),
                self._result("chunk-2", current, "Albert Einstein"),
            ]
        )

        self.assertEqual(results[1].text, current)

    def test_overlap_cleanup_keeps_original_if_cleaned_text_is_too_short(self) -> None:
        """Cleanup should avoid removing all useful text from a result."""

        text = "one two three four five six seven eight nine ten"

        results = deduplicate_retrieved_overlaps(
            [
                self._result("chunk-1", text, "Albert Einstein"),
                self._result("chunk-2", text, "Albert Einstein"),
            ]
        )

        self.assertEqual(results[1].text, text)

    def test_retrieve_returns_cleaned_overlap_text(self) -> None:
        """RetrievedContext should contain overlap-cleaned result text."""

        vector_store = FakeVectorStore()
        vector_store.intro_results = [
            self._result("chunk-intro-einstein", self.OVERLAP_PREVIOUS, "Albert Einstein")
        ]
        vector_store.results = [
            self._result("chunk-overlap-einstein", self.OVERLAP_CURRENT, "Albert Einstein")
        ]
        retriever = RAGRetriever(vector_store=vector_store)

        context = retriever.retrieve("Who was Albert Einstein?", top_k=2)

        self.assertTrue(context.results[1].text.startswith("He acquired"))

    def test_source_summary_uses_cleaned_result_preview(self) -> None:
        """Source summaries should preview cleaned result text after retrieval."""

        vector_store = FakeVectorStore()
        vector_store.intro_results = [
            self._result("chunk-intro-einstein", self.OVERLAP_PREVIOUS, "Albert Einstein")
        ]
        vector_store.results = [
            self._result("chunk-overlap-einstein", self.OVERLAP_CURRENT, "Albert Einstein")
        ]
        retriever = RAGRetriever(vector_store=vector_store)

        context = retriever.retrieve("Who was Albert Einstein?", top_k=2)
        summary = retriever.get_source_summary(context.results)

        self.assertTrue(summary[1]["preview"].startswith("He acquired"))

    def _result(
        self,
        vector_id: str,
        text: str,
        entity: str,
        distance: float | None = None,
    ) -> VectorSearchResult:
        """Build a representative search result for overlap-cleanup tests."""

        return VectorSearchResult(
            vector_id=vector_id,
            text=text,
            metadata={
                "chunk_id": vector_id,
                "entity": entity,
                "entity_type": "person",
                "source_url": "https://example.test/source",
            },
            distance=distance,
        )
