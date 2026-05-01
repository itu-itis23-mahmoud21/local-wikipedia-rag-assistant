"""Tests for retrieval orchestration."""

import unittest

from src.query_router import ROUTE_BOTH, ROUTE_PERSON, ROUTE_PLACE, ROUTE_UNKNOWN
from src.retriever import RAGRetriever, RetrievedContext
from src.vector_store import VectorSearchResult


class FakeVectorStore:
    """Fake vector store that records search calls."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
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
                },
                distance=0.25,
            ),
        ]

    def search(
        self,
        query: str,
        top_k: int = 5,
        entity_type: str | None = None,
    ) -> list[VectorSearchResult]:
        """Record search parameters and return fake results."""

        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "entity_type": entity_type,
            }
        )
        return self.results[:top_k]


class TestRAGRetriever(unittest.TestCase):
    """Tests for routed retrieval."""

    def test_retrieve_person_query_applies_person_filter(self) -> None:
        """Person-routed queries should filter entity_type=person."""

        vector_store = FakeVectorStore()
        retriever = RAGRetriever(vector_store=vector_store)

        context = retriever.retrieve("What did Albert Einstein discover?")

        self.assertEqual(context.route.route, ROUTE_PERSON)
        self.assertEqual(vector_store.calls[0]["entity_type"], "person")

    def test_retrieve_place_query_applies_place_filter(self) -> None:
        """Place-routed queries should filter entity_type=place."""

        vector_store = FakeVectorStore()
        retriever = RAGRetriever(vector_store=vector_store)

        context = retriever.retrieve("Where is the Eiffel Tower located?")

        self.assertEqual(context.route.route, ROUTE_PLACE)
        self.assertEqual(vector_store.calls[0]["entity_type"], "place")

    def test_retrieve_both_query_applies_no_filter(self) -> None:
        """Both-routed queries should search without entity_type filter."""

        vector_store = FakeVectorStore()
        retriever = RAGRetriever(vector_store=vector_store)

        context = retriever.retrieve("Compare Albert Einstein and Nikola Tesla")

        self.assertEqual(context.route.route, ROUTE_BOTH)
        self.assertIsNone(vector_store.calls[0]["entity_type"])

    def test_retrieve_unknown_query_applies_no_filter(self) -> None:
        """Unknown-routed queries should search without entity_type filter."""

        vector_store = FakeVectorStore()
        retriever = RAGRetriever(vector_store=vector_store)

        context = retriever.retrieve("Tell me about a random concept")

        self.assertEqual(context.route.route, ROUTE_UNKNOWN)
        self.assertIsNone(vector_store.calls[0]["entity_type"])

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
        self.assertIn("Albert Einstein was a physicist.", context)

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
