"""Retrieval orchestration for routed Wikipedia RAG queries."""

from __future__ import annotations

from dataclasses import dataclass

from src import config
from src.query_router import (
    QueryRoute,
    ROUTE_BOTH,
    ROUTE_PERSON,
    ROUTE_PLACE,
    route_query,
)
from src.vector_store import ChromaVectorStore, VectorSearchResult


@dataclass(frozen=True)
class RetrievedContext:
    """Retrieved chunks and routing metadata for one user query."""

    query: str
    route: QueryRoute
    results: list[VectorSearchResult]


class RAGRetriever:
    """Route user questions and retrieve matching Chroma chunks."""

    def __init__(
        self,
        vector_store: ChromaVectorStore | None = None,
        top_k: int = config.DEFAULT_TOP_K,
    ) -> None:
        """Create a retriever with a vector store and default result count."""

        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        self.vector_store = vector_store or ChromaVectorStore()
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        show_route: bool = False,
    ) -> RetrievedContext:
        """Route a query and retrieve relevant vector-search results."""

        effective_top_k = top_k if top_k is not None else self.top_k
        if effective_top_k <= 0:
            raise ValueError("top_k must be positive.")

        route = route_query(query)
        entity_type_filter = _entity_type_filter_for_route(route)
        entity_name_filter = _entity_names_for_route(route)
        results = self.vector_store.search(
            query,
            top_k=effective_top_k,
            entity_type=entity_type_filter,
            entity_names=entity_name_filter,
        )

        return RetrievedContext(query=query, route=route, results=results)

    def format_context(
        self,
        results: list[VectorSearchResult],
        max_chars: int = 4000,
    ) -> str:
        """Format retrieved chunks into a context block for a later LLM."""

        if max_chars <= 0:
            raise ValueError("max_chars must be positive.")
        if not results:
            return ""

        sections: list[str] = []
        current_length = 0

        for index, result in enumerate(results, start=1):
            metadata = result.metadata
            section = (
                f"[Source {index}] "
                f"entity={metadata.get('entity', '')}, "
                f"type={metadata.get('entity_type', '')}, "
                f"url={metadata.get('source_url', '')}\n"
                f"{result.text.strip()}"
            ).strip()

            additional_length = len(section) + (2 if sections else 0)
            if current_length + additional_length > max_chars:
                break

            sections.append(section)
            current_length += additional_length

        return "\n\n".join(sections)

    def get_source_summary(
        self,
        results: list[VectorSearchResult],
    ) -> list[dict]:
        """Return compact source metadata for a future UI."""

        summaries: list[dict] = []
        for rank, result in enumerate(results, start=1):
            metadata = result.metadata
            summaries.append(
                {
                    "rank": rank,
                    "entity": metadata.get("entity"),
                    "entity_type": metadata.get("entity_type"),
                    "source_url": metadata.get("source_url"),
                    "chunk_id": metadata.get("chunk_id"),
                    "distance": result.distance,
                    "preview": _preview_text(result.text),
                }
            )
        return summaries


def retrieve_context(query: str) -> list[dict]:
    """Compatibility helper returning source summaries for a query."""

    retriever = RAGRetriever()
    context = retriever.retrieve(query)
    return retriever.get_source_summary(context.results)


def _entity_type_filter_for_route(route: QueryRoute) -> str | None:
    """Return a Chroma metadata filter value for a routing decision."""

    if route.route == ROUTE_PERSON:
        return "person"
    if route.route == ROUTE_PLACE:
        return "place"
    if route.route == ROUTE_BOTH:
        return None
    return None


def _entity_names_for_route(route: QueryRoute) -> list[str] | None:
    """Return exact configured entity names mentioned in the query, if any."""

    if route.route == ROUTE_PERSON and route.matched_people:
        return route.matched_people
    if route.route == ROUTE_PLACE and route.matched_places:
        return route.matched_places
    if route.route == ROUTE_BOTH:
        matched_entities = [*route.matched_people, *route.matched_places]
        return matched_entities or None
    return None


def _preview_text(text: str, max_chars: int = 160) -> str:
    """Return a compact one-line preview."""

    preview = " ".join(text.split())
    if len(preview) <= max_chars:
        return preview
    return preview[: max_chars - 3].rstrip() + "..."
