"""Retrieval orchestration for routed Wikipedia RAG queries."""

from __future__ import annotations

from dataclasses import dataclass
import re

from src import config
from src.query_router import (
    QueryRoute,
    ROUTE_BOTH,
    ROUTE_PERSON,
    ROUTE_PLACE,
    route_query,
)
from src.vector_store import ChromaVectorStore, VectorSearchResult


_OVERLAP_WORD_PATTERN = re.compile(r"\w+", re.UNICODE)
_MIN_OVERLAP_WORDS = 8
_MIN_CLEANED_WORDS = 4


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
        intro_results = []
        if entity_name_filter:
            intro_results = self.vector_store.get_intro_chunks(
                entity_name_filter,
                per_entity=1,
                entity_type=entity_type_filter,
            )

        semantic_results = self.vector_store.search(
            query,
            top_k=effective_top_k,
            entity_type=entity_type_filter,
            entity_names=entity_name_filter,
        )
        results = _merge_results(intro_results, semantic_results, effective_top_k)
        results = deduplicate_retrieved_overlaps(results)

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


def deduplicate_retrieved_overlaps(
    results: list[VectorSearchResult],
) -> list[VectorSearchResult]:
    """Remove repeated overlap prefixes from retrieved chunks for display/LLM context."""

    cleaned_results: list[VectorSearchResult] = []

    for result in results:
        cleaned_text = result.text
        current_entity = _entity_key(result)

        if current_entity:
            for previous_result in reversed(cleaned_results):
                if _entity_key(previous_result) != current_entity:
                    continue

                candidate_text = _remove_repeated_prefix(
                    previous_result.text,
                    cleaned_text,
                )
                if candidate_text != cleaned_text:
                    cleaned_text = candidate_text
                    break

        cleaned_results.append(
            VectorSearchResult(
                vector_id=result.vector_id,
                text=cleaned_text,
                metadata=result.metadata,
                distance=result.distance,
            )
        )

    return cleaned_results


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


def _merge_results(
    intro_results: list[VectorSearchResult],
    semantic_results: list[VectorSearchResult],
    limit: int,
) -> list[VectorSearchResult]:
    """Merge intro chunks before semantic chunks, de-duplicating by vector id."""

    merged_results: list[VectorSearchResult] = []
    seen_vector_ids: set[str] = set()

    for result in [*intro_results, *semantic_results]:
        if result.vector_id in seen_vector_ids:
            continue
        merged_results.append(result)
        seen_vector_ids.add(result.vector_id)
        if len(merged_results) >= limit:
            break

    return merged_results


def _entity_key(result: VectorSearchResult) -> str:
    """Return a normalized entity key for same-entity overlap cleanup."""

    entity = result.metadata.get("entity")
    return str(entity or "").strip().casefold()


def _remove_repeated_prefix(previous_text: str, current_text: str) -> str:
    """Remove a repeated word-overlap prefix from current_text when safe."""

    prefix_end = _find_word_overlap_prefix(previous_text, current_text)
    if prefix_end <= 0:
        return current_text

    cleaned_text = current_text[prefix_end:].lstrip(" \t\r\n.,;:!?)]}")
    if len(_word_tokens_for_overlap(cleaned_text)) < _MIN_CLEANED_WORDS:
        return current_text

    return cleaned_text.strip()


def _find_word_overlap_prefix(previous_text: str, current_text: str) -> int:
    """Return the character offset after a repeated current-text prefix."""

    previous_tokens = _word_tokens_for_overlap(previous_text)
    current_tokens = _word_tokens_for_overlap(current_text)
    max_overlap = min(len(previous_tokens), len(current_tokens))

    for word_count in range(max_overlap, _MIN_OVERLAP_WORDS - 1, -1):
        previous_suffix = [token for token, _, _ in previous_tokens[-word_count:]]
        current_prefix = [token for token, _, _ in current_tokens[:word_count]]
        if previous_suffix == current_prefix:
            return current_tokens[word_count - 1][2]

    return 0


def _word_tokens_for_overlap(text: str) -> list[tuple[str, int, int]]:
    """Return normalized word tokens with original character spans."""

    return [
        (match.group(0).casefold(), match.start(), match.end())
        for match in _OVERLAP_WORD_PATTERN.finditer(text)
    ]


def _preview_text(text: str, max_chars: int = 160) -> str:
    """Return a compact one-line preview."""

    preview = " ".join(text.split())
    if len(preview) <= max_chars:
        return preview
    return preview[: max_chars - 3].rstrip() + "..."
