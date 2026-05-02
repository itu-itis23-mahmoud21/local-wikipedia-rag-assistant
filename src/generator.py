"""Local grounded answer generation with Ollama.

This module turns retrieved context into a concise answer using a local Ollama
language model. It does not call external LLM APIs.
"""

from __future__ import annotations

from dataclasses import dataclass

from src import config
from src.retriever import RAGRetriever

try:
    import ollama
except ImportError:  # pragma: no cover - exercised through GenerationError tests.
    ollama = None


class GenerationError(Exception):
    """Raised when local answer generation fails."""


@dataclass(frozen=True)
class GeneratedAnswer:
    """Generated answer and its retrieval metadata."""

    query: str
    answer: str
    route: str
    context: str
    sources: list[dict]
    model: str


class OllamaAnswerGenerator:
    """Generate grounded answers with a local Ollama model."""

    def __init__(
        self,
        model: str = config.OLLAMA_GENERATION_MODEL,
        temperature: float = config.DEFAULT_GENERATION_TEMPERATURE,
    ) -> None:
        """Create a local answer generator."""

        if not model.strip():
            raise ValueError("model must not be blank.")
        if temperature < 0 or temperature > 1:
            raise ValueError("temperature must be between 0 and 1.")

        self.model = model
        self.temperature = temperature

    def build_prompt(self, query: str, context: str) -> str:
        """Build a grounded RAG prompt for the local language model."""

        clean_query = _validate_query(query)
        clean_context = context.strip()

        return (
            "You are a local Wikipedia RAG assistant. Use only the retrieved "
            "context below. Do not use outside knowledge. If the context does "
            "not contain the answer, say \"I don't know.\" Be concise but "
            "complete. For comparison questions, compare only using available "
            "context. Keep facts separated by entity, describe each compared "
            "entity separately first, and do not transfer facts from one entity "
            "to another. For 'which person' or 'which place' questions, identify "
            "the main configured entity represented by the retrieved sources. "
            "Do not list every person or place merely mentioned inside the "
            "context. If one retrieved entity is clearly the best match, answer "
            "with that entity and explain briefly. If the retrieved context does "
            "not support a claim, say \"I don't know\" or explain that the "
            "context does not support it. Include no fake citations.\n\n"
            "Retrieved context:\n"
            f"{clean_context}\n\n"
            "User question:\n"
            f"{clean_query}\n\n"
            "Answer:"
        )

    def generate_from_context(self, query: str, context: str) -> str:
        """Generate an answer from retrieved context with local Ollama."""

        _validate_query(query)
        if not context.strip():
            return "I don't know."

        prompt = self.build_prompt(query, context)
        response = self._call_ollama(prompt)
        answer = self._extract_response_text(response)
        if not answer:
            raise GenerationError(
                f"Ollama local generation model '{self.model}' returned a blank answer."
            )
        return answer

    def answer_query(
        self,
        query: str,
        retriever: RAGRetriever | None = None,
        top_k: int | None = None,
    ) -> GeneratedAnswer:
        """Retrieve context for a query and generate a grounded answer."""

        active_retriever = retriever or RAGRetriever()
        retrieved_context = active_retriever.retrieve(query, top_k=top_k)
        context = active_retriever.format_context(
            retrieved_context.results,
            max_chars=config.DEFAULT_CONTEXT_MAX_CHARS,
        )
        sources = active_retriever.get_source_summary(retrieved_context.results)
        answer = self.generate_from_context(query, context)

        return GeneratedAnswer(
            query=query,
            answer=answer,
            route=retrieved_context.route.route,
            context=context,
            sources=sources,
            model=self.model,
        )

    def _call_ollama(self, prompt: str) -> object:
        """Call local Ollama generation."""

        if ollama is None:
            raise GenerationError(
                "Ollama Python package is not available for local generation "
                f"model '{self.model}'."
            )

        try:
            return ollama.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": self.temperature},
            )
        except Exception as exc:
            raise GenerationError(
                f"Ollama local generation model '{self.model}' failed: {exc}"
            ) from exc

    def _extract_response_text(self, response: object) -> str:
        """Extract answer text from common Ollama response shapes."""

        if isinstance(response, dict):
            if "response" not in response:
                raise GenerationError(
                    f"Ollama local generation model '{self.model}' returned "
                    "malformed output."
                )
            return str(response["response"] or "").strip()

        object_response = getattr(response, "response", None)
        if object_response is None:
            raise GenerationError(
                f"Ollama local generation model '{self.model}' returned "
                "malformed output."
            )
        return str(object_response or "").strip()


def generate_answer(query: str) -> GeneratedAnswer:
    """Generate a grounded answer for a query using default components."""

    return OllamaAnswerGenerator().answer_query(query)


def _validate_query(query: str) -> str:
    """Validate and strip a user query."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be blank.")
    return query.strip()
