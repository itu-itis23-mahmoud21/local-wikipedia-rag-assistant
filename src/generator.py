"""Local answer generation placeholder.

This module will call the local Ollama generation model, `llama3.2:3b`, with
retrieved context and instructions to answer only from grounded evidence.
"""


def generate_answer(query: str, context: list[dict[str, str]]) -> str:
    """Generate a grounded answer from retrieved context."""

    raise NotImplementedError("Answer generation is not implemented yet.")
