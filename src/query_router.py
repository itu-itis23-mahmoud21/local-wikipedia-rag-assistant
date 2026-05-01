"""Query routing placeholder.

This module will decide whether a user question is about a person, a place, or
both, then expose that routing decision to retrieval.
"""


def route_query(query: str) -> str:
    """Classify a query as `person`, `place`, or `both`."""

    raise NotImplementedError("Query routing is not implemented yet.")
