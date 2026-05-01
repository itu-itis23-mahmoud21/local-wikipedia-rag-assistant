"""Entity configuration for famous people and famous places.

The final project will define 50 famous people and 50 famous places here. The
lists are intentionally empty in this skeleton step.
"""

PEOPLE: list[str] = []
PLACES: list[str] = []
ENTITY_TYPES = ("person", "place")


def get_all_entities() -> list[str]:
    """Return all configured entity names.

    This placeholder will become useful after the people and places lists are
    populated in a later implementation step.
    """

    return PEOPLE + PLACES
