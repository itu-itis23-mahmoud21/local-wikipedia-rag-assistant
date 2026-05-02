"""Entity configuration for famous people and famous places.

The assistant ingests Wikipedia pages for these configured entities. This
module intentionally contains only static entity metadata and small helper
functions.
"""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Entity:
    """A configured Wikipedia entity and its category."""

    name: str
    entity_type: str


PEOPLE: list[str] = [
    "Albert Einstein",
    "Marie Curie",
    "Leonardo da Vinci",
    "William Shakespeare",
    "Ada Lovelace",
    "Nikola Tesla",
    "Lionel Messi",
    "Cristiano Ronaldo",
    "Taylor Swift",
    "Frida Kahlo",
    "Mohamed Salah",
    "Isaac Newton",
    "Charles Darwin",
    "Alan Turing",
    "Grace Hopper",
    "Aristotle",
    "Mahatma Gandhi",
    "Nelson Mandela",
    "Martin Luther King Jr.",
    "Abraham Lincoln",
    "Cleopatra",
    "Queen Elizabeth II",
    "Pablo Picasso",
    "Wolfgang Amadeus Mozart",
    "Jane Austen",
]

PLACES: list[str] = [
    "Eiffel Tower",
    "Great Wall of China",
    "Taj Mahal",
    "Grand Canyon",
    "Machu Picchu",
    "Colosseum",
    "Hagia Sophia",
    "Statue of Liberty",
    "Pyramids of Giza",
    "Mount Everest",
    "Burj Khalifa",
    "Big Ben",
    "Sydney Opera House",
    "Sagrada Família",
    "Stonehenge",
    "Petra",
    "Angkor Wat",
    "Acropolis of Athens",
    "Louvre",
    "Vatican City",
    "Mount Fuji",
    "Niagara Falls",
    "Alhambra",
    "Blue Mosque",
    "Topkapı Palace",
]

PLACE_LOCATION_HINTS: dict[str, list[str]] = {
    "Eiffel Tower": ["france", "paris"],
    "Great Wall of China": ["china"],
    "Taj Mahal": ["india", "agra"],
    "Grand Canyon": ["united states", "usa", "u.s.", "arizona"],
    "Machu Picchu": ["peru", "cusco"],
    "Colosseum": ["italy", "rome"],
    "Hagia Sophia": ["turkey", "türkiye", "istanbul"],
    "Statue of Liberty": ["united states", "usa", "u.s.", "new york"],
    "Pyramids of Giza": ["egypt", "giza", "cairo"],
    "Mount Everest": ["nepal", "china", "tibet", "himalayas"],
    "Burj Khalifa": ["united arab emirates", "uae", "dubai"],
    "Big Ben": ["united kingdom", "uk", "england", "london"],
    "Sydney Opera House": ["australia", "sydney"],
    "Sagrada Família": ["spain", "barcelona"],
    "Stonehenge": ["united kingdom", "uk", "england", "wiltshire"],
    "Petra": ["jordan"],
    "Angkor Wat": ["cambodia", "siem reap"],
    "Acropolis of Athens": ["greece", "athens"],
    "Louvre": ["france", "paris"],
    "Vatican City": ["vatican", "vatican city", "rome", "italy"],
    "Mount Fuji": ["japan"],
    "Niagara Falls": ["canada", "united states", "usa", "u.s.", "new york", "ontario"],
    "Alhambra": ["spain", "granada"],
    "Blue Mosque": ["turkey", "türkiye", "istanbul"],
    "Topkapı Palace": ["turkey", "türkiye", "istanbul"],
}

ENTITY_TYPES = ("person", "place")


def get_people() -> list[str]:
    """Return a copy of the configured people list."""

    return list(PEOPLE)


def get_places() -> list[str]:
    """Return a copy of the configured places list."""

    return list(PLACES)


def get_all_entities() -> list[str]:
    """Return a copy of all configured entity names."""

    return get_people() + get_places()


def get_entity_records() -> list[Entity]:
    """Return all configured entities as typed records."""

    return [Entity(name, "person") for name in PEOPLE] + [
        Entity(name, "place") for name in PLACES
    ]


def get_places_for_location_query(query: str) -> list[str]:
    """Return configured places whose location hints appear in a query."""

    normalized_query = _normalize_location_text(query)
    if not normalized_query:
        return []

    matched_places: list[str] = []
    for place in PLACES:
        for hint in PLACE_LOCATION_HINTS.get(place, []):
            if _contains_location_hint(normalized_query, hint):
                matched_places.append(place)
                break

    return matched_places


def get_entity_type(name: str) -> str | None:
    """Return the configured entity type for a name, case-insensitively."""

    normalized_name = _normalize_name(name)
    for person in PEOPLE:
        if _normalize_name(person) == normalized_name:
            return "person"
    for place in PLACES:
        if _normalize_name(place) == normalized_name:
            return "place"
    return None


def is_known_entity(name: str) -> bool:
    """Return whether a name is configured as a person or place."""

    return get_entity_type(name) is not None


def validate_entities() -> None:
    """Validate entity counts, names, uniqueness, and place metadata."""

    if len(PEOPLE) != 25:
        raise ValueError(f"Expected 25 people, found {len(PEOPLE)}.")
    if len(PLACES) != 25:
        raise ValueError(f"Expected 25 places, found {len(PLACES)}.")

    all_names = PEOPLE + PLACES
    blank_names = [name for name in all_names if not name.strip()]
    if blank_names:
        raise ValueError("Entity names must not be blank.")

    seen: set[str] = set()
    duplicates: list[str] = []
    for name in all_names:
        normalized_name = _normalize_name(name)
        if normalized_name in seen:
            duplicates.append(name)
        seen.add(normalized_name)

    if duplicates:
        duplicate_text = ", ".join(duplicates)
        raise ValueError(f"Duplicate entity names found: {duplicate_text}.")

    _validate_place_location_hints()


def _normalize_name(name: str) -> str:
    """Normalize entity names for case-insensitive matching."""

    return name.strip().casefold()


def _validate_place_location_hints() -> None:
    """Validate that every configured place has non-blank location hints."""

    place_names = set(PLACES)
    hint_names = set(PLACE_LOCATION_HINTS)
    unknown_places = sorted(hint_names - place_names)
    missing_places = sorted(place_names - hint_names)

    if unknown_places:
        raise ValueError(
            "Location hints include unknown places: "
            f"{', '.join(unknown_places)}."
        )
    if missing_places:
        raise ValueError(
            "Location hints are missing configured places: "
            f"{', '.join(missing_places)}."
        )

    blank_hints = [
        place
        for place, hints in PLACE_LOCATION_HINTS.items()
        if not hints or any(not hint.strip() for hint in hints)
    ]
    if blank_hints:
        raise ValueError(
            "Location hints must not be blank for places: "
            f"{', '.join(blank_hints)}."
        )


def _normalize_location_text(text: str) -> str:
    """Normalize location query/hint text for deterministic matching."""

    return " ".join(text.casefold().strip().split())


def _contains_location_hint(normalized_query: str, hint: str) -> bool:
    """Return whether a normalized location hint appears with safe boundaries."""

    normalized_hint = _normalize_location_text(hint)
    if not normalized_hint:
        return False
    pattern = rf"(?<!\w){re.escape(normalized_hint)}(?!\w)"
    return re.search(pattern, normalized_query) is not None


validate_entities()
