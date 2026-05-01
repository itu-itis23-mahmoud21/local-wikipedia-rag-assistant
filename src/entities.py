"""Entity configuration for famous people and famous places.

The assistant will ingest Wikipedia pages for these configured entities in a
later step. This module intentionally contains only static entity metadata and
small helper functions.
"""

from dataclasses import dataclass


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
    "Galileo Galilei",
    "Stephen Hawking",
    "Alan Turing",
    "Grace Hopper",
    "Steve Jobs",
    "Bill Gates",
    "Elon Musk",
    "Aristotle",
    "Plato",
    "Socrates",
    "Mahatma Gandhi",
    "Nelson Mandela",
    "Martin Luther King Jr.",
    "Abraham Lincoln",
    "Winston Churchill",
    "Cleopatra",
    "Alexander the Great",
    "Napoleon",
    "Julius Caesar",
    "Queen Elizabeth II",
    "Vincent van Gogh",
    "Pablo Picasso",
    "Michelangelo",
    "Ludwig van Beethoven",
    "Wolfgang Amadeus Mozart",
    "Michael Jackson",
    "Elvis Presley",
    "Beyoncé",
    "Bob Marley",
    "J. K. Rowling",
    "Agatha Christie",
    "Jane Austen",
    "Mark Twain",
    "Walt Disney",
    "Serena Williams",
    "Usain Bolt",
    "Michael Jordan",
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
    "Tower Bridge",
    "Sydney Opera House",
    "Sagrada Família",
    "Stonehenge",
    "Petra",
    "Angkor Wat",
    "Acropolis of Athens",
    "Louvre",
    "Vatican City",
    "Notre-Dame de Paris",
    "Palace of Versailles",
    "Buckingham Palace",
    "Kremlin",
    "Leaning Tower of Pisa",
    "Christ the Redeemer",
    "Golden Gate Bridge",
    "Mount Fuji",
    "Niagara Falls",
    "Amazon Rainforest",
    "Sahara",
    "Nile",
    "Amazon River",
    "Dead Sea",
    "Lake Baikal",
    "Mount Kilimanjaro",
    "Santorini",
    "Maldives",
    "Bali",
    "Central Park",
    "Times Square",
    "Hollywood Sign",
    "Disneyland",
    "Las Vegas Strip",
    "Alhambra",
    "Blue Mosque",
    "Topkapı Palace",
    "Cappadocia",
    "Mount Rushmore",
]

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
    """Validate entity counts, names, and uniqueness."""

    if len(PEOPLE) != 50:
        raise ValueError(f"Expected 50 people, found {len(PEOPLE)}.")
    if len(PLACES) != 50:
        raise ValueError(f"Expected 50 places, found {len(PLACES)}.")

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


def _normalize_name(name: str) -> str:
    """Normalize entity names for case-insensitive matching."""

    return name.strip().casefold()


validate_entities()
