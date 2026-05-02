"""Rule-based query routing for person/place Wikipedia questions."""

from __future__ import annotations

from dataclasses import dataclass
import re

from src.entities import get_people, get_places


ROUTE_PERSON = "person"
ROUTE_PLACE = "place"
ROUTE_BOTH = "both"
ROUTE_UNKNOWN = "unknown"

COMPARISON_INDICATORS = (
    "compare",
    "comparison",
    "versus",
    "vs",
    "difference",
    "similarities",
    "similar",
    "contrast",
)

PERSON_KEYWORDS = (
    "who",
    "person",
    "born",
    "died",
    "discovered",
    "invented",
    "wrote",
    "singer",
    "footballer",
    "athlete",
    "scientist",
    "artist",
    "king",
    "queen",
    "president",
)

PLACE_KEYWORDS = (
    "where",
    "located",
    "built",
    "monument",
    "city",
    "country",
    "mountain",
    "river",
    "museum",
    "palace",
    "tower",
    "landmark",
    "place",
)

BROAD_MIXED_CLUES = (
    "associated with",
    "known for",
)

LOCATION_ENTITY_HINTS = {
    "turkey": ["Hagia Sophia", "Blue Mosque", "Topkapı Palace"],
    "türkiye": ["Hagia Sophia", "Blue Mosque", "Topkapı Palace"],
    "turkish": ["Hagia Sophia", "Blue Mosque", "Topkapı Palace"],
    "istanbul": ["Hagia Sophia", "Blue Mosque", "Topkapı Palace"],
}

MIN_ALIAS_LENGTH = 3
ALIAS_STOPWORDS = {
    "about",
    "and",
    "are",
    "associated",
    "born",
    "built",
    "city",
    "compare",
    "country",
    "difference",
    "famous",
    "for",
    "from",
    "great",
    "how",
    "known",
    "located",
    "monument",
    "museum",
    "person",
    "place",
    "president",
    "the",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
SURNAME_PARTICLES = {"da", "de", "del", "la", "le", "van", "von"}


@dataclass(frozen=True)
class QueryRoute:
    """Routing decision and matched configured entities for a query."""

    route: str
    matched_people: list[str]
    matched_places: list[str]
    reason: str


def normalize_query(text: str) -> str:
    """Normalize query text for simple rule matching."""

    return " ".join(text.casefold().strip().split())


def find_entity_mentions(query: str, candidates: list[str]) -> list[str]:
    """Return configured entity names mentioned in the query or by safe alias."""

    normalized_query = normalize_query(query)
    aliases = build_entity_aliases(candidates)
    mentions: list[str] = []
    seen: set[str] = set()

    for alias, canonical_name in aliases.items():
        normalized_canonical = normalize_query(canonical_name)
        if normalized_canonical in seen:
            continue
        if _contains_phrase(normalized_query, alias):
            mentions.append(canonical_name)
            seen.add(normalized_canonical)

    return mentions


def build_entity_aliases(candidates: list[str]) -> dict[str, str]:
    """Build safe, unambiguous aliases that map to canonical entity names."""

    alias_candidates: dict[str, list[str]] = {}
    allow_partial_aliases = _all_candidates_are_people(candidates)

    for candidate in candidates:
        for alias in _candidate_aliases(candidate, allow_partial_aliases):
            alias_candidates.setdefault(alias, []).append(candidate)

    aliases: dict[str, str] = {}
    for alias, names in alias_candidates.items():
        unique_names = list(dict.fromkeys(names))
        if len(unique_names) == 1:
            aliases[alias] = unique_names[0]

    return aliases


def route_query(query: str) -> QueryRoute:
    """Classify a query as person, place, both, or unknown."""

    normalized_query = normalize_query(query)
    if not normalized_query:
        raise ValueError("query must not be blank.")

    matched_people = find_entity_mentions(query, get_people())
    matched_places = find_entity_mentions(query, get_places())
    total_matches = len(matched_people) + len(matched_places)
    has_comparison = _contains_any_keyword(normalized_query, COMPARISON_INDICATORS)

    if has_comparison and total_matches >= 2:
        return QueryRoute(
            route=ROUTE_BOTH,
            matched_people=matched_people,
            matched_places=matched_places,
            reason="comparison query with multiple known entities",
        )

    if matched_people and matched_places:
        return QueryRoute(
            route=ROUTE_BOTH,
            matched_people=matched_people,
            matched_places=matched_places,
            reason="known person and place mentioned",
        )

    if matched_people:
        return QueryRoute(
            route=ROUTE_PERSON,
            matched_people=matched_people,
            matched_places=matched_places,
            reason="known person mentioned",
        )

    if matched_places:
        return QueryRoute(
            route=ROUTE_PLACE,
            matched_people=matched_people,
            matched_places=matched_places,
            reason="known place mentioned",
        )

    if _contains_any_keyword(normalized_query, PERSON_KEYWORDS):
        return QueryRoute(
            route=ROUTE_PERSON,
            matched_people=matched_people,
            matched_places=matched_places,
            reason="person-related keyword detected",
        )

    if _contains_any_keyword(normalized_query, PLACE_KEYWORDS):
        return QueryRoute(
            route=ROUTE_PLACE,
            matched_people=matched_people,
            matched_places=matched_places,
            reason="place-related keyword detected",
        )

    if _contains_any_phrase(normalized_query, BROAD_MIXED_CLUES):
        return QueryRoute(
            route=ROUTE_BOTH,
            matched_people=matched_people,
            matched_places=matched_places,
            reason="broad mixed clue detected",
        )

    return QueryRoute(
        route=ROUTE_UNKNOWN,
        matched_people=matched_people,
        matched_places=matched_places,
        reason="no configured entity or routing keyword matched",
    )


def get_location_entity_hints(query: str) -> list[str]:
    """Return configured place hints implied by a location term in the query."""

    normalized_query = normalize_query(query)
    if not normalized_query:
        return []

    configured_places = set(get_places())
    hinted_entities: list[str] = []
    seen: set[str] = set()

    for location_term, entity_names in LOCATION_ENTITY_HINTS.items():
        if not _contains_phrase(normalized_query, normalize_query(location_term)):
            continue

        for entity_name in entity_names:
            if entity_name not in configured_places or entity_name in seen:
                continue
            hinted_entities.append(entity_name)
            seen.add(entity_name)

    return hinted_entities


def _contains_phrase(text: str, phrase: str) -> bool:
    """Return whether a normalized phrase appears with safe boundaries."""

    pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
    return re.search(pattern, text) is not None


def _contains_any_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    """Return whether any keyword appears with safe word boundaries."""

    return any(_contains_phrase(text, keyword) for keyword in keywords)


def _contains_any_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    """Return whether any phrase appears in text."""

    return any(phrase in text for phrase in phrases)


def _candidate_aliases(candidate: str, allow_partial_aliases: bool) -> list[str]:
    """Return normalized alias candidates for one configured entity."""

    normalized_candidate = normalize_query(candidate)
    aliases = [normalized_candidate]

    if allow_partial_aliases:
        tokens = _alias_tokens(normalized_candidate)
        if tokens:
            aliases.extend([tokens[0], tokens[-1]])
            aliases.extend(_particle_aliases(tokens))

    safe_aliases: list[str] = []
    for alias in aliases:
        if alias not in safe_aliases and _is_safe_alias(alias):
            safe_aliases.append(alias)

    return safe_aliases


def _alias_tokens(normalized_name: str) -> list[str]:
    """Return simple alphanumeric tokens from a normalized entity name."""

    return re.findall(r"\w+", normalized_name)


def _particle_aliases(tokens: list[str]) -> list[str]:
    """Return multi-word surname aliases such as 'da vinci' or 'van gogh'."""

    aliases: list[str] = []
    for index, token in enumerate(tokens[:-1]):
        if token in SURNAME_PARTICLES:
            aliases.append(" ".join(tokens[index:]))
    return aliases


def _is_safe_alias(alias: str) -> bool:
    """Return whether an alias is specific enough for query matching."""

    if len(alias) < MIN_ALIAS_LENGTH:
        return False
    if alias in ALIAS_STOPWORDS:
        return False
    return True


def _all_candidates_are_people(candidates: list[str]) -> bool:
    """Return whether the provided candidates all come from the people list."""

    if not candidates:
        return False
    people = set(get_people())
    places = set(get_places())
    return all(candidate in people for candidate in candidates) or not any(
        candidate in places for candidate in candidates
    )
