"""Tests for rule-based query routing."""

import unittest

from src.query_router import (
    ROUTE_BOTH,
    ROUTE_PERSON,
    ROUTE_PLACE,
    ROUTE_UNKNOWN,
    build_entity_aliases,
    find_entity_mentions,
    route_query,
)


class TestQueryRouter(unittest.TestCase):
    """Tests for person/place/both/unknown routing."""

    def test_blank_query_raises_value_error(self) -> None:
        """Blank queries should be rejected."""

        with self.assertRaises(ValueError):
            route_query("  ")

    def test_exact_person_mention_routes_person(self) -> None:
        """A known person mention should route to person."""

        route = route_query("What did Albert Einstein discover?")

        self.assertEqual(route.route, ROUTE_PERSON)
        self.assertIn("Albert Einstein", route.matched_people)
        self.assertEqual(route.matched_places, [])

    def test_exact_place_mention_routes_place(self) -> None:
        """A known place mention should route to place."""

        route = route_query("Where is the Eiffel Tower located?")

        self.assertEqual(route.route, ROUTE_PLACE)
        self.assertIn("Eiffel Tower", route.matched_places)
        self.assertEqual(route.matched_people, [])

    def test_person_and_place_mention_routes_both(self) -> None:
        """Mentioning a person and place should route to both."""

        route = route_query("Did Leonardo da Vinci ever visit the Louvre?")

        self.assertEqual(route.route, ROUTE_BOTH)
        self.assertIn("Leonardo da Vinci", route.matched_people)
        self.assertIn("Louvre", route.matched_places)

    def test_compare_einstein_and_tesla_routes_both(self) -> None:
        """Comparison of two people should route to both."""

        route = route_query("Compare Albert Einstein and Nikola Tesla")

        self.assertEqual(route.route, ROUTE_BOTH)
        self.assertIn("Albert Einstein", route.matched_people)
        self.assertIn("Nikola Tesla", route.matched_people)

    def test_compare_two_places_routes_both(self) -> None:
        """Comparison of two places should route to both."""

        route = route_query("Compare the Eiffel Tower and the Statue of Liberty")

        self.assertEqual(route.route, ROUTE_BOTH)
        self.assertIn("Eiffel Tower", route.matched_places)
        self.assertIn("Statue of Liberty", route.matched_places)

    def test_famous_place_in_turkey_routes_place(self) -> None:
        """A place clue without exact entity mention should route to place."""

        route = route_query("Which famous place is located in Turkey?")

        self.assertEqual(route.route, ROUTE_PLACE)

    def test_person_associated_with_electricity_routes_person(self) -> None:
        """A person clue should route to person."""

        route = route_query("Which person is associated with electricity?")

        self.assertEqual(route.route, ROUTE_PERSON)

    def test_unknown_neutral_query_routes_unknown(self) -> None:
        """Neutral queries without clues should route unknown."""

        route = route_query("Tell me about a random concept")

        self.assertEqual(route.route, ROUTE_UNKNOWN)

    def test_matching_is_case_insensitive(self) -> None:
        """Configured entity matching should ignore case."""

        route = route_query("what is ALBERT EINSTEIN known for?")

        self.assertEqual(route.route, ROUTE_PERSON)
        self.assertIn("Albert Einstein", route.matched_people)

    def test_find_entity_mentions_does_not_duplicate_names(self) -> None:
        """Repeated query mentions should not duplicate candidate names."""

        mentions = find_entity_mentions(
            "Albert Einstein and Albert Einstein",
            ["Albert Einstein"],
        )

        self.assertEqual(mentions, ["Albert Einstein"])

    def test_last_name_alias_einstein_routes_person(self) -> None:
        """A safe last-name alias should match the configured person."""

        route = route_query("Who is Einstein?")

        self.assertEqual(route.route, ROUTE_PERSON)
        self.assertIn("Albert Einstein", route.matched_people)

    def test_last_name_alias_curie_routes_person(self) -> None:
        """Curie should resolve to Marie Curie."""

        route = route_query("Who is Curie?")

        self.assertEqual(route.route, ROUTE_PERSON)
        self.assertIn("Marie Curie", route.matched_people)

    def test_last_name_alias_shakespeare_routes_person(self) -> None:
        """Shakespeare should resolve to William Shakespeare."""

        route = route_query("Who is Shakespeare?")

        self.assertEqual(route.route, ROUTE_PERSON)
        self.assertIn("William Shakespeare", route.matched_people)

    def test_last_name_alias_lovelace_routes_person(self) -> None:
        """Lovelace should resolve to Ada Lovelace."""

        route = route_query("Who is Lovelace?")

        self.assertEqual(route.route, ROUTE_PERSON)
        self.assertIn("Ada Lovelace", route.matched_people)

    def test_multi_word_alias_da_vinci_routes_person(self) -> None:
        """A multi-word surname alias should resolve to Leonardo da Vinci."""

        route = route_query("Who is da Vinci?")

        self.assertEqual(route.route, ROUTE_PERSON)
        self.assertIn("Leonardo da Vinci", route.matched_people)

    def test_unambiguous_first_name_alias_routes_person(self) -> None:
        """Unambiguous first-name aliases should resolve to the configured name."""

        route = route_query("Who is Ada?")

        self.assertEqual(route.route, ROUTE_PERSON)
        self.assertIn("Ada Lovelace", route.matched_people)

    def test_alias_matching_is_case_insensitive(self) -> None:
        """Alias matching should ignore case."""

        route = route_query("who is eInStEiN?")

        self.assertEqual(route.route, ROUTE_PERSON)
        self.assertIn("Albert Einstein", route.matched_people)

    def test_alias_and_full_name_do_not_duplicate_mentions(self) -> None:
        """A full name and alias in one query should produce one canonical mention."""

        mentions = find_entity_mentions(
            "Albert Einstein and Einstein",
            ["Albert Einstein"],
        )

        self.assertEqual(mentions, ["Albert Einstein"])

    def test_ambiguous_alias_is_not_used(self) -> None:
        """Single-token aliases should be ignored when they are ambiguous."""

        aliases = build_entity_aliases(["Michael Jackson", "Michael Jordan"])

        self.assertNotIn("michael", aliases)
        self.assertEqual(aliases["jackson"], "Michael Jackson")
        self.assertEqual(aliases["jordan"], "Michael Jordan")
