"""Tests for entity configuration."""

import unittest

from src import entities


REQUIRED_HW_PEOPLE = [
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
]

REQUIRED_HW_PLACES = [
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
]


class TestEntities(unittest.TestCase):
    """Tests for the configured people and places."""

    def test_entity_placeholders_are_lists(self) -> None:
        """The module should expose person and place lists."""

        self.assertIsInstance(entities.PEOPLE, list)
        self.assertIsInstance(entities.PLACES, list)

    def test_people_count_is_50(self) -> None:
        """The homework configuration should include exactly 50 people."""

        self.assertEqual(len(entities.PEOPLE), 50)

    def test_places_count_is_50(self) -> None:
        """The homework configuration should include exactly 50 places."""

        self.assertEqual(len(entities.PLACES), 50)

    def test_total_entities_count_is_100(self) -> None:
        """People and places together should total 100 entities."""

        self.assertEqual(len(entities.get_all_entities()), 100)

    def test_required_hw_people_are_included(self) -> None:
        """The first required HW people should be present."""

        for person in REQUIRED_HW_PEOPLE:
            with self.subTest(person=person):
                self.assertIn(person, entities.PEOPLE)

    def test_required_hw_places_are_included(self) -> None:
        """The first required HW places should be present."""

        for place in REQUIRED_HW_PLACES:
            with self.subTest(place=place):
                self.assertIn(place, entities.PLACES)

    def test_mohamed_salah_is_included(self) -> None:
        """Mohamed Salah should be part of the people configuration."""

        self.assertIn("Mohamed Salah", entities.PEOPLE)

    def test_accented_homework_names_are_exact(self) -> None:
        """Names with non-ASCII characters should match the homework list."""

        self.assertIn("Beyoncé", entities.PEOPLE)
        self.assertIn("Sagrada Família", entities.PLACES)
        self.assertIn("Topkapı Palace", entities.PLACES)

    def test_get_entity_type_is_case_insensitive(self) -> None:
        """Entity type lookup should ignore case."""

        self.assertEqual(entities.get_entity_type("albert einstein"), "person")
        self.assertEqual(entities.get_entity_type("MOUNT EVEREST"), "place")
        self.assertIsNone(entities.get_entity_type("Unknown Entity"))

    def test_is_known_entity_is_case_insensitive(self) -> None:
        """Known entity lookup should ignore case."""

        self.assertTrue(entities.is_known_entity("taylor swift"))
        self.assertTrue(entities.is_known_entity("eiffel tower"))
        self.assertFalse(entities.is_known_entity("Unknown Entity"))

    def test_get_entity_records_returns_100_entity_records(self) -> None:
        """Entity records should include all configured people and places."""

        records = entities.get_entity_records()

        self.assertEqual(len(records), 100)
        self.assertTrue(all(isinstance(record, entities.Entity) for record in records))

    def test_returned_lists_are_copies(self) -> None:
        """Helper list results should not mutate module-level lists."""

        people = entities.get_people()
        places = entities.get_places()
        all_entities = entities.get_all_entities()

        people.append("Temporary Person")
        places.append("Temporary Place")
        all_entities.append("Temporary Entity")

        self.assertNotIn("Temporary Person", entities.PEOPLE)
        self.assertNotIn("Temporary Place", entities.PLACES)
        self.assertNotIn("Temporary Entity", entities.PEOPLE)
        self.assertNotIn("Temporary Entity", entities.PLACES)
        self.assertEqual(len(entities.PEOPLE), 50)
        self.assertEqual(len(entities.PLACES), 50)

    def test_validate_entities_does_not_raise(self) -> None:
        """The current configuration should pass validation."""

        entities.validate_entities()
