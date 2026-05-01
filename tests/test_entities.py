"""Placeholder tests for entity configuration."""

import unittest

from src import entities


class TestEntityPlaceholders(unittest.TestCase):
    """Tests for the initial empty entity configuration."""

    def test_entity_placeholders_are_lists(self) -> None:
        """The skeleton should expose person and place lists without data."""

        self.assertIsInstance(entities.PEOPLE, list)
        self.assertIsInstance(entities.PLACES, list)
