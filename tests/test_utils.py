"""Tests for shared utility helpers."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.utils import resolve_project_path, serialize_project_path


class TestPathUtilities(unittest.TestCase):
    """Tests for portable project path handling."""

    def test_serialize_project_path_returns_project_relative_string(self) -> None:
        """Project-local absolute paths should be stored as relative strings."""

        with TemporaryDirectory() as temporary_dir:
            project_root = Path(temporary_dir)
            file_path = project_root / "data" / "processed" / "person" / "einstein.txt"

            stored_path = serialize_project_path(file_path, project_root=project_root)

        self.assertEqual(stored_path, "data/processed/person/einstein.txt")

    def test_serialize_project_path_keeps_relative_string_portable(self) -> None:
        """Already-relative paths should remain relative and normalized."""

        stored_path = serialize_project_path(
            Path("data") / "raw" / "place" / "louvre.txt",
            project_root=Path.cwd(),
        )

        self.assertEqual(stored_path, "data/raw/place/louvre.txt")

    def test_resolve_project_path_resolves_relative_from_project_root(self) -> None:
        """Relative stored paths should resolve from the current project root."""

        with TemporaryDirectory() as temporary_dir:
            project_root = Path(temporary_dir)

            resolved_path = resolve_project_path(
                "data/processed/person/einstein.txt",
                project_root=project_root,
            )

            self.assertEqual(
                resolved_path,
                project_root / "data" / "processed" / "person" / "einstein.txt",
            )

    def test_resolve_project_path_recovers_legacy_absolute_data_suffix(self) -> None:
        """Stale absolute paths should recover their data/... suffix when possible."""

        with TemporaryDirectory() as temporary_dir:
            project_root = Path(temporary_dir)
            legacy_path = Path("C:/old/project/data/processed/person/einstein.txt")

            resolved_path = resolve_project_path(legacy_path, project_root=project_root)

            self.assertEqual(
                resolved_path,
                project_root / "data" / "processed" / "person" / "einstein.txt",
            )

    def test_resolve_project_path_returns_none_for_blank_values(self) -> None:
        """Blank stored paths should resolve to None."""

        self.assertIsNone(resolve_project_path("  "))
        self.assertIsNone(resolve_project_path(None))


if __name__ == "__main__":
    unittest.main()
