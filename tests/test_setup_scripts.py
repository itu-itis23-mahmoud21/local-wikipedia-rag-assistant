"""Tests for setup and reset helper scripts."""

from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

from scripts import reset_system, setup_all


def _setup_args(**overrides) -> Namespace:
    """Build a setup_all argparse namespace for tests."""

    values = {
        "entity_type": "all",
        "limit": None,
        "force_ingest": False,
        "reset_db": False,
        "reset_chunks": False,
        "reset_collection": False,
        "skip_ingest": False,
        "skip_chunk": False,
        "skip_vector_store": False,
    }
    values.update(overrides)
    return Namespace(**values)


class TestSetupAllScript(unittest.TestCase):
    """Tests for setup_all.py command construction and pipeline behavior."""

    def test_builds_expected_ingest_command_with_executable_and_script_path(self) -> None:
        """The first command should run ingest_wikipedia.py with sys.executable."""

        commands = setup_all.build_pipeline_commands(_setup_args())

        self.assertEqual(commands[0][0], sys.executable)
        self.assertEqual(commands[0][1], str(setup_all.SCRIPTS_DIR / "ingest_wikipedia.py"))

    def test_passes_entity_type_limit_force_and_reset_db_to_ingest(self) -> None:
        """Ingest-specific arguments should be passed to ingestion."""

        commands = setup_all.build_pipeline_commands(
            _setup_args(
                entity_type="person",
                limit=3,
                force_ingest=True,
                reset_db=True,
            )
        )
        ingest_command = commands[0]

        self.assertIn("--entity-type", ingest_command)
        self.assertIn("person", ingest_command)
        self.assertIn("--limit", ingest_command)
        self.assertIn("3", ingest_command)
        self.assertIn("--force", ingest_command)
        self.assertIn("--reset-db", ingest_command)

    def test_passes_reset_chunks_to_chunk_script(self) -> None:
        """Chunk reset flag should be passed only to chunk_documents.py."""

        commands = setup_all.build_pipeline_commands(_setup_args(reset_chunks=True))
        chunk_command = commands[1]

        self.assertEqual(chunk_command[1], str(setup_all.SCRIPTS_DIR / "chunk_documents.py"))
        self.assertIn("--reset-chunks", chunk_command)
        self.assertNotIn("--reset-chunks", commands[0])

    def test_passes_reset_collection_to_vector_store_script(self) -> None:
        """Vector reset flag should be passed to build_vector_store.py."""

        commands = setup_all.build_pipeline_commands(
            _setup_args(reset_collection=True)
        )
        vector_command = commands[2]

        self.assertEqual(
            vector_command[1],
            str(setup_all.SCRIPTS_DIR / "build_vector_store.py"),
        )
        self.assertIn("--reset-collection", vector_command)

    def test_respects_skip_flags(self) -> None:
        """Skipped stages should not appear in built commands."""

        commands = setup_all.build_pipeline_commands(
            _setup_args(skip_ingest=True, skip_vector_store=True)
        )

        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][1], str(setup_all.SCRIPTS_DIR / "chunk_documents.py"))

    def test_run_pipeline_stops_and_returns_failing_exit_code(self) -> None:
        """Pipeline should stop at the first failing subprocess."""

        args = _setup_args()

        with patch("scripts.setup_all.run_command", side_effect=[0, 7, 0]) as mock_run:
            exit_code = setup_all.run_pipeline(args)

        self.assertEqual(exit_code, 7)
        self.assertEqual(mock_run.call_count, 2)


class TestResetSystemScript(unittest.TestCase):
    """Tests for reset_system.py target selection and safe deletion."""

    def test_default_targets_data_and_chroma(self) -> None:
        """Default reset should target data and chroma_db."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            targets = reset_system.get_reset_targets(
                data_only=False,
                chroma_only=False,
                project_root=root,
            )

        self.assertEqual(targets, [root / "data", root / "chroma_db"])

    def test_data_only_targets_only_data(self) -> None:
        """--data-only should target only data."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            targets = reset_system.get_reset_targets(
                data_only=True,
                chroma_only=False,
                project_root=root,
            )

        self.assertEqual(targets, [root / "data"])

    def test_chroma_only_targets_only_chroma_db(self) -> None:
        """--chroma-only should target only chroma_db."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            targets = reset_system.get_reset_targets(
                data_only=False,
                chroma_only=True,
                project_root=root,
            )

        self.assertEqual(targets, [root / "chroma_db"])

    def test_yes_removes_temporary_fake_data_and_chroma_directories(self) -> None:
        """--yes flow should remove fake generated directories without prompting."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            chroma_dir = root / "chroma_db"
            data_dir.mkdir()
            chroma_dir.mkdir()

            with patch(
                "scripts.reset_system.get_reset_targets",
                return_value=[data_dir, chroma_dir],
            ):
                exit_code = reset_system.main(["--yes"])

            self.assertEqual(exit_code, 0)
            self.assertFalse(data_dir.exists())
            self.assertFalse(chroma_dir.exists())

    def test_without_yes_can_cancel_without_deletion(self) -> None:
        """A negative confirmation should leave targets in place."""

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "data"
            target.mkdir()

            with patch("builtins.input", return_value="n"):
                confirmed = reset_system.confirm_reset([target], assume_yes=False)

            self.assertFalse(confirmed)
            self.assertTrue(target.exists())
