"""Project structure tests for the initial repository skeleton."""

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


EXPECTED_PATHS = [
    "app.py",
    "README.md",
    "product_prd.md",
    "recommendation.md",
    "DEMO_CHECKLIST.md",
    "requirements.txt",
    ".gitignore",
    "src/__init__.py",
    "src/config.py",
    "src/entities.py",
    "src/database.py",
    "src/wiki_client.py",
    "src/chunker.py",
    "src/embeddings.py",
    "src/vector_store.py",
    "src/query_router.py",
    "src/retriever.py",
    "src/generator.py",
    "src/utils.py",
    "scripts/ingest_wikipedia.py",
    "scripts/chunk_documents.py",
    "scripts/build_vector_store.py",
    "scripts/setup_all.py",
    "scripts/reset_system.py",
    "tests/__init__.py",
    "tests/test_entities.py",
    "tests/test_project_structure.py",
]


class TestProjectStructure(unittest.TestCase):
    """Tests for the requested repository skeleton."""

    def test_expected_project_paths_exist(self) -> None:
        """Verify the requested skeleton files and folders exist."""

        missing_paths = [
            relative_path
            for relative_path in EXPECTED_PATHS
            if not (PROJECT_ROOT / relative_path).exists()
        ]
        self.assertEqual(missing_paths, [])
