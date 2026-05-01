"""Tests for the SQLite metadata database layer."""

from contextlib import closing
from pathlib import Path
import tempfile
import unittest

from src.database import MetadataDB


class TestMetadataDB(unittest.TestCase):
    """Tests for MetadataDB using isolated temporary databases."""

    def setUp(self) -> None:
        """Create a fresh temporary database for each test."""

        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        db_path = Path(self.temp_dir.name) / "rag_metadata.sqlite"
        self.db = MetadataDB(db_path)
        self.db.init_schema()

    def test_schema_initialization_creates_required_tables(self) -> None:
        """Schema initialization should create all metadata tables."""

        with closing(self.db.connect()) as connection:
            rows = connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table';
                """
            ).fetchall()

        table_names = {row["name"] for row in rows}
        self.assertIn("entities", table_names)
        self.assertIn("documents", table_names)
        self.assertIn("chunks", table_names)
        self.assertIn("ingestion_runs", table_names)

    def test_upsert_entity_inserts_and_returns_id(self) -> None:
        """upsert_entity should insert an entity and return its id."""

        entity_id = self.db.upsert_entity(
            "Albert Einstein",
            "person",
            "https://en.wikipedia.org/wiki/Albert_Einstein",
        )

        self.assertIsInstance(entity_id, int)
        entity = self.db.get_entity("Albert Einstein", "person")
        self.assertIsNotNone(entity)
        self.assertEqual(entity["id"], entity_id)
        self.assertEqual(entity["source"], "wikipedia")

    def test_upsert_entity_updates_existing_without_duplicate(self) -> None:
        """upsert_entity should update an existing entity row."""

        first_id = self.db.upsert_entity(
            "Albert Einstein",
            "person",
            "https://old.example/albert",
        )
        second_id = self.db.upsert_entity(
            "Albert Einstein",
            "person",
            "https://new.example/albert",
        )

        self.assertEqual(first_id, second_id)
        entities = self.db.list_entities()
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]["source_url"], "https://new.example/albert")

    def test_get_entity_is_case_insensitive(self) -> None:
        """Entity lookup should ignore name case."""

        self.db.upsert_entity("Marie Curie", "person")

        entity = self.db.get_entity("marie curie")

        self.assertIsNotNone(entity)
        self.assertEqual(entity["name"], "Marie Curie")

    def test_list_entities_filters_by_type(self) -> None:
        """list_entities should filter people and places."""

        self.db.upsert_entity("Ada Lovelace", "person")
        self.db.upsert_entity("Eiffel Tower", "place")

        people = self.db.list_entities("person")
        places = self.db.list_entities("place")

        self.assertEqual([person["name"] for person in people], ["Ada Lovelace"])
        self.assertEqual([place["name"] for place in places], ["Eiffel Tower"])

    def test_create_document_and_get_document_work(self) -> None:
        """Document creation and lookup should store document metadata."""

        entity_id = self.db.upsert_entity("Nikola Tesla", "person")
        document_id = self.db.create_document(
            entity_id=entity_id,
            title="Nikola Tesla",
            source_url="https://en.wikipedia.org/wiki/Nikola_Tesla",
            raw_path="data/raw/nikola_tesla.txt",
            processed_path="data/processed/nikola_tesla.txt",
            status="success",
        )

        document = self.db.get_document(document_id)

        self.assertIsNotNone(document)
        self.assertEqual(document["entity_id"], entity_id)
        self.assertEqual(document["title"], "Nikola Tesla")
        self.assertEqual(document["status"], "success")

    def test_update_document_status_works(self) -> None:
        """Document status updates should persist."""

        entity_id = self.db.upsert_entity("Taylor Swift", "person")
        document_id = self.db.create_document(
            entity_id,
            "Taylor Swift",
            None,
            None,
            None,
            "pending",
        )

        self.db.update_document_status(document_id, "failed", "Network timeout")
        document = self.db.get_document(document_id)

        self.assertEqual(document["status"], "failed")
        self.assertEqual(document["error_message"], "Network timeout")

    def test_add_chunk_computes_char_count_and_token_estimate(self) -> None:
        """Chunk insertion should compute text length metadata."""

        entity_id, document_id = self._create_entity_and_document()
        text = "abcd" * 5

        chunk_id = self.db.add_chunk(document_id, entity_id, 0, text, "vec-1")
        chunks = self.db.list_chunks()

        self.assertIsInstance(chunk_id, int)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["char_count"], 20)
        self.assertEqual(chunks[0]["token_estimate"], 5)
        self.assertEqual(chunks[0]["vector_id"], "vec-1")

    def test_list_chunks_filters_by_document_id_and_entity_id(self) -> None:
        """Chunk listing should support document and entity filters."""

        first_entity_id, first_document_id = self._create_entity_and_document()
        second_entity_id = self.db.upsert_entity("Grand Canyon", "place")
        second_document_id = self.db.create_document(
            second_entity_id,
            "Grand Canyon",
            None,
            None,
            None,
            "success",
        )

        self.db.add_chunk(first_document_id, first_entity_id, 0, "first")
        self.db.add_chunk(first_document_id, first_entity_id, 1, "second")
        self.db.add_chunk(second_document_id, second_entity_id, 0, "third")

        chunks_for_first_document = self.db.list_chunks(document_id=first_document_id)
        chunks_for_second_entity = self.db.list_chunks(entity_id=second_entity_id)

        self.assertEqual(len(chunks_for_first_document), 2)
        self.assertEqual(len(chunks_for_second_entity), 1)
        self.assertEqual(chunks_for_second_entity[0]["text"], "third")

    def test_delete_chunks_for_document_deletes_only_that_document(self) -> None:
        """delete_chunks(document_id) should leave other document chunks intact."""

        first_entity_id, first_document_id = self._create_entity_and_document()
        second_entity_id = self.db.upsert_entity("Grand Canyon", "place")
        second_document_id = self.db.create_document(
            second_entity_id,
            "Grand Canyon",
            None,
            None,
            None,
            "success",
        )
        self.db.add_chunk(first_document_id, first_entity_id, 0, "first")
        self.db.add_chunk(first_document_id, first_entity_id, 1, "second")
        self.db.add_chunk(second_document_id, second_entity_id, 0, "third")

        deleted_count = self.db.delete_chunks(first_document_id)

        self.assertEqual(deleted_count, 2)
        self.assertEqual(self.db.list_chunks(document_id=first_document_id), [])
        remaining_chunks = self.db.list_chunks(document_id=second_document_id)
        self.assertEqual(len(remaining_chunks), 1)
        self.assertEqual(remaining_chunks[0]["text"], "third")

    def test_delete_chunks_without_document_deletes_all_chunks(self) -> None:
        """delete_chunks() should clear all chunk rows."""

        first_entity_id, first_document_id = self._create_entity_and_document()
        second_entity_id = self.db.upsert_entity("Grand Canyon", "place")
        second_document_id = self.db.create_document(
            second_entity_id,
            "Grand Canyon",
            None,
            None,
            None,
            "success",
        )
        self.db.add_chunk(first_document_id, first_entity_id, 0, "first")
        self.db.add_chunk(second_document_id, second_entity_id, 0, "second")

        deleted_count = self.db.delete_chunks()

        self.assertEqual(deleted_count, 2)
        self.assertEqual(self.db.list_chunks(), [])

    def test_ingestion_run_start_and_finish_work(self) -> None:
        """Ingestion run lifecycle metadata should persist."""

        run_id = self.db.start_ingestion_run(100, "Initial run")
        self.db.finish_ingestion_run(
            run_id,
            "success",
            successful_entities=98,
            failed_entities=2,
            notes="Finished with two failures",
        )

        run = self.db.get_ingestion_run(run_id)

        self.assertIsNotNone(run)
        self.assertEqual(run["status"], "success")
        self.assertEqual(run["total_entities"], 100)
        self.assertEqual(run["successful_entities"], 98)
        self.assertEqual(run["failed_entities"], 2)
        self.assertIsNotNone(run["finished_at"])
        self.assertEqual(run["notes"], "Finished with two failures")

    def test_get_summary_counts_returns_correct_counts(self) -> None:
        """Summary counts should reflect inserted metadata."""

        person_id = self.db.upsert_entity("Albert Einstein", "person")
        place_id = self.db.upsert_entity("Eiffel Tower", "place")
        success_document_id = self.db.create_document(
            person_id,
            "Albert Einstein",
            None,
            None,
            None,
            "success",
        )
        failed_document_id = self.db.create_document(
            place_id,
            "Eiffel Tower",
            None,
            None,
            None,
            "failed",
        )
        self.db.create_document(place_id, "Eiffel Tower Retry", None, None, None, "pending")
        self.db.add_chunk(success_document_id, person_id, 0, "alpha")
        self.db.add_chunk(failed_document_id, place_id, 0, "beta")

        counts = self.db.get_summary_counts()

        self.assertEqual(counts["entities"], 2)
        self.assertEqual(counts["people"], 1)
        self.assertEqual(counts["places"], 1)
        self.assertEqual(counts["documents"], 3)
        self.assertEqual(counts["chunks"], 2)
        self.assertEqual(counts["successful_documents"], 1)
        self.assertEqual(counts["failed_documents"], 1)

    def test_reset_drops_and_recreates_tables(self) -> None:
        """reset should clear data and leave the schema usable."""

        entity_id, document_id = self._create_entity_and_document()
        self.db.add_chunk(document_id, entity_id, 0, "chunk")

        self.db.reset()
        counts = self.db.get_summary_counts()
        new_entity_id = self.db.upsert_entity("Marie Curie", "person")

        self.assertEqual(counts["entities"], 0)
        self.assertEqual(counts["documents"], 0)
        self.assertEqual(counts["chunks"], 0)
        self.assertIsInstance(new_entity_id, int)

    def test_invalid_entity_type_raises_value_error(self) -> None:
        """Unsupported entity types should be rejected."""

        with self.assertRaises(ValueError):
            self.db.upsert_entity("Unknown", "landmark")

    def test_blank_status_raises_value_error(self) -> None:
        """Blank document statuses should be rejected."""

        entity_id = self.db.upsert_entity("Frida Kahlo", "person")

        with self.assertRaises(ValueError):
            self.db.create_document(entity_id, "Frida Kahlo", None, None, None, "  ")

    def _create_entity_and_document(self) -> tuple[int, int]:
        """Create a standard entity and document pair for tests."""

        entity_id = self.db.upsert_entity("Machu Picchu", "place")
        document_id = self.db.create_document(
            entity_id,
            "Machu Picchu",
            "https://en.wikipedia.org/wiki/Machu_Picchu",
            None,
            None,
            "success",
        )
        return entity_id, document_id
