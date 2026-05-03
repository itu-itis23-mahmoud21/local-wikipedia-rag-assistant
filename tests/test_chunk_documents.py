"""Tests for the document chunking script helpers."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.chunk_documents import chunk_document_rows
from src.chunker import TextChunk


class FakeChunkDB:
    """Small fake database for chunk document helper tests."""

    def __init__(self, existing_chunks: dict[int, list[dict]] | None = None) -> None:
        self.existing_chunks = existing_chunks or {}
        self.added_chunks: list[dict] = []

    def list_chunks(
        self,
        document_id: int | None = None,
        entity_id: int | None = None,
    ) -> list[dict]:
        """Return fake existing chunks for a document."""

        if document_id is None:
            return []
        return self.existing_chunks.get(document_id, [])

    def add_chunk(
        self,
        document_id: int,
        entity_id: int,
        chunk_index: int,
        text: str,
    ) -> int:
        """Record a fake chunk insert."""

        self.added_chunks.append(
            {
                "document_id": document_id,
                "entity_id": entity_id,
                "chunk_index": chunk_index,
                "text": text,
            }
        )
        return len(self.added_chunks)


class TestChunkDocumentsScript(unittest.TestCase):
    """Tests for script-level chunking behavior."""

    def test_existing_chunks_skip_before_checking_stale_processed_path(self) -> None:
        """Already-chunked documents with stale paths should be skipped, not failed."""

        db = FakeChunkDB(existing_chunks={10: [{"id": 1, "text": "existing"}]})
        chunker_calls: list[Path] = []

        def unexpected_chunk_file(
            path: Path,
            chunk_size: int,
            overlap: int,
        ) -> list[TextChunk]:
            _ = chunk_size, overlap
            chunker_calls.append(path)
            raise AssertionError(
                "chunk_file_func should not be called for skipped documents"
            )

        summary = chunk_document_rows(
            db=db,
            documents=[
                {
                    "id": 10,
                    "entity_id": 20,
                    "processed_path": "C:/old/project/data/processed/person/einstein.txt",
                }
            ],
            reset_chunks=False,
            chunk_size=900,
            overlap=150,
            chunk_file_func=unexpected_chunk_file,
        )

        self.assertEqual(summary["skipped_documents"], 1)
        self.assertEqual(summary["failed_documents"], 0)
        self.assertEqual(summary["chunks_created"], 0)
        self.assertEqual(chunker_calls, [])
        self.assertEqual(db.added_chunks, [])

    def test_document_without_chunks_uses_chunk_file_and_inserts_chunks(self) -> None:
        """Documents without chunks should read the processed path and insert chunks."""

        db = FakeChunkDB()
        with TemporaryDirectory() as temporary_dir:
            processed_path = Path(temporary_dir) / "einstein.txt"
            processed_path.write_text("Albert Einstein text.", encoding="utf-8")

            summary = chunk_document_rows(
                db=db,
                documents=[
                    {
                        "id": 10,
                        "entity_id": 20,
                        "processed_path": str(processed_path),
                    }
                ],
                reset_chunks=False,
                chunk_size=900,
                overlap=150,
                chunk_file_func=lambda path, chunk_size, overlap: [
                    TextChunk(
                        chunk_index=0,
                        text="Albert Einstein text.",
                        char_count=21,
                        token_estimate=5,
                    )
                ],
            )

        self.assertEqual(summary["chunked_documents"], 1)
        self.assertEqual(summary["failed_documents"], 0)
        self.assertEqual(summary["chunks_created"], 1)
        self.assertEqual(db.added_chunks[0]["document_id"], 10)
        self.assertEqual(db.added_chunks[0]["entity_id"], 20)
        self.assertEqual(db.added_chunks[0]["text"], "Albert Einstein text.")


if __name__ == "__main__":
    unittest.main()
