"""Tests for the Chroma vector-store wrapper."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from src.database import MetadataDB
from src.vector_store import ChromaVectorStore, VectorSearchResult, VectorStoreError


class FakeEmbeddingClient:
    """Deterministic embedding client for vector-store tests."""

    def __init__(self) -> None:
        self.model = "test-embedding-model"
        self.calls: list[str] = []

    def embed_text(self, text: str) -> list[float]:
        """Return a tiny deterministic embedding."""

        self.calls.append(text)
        return [float(len(text)), 1.0]


class FakeCollection:
    """Small in-memory stand-in for a Chroma collection."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.items: dict[str, dict] = {}
        self.last_where: dict | None = None

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Store items by id."""

        for index, item_id in enumerate(ids):
            self.items[item_id] = {
                "embedding": embeddings[index],
                "document": documents[index],
                "metadata": metadatas[index],
            }

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict | None = None,
    ) -> dict:
        """Return stored items in insertion order, optionally metadata-filtered."""

        self.last_where = where
        items = list(self.items.items())
        if where:
            items = [
                item
                for item in items
                if self._matches_where(item[1]["metadata"], where)
            ]
        items = items[:n_results]

        return {
            "ids": [[item_id for item_id, _ in items]],
            "documents": [[item["document"] for _, item in items]],
            "metadatas": [[item["metadata"] for _, item in items]],
            "distances": [[0.1 for _ in items]],
        }

    def _matches_where(self, metadata: dict, where: dict) -> bool:
        """Apply a small subset of Chroma where filtering for tests."""

        if "$and" in where:
            return all(self._matches_where(metadata, condition) for condition in where["$and"])

        for key, expected_value in where.items():
            actual_value = metadata.get(key)
            if isinstance(expected_value, dict) and "$in" in expected_value:
                if actual_value not in expected_value["$in"]:
                    return False
            elif actual_value != expected_value:
                return False
        return True

    def count(self) -> int:
        """Return item count."""

        return len(self.items)


class FakePersistentClient:
    """Small in-memory stand-in for chromadb.PersistentClient."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.collections: dict[str, FakeCollection] = {}

    def get_or_create_collection(self, name: str) -> FakeCollection:
        """Return an existing fake collection or create one."""

        if name not in self.collections:
            self.collections[name] = FakeCollection(name)
        return self.collections[name]

    def delete_collection(self, name: str) -> None:
        """Delete a fake collection."""

        self.collections.pop(name, None)


class TestChromaVectorStore(unittest.TestCase):
    """Tests for ChromaVectorStore using fake Chroma and fake embeddings."""

    def setUp(self) -> None:
        """Create a temporary directory and fake Chroma namespace."""

        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.fake_chromadb = SimpleNamespace(PersistentClient=FakePersistentClient)

    def test_build_metadata_includes_required_fields(self) -> None:
        """build_metadata should include Chroma metadata fields."""

        with patch("src.vector_store.chromadb", self.fake_chromadb):
            store = self._make_store()

        metadata = store.build_metadata(
            self._chunk(),
            self._document(),
            self._entity(),
            "test-model",
        )

        self.assertEqual(metadata["chunk_id"], 1)
        self.assertEqual(metadata["document_id"], 2)
        self.assertEqual(metadata["entity_id"], 3)
        self.assertEqual(metadata["entity"], "Albert Einstein")
        self.assertEqual(metadata["entity_type"], "person")
        self.assertEqual(metadata["source_url"], "https://example.test/wiki")
        self.assertEqual(metadata["chunk_index"], 0)
        self.assertEqual(metadata["model"], "test-model")

    def test_add_chunk_stores_one_item_and_count_becomes_one(self) -> None:
        """add_chunk should upsert one Chroma item."""

        with patch("src.vector_store.chromadb", self.fake_chromadb):
            store = self._make_store()
            store.add_chunk(
                self._chunk(),
                self._document(),
                self._entity(),
                embedding=[0.1, 0.2],
            )

        self.assertEqual(store.count(), 1)

    def test_add_chunk_returns_stable_vector_id(self) -> None:
        """add_chunk should return chunk-<id>."""

        with patch("src.vector_store.chromadb", self.fake_chromadb):
            store = self._make_store()
            vector_id = store.add_chunk(
                self._chunk(id=42),
                self._document(),
                self._entity(),
                embedding=[0.1, 0.2],
            )

        self.assertEqual(vector_id, "chunk-42")

    def test_search_embeds_query_and_returns_results(self) -> None:
        """search should embed the query and return result records."""

        embedding_client = FakeEmbeddingClient()
        with patch("src.vector_store.chromadb", self.fake_chromadb):
            store = self._make_store(embedding_client)
            store.add_chunk(
                self._chunk(text="stored text"),
                self._document(),
                self._entity(),
                embedding=[0.1, 0.2],
            )
            results = store.search("who was einstein?", top_k=5)

        self.assertEqual(embedding_client.calls, ["who was einstein?"])
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], VectorSearchResult)
        self.assertEqual(results[0].vector_id, "chunk-1")
        self.assertEqual(results[0].text, "stored text")
        self.assertEqual(results[0].distance, 0.1)

    def test_search_with_entity_type_applies_metadata_filter(self) -> None:
        """search should pass an entity_type filter to Chroma."""

        embedding_client = FakeEmbeddingClient()
        with patch("src.vector_store.chromadb", self.fake_chromadb):
            store = self._make_store(embedding_client)
            store.add_chunk(
                self._chunk(id=1, text="person text"),
                self._document(),
                self._entity(entity_type="person"),
                embedding=[0.1, 0.2],
            )
            store.add_chunk(
                self._chunk(id=2, text="place text"),
                self._document(),
                self._entity(name="Eiffel Tower", entity_type="place"),
                embedding=[0.3, 0.4],
            )
            results = store.search("tower", top_k=5, entity_type="place")

        self.assertEqual(store.collection.last_where, {"entity_type": "place"})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["entity_type"], "place")

    def test_search_with_single_entity_name_applies_entity_filter(self) -> None:
        """search should filter to one exact configured entity name."""

        embedding_client = FakeEmbeddingClient()
        with patch("src.vector_store.chromadb", self.fake_chromadb):
            store = self._make_store(embedding_client)
            store.add_chunk(
                self._chunk(id=1, text="Einstein text"),
                self._document(),
                self._entity(name="Albert Einstein"),
                embedding=[0.1, 0.2],
            )
            store.add_chunk(
                self._chunk(id=2, text="Tesla text"),
                self._document(),
                self._entity(name="Nikola Tesla"),
                embedding=[0.3, 0.4],
            )
            results = store.search(
                "who was einstein?",
                top_k=5,
                entity_names=["Albert Einstein"],
            )

        self.assertEqual(store.collection.last_where, {"entity": "Albert Einstein"})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["entity"], "Albert Einstein")

    def test_search_with_multiple_entity_names_applies_in_filter(self) -> None:
        """search should filter to a set of mentioned configured entities."""

        embedding_client = FakeEmbeddingClient()
        with patch("src.vector_store.chromadb", self.fake_chromadb):
            store = self._make_store(embedding_client)
            store.add_chunk(
                self._chunk(id=1, text="Einstein text"),
                self._document(),
                self._entity(name="Albert Einstein"),
                embedding=[0.1, 0.2],
            )
            store.add_chunk(
                self._chunk(id=2, text="Tesla text"),
                self._document(),
                self._entity(name="Nikola Tesla"),
                embedding=[0.3, 0.4],
            )
            store.add_chunk(
                self._chunk(id=3, text="Curie text"),
                self._document(),
                self._entity(name="Marie Curie"),
                embedding=[0.5, 0.6],
            )
            results = store.search(
                "compare einstein and tesla",
                top_k=5,
                entity_names=["Albert Einstein", "Nikola Tesla"],
            )

        self.assertEqual(
            store.collection.last_where,
            {"entity": {"$in": ["Albert Einstein", "Nikola Tesla"]}},
        )
        self.assertEqual([result.metadata["entity"] for result in results], [
            "Albert Einstein",
            "Nikola Tesla",
        ])

    def test_search_combines_entity_type_and_entity_names(self) -> None:
        """search should combine type and exact entity filters with $and."""

        embedding_client = FakeEmbeddingClient()
        with patch("src.vector_store.chromadb", self.fake_chromadb):
            store = self._make_store(embedding_client)
            store.add_chunk(
                self._chunk(id=1, text="Einstein text"),
                self._document(),
                self._entity(name="Albert Einstein", entity_type="person"),
                embedding=[0.1, 0.2],
            )
            store.add_chunk(
                self._chunk(id=2, text="Eiffel text"),
                self._document(),
                self._entity(name="Eiffel Tower", entity_type="place"),
                embedding=[0.3, 0.4],
            )
            results = store.search(
                "who was einstein?",
                top_k=5,
                entity_type="person",
                entity_names=["Albert Einstein"],
            )

        self.assertEqual(
            store.collection.last_where,
            {"$and": [{"entity_type": "person"}, {"entity": "Albert Einstein"}]},
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["entity"], "Albert Einstein")

    def test_search_ignores_blank_entity_names(self) -> None:
        """Blank entity names should not add metadata filters."""

        embedding_client = FakeEmbeddingClient()
        with patch("src.vector_store.chromadb", self.fake_chromadb):
            store = self._make_store(embedding_client)
            store.add_chunk(
                self._chunk(id=1, text="Einstein text"),
                self._document(),
                self._entity(name="Albert Einstein", entity_type="person"),
                embedding=[0.1, 0.2],
            )
            results = store.search(
                "person question",
                top_k=5,
                entity_type="person",
                entity_names=[" ", ""],
            )

        self.assertEqual(store.collection.last_where, {"entity_type": "person"})
        self.assertEqual(len(results), 1)

    def test_search_rejects_blank_query(self) -> None:
        """Blank query text should be rejected."""

        with patch("src.vector_store.chromadb", self.fake_chromadb):
            store = self._make_store()

        with self.assertRaises(ValueError):
            store.search("  ")

    def test_reset_collection_clears_collection(self) -> None:
        """reset_collection should clear stored items."""

        with patch("src.vector_store.chromadb", self.fake_chromadb):
            store = self._make_store()
            store.add_chunk(
                self._chunk(),
                self._document(),
                self._entity(),
                embedding=[0.1, 0.2],
            )
            self.assertEqual(store.count(), 1)

            store.reset_collection()

        self.assertEqual(store.count(), 0)

    def test_add_chunks_integrates_with_db_and_updates_vector_id(self) -> None:
        """add_chunks should read metadata from SQLite and store vector ids."""

        with tempfile.TemporaryDirectory() as temp_dir:
            db = MetadataDB(Path(temp_dir) / "metadata.sqlite")
            db.init_schema()
            entity_id = db.upsert_entity("Albert Einstein", "person")
            document_id = db.create_document(
                entity_id,
                "Albert Einstein",
                "https://example.test/wiki",
                None,
                None,
                "success",
            )
            chunk_id = db.add_chunk(document_id, entity_id, 0, "chunk text")
            chunks = db.list_chunks()

            with patch("src.vector_store.chromadb", self.fake_chromadb):
                store = self._make_store()
                added_count = store.add_chunks(chunks, db)

            updated_chunk = db.list_chunks()[0]

        self.assertEqual(added_count, 1)
        self.assertEqual(store.count(), 1)
        self.assertEqual(updated_chunk["vector_id"], f"chunk-{chunk_id}")

    def test_count_returns_collection_count(self) -> None:
        """count should return collection item count."""

        with patch("src.vector_store.chromadb", self.fake_chromadb):
            store = self._make_store()
            self.assertEqual(store.count(), 0)
            store.add_chunk(
                self._chunk(),
                self._document(),
                self._entity(),
                embedding=[0.1, 0.2],
            )

        self.assertEqual(store.count(), 1)

    def test_add_chunks_raises_for_missing_document_metadata(self) -> None:
        """add_chunks should fail clearly when SQLite metadata is missing."""

        with tempfile.TemporaryDirectory() as temp_dir:
            db = MetadataDB(Path(temp_dir) / "metadata.sqlite")
            db.init_schema()
            chunk = {
                "id": 99,
                "document_id": 123,
                "entity_id": 456,
                "chunk_index": 0,
                "text": "orphan chunk",
            }

            with patch("src.vector_store.chromadb", self.fake_chromadb):
                store = self._make_store()
                with self.assertRaises(VectorStoreError):
                    store.add_chunks([chunk], db)

    def _make_store(
        self,
        embedding_client: FakeEmbeddingClient | None = None,
    ) -> ChromaVectorStore:
        """Create a vector store with fake Chroma and fake embeddings."""

        return ChromaVectorStore(
            persist_directory=Path(self.temp_dir.name) / "chroma",
            collection_name="test_collection",
            embedding_client=embedding_client or FakeEmbeddingClient(),
        )

    def _chunk(
        self,
        id: int = 1,
        text: str = "Albert Einstein was a physicist.",
    ) -> dict:
        """Return a representative chunk row."""

        return {
            "id": id,
            "document_id": 2,
            "entity_id": 3,
            "chunk_index": 0,
            "text": text,
        }

    def _document(self) -> dict:
        """Return representative document metadata."""

        return {
            "id": 2,
            "entity_id": 3,
            "title": "Albert Einstein",
            "source_url": "https://example.test/wiki",
        }

    def _entity(
        self,
        name: str = "Albert Einstein",
        entity_type: str = "person",
    ) -> dict:
        """Return representative entity metadata."""

        return {
            "id": 3,
            "name": name,
            "entity_type": entity_type,
            "source_url": "https://example.test/entity",
        }
