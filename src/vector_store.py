"""Persistent Chroma vector store for Wikipedia chunk embeddings.

This module stores one vector item per SQLite chunk in a single Chroma
collection. Embeddings are generated through the local Ollama embedding client;
Chroma's built-in embedding functions are intentionally not used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src import config
from src.database import MetadataDB
from src.embeddings import OllamaEmbeddingClient, ensure_embedding_vector
from src.entities import ENTITY_TYPES

try:
    import chromadb
except ImportError:  # pragma: no cover - production dependency may be absent in tests.
    chromadb = None


class VectorStoreError(Exception):
    """Raised when Chroma vector-store operations fail."""


@dataclass(frozen=True)
class VectorSearchResult:
    """One vector search result from Chroma."""

    vector_id: str
    text: str
    metadata: dict
    distance: float | None = None


class ChromaVectorStore:
    """Persistent Chroma collection for local Wikipedia RAG chunks."""

    def __init__(
        self,
        persist_directory: str | Path = config.CHROMA_DB_DIR,
        collection_name: str = config.CHROMA_COLLECTION_NAME,
        embedding_client: OllamaEmbeddingClient | None = None,
    ) -> None:
        """Create or open the configured Chroma collection."""

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_client = embedding_client or OllamaEmbeddingClient()

        if chromadb is None:
            raise VectorStoreError(
                "chromadb is not installed. Install requirements.txt to use "
                "the local Chroma vector store."
            )

        try:
            self.client = chromadb.PersistentClient(path=str(self.persist_directory))
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to initialize Chroma: {exc}") from exc

    def reset_collection(self) -> None:
        """Delete and recreate the configured Chroma collection."""

        try:
            try:
                self.client.delete_collection(name=self.collection_name)
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to reset Chroma collection: {exc}") from exc

    def build_metadata(
        self,
        chunk: dict,
        document: dict,
        entity: dict,
        model: str,
    ) -> dict:
        """Build Chroma-safe metadata for one chunk."""

        return {
            "chunk_id": _require_int(chunk, "id"),
            "document_id": _require_int(chunk, "document_id"),
            "entity_id": _require_int(chunk, "entity_id"),
            "entity": str(entity.get("name") or ""),
            "entity_type": str(entity.get("entity_type") or ""),
            "source_url": str(document.get("source_url") or entity.get("source_url") or ""),
            "chunk_index": _require_int(chunk, "chunk_index"),
            "model": str(model),
        }

    def add_chunk(
        self,
        chunk: dict,
        document: dict,
        entity: dict,
        embedding: list[float] | None = None,
    ) -> str:
        """Add or update one chunk item in Chroma and return its vector id."""

        chunk_id = _require_int(chunk, "id")
        text = _require_text(chunk, "text")
        vector_id = f"chunk-{chunk_id}"

        try:
            vector = (
                ensure_embedding_vector(embedding)
                if embedding is not None
                else self.embedding_client.embed_text(text)
            )
            metadata = self.build_metadata(
                chunk,
                document,
                entity,
                model=self.embedding_client.model,
            )
            self.collection.upsert(
                ids=[vector_id],
                embeddings=[vector],
                documents=[text],
                metadatas=[metadata],
            )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to add chunk {chunk_id} to Chroma: {exc}"
            ) from exc

        return vector_id

    def add_chunks(
        self,
        chunks: list[dict],
        db: MetadataDB,
        limit: int | None = None,
    ) -> int:
        """Add SQLite chunk rows to Chroma and update their vector ids."""

        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative.")

        selected_chunks = chunks[:limit] if limit is not None else chunks
        added_count = 0

        for chunk in selected_chunks:
            chunk_id = _require_int(chunk, "id")
            document_id = _require_int(chunk, "document_id")
            entity_id = _require_int(chunk, "entity_id")

            document = db.get_document(document_id)
            if document is None:
                raise VectorStoreError(
                    f"Missing document metadata for chunk {chunk_id}."
                )

            entity = db.get_entity_by_id(entity_id)
            if entity is None:
                raise VectorStoreError(f"Missing entity metadata for chunk {chunk_id}.")

            vector_id = self.add_chunk(chunk, document, entity)
            if hasattr(db, "update_chunk_vector_id"):
                db.update_chunk_vector_id(chunk_id, vector_id)
            added_count += 1

        return added_count

    def search(
        self,
        query: str,
        top_k: int = 5,
        entity_type: str | None = None,
    ) -> list[VectorSearchResult]:
        """Search Chroma by embedded query text."""

        if not query.strip():
            raise ValueError("query must not be blank.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if entity_type is not None and entity_type not in ENTITY_TYPES:
            allowed = ", ".join(ENTITY_TYPES)
            raise ValueError(f"entity_type must be one of: {allowed}.")

        where = {"entity_type": entity_type} if entity_type is not None else None

        try:
            query_embedding = self.embedding_client.embed_text(query)
            query_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
            }
            if where is not None:
                query_kwargs["where"] = where
            result = self.collection.query(**query_kwargs)
        except Exception as exc:
            raise VectorStoreError(f"Failed to search Chroma collection: {exc}") from exc

        return _parse_search_results(result)

    def count(self) -> int:
        """Return the number of items in the Chroma collection."""

        try:
            return int(self.collection.count())
        except Exception as exc:
            raise VectorStoreError(f"Failed to count Chroma collection: {exc}") from exc


def get_collection():
    """Return the configured Chroma collection."""

    return ChromaVectorStore().collection


def _parse_search_results(result: dict) -> list[VectorSearchResult]:
    """Parse Chroma query output into VectorSearchResult objects."""

    ids = _first_result_list(result.get("ids"))
    documents = _first_result_list(result.get("documents"))
    metadatas = _first_result_list(result.get("metadatas"))
    distances = _first_result_list(result.get("distances"))

    search_results: list[VectorSearchResult] = []
    for index, vector_id in enumerate(ids):
        text = documents[index] if index < len(documents) else ""
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None
        search_results.append(
            VectorSearchResult(
                vector_id=str(vector_id),
                text=str(text or ""),
                metadata=dict(metadata or {}),
                distance=float(distance) if distance is not None else None,
            )
        )

    return search_results


def _first_result_list(value: object) -> list:
    """Return the first nested Chroma result list or an empty list."""

    if not value:
        return []
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    if isinstance(value, list):
        return value
    return []


def _require_int(row: dict, key: str) -> int:
    """Read a required integer-like value from a row dictionary."""

    if key not in row or row[key] is None:
        raise VectorStoreError(f"Missing required metadata field: {key}.")
    try:
        return int(row[key])
    except (TypeError, ValueError) as exc:
        raise VectorStoreError(f"Invalid integer metadata field: {key}.") from exc


def _require_text(row: dict, key: str) -> str:
    """Read a required non-blank text value from a row dictionary."""

    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise VectorStoreError(f"Missing required text field: {key}.")
    return value
