"""Persistent Chroma vector store for Wikipedia chunk embeddings.

This module stores one vector item per SQLite chunk across deterministic Chroma
shards. Embeddings are generated through the local Ollama embedding client;
Chroma's built-in embedding functions are intentionally not used.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

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
        embedding_client: Any | None = None,
        shard_count: int = config.DEFAULT_CHROMA_SHARD_COUNT,
    ) -> None:
        """Create or open the configured sharded Chroma collections."""

        if shard_count <= 0:
            raise ValueError("shard_count must be positive.")
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.shard_count = shard_count
        self.embedding_client = embedding_client or OllamaEmbeddingClient()

        if chromadb is None:
            raise VectorStoreError(
                "chromadb is not installed. Install requirements.txt to use "
                "the local Chroma vector store."
            )

        try:
            self.client = chromadb.PersistentClient(path=str(self.persist_directory))
            self.collections = {
                shard_index: self.client.get_or_create_collection(
                    name=_get_shard_collection_name(
                        self.collection_name,
                        shard_index,
                    )
                )
                for shard_index in range(self.shard_count)
            }
            self.collection = self.collections[0]
        except Exception as exc:
            raise VectorStoreError(f"Failed to initialize Chroma: {exc}") from exc

    def reset_collection(self) -> None:
        """Delete and recreate all configured Chroma shard collections."""

        try:
            collection_names = {self.collection_name}
            collection_names.update(
                _get_shard_collection_name(self.collection_name, shard_index)
                for shard_index in range(self.shard_count)
            )
            collection_names.update(
                name
                for name in _list_existing_collection_names(self.client)
                if _is_shard_collection_name(name, self.collection_name)
            )

            for collection_name in collection_names:
                try:
                    self.client.delete_collection(name=collection_name)
                except Exception:
                    pass

            self.collections = {
                shard_index: self.client.get_or_create_collection(
                    name=_get_shard_collection_name(
                        self.collection_name,
                        shard_index,
                    )
                )
                for shard_index in range(self.shard_count)
            }
            self.collection = self.collections[0]
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to reset Chroma shard collections: {exc}"
            ) from exc

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
            collection = self._get_entity_shard_collection(
                int(metadata["entity_id"])
            )
            collection.upsert(
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
        progress_callback: Callable[[int, int, dict], None] | None = None,
        batch_size: int = 50,
    ) -> int:
        """Add SQLite chunk rows to Chroma and update their vector ids."""

        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        selected_chunks = chunks[:limit] if limit is not None else chunks
        added_count = 0
        total_count = len(selected_chunks)

        for batch in _batched(selected_chunks, batch_size):
            prepared_batch = self._prepare_chunk_batch(batch, db)
            self._upsert_prepared_batch(prepared_batch)

            for prepared in prepared_batch:
                if hasattr(db, "update_chunk_vector_id"):
                    db.update_chunk_vector_id(
                        prepared["chunk_id"],
                        prepared["vector_id"],
                    )
                added_count += 1
                if progress_callback is not None:
                    progress_callback(added_count, total_count, prepared["chunk"])

        return added_count

    def _prepare_chunk_batch(
        self,
        chunks: list[dict],
        db: MetadataDB,
    ) -> list[dict]:
        """Prepare chunk text, metadata, ids, and embeddings for one upsert batch."""

        prepared_batch: list[dict] = []

        for chunk in chunks:
            chunk_id = _require_int(chunk, "id")
            document_id = _require_int(chunk, "document_id")
            entity_id = _require_int(chunk, "entity_id")
            text = _require_text(chunk, "text")

            document = db.get_document(document_id)
            if document is None:
                raise VectorStoreError(
                    f"Missing document metadata for chunk {chunk_id}."
                )

            entity = db.get_entity_by_id(entity_id)
            if entity is None:
                raise VectorStoreError(f"Missing entity metadata for chunk {chunk_id}.")

            try:
                embedding = self.embedding_client.embed_text(text)
            except Exception as exc:
                raise VectorStoreError(
                    f"Failed to embed chunk {chunk_id} for Chroma: {exc}"
                ) from exc

            vector_id = f"chunk-{chunk_id}"
            metadata = self.build_metadata(
                chunk,
                document,
                entity,
                model=self.embedding_client.model,
            )
            prepared_batch.append(
                {
                    "chunk": chunk,
                    "chunk_id": chunk_id,
                    "entity_id": entity_id,
                    "shard_index": _get_shard_index(entity_id, self.shard_count),
                    "vector_id": vector_id,
                    "text": text,
                    "embedding": ensure_embedding_vector(embedding),
                    "metadata": metadata,
                }
            )

        return prepared_batch

    def _upsert_prepared_batch(self, prepared_batch: list[dict]) -> None:
        """Upsert one prepared batch into Chroma."""

        if not prepared_batch:
            return

        grouped_items: dict[int, list[dict]] = {}
        for item in prepared_batch:
            grouped_items.setdefault(int(item["shard_index"]), []).append(item)

        for shard_index, shard_items in grouped_items.items():
            collection = self.collections[shard_index]
            collection_name = _get_shard_collection_name(
                self.collection_name,
                shard_index,
            )
            try:
                collection.upsert(
                    ids=[item["vector_id"] for item in shard_items],
                    embeddings=[item["embedding"] for item in shard_items],
                    documents=[item["text"] for item in shard_items],
                    metadatas=[item["metadata"] for item in shard_items],
                )
            except Exception as exc:
                first_chunk_id = shard_items[0]["chunk_id"]
                raise VectorStoreError(
                    "Failed to add Chroma batch starting at chunk "
                    f"{first_chunk_id} to {collection_name}: {exc}"
                ) from exc

    def _get_entity_shard_collection(self, entity_id: int):
        """Return the shard collection for an entity id."""

        shard_index = _get_shard_index(entity_id, self.shard_count)
        return self.collections[shard_index]

    def search(
        self,
        query: str,
        top_k: int = 5,
        entity_type: str | None = None,
        entity_names: list[str] | None = None,
    ) -> list[VectorSearchResult]:
        """Search Chroma by embedded query text."""

        if not query.strip():
            raise ValueError("query must not be blank.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if entity_type is not None and entity_type not in ENTITY_TYPES:
            allowed = ", ".join(ENTITY_TYPES)
            raise ValueError(f"entity_type must be one of: {allowed}.")

        where = _build_where_filter(entity_type, entity_names)

        try:
            query_embedding = self.embedding_client.embed_text(query)
            search_results: list[VectorSearchResult] = []
            for shard_index, collection in self.collections.items():
                collection_name = _get_shard_collection_name(
                    self.collection_name,
                    shard_index,
                )
                query_kwargs = {
                    "query_embeddings": [query_embedding],
                    "n_results": top_k,
                }
                if where is not None:
                    query_kwargs["where"] = where
                try:
                    if int(collection.count()) == 0:
                        continue
                    result = collection.query(**query_kwargs)
                except Exception as exc:
                    raise VectorStoreError(
                        f"Failed to search Chroma shard {collection_name}: {exc}"
                    ) from exc
                search_results.extend(_parse_search_results(result))
        except Exception as exc:
            if isinstance(exc, VectorStoreError):
                raise
            raise VectorStoreError(f"Failed to search Chroma collection: {exc}") from exc

        search_results.sort(key=_distance_sort_key)
        return search_results[:top_k]

    def get_intro_chunks(
        self,
        entity_names: list[str],
        per_entity: int = 1,
        entity_type: str | None = None,
    ) -> list[VectorSearchResult]:
        """Return the first stored chunk(s) for exact entity mentions."""

        if per_entity <= 0:
            raise ValueError("per_entity must be positive.")
        if entity_type is not None and entity_type not in ENTITY_TYPES:
            allowed = ", ".join(ENTITY_TYPES)
            raise ValueError(f"entity_type must be one of: {allowed}.")

        clean_entity_names = _normalize_entity_names(entity_names)
        if not clean_entity_names:
            return []

        where = _build_where_filter(entity_type, clean_entity_names)

        results: list[VectorSearchResult] = []
        for shard_index, collection in self.collections.items():
            collection_name = _get_shard_collection_name(
                self.collection_name,
                shard_index,
            )
            try:
                result = collection.get(
                    where=where,
                    include=["documents", "metadatas"],
                )
            except Exception as exc:
                raise VectorStoreError(
                    f"Failed to get intro chunks from {collection_name}: {exc}"
                ) from exc
            results.extend(_parse_get_results(result))

        results.sort(key=_intro_sort_key)

        selected_results: list[VectorSearchResult] = []
        selected_counts: dict[str, int] = {}
        allowed_names = {name.casefold() for name in clean_entity_names}

        for result in results:
            entity_name = str(result.metadata.get("entity") or "")
            if entity_name.casefold() not in allowed_names:
                continue

            entity_key = entity_name.casefold()
            current_count = selected_counts.get(entity_key, 0)
            if current_count >= per_entity:
                continue

            selected_results.append(result)
            selected_counts[entity_key] = current_count + 1

        return selected_results

    def count(self) -> int:
        """Return the number of items across all Chroma shard collections."""

        return sum(self.get_shard_counts().values())

    def get_shard_counts(self) -> dict[str, int]:
        """Return item counts keyed by shard collection name."""

        counts: dict[str, int] = {}
        for shard_index, collection in self.collections.items():
            collection_name = _get_shard_collection_name(
                self.collection_name,
                shard_index,
            )
            try:
                counts[collection_name] = int(collection.count())
            except Exception as exc:
                raise VectorStoreError(
                    f"Failed to count Chroma shard {collection_name}: {exc}"
                ) from exc
        return counts

    def get_sample_results(self, limit: int = 1) -> list[VectorSearchResult]:
        """Return up to limit stored items from the shard collections."""

        if limit <= 0:
            raise ValueError("limit must be positive.")

        sample_results: list[VectorSearchResult] = []
        for shard_index, collection in self.collections.items():
            remaining = limit - len(sample_results)
            if remaining <= 0:
                break
            collection_name = _get_shard_collection_name(
                self.collection_name,
                shard_index,
            )
            try:
                result = collection.get(
                    limit=remaining,
                    include=["documents", "metadatas"],
                )
            except Exception as exc:
                raise VectorStoreError(
                    f"Failed to sample Chroma shard {collection_name}: {exc}"
                ) from exc
            sample_results.extend(_parse_get_results(result))

        return sample_results[:limit]

    def _count_one_collection(self, collection_name: str, collection) -> int:
        """Return one collection count with a shard-aware error message."""

        try:
            return int(collection.count())
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to count Chroma shard {collection_name}: {exc}"
            ) from exc


def get_collection():
    """Return the configured Chroma collection."""

    return ChromaVectorStore().collection


def _get_shard_index(entity_id: int, shard_count: int) -> int:
    """Return a deterministic shard index for an entity id."""

    if shard_count <= 0:
        raise ValueError("shard_count must be positive.")
    digest = hashlib.sha256(str(entity_id).encode("utf-8")).hexdigest()
    return int(digest, 16) % shard_count


def _get_shard_collection_name(base_name: str, shard_index: int) -> str:
    """Return the Chroma collection name for a shard index."""

    return f"{base_name}_shard_{shard_index:02d}"


def _is_shard_collection_name(collection_name: str, base_name: str) -> bool:
    """Return True if collection_name looks like one of this store's shards."""

    prefix = f"{base_name}_shard_"
    suffix = collection_name.removeprefix(prefix)
    return collection_name.startswith(prefix) and suffix.isdigit()


def _list_existing_collection_names(client: object) -> list[str]:
    """Return existing Chroma collection names when the client supports listing."""

    list_collections = getattr(client, "list_collections", None)
    if list_collections is None:
        return []

    try:
        collections = list_collections()
    except Exception:
        return []

    names: list[str] = []
    for collection in collections:
        if isinstance(collection, str):
            names.append(collection)
            continue
        name = getattr(collection, "name", None)
        if name is not None:
            names.append(str(name))
    return names


def _build_where_filter(
    entity_type: str | None,
    entity_names: list[str] | None,
) -> dict | None:
    """Build a Chroma metadata filter from route/entity constraints."""

    conditions: list[dict] = []
    clean_entity_names = _normalize_entity_names(entity_names)

    if entity_type is not None:
        conditions.append({"entity_type": entity_type})

    if len(clean_entity_names) == 1:
        conditions.append({"entity": clean_entity_names[0]})
    elif len(clean_entity_names) > 1:
        conditions.append({"entity": {"$in": clean_entity_names}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _normalize_entity_names(entity_names: list[str] | None) -> list[str]:
    """Clean blank names and remove duplicates while preserving order."""

    if not entity_names:
        return []

    names: list[str] = []
    seen: set[str] = set()
    for name in entity_names:
        clean_name = str(name).strip()
        if not clean_name:
            continue
        name_key = clean_name.casefold()
        if name_key in seen:
            continue
        names.append(clean_name)
        seen.add(name_key)
    return names


def _batched(items: list[dict], batch_size: int) -> list[list[dict]]:
    """Split items into deterministic non-empty batches."""

    return [
        items[index : index + batch_size]
        for index in range(0, len(items), batch_size)
    ]


def _parse_search_results(result: Any) -> list[VectorSearchResult]:
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


def _parse_get_results(result: Any) -> list[VectorSearchResult]:
    """Parse Chroma get output into VectorSearchResult objects."""

    ids = _first_result_list(result.get("ids"))
    documents = _first_result_list(result.get("documents"))
    metadatas = _first_result_list(result.get("metadatas"))

    get_results: list[VectorSearchResult] = []
    for index, vector_id in enumerate(ids):
        text = documents[index] if index < len(documents) else ""
        metadata = metadatas[index] if index < len(metadatas) else {}
        get_results.append(
            VectorSearchResult(
                vector_id=str(vector_id),
                text=str(text or ""),
                metadata=dict(metadata or {}),
                distance=None,
            )
        )

    return get_results


def _intro_sort_key(result: VectorSearchResult) -> tuple[str, int]:
    """Sort intro candidates by entity name and chunk index."""

    entity_name = str(result.metadata.get("entity") or "").casefold()
    try:
        chunk_index = int(result.metadata.get("chunk_index", 0))
    except (TypeError, ValueError):
        chunk_index = 0
    return (entity_name, chunk_index)


def _distance_sort_key(result: VectorSearchResult) -> tuple[bool, float]:
    """Sort semantic results by distance, with missing distances last."""

    if result.distance is None:
        return (True, float("inf"))
    return (False, result.distance)


def _first_result_list(value: Any) -> list[Any]:
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
