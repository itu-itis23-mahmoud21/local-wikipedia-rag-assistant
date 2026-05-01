"""Build the Chroma vector store from SQLite chunks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.database import MetadataDB
from src.vector_store import ChromaVectorStore


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Embed SQLite chunks and store them in Chroma."
    )
    parser.add_argument(
        "--reset-collection",
        action="store_true",
        help="Delete and recreate the Chroma collection before adding chunks.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of chunks to add.",
    )
    parser.add_argument(
        "--entity-type",
        choices=["all", "person", "place"],
        default="all",
        help="Filter chunks by entity type.",
    )

    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative.")
    return args


def main() -> None:
    """Build the persistent Chroma collection from SQLite chunk rows."""

    args = parse_args()

    db = MetadataDB()
    db.init_schema()

    vector_store = ChromaVectorStore()
    if args.reset_collection:
        vector_store.reset_collection()

    all_chunks = db.list_chunks()
    selected_chunks = _filter_chunks_by_entity_type(db, all_chunks, args.entity_type)
    if args.limit is not None:
        selected_chunks = selected_chunks[: args.limit]

    chunks_added = vector_store.add_chunks(selected_chunks, db)

    print("Vector store build summary")
    print(f"Chunks available: {len(all_chunks)}")
    print(f"Chunks selected: {len(selected_chunks)}")
    print(f"Chunks added: {chunks_added}")
    print(f"Collection count: {vector_store.count()}")
    print(f"Chroma path: {config.CHROMA_DB_DIR}")
    print(f"Collection name: {config.CHROMA_COLLECTION_NAME}")
    print(f"Database path: {db.db_path}")


def _filter_chunks_by_entity_type(
    db: MetadataDB,
    chunks: list[dict],
    entity_type: str,
) -> list[dict]:
    """Return chunks matching the selected entity type."""

    if entity_type == "all":
        return chunks

    selected_chunks: list[dict] = []
    for chunk in chunks:
        entity = db.get_entity_by_id(int(chunk["entity_id"]))
        if entity is not None and entity["entity_type"] == entity_type:
            selected_chunks.append(chunk)

    return selected_chunks


if __name__ == "__main__":
    main()
