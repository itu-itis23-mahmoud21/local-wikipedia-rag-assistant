"""Chunk successful processed documents and store chunk metadata in SQLite."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.chunker import chunk_file
from src.database import MetadataDB


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Chunk successful processed Wikipedia documents."
    )
    parser.add_argument(
        "--reset-chunks",
        action="store_true",
        help="Delete existing chunk rows before recreating chunks.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of successful documents to chunk.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=config.DEFAULT_CHUNK_SIZE_CHARS,
        help="Target chunk size in characters.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=config.DEFAULT_CHUNK_OVERLAP_CHARS,
        help="Overlap between neighboring chunks in characters.",
    )

    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative.")

    return args


def main() -> None:
    """Run document chunking for successful processed documents."""

    args = parse_args()

    db = MetadataDB()
    db.init_schema()

    deleted_chunks = 0
    if args.reset_chunks:
        deleted_chunks = db.delete_chunks()

    documents = db.list_documents(status="success")
    if args.limit is not None:
        documents = documents[: args.limit]

    chunked_documents = 0
    skipped_documents = 0
    failed_documents = 0
    chunks_created = 0

    for document in documents:
        document_id = int(document["id"])
        entity_id = int(document["entity_id"])
        processed_path_value = document["processed_path"]

        if not processed_path_value:
            failed_documents += 1
            print(f"[failed] document {document_id}: missing processed_path")
            continue

        processed_path = Path(processed_path_value)
        if not processed_path.exists():
            failed_documents += 1
            print(f"[failed] document {document_id}: file not found: {processed_path}")
            continue

        existing_chunks = db.list_chunks(document_id=document_id)
        if existing_chunks and not args.reset_chunks:
            skipped_documents += 1
            print(f"[skipped] document {document_id}: chunks already exist")
            continue

        try:
            chunks = chunk_file(
                processed_path,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
            )
            if not chunks:
                failed_documents += 1
                print(f"[failed] document {document_id}: no chunks created")
                continue

            for chunk in chunks:
                db.add_chunk(
                    document_id=document_id,
                    entity_id=entity_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                )

            chunked_documents += 1
            chunks_created += len(chunks)
            print(f"[chunked] document {document_id}: {len(chunks)} chunks")
        except Exception as exc:
            failed_documents += 1
            print(f"[failed] document {document_id}: {exc}")

    print()
    print("Document chunking summary")
    print(f"Documents considered: {len(documents)}")
    print(f"Chunked documents: {chunked_documents}")
    print(f"Skipped documents: {skipped_documents}")
    print(f"Failed documents: {failed_documents}")
    print(f"Chunks deleted: {deleted_chunks}")
    print(f"Chunks created: {chunks_created}")
    print(f"Database path: {db.db_path}")


if __name__ == "__main__":
    main()
