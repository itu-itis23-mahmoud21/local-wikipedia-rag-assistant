"""Chunk successful processed documents and store chunk metadata in SQLite."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.chunker import TextChunk, chunk_file
from src.database import MetadataDB
from src.utils import resolve_project_path


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

    summary = chunk_document_rows(
        db=db,
        documents=documents,
        reset_chunks=args.reset_chunks,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    print()
    print("Document chunking summary")
    print(f"Documents considered: {len(documents)}")
    print(f"Chunked documents: {summary['chunked_documents']}")
    print(f"Skipped documents: {summary['skipped_documents']}")
    print(f"Failed documents: {summary['failed_documents']}")
    print(f"Chunks deleted: {deleted_chunks}")
    print(f"Chunks created: {summary['chunks_created']}")
    print(f"Database path: {db.db_path}")


def chunk_document_rows(
    db: Any,
    documents: list[dict],
    reset_chunks: bool,
    chunk_size: int,
    overlap: int,
    chunk_file_func: Callable[..., list[TextChunk]] = chunk_file,
) -> dict[str, int]:
    """Chunk document rows and return processing counts."""

    chunked_documents = 0
    skipped_documents = 0
    failed_documents = 0
    chunks_created = 0

    for document in documents:
        document_id = int(document["id"])
        entity_id = int(document["entity_id"])

        existing_chunks = db.list_chunks(document_id=document_id)
        if existing_chunks and not reset_chunks:
            skipped_documents += 1
            print(f"[skipped] document {document_id}: chunks already exist")
            continue

        processed_path_value = document["processed_path"]
        if not processed_path_value:
            failed_documents += 1
            print(f"[failed] document {document_id}: missing processed_path")
            continue

        processed_path = resolve_project_path(processed_path_value)
        if processed_path is None:
            failed_documents += 1
            print(f"[failed] document {document_id}: missing processed_path")
            continue

        if not processed_path.exists():
            failed_documents += 1
            print(f"[failed] document {document_id}: file not found: {processed_path}")
            continue

        try:
            chunks = chunk_file_func(
                processed_path,
                chunk_size=chunk_size,
                overlap=overlap,
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

    return {
        "chunked_documents": chunked_documents,
        "skipped_documents": skipped_documents,
        "failed_documents": failed_documents,
        "chunks_created": chunks_created,
    }


if __name__ == "__main__":
    main()
