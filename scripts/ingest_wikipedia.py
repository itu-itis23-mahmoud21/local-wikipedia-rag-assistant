"""Ingest configured Wikipedia pages into local text files and SQLite."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.database import MetadataDB
from src.entities import get_people, get_places
from src.wiki_client import (
    WikipediaClient,
    WikipediaClientError,
    build_wikipedia_page_url,
    normalize_wikipedia_text,
    safe_filename,
    write_text_file,
)


EntitySelection = list[tuple[str, str]]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Ingest configured Wikipedia pages into local storage."
    )
    parser.add_argument(
        "--entity-type",
        choices=["all", "person", "place"],
        default="all",
        help="Select which configured entities to ingest.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of selected entities to ingest.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refetch pages even when a processed local file already exists.",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Reset SQLite metadata before ingestion.",
    )

    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative.")

    return args


def main() -> None:
    """Run the Wikipedia ingestion command."""

    args = parse_args()
    selected_entities = select_entities(args.entity_type, args.limit)

    db = MetadataDB()
    if args.reset_db:
        db.reset()
    else:
        db.init_schema()

    client = WikipediaClient()
    run_id = db.start_ingestion_run(
        total_entities=len(selected_entities),
        notes=(
            f"entity_type={args.entity_type}; "
            f"limit={args.limit}; force={args.force}"
        ),
    )

    successful = 0
    skipped = 0
    failed = 0

    for entity_name, entity_type in selected_entities:
        raw_path = _entity_text_path(config.RAW_DATA_DIR, entity_type, entity_name)
        processed_path = _entity_text_path(
            config.PROCESSED_DATA_DIR,
            entity_type,
            entity_name,
        )

        existing_entity = db.get_entity(entity_name, entity_type)
        if existing_entity is None:
            entity_id = db.upsert_entity(entity_name, entity_type)
            existing_source_url = None
        else:
            entity_id = int(existing_entity["id"])
            existing_source_url = existing_entity["source_url"]

        if processed_path.exists() and not args.force:
            source_url = existing_source_url or build_wikipedia_page_url(entity_name)
            entity_id = db.upsert_entity(
                entity_name,
                entity_type,
                source_url=source_url,
            )
            db.create_document(
                entity_id=entity_id,
                title=entity_name,
                source_url=source_url,
                raw_path=str(raw_path) if raw_path.exists() else None,
                processed_path=str(processed_path),
                status="success",
                error_message="Skipped because processed file already exists.",
            )
            skipped += 1
            print(f"[skipped] {entity_type}: {entity_name}")
            continue

        try:
            page = client.fetch_page(entity_name)
            processed_text = normalize_wikipedia_text(page.extract)

            write_text_file(raw_path, page.extract)
            write_text_file(processed_path, processed_text)

            entity_id = db.upsert_entity(
                entity_name,
                entity_type,
                source_url=page.source_url,
            )
            db.create_document(
                entity_id=entity_id,
                title=page.title,
                source_url=page.source_url,
                raw_path=str(raw_path),
                processed_path=str(processed_path),
                status="success",
            )
            successful += 1
            print(f"[success] {entity_type}: {entity_name}")
        except WikipediaClientError as exc:
            db.create_document(
                entity_id=entity_id,
                title=entity_name,
                source_url=existing_source_url,
                raw_path=str(raw_path),
                processed_path=str(processed_path),
                status="failed",
                error_message=str(exc),
            )
            failed += 1
            print(f"[failed] {entity_type}: {entity_name} - {exc}")

    completed = successful + skipped
    run_status = _final_status(completed, failed)
    db.finish_ingestion_run(
        run_id,
        status=run_status,
        successful_entities=completed,
        failed_entities=failed,
        notes=(
            f"successful={successful}; skipped={skipped}; "
            f"failed={failed}; status={run_status}"
        ),
    )

    print()
    print("Wikipedia ingestion summary")
    print(f"Total selected: {len(selected_entities)}")
    print(f"Successful: {successful}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Run status: {run_status}")
    print(f"Database path: {db.db_path}")
    print(f"Raw data path: {config.RAW_DATA_DIR}")
    print(f"Processed data path: {config.PROCESSED_DATA_DIR}")


def select_entities(entity_type: str, limit: int | None = None) -> EntitySelection:
    """Return configured entities matching the requested type."""

    entities: EntitySelection = []
    if entity_type in {"all", "person"}:
        entities.extend((name, "person") for name in get_people())
    if entity_type in {"all", "place"}:
        entities.extend((name, "place") for name in get_places())

    if limit is not None:
        entities = entities[:limit]

    return entities


def _entity_text_path(base_dir: Path, entity_type: str, entity_name: str) -> Path:
    """Build a local text path for an entity."""

    return base_dir / entity_type / f"{safe_filename(entity_name)}.txt"


def _final_status(completed: int, failed: int) -> str:
    """Return the ingestion run status from completed and failed counts."""

    if failed == 0:
        return "success"
    if completed > 0:
        return "partial_success"
    return "failed"


if __name__ == "__main__":
    main()
