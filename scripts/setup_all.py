"""Run the full local setup pipeline for the RAG assistant."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse setup pipeline command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Prepare the local Wikipedia RAG system end to end."
    )
    parser.add_argument(
        "--entity-type",
        choices=["all", "person", "place"],
        default="all",
        help="Entity type to process.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick testing.",
    )
    parser.add_argument(
        "--force-ingest",
        action="store_true",
        help="Pass --force to the Wikipedia ingestion stage.",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Reset SQLite metadata before ingestion.",
    )
    parser.add_argument(
        "--reset-chunks",
        action="store_true",
        help="Delete existing chunk rows before chunking.",
    )
    parser.add_argument(
        "--reset-collection",
        action="store_true",
        help="Delete and recreate the Chroma collection before vector build.",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip Wikipedia ingestion.",
    )
    parser.add_argument(
        "--skip-chunk",
        action="store_true",
        help="Skip document chunking.",
    )
    parser.add_argument(
        "--skip-vector-store",
        action="store_true",
        help="Skip Chroma vector-store build.",
    )

    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative.")

    return args


def build_pipeline_commands(args: argparse.Namespace) -> list[list[str]]:
    """Build enabled pipeline commands in execution order."""

    return [command for _, command in build_pipeline_stages(args)]


def build_pipeline_stages(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """Build enabled pipeline stages with human-readable names."""

    stages: list[tuple[str, list[str]]] = []

    if not args.skip_ingest:
        command = [
            sys.executable,
            str(SCRIPTS_DIR / "ingest_wikipedia.py"),
            "--entity-type",
            args.entity_type,
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        if args.force_ingest:
            command.append("--force")
        if args.reset_db:
            command.append("--reset-db")
        stages.append(("Ingest Wikipedia pages", command))

    if not args.skip_chunk:
        command = [
            sys.executable,
            str(SCRIPTS_DIR / "chunk_documents.py"),
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        if args.reset_chunks:
            command.append("--reset-chunks")
        stages.append(("Chunk processed documents", command))

    if not args.skip_vector_store:
        command = [
            sys.executable,
            str(SCRIPTS_DIR / "build_vector_store.py"),
            "--entity-type",
            args.entity_type,
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        if args.reset_collection:
            command.append("--reset-collection")
        stages.append(("Build Chroma vector store", command))

    return stages


def run_command(command: list[str]) -> int:
    """Run one pipeline command and return its exit code."""

    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return int(completed.returncode)


def run_pipeline(args: argparse.Namespace) -> int:
    """Run enabled pipeline stages and stop at the first failure."""

    skipped_stages = []
    if args.skip_ingest:
        skipped_stages.append("Ingest Wikipedia pages")
    if args.skip_chunk:
        skipped_stages.append("Chunk processed documents")
    if args.skip_vector_store:
        skipped_stages.append("Build Chroma vector store")

    for stage_name in skipped_stages:
        print(f"\n=== Skipped: {stage_name} ===")

    for stage_name, command in build_pipeline_stages(args):
        print(f"\n=== {stage_name} ===")
        print("Command:", _format_command_for_display(command))
        exit_code = run_command(command)
        if exit_code != 0:
            print(f"Stage failed with exit code {exit_code}: {stage_name}")
            return exit_code

    print("\nLocal RAG setup completed successfully.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the setup command-line interface."""

    args = parse_args(argv)
    return run_pipeline(args)


def _format_command_for_display(command: list[str]) -> str:
    """Format commands without printing non-ASCII project-root paths."""

    display_parts: list[str] = []
    for part in command:
        display_parts.append(_project_relative_display(part))
    return " ".join(display_parts)


def _project_relative_display(value: str) -> str:
    """Return a project-relative display path when possible."""

    try:
        path = Path(value)
        if path.is_absolute():
            return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        pass
    return value


if __name__ == "__main__":
    sys.exit(main())
