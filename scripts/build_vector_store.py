"""Build the Chroma vector store from SQLite chunks."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.database import MetadataDB
from src.vector_store import ChromaVectorStore


CommandRunner = Callable[[list[str], int], tuple[bool, str]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print vector build progress every N chunks. Use <= 0 to disable.",
    )
    parser.add_argument(
        "--gpu-check",
        dest="gpu_check",
        action="store_true",
        default=True,
        help="Print NVIDIA/Ollama preflight status before building vectors.",
    )
    parser.add_argument(
        "--skip-gpu-check",
        dest="gpu_check",
        action="store_false",
        help="Skip NVIDIA/Ollama preflight status output.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail early if NVIDIA GPU or Ollama visibility cannot be detected.",
    )

    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative.")
    if args.require_gpu and not args.gpu_check:
        parser.error("--require-gpu cannot be used with --skip-gpu-check.")
    return args


def main(argv: list[str] | None = None) -> int:
    """Build the persistent Chroma collection from SQLite chunk rows."""

    args = parse_args(argv)

    if args.gpu_check:
        gpu_ready = run_gpu_preflight(require_gpu=args.require_gpu)
        if args.require_gpu and not gpu_ready:
            return 2

    db = MetadataDB()
    db.init_schema()

    vector_store = ChromaVectorStore()
    if args.reset_collection:
        vector_store.reset_collection()
        cleared_count = db.clear_chunk_vector_ids()
        print(
            "Cleared "
            f"{cleared_count} stale vector IDs from SQLite because the Chroma "
            "collection was reset."
        )

    all_chunks = db.list_chunks()
    selected_chunks = _filter_chunks_by_entity_type(db, all_chunks, args.entity_type)
    if args.limit is not None:
        selected_chunks = selected_chunks[: args.limit]

    print_vector_build_start(
        selected_count=len(selected_chunks),
        progress_every=args.progress_every,
        db_path=db.db_path,
    )

    progress_callback = make_progress_callback(
        total_chunks=len(selected_chunks),
        progress_every=args.progress_every,
    )
    chunks_added = vector_store.add_chunks(
        selected_chunks,
        db,
        progress_callback=progress_callback,
    )

    print("Vector store build summary")
    print(f"Chunks available: {len(all_chunks)}")
    print(f"Chunks selected: {len(selected_chunks)}")
    print(f"Chunks added: {chunks_added}")
    print(f"Collection count: {vector_store.count()}")
    print(f"Chroma path: {config.CHROMA_DB_DIR}")
    print(f"Collection name: {config.CHROMA_COLLECTION_NAME}")
    print(f"Database path: {db.db_path}")
    return 0


def print_vector_build_start(
    selected_count: int,
    progress_every: int,
    db_path: Path,
) -> None:
    """Print a clear start message before embedding begins."""

    interval = (
        f"every {progress_every} chunks"
        if progress_every > 0
        else "disabled"
    )
    print("Vector store build starting")
    print(f"Chunks selected: {selected_count}")
    print(f"Collection: {config.CHROMA_COLLECTION_NAME}")
    print(f"Chroma path: {config.CHROMA_DB_DIR}")
    print(f"Database path: {db_path}")
    print(f"Progress interval: {interval}")


def make_progress_callback(
    total_chunks: int,
    progress_every: int,
) -> Callable[[int, int, dict], None] | None:
    """Create a progress callback for ChromaVectorStore.add_chunks."""

    if progress_every <= 0:
        return None

    started_at = time.monotonic()

    def progress_callback(processed_count: int, total_count: int, chunk: dict) -> None:
        del chunk
        if processed_count % progress_every != 0 and processed_count != total_count:
            return
        elapsed_seconds = time.monotonic() - started_at
        print(
            format_progress_line(
                processed_count=processed_count,
                total_count=total_count,
                chunks_added=processed_count,
                elapsed_seconds=elapsed_seconds,
            ),
            flush=True,
        )

    return progress_callback


def format_progress_line(
    processed_count: int,
    total_count: int,
    chunks_added: int,
    elapsed_seconds: float,
) -> str:
    """Format one vector-build progress line."""

    safe_total = max(0, int(total_count))
    safe_processed = max(0, int(processed_count))
    safe_added = max(0, int(chunks_added))
    elapsed_seconds = max(0.0, float(elapsed_seconds))

    percent_complete = (
        (safe_processed / safe_total) * 100.0 if safe_total else 0.0
    )
    elapsed_minutes = elapsed_seconds / 60.0
    chunks_per_minute = safe_added / elapsed_minutes if elapsed_minutes > 0 else 0.0
    remaining_chunks = max(0, safe_total - safe_processed)
    eta_seconds = (
        (remaining_chunks / chunks_per_minute) * 60.0
        if chunks_per_minute > 0
        else None
    )

    return (
        "[vector-build] "
        f"{safe_processed}/{safe_total} chunks added ({percent_complete:.2f}%) "
        f"| elapsed {_format_duration(elapsed_seconds)} "
        f"| rate {chunks_per_minute:.1f} chunks/min "
        f"| ETA {_format_duration(eta_seconds) if eta_seconds is not None else 'unknown'}"
    )


def run_gpu_preflight(
    require_gpu: bool = False,
    command_runner: CommandRunner | None = None,
) -> bool:
    """Print NVIDIA/Ollama status without claiming Python controls GPU use."""

    runner = command_runner or run_command_capture

    print("GPU preflight")
    print("GPU usage is controlled by the Ollama server/runtime, not Python.")

    nvidia_ok, nvidia_output = runner(["nvidia-smi"], 8)
    if nvidia_ok:
        print("NVIDIA GPU detected.")
        summary = _first_non_empty_line(nvidia_output)
        if summary:
            print(summary)
        if "ollama" in nvidia_output.casefold():
            print("Ollama appears in nvidia-smi output.")
        else:
            print(
                "Ollama is not visible in nvidia-smi yet. It may appear only "
                "while a model is loaded or generating embeddings."
            )
    else:
        print("NVIDIA GPU was not detected by nvidia-smi.")
        if nvidia_output:
            print(nvidia_output)

    ollama_ok, ollama_output = runner(["ollama", "ps"], 8)
    if ollama_ok:
        print("Ollama is running. Current Ollama models/processes:")
        print(ollama_output.strip() or "(ollama ps returned no active models)")
    else:
        print("Ollama status could not be verified with 'ollama ps'.")
        if ollama_output:
            print(ollama_output)

    if not nvidia_ok or not ollama_ok:
        print(
            "GPU/Ollama warning: start Ollama after updating the NVIDIA driver, "
            "then check nvidia-smi for ollama.exe during embedding."
        )
        if require_gpu:
            print("--require-gpu was set, so the vector build will stop.")
            return False

    return nvidia_ok and ollama_ok


def run_command_capture(command: list[str], timeout_seconds: int) -> tuple[bool, str]:
    """Run a small status command and return success plus combined output."""

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
        return False, str(exc)

    output = "\n".join(
        part.strip()
        for part in (completed.stdout, completed.stderr)
        if part and part.strip()
    )
    return completed.returncode == 0, output


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


def _format_duration(seconds: float | None) -> str:
    """Format seconds as HH:MM:SS."""

    if seconds is None:
        return "unknown"
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _first_non_empty_line(text: str) -> str:
    """Return the first non-empty line from command output."""

    for line in text.splitlines():
        clean_line = line.strip()
        if clean_line:
            return clean_line
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
