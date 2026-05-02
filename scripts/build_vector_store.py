"""Build the Chroma vector store from SQLite chunks."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import shutil
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
        "--batch-size",
        type=int,
        default=50,
        help="Number of vectors to upsert to Chroma per batch.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=config.DEFAULT_CHROMA_SHARD_COUNT,
        help="Number of Chroma shard collections to use.",
    )
    parser.add_argument(
        "--reset-chroma-dir",
        action="store_true",
        help="Delete the whole Chroma persistent directory before building.",
    )
    parser.add_argument(
        "--post-build-health-check",
        dest="post_build_health_check",
        action="store_true",
        default=True,
        help="Reopen Chroma in a fresh Python process and verify querying works.",
    )
    parser.add_argument(
        "--skip-post-build-health-check",
        dest="post_build_health_check",
        action="store_false",
        help="Skip the fresh-process Chroma health check.",
    )
    parser.add_argument(
        "--health-check-query",
        default="Who is Albert Einstein?",
        help="Query used by the post-build Chroma health check.",
    )
    parser.add_argument(
        "--health-check-entity",
        default="",
        help="Optional exact entity filter used by the post-build health check.",
    )
    parser.add_argument(
        "--health-check-entity-type",
        choices=["all", "person", "place"],
        default="all",
        help="Optional entity type filter used by the post-build health check.",
    )
    parser.add_argument(
        "--post-build-settle-seconds",
        type=float,
        default=10.0,
        help="Seconds to wait after upserts before fresh-process health checking.",
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
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.shard_count <= 0:
        parser.error("--shard-count must be positive.")
    if args.post_build_settle_seconds < 0:
        parser.error("--post-build-settle-seconds must be non-negative.")
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

    if args.reset_chroma_dir:
        reset_chroma_directory(config.CHROMA_DB_DIR)

    vector_store = ChromaVectorStore(shard_count=args.shard_count)
    if args.reset_collection:
        vector_store.reset_collection()

    if args.reset_collection or args.reset_chroma_dir:
        cleared_count = db.clear_chunk_vector_ids()
        print(
            "Cleared "
            f"{cleared_count} stale vector IDs from SQLite because Chroma "
            "storage was reset."
        )

    all_chunks = db.list_chunks()
    selected_chunks = _filter_chunks_by_entity_type(db, all_chunks, args.entity_type)
    if args.limit is not None:
        selected_chunks = selected_chunks[: args.limit]

    print_vector_build_start(
        selected_count=len(selected_chunks),
        progress_every=args.progress_every,
        db_path=db.db_path,
        batch_size=args.batch_size,
        shard_count=args.shard_count,
        post_build_health_check=args.post_build_health_check,
        reset_chroma_dir=args.reset_chroma_dir,
    )

    progress_callback = make_progress_callback(
        total_chunks=len(selected_chunks),
        progress_every=args.progress_every,
    )
    chunks_added = vector_store.add_chunks(
        selected_chunks,
        db,
        progress_callback=progress_callback,
        batch_size=args.batch_size,
    )

    collection_count = vector_store.count()
    shard_counts = vector_store.get_shard_counts()
    if args.post_build_settle_seconds > 0:
        print(
            "Waiting "
            f"{args.post_build_settle_seconds:.1f}s for Chroma persistence "
            "to settle before verification..."
        )
        time.sleep(args.post_build_settle_seconds)
        collection_count = vector_store.count()
        shard_counts = vector_store.get_shard_counts()

    sqlite_vector_id_count = db.count_chunk_vector_ids()

    print("Vector store build summary")
    print(f"Chunks available: {len(all_chunks)}")
    print(f"Chunks selected: {len(selected_chunks)}")
    print(f"Chunks added: {chunks_added}")
    print(f"Collection count: {collection_count}")
    print("Per-shard counts:")
    for shard_name, shard_count in shard_counts.items():
        print(f"- {shard_name}: {shard_count}")
    print(f"SQLite vector_id count: {sqlite_vector_id_count}")
    print(f"Chroma path: {config.CHROMA_DB_DIR}")
    print(f"Collection name: {config.CHROMA_COLLECTION_NAME}")
    print(f"Database path: {db.db_path}")

    if args.post_build_health_check:
        health_ok = run_post_build_health_check(
            query=args.health_check_query,
            entity=args.health_check_entity,
            entity_type=args.health_check_entity_type,
            shard_count=args.shard_count,
        )
        print(
            "Post-build health check result: "
            f"{'passed' if health_ok else 'failed'}"
        )
        if not health_ok:
            print(
                "Vector build completed, but reopened Chroma query health check "
                "failed. The persisted HNSW index is not queryable."
            )
            return 3
    else:
        print("Post-build health check result: skipped")

    return 0


def print_vector_build_start(
    selected_count: int,
    progress_every: int,
    db_path: Path,
    batch_size: int,
    shard_count: int,
    post_build_health_check: bool,
    reset_chroma_dir: bool,
) -> None:
    """Print a clear start message before embedding begins."""

    interval = (
        f"every {progress_every} chunks"
        if progress_every > 0
        else "disabled"
    )
    print("Vector store build starting")
    print(f"chromadb version: {get_chromadb_version()}")
    print(f"Chunks selected: {selected_count}")
    print(f"Collection: {config.CHROMA_COLLECTION_NAME}")
    print(f"Chroma path: {config.CHROMA_DB_DIR}")
    print(f"Database path: {db_path}")
    print(f"Batch size: {batch_size}")
    print(f"Shard count: {shard_count}")
    print(f"Progress interval: {interval}")
    print(
        "Post-build health check: "
        f"{'enabled' if post_build_health_check else 'disabled'}"
    )
    print(f"Reset Chroma directory: {'yes' if reset_chroma_dir else 'no'}")


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


def reset_chroma_directory(chroma_dir: Path) -> bool:
    """Delete the Chroma persistent directory if it exists."""

    if not chroma_dir.exists():
        print(f"Chroma directory already absent: {chroma_dir}")
        return False
    if not chroma_dir.is_dir():
        raise RuntimeError(f"Chroma path is not a directory: {chroma_dir}")

    shutil.rmtree(chroma_dir)
    print(f"Removed Chroma directory: {chroma_dir}")
    return True


def get_chromadb_version() -> str:
    """Return the installed chromadb version if available."""

    try:
        import chromadb
    except ImportError:
        return "not installed"
    return str(getattr(chromadb, "__version__", "unknown"))


def run_post_build_health_check(
    query: str,
    entity: str | None = None,
    entity_type: str = "all",
    shard_count: int = config.DEFAULT_CHROMA_SHARD_COUNT,
    timeout_seconds: int = 120,
    command_runner: CommandRunner | None = None,
) -> bool:
    """Run a fresh-process Chroma query health check."""

    runner = command_runner or run_command_capture
    command = build_health_check_command(query, entity, entity_type, shard_count)

    print("Running post-build Chroma health check in a fresh Python process...")
    ok, output = runner(command, timeout_seconds)
    if output:
        print(output)
    return ok


def build_health_check_command(
    query: str,
    entity: str | None,
    entity_type: str,
    shard_count: int = config.DEFAULT_CHROMA_SHARD_COUNT,
) -> list[str]:
    """Build the subprocess command for the Chroma health check."""

    health_check_code = """
import argparse

from src.vector_store import ChromaVectorStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--entity", default="")
    parser.add_argument("--entity-type", default="all")
    parser.add_argument("--shard-count", type=int, default=10)
    args = parser.parse_args()

    store = ChromaVectorStore(shard_count=args.shard_count)
    print(f"Reopened Chroma count: {store.count()}")
    for shard_name, shard_count in store.get_shard_counts().items():
        print(f"{shard_name}: {shard_count}")

    sample_results = store.get_sample_results(limit=1)
    print(f"Fresh-process sample get returned {len(sample_results)} item(s).")

    entity_names = [args.entity] if args.entity.strip() else None
    entity_type_filter = None if args.entity_type == "all" else args.entity_type
    results = store.search(
        args.query,
        top_k=3,
        entity_type=entity_type_filter,
        entity_names=entity_names,
    )
    print(f"Fresh-process search returned {len(results)} result(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""".strip()

    command = [
        sys.executable,
        "-c",
        health_check_code,
        "--query",
        query,
        "--entity-type",
        entity_type,
        "--shard-count",
        str(shard_count),
    ]
    if entity is not None and entity.strip():
        command.extend(["--entity", entity.strip()])
    return command


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
            cwd=PROJECT_ROOT,
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
