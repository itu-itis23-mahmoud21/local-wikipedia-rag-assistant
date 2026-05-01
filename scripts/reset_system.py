"""Safely remove generated local RAG artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATED_TARGET_NAMES = {"data", "chroma_db"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse reset command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Remove generated local data and Chroma artifacts."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation.",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Remove only the data directory.",
    )
    parser.add_argument(
        "--chroma-only",
        action="store_true",
        help="Remove only the chroma_db directory.",
    )
    return parser.parse_args(argv)


def get_reset_targets(
    data_only: bool,
    chroma_only: bool,
    project_root: Path = PROJECT_ROOT,
) -> list[Path]:
    """Return generated artifact targets selected for reset."""

    if data_only and not chroma_only:
        return [project_root / "data"]
    if chroma_only and not data_only:
        return [project_root / "chroma_db"]
    return [project_root / "data", project_root / "chroma_db"]


def confirm_reset(targets: list[Path], assume_yes: bool) -> bool:
    """Ask for confirmation unless --yes was provided."""

    print("The following generated directories will be removed:")
    for target in targets:
        print(f"- {_display_path(target)}")

    if assume_yes:
        return True

    answer = input("Proceed? Type 'yes' to continue: ").strip().casefold()
    return answer in {"y", "yes"}


def reset_targets(targets: list[Path]) -> None:
    """Remove selected generated artifact directories."""

    for target in targets:
        _validate_generated_target(target)
        if not target.exists():
            print(f"Already absent: {_display_path(target)}")
            continue
        shutil.rmtree(target)
        print(f"Removed: {_display_path(target)}")


def main(argv: list[str] | None = None) -> int:
    """Run the reset command-line interface."""

    args = parse_args(argv)
    targets = get_reset_targets(args.data_only, args.chroma_only)

    if not confirm_reset(targets, assume_yes=args.yes):
        print("Reset cancelled.")
        return 0

    reset_targets(targets)
    print("Reset completed.")
    return 0


def _validate_generated_target(target: Path) -> None:
    """Ensure only known generated artifact directories are removed."""

    if target.name not in GENERATED_TARGET_NAMES:
        raise ValueError(f"Refusing to remove unexpected target: {target}")


def _display_path(target: Path) -> str:
    """Return a console-safe project-relative path when possible."""

    try:
        return str(target.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(target)


if __name__ == "__main__":
    sys.exit(main())
