"""Shared utility helpers for the local RAG project."""

from __future__ import annotations

from pathlib import Path

from src import config


def serialize_project_path(
    path: str | Path | None,
    project_root: str | Path = config.PROJECT_ROOT,
) -> str | None:
    """Return a portable project-relative path string when possible.

    Paths inside the project are stored relative to the project root using
    forward slashes. External paths are left absolute so legacy callers still
    receive a useful value.
    """

    if path is None:
        return None

    path_text = str(path).strip()
    if not path_text:
        return None

    root = Path(project_root).resolve()
    candidate = Path(path_text)
    absolute_candidate = candidate if candidate.is_absolute() else root / candidate

    try:
        relative_path = absolute_candidate.resolve().relative_to(root)
    except ValueError:
        if candidate.is_absolute():
            return str(candidate)
        return candidate.as_posix()

    return relative_path.as_posix()


def resolve_project_path(
    stored_path: str | Path | None,
    project_root: str | Path = config.PROJECT_ROOT,
) -> Path | None:
    """Resolve a stored project path to an absolute path.

    Relative stored paths are resolved from the project root. Existing absolute
    paths are returned directly. For legacy absolute paths from another machine,
    the helper tries to recover the project-local ``data/...`` suffix.
    """

    if stored_path is None:
        return None

    path_text = str(stored_path).strip()
    if not path_text:
        return None

    root = Path(project_root).resolve()
    candidate = Path(path_text)

    if not candidate.is_absolute():
        return root / candidate

    if candidate.exists():
        return candidate

    legacy_data_path = _recover_project_data_path(candidate, root)
    if legacy_data_path is not None:
        return legacy_data_path

    return candidate


def _recover_project_data_path(path: Path, project_root: Path) -> Path | None:
    """Recover a current-project data path from a stale absolute path."""

    parts = path.parts
    for index, part in enumerate(parts):
        if part.casefold() == "data":
            return project_root.joinpath(*parts[index:])
    return None
