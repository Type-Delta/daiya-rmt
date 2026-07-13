"""Path identity checks for non-mutating validation artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def resolve_path(path: str | Path) -> Path:
    """Resolve an existing path and all existing ancestors, including links."""
    return Path(path).expanduser().resolve(strict=False)


def paths_alias(left: str | Path, right: str | Path) -> bool:
    """Return whether two paths identify the same resolved filesystem path."""
    return os.path.normcase(os.fspath(resolve_path(left))) == os.path.normcase(os.fspath(resolve_path(right)))


def path_is_within(path: str | Path, directory: str | Path) -> bool:
    """Return whether path is directory itself or below its resolved path."""
    candidate = resolve_path(path)
    root = resolve_path(directory)
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def reject_output_aliases(
    output: str | Path,
    *,
    files: Iterable[tuple[str, str | Path | None]] = (),
    directories: Iterable[tuple[str, str | Path | None]] = (),
) -> Path:
    """Reject an output that aliases an input or is inside an input directory."""
    resolved_output = resolve_path(output)
    for label, path in files:
        if path is not None and paths_alias(resolved_output, path):
            raise ValueError(f"output path aliases {label}: {output}")
    for label, path in directories:
        if path is not None and path_is_within(resolved_output, path):
            raise ValueError(f"output path aliases or is inside {label}: {output}")
    return resolved_output
