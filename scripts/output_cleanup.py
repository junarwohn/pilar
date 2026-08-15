from __future__ import annotations

import shutil
from datetime import date, datetime, timedelta
from pathlib import Path


def remove_old_output_dirs(
    output_root: Path,
    *,
    retention_days: int = 7,
    today: date | None = None,
) -> list[Path]:
    """Remove dated output directories that are at least retention_days old."""
    if retention_days < 1:
        raise ValueError("retention_days must be at least 1")

    root = output_root.expanduser().resolve()
    if not root.is_dir():
        return []

    cutoff = (today or date.today()) - timedelta(days=retention_days)
    removed: list[Path] = []
    for entry in root.iterdir():
        if not entry.is_dir() or entry.is_symlink():
            continue
        try:
            directory_date = datetime.strptime(entry.name, "%y-%m-%d").date()
        except ValueError:
            continue
        if directory_date <= cutoff:
            shutil.rmtree(entry)
            removed.append(entry)

    return sorted(removed)
