#!/usr/bin/env python3
import argparse
import datetime as dt
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Create date-based directory and print its path."
    )
    p.add_argument(
        "--base",
        type=Path,
        default=Path("out/video"),
        help="Base directory where the date folders will be created (default: out/video)",
    )
    p.add_argument(
        "--style",
        choices=["yyyy/mm/dd", "yyyy-mm-dd"],
        default="yyyy/mm/dd",
        help="Date folder style (default: yyyy/mm/dd)",
    )
    p.add_argument(
        "--date",
        type=str,
        default=None,
        help="Override date in YYYY-MM-DD format (default: today)",
    )
    p.add_argument(
        "--scaffold",
        action="store_true",
        help="Create common subfolders inside the date dir (raw, clips, thumbs, meta)",
    )
    p.add_argument(
        "--print-env",
        action="store_true",
        help="Print an export line for shell usage (VIDEO_DIR=...) in addition to the path",
    )
    return p.parse_args()


def resolve_date(date_str: str | None) -> dt.date:
    if date_str:
        return dt.date.fromisoformat(date_str)
    return dt.date.today()


def build_path(base: Path, date_obj: dt.date, style: str) -> Path:
    if style == "yyyy/mm/dd":
        return base / f"{date_obj:%Y}" / f"{date_obj:%m}" / f"{date_obj:%d}"
    else:  # yyyy-mm-dd
        return base / f"{date_obj:%Y-%m-%d}"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    date_obj = resolve_date(args.date)
    target = build_path(args.base.expanduser(), date_obj, args.style)
    ensure_dir(target)

    if args.scaffold:
        for sub in ("raw", "clips", "thumbs", "meta"):
            ensure_dir(target / sub)

    # Print the created/ensured directory path for piping into other tools.
    print(str(target))
    if args.print-env:
        # Provide a convenient export line
        print(f"export VIDEO_DIR='{target}'")


if __name__ == "__main__":
    main()

