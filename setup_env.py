#!/usr/bin/env python3
from __future__ import annotations

import configparser
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SOURCE_CANDIDATES = [
    ROOT_DIR / "src" / "user_config.ini",
    ROOT_DIR / "res" / "user_config.ini",
]
OUTPUT_FILE = ROOT_DIR / "user_config.ini"


def find_source_file() -> Path:
    for candidate in SOURCE_CANDIDATES:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in SOURCE_CANDIDATES)
    raise FileNotFoundError(f"user_config.ini source file not found. checked: {searched}")


def load_global_section(source_file: Path) -> dict[str, str]:
    parser = configparser.ConfigParser()
    parser.read(source_file, encoding="utf-8")

    if "global" not in parser:
        raise KeyError(f"[global] section not found in {source_file}")

    return dict(parser["global"])


def write_output(global_values: dict[str, str], output_file: Path) -> None:
    parser = configparser.ConfigParser()
    parser["global"] = global_values

    with output_file.open("w", encoding="utf-8") as file:
        parser.write(file)


def prompt_global_values(global_values: dict[str, str]) -> dict[str, str]:
    prompted_values: dict[str, str] = {}

    print("Fill values for [global]. Press Enter to keep current value.")
    for key, current_value in global_values.items():
        user_input = input(f"{key} [{current_value}]: ").strip()
        prompted_values[key] = user_input if user_input else current_value

    return prompted_values


def main() -> None:
    source_file = find_source_file()
    global_values = load_global_section(source_file)
    prompted_values = prompt_global_values(global_values)
    write_output(prompted_values, OUTPUT_FILE)
    print(f"Generated {OUTPUT_FILE} from {source_file}")


if __name__ == "__main__":
    main()
