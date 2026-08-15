#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import sys
from datetime import datetime
from pathlib import Path

from scripts.kakao_admin_uploader import upload_to_kakao
from scripts.output_cleanup import remove_old_output_dirs


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = ROOT_DIR / "user_config.ini"


def str_to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    v = value.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return default


def load_global_config(path: Path) -> configparser.SectionProxy:
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    if "global" not in parser:
        raise KeyError(f"[global] section not found in {path}")
    return parser["global"]


def default_image_dir() -> Path:
    day = datetime.now().strftime("%y-%m-%d")
    return ROOT_DIR / "out" / day


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Standalone Kakao uploader (visible Chrome by default)."
    )
    p.add_argument(
        "--config",
        default=str(CONFIG_FILE),
        help=f"Path to user config ini (default: {CONFIG_FILE})",
    )
    p.add_argument(
        "--image-dir",
        default=None,
        help="Directory containing YYMMDD_XX.jpg files (default: out/YY-MM-DD)",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Optional post title (default: today YYYY-MM-DD-요일)",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="Run Chrome in headless mode (default: visible Chrome).",
    )
    p.add_argument(
        "--manual-login",
        action="store_true",
        help="Wait for manual Kakao login (default: True unless config overrides).",
    )
    p.add_argument("--user-data-dir", default=None, help="Chrome user data dir")
    p.add_argument("--profile-directory", default=None, help="Chrome profile name")
    p.add_argument("--driver-path", default=None, help="ChromeDriver path")
    p.add_argument("--email", default=None, help="Kakao email for auto-login")
    p.add_argument("--password", default=None, help="Kakao password for auto-login")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    removed = remove_old_output_dirs(ROOT_DIR / "out")
    if removed:
        print(f"Removed {len(removed)} output director{'y' if len(removed) == 1 else 'ies'} older than one week.")

    cfg_path = Path(args.config).expanduser().resolve()
    global_cfg = load_global_config(cfg_path)

    kakao_url = (global_cfg.get("kakao_url") or "").strip()
    if not kakao_url:
        print("Error: set [global] kakao_url in user_config.ini", file=sys.stderr)
        return 2

    image_dir = Path(args.image_dir).expanduser().resolve() if args.image_dir else default_image_dir()
    if not image_dir.exists():
        print(f"Error: image directory not found: {image_dir}", file=sys.stderr)
        return 2

    headless = bool(args.headless)
    if not args.headless:
        headless = str_to_bool(global_cfg.get("headless"), False)

    manual_login = bool(args.manual_login)
    if not args.manual_login:
        manual_login = str_to_bool(global_cfg.get("manual_login"), True)

    user_data_dir = args.user_data_dir or (global_cfg.get("user_data_dir") or None)
    profile_directory = args.profile_directory or (global_cfg.get("profile_directory") or None)
    driver_path = args.driver_path or (global_cfg.get("driver_path") or None)

    email = args.email if args.email is not None else (global_cfg.get("email") or "")
    password = args.password if args.password is not None else (global_cfg.get("password") or "")

    if not manual_login and (not email or not password):
        print(
            "Error: auto-login requires email/password. Set --manual-login or provide credentials.",
            file=sys.stderr,
        )
        return 2

    title = args.title or (global_cfg.get("title") or None)

    print("Starting upload with visible Chrome." if not headless else "Starting upload in headless mode.")
    print(f"- url: {kakao_url}")
    print(f"- image_dir: {image_dir}")
    print(f"- manual_login: {manual_login}")

    upload_to_kakao(
        url=kakao_url,
        email=email,
        password=password,
        image_dir=str(image_dir),
        headless=headless,
        user_data_dir=user_data_dir,
        profile_directory=profile_directory,
        driver_path=driver_path,
        title=title,
        manual_login=manual_login,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
