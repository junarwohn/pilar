import os
import time
from pathlib import Path

from pilar.web.server import create_app


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y", "on")


def _resolve_base_dir() -> str:
    override = os.getenv("PILAR_BASE_DIR")
    if override:
        return override
    day_info = time.strftime('%Y-%m-%d', time.localtime())[2:]
    # repo root: .../pilar/web/wsgi.py -> parents[2]
    repo_root = Path(__file__).resolve().parents[2]
    return str((repo_root / "out" / day_info).resolve())


# Expose WSGI app for Gunicorn
app = create_app(
    base_dir=_resolve_base_dir(),
    no_gui=_env_bool("PILAR_NO_GUI", True),
)

