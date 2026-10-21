"""Settings from the environment and the git-ignored .env file (see .env.example).

A tiny .env reader, so no extra dependency: KEY=VALUE lines, # comments, optional quotes.
Values already in the environment win over the file.
"""

from __future__ import annotations

import os
from pathlib import Path

from nextcour.core import paths


def load_env(path: Path | None = None) -> None:
    path = path or paths.ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value:
            os.environ.setdefault(key.strip(), value)


def gemini_api_key() -> str | None:
    load_env()
    return os.environ.get("GEMINI_API_KEY") or None
