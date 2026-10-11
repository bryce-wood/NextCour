"""The data manifest (manifests/datasets.json): each dataset's source, license, files, sizes and
SHA-256 checksums, so every result can name the exact data it used. It's committed; the data
itself never is.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from nextcour.core import paths


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path | None = None) -> dict[str, Any]:
    path = path or paths.DATASET_MANIFEST
    if not path.exists():
        return {
            "about": "Sources and checksums of every dataset NextCour uses. Data lives in data/.",
            "datasets": {},
        }
    return json.loads(path.read_text(encoding="utf-8"))


def update(name: str, record: dict[str, Any], path: Path | None = None) -> None:
    path = path or paths.DATASET_MANIFEST
    manifest = load(path)
    manifest["datasets"][name] = record
    manifest["datasets"] = dict(sorted(manifest["datasets"].items()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
