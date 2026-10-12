"""Where things live on disk.

Datasets and everything generated from them go under data/, which git ignores. The data manifest
is committed. My frozen test set (frozen/) is personal, so git ignores it too.
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """The repo root: the nearest folder above this file that holds pyproject.toml."""
    for folder in Path(__file__).resolve().parents:
        if (folder / "pyproject.toml").is_file():
            return folder
    raise RuntimeError("Can't find the project root (no pyproject.toml above this file)")


ROOT = project_root()
DATA = Path(os.environ.get("NEXTCOUR_DATA_DIR", ROOT / "data"))

RAW = DATA / "raw"
PARQUET = DATA / "parquet"
PROCESSED = DATA / "processed"
FROZEN_SNAPSHOT_DIR = DATA / "frozen"
APP_DB = DATA / "nextcour.sqlite"
RESULTS = DATA / "results"  # evaluation results; they hold my labels, so they stay private

MANIFESTS = ROOT / "manifests"
DATASET_MANIFEST = MANIFESTS / "datasets.json"
FROZEN_TEST_SET = ROOT / "frozen" / "my_test_set.json"
