"""Tiny config loader shared by the data pipeline.

Resolves all `paths.*` entries to absolute paths against the repo root so the
scripts work regardless of the current working directory.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

# repo root = two levels up from this file (src/data/config.py -> repo/)
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | os.PathLike = "configs/data.yaml") -> dict:
    cfg_path = (REPO_ROOT / path) if not Path(path).is_absolute() else Path(path)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve every path entry relative to the repo root.
    cfg["paths"] = {k: str((REPO_ROOT / v).resolve()) for k, v in cfg["paths"].items()}
    cfg["_repo_root"] = str(REPO_ROOT)
    return cfg
