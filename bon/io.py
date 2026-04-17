"""Small JSON / gzip-JSON I/O helpers used across the pipeline."""

from __future__ import annotations

import gzip
import json
import os
from typing import Any


def save_json(data: Any, filename: str | os.PathLike) -> None:
    filename = os.fspath(filename)
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, "w") as f:
        json.dump(data, f)


def load_json(filename: str | os.PathLike) -> Any:
    with open(os.fspath(filename), "r") as f:
        return json.load(f)


def save_gzip(data: Any, filename: str | os.PathLike) -> None:
    filename = os.fspath(filename)
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with gzip.open(filename, "wt", encoding="utf-8") as f:
        json.dump(data, f)


def load_gzip(filename: str | os.PathLike) -> Any:
    with gzip.open(os.fspath(filename), "rt", encoding="utf-8") as f:
        return json.load(f)
