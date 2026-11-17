"""Record which files the warmed service actually loaded, to prune the packed runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def loaded_files() -> dict:
    modules = {
        os.path.realpath(getattr(module, "__file__", None))
        for module in list(sys.modules.values())
        if getattr(module, "__file__", None)
    }
    mapped = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith("/"):
            mapped.add(os.path.realpath(parts[5]))
    return {"modules": sorted(modules), "mapped": sorted(mapped)}


def write_trace(path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(loaded_files()))
