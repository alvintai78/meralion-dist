"""Remove optional AOT caches for precisions/architectures outside this BF16 A100 service."""

import importlib.metadata
import json
from pathlib import Path
import shutil


def unused_cache(name: str) -> bool:
    return any(marker in name for marker in (
        "_dtype_q_f16_", "_input_f16_", "e4m3", "e5m2",
        "_sm90", "_sm100", "_sm120",
    ))


def prune(cache: Path) -> dict:
    if not cache.is_dir():
        raise FileNotFoundError(cache)
    removed = []
    removed_bytes = 0
    for entry in sorted(cache.iterdir()):
        if entry.is_dir() and not entry.is_symlink() and unused_cache(entry.name):
            removed_bytes += sum(path.stat().st_size for path in entry.rglob("*") if path.is_file())
            removed.append(entry.name)
            shutil.rmtree(entry)
    return {"removed_bytes": removed_bytes, "removed_modules": removed}


if __name__ == "__main__":
    version = importlib.metadata.version("flashinfer-jit-cache")
    if version.split("+", 1)[0] != "0.6.3":
        raise RuntimeError(f"Expected FlashInfer JIT cache 0.6.3, got {version}")
    print(json.dumps(prune(Path(
        "/usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache"
    ))))
