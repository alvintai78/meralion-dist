"""Pack the Python/CUDA runtime from a built GPU image into Blob tar shards.

Run inside a container of the full runtime image with a managed identity that
can write to the models container. The shards are plain tar files so that the
bootstrap image can stream them with ``curl | tar`` in parallel.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
from pathlib import Path
import ssl
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from urllib.parse import quote, urlencode, urlparse
import urllib.request

ROOTS = [
    "usr/local/lib/python3.12/dist-packages",
    "usr/local/bin",
    # Warm Triton/vLLM/numba caches so first-request compilation is not paid at startup.
    "app/home/.cache",
    "app/home/.triton",
    "app/model-cache/modules",
]
# Packages the in-process Transformers engine never imports. Set PACK_EXCLUDES="" to
# pack everything (needed for the vLLM engines; PACK_TRACE_FILE then prunes by observed use). torch's own nvidia/* wheels are kept:
# libtorch_cuda links them at import time.
_DP = "usr/local/lib/python3.12/dist-packages/"
DEFAULT_EXCLUDES = ",".join(
    _DP + name
    for name in (
        "vllm", "triton", "flashinfer", "flashinfer_jit_cache", "flashinfer_cubin", "xgrammar", "cuda",
        "cupy", "cupyx", "cupy_backends", "ray", "bitsandbytes", "nixl", "nixl_cu12", "nixl_cu12.libs",
        "deep_gemm", "pplx_kernels", "lmcache", "modelscope", "outlines_core", "llguidance",
        "mistral_common", "openai", "anthropic", "mcp", "depyf", "nvidia_cutlass_dsl",
    )
)


def token(client_id: str) -> str:
    query = urlencode({"resource": "https://storage.azure.com/", "client_id": client_id, "api-version": "2019-08-01"})
    request = urllib.request.Request(
        os.environ["IDENTITY_ENDPOINT"] + "?" + query,
        headers={"X-IDENTITY-HEADER": os.environ["IDENTITY_HEADER"]},
    )
    with urllib.request.urlopen(request) as response:  # nosec B310
        return json.load(response)["access_token"]


def put_blob(container_url: str, name: str, path: str, client_id: str) -> int:
    parsed = urlparse(container_url)
    size = os.path.getsize(path)
    connection = http.client.HTTPSConnection(parsed.hostname, context=ssl.create_default_context(), timeout=600)
    with open(path, "rb") as source:
        connection.request(
            "PUT",
            f"{parsed.path}/{quote(name)}",
            body=source,
            headers={
                "Authorization": "Bearer " + token(client_id),
                "x-ms-version": "2023-11-03",
                "x-ms-blob-type": "BlockBlob",
                "Content-Length": str(size),
            },
        )
        response = connection.getresponse()
        response.read()
    if response.status not in (200, 201):
        raise RuntimeError(f"upload of {name} failed with HTTP {response.status}")
    return size


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack_shard(args):
    shard_name, members, container_url, prefix, client_id, workdir = args
    listing = os.path.join(workdir, f"{shard_name}.list")
    archive = os.path.join(workdir, shard_name)
    with open(listing, "w") as handle:
        handle.write("\n".join(members) + "\n")
    subprocess.run(
        ["tar", "--no-recursion", "-C", "/", "-cf", archive, "-T", listing],
        check=True,
    )
    size = put_blob(container_url, f"{prefix}/{shard_name}", archive, client_id)
    checksum = sha256(archive)
    os.unlink(archive)
    return {"name": shard_name, "size": size, "sha256": checksum, "files": len(members)}


MOUNT_POINTS = {"usr/local/lib/python3.12/dist-packages"}
TRACE_MIN_BYTES = 5 << 20


def directory_bytes(path: str) -> int:
    return sum(
        os.path.getsize(os.path.join(directory, name))
        for directory, _dirs, names in os.walk(path)
        for name in names
        if not os.path.islink(os.path.join(directory, name))
    )


def trace_excludes(trace_file: str) -> list[str]:
    """Exclude large packages (and unused FlashInfer kernel modules) the warmed service never loaded.

    The trace comes from ``model_service.runtime_trace`` after the warm-up transcription, so
    everything the transcription path needs is kept; only dead weight (ray, cupy, unused AOT
    kernels, ...) is dropped from the cold-start download.
    """
    trace = json.loads(Path(trace_file).read_text())
    used = [*trace["modules"], *trace["mapped"]]

    def in_use(directory: str) -> bool:
        prefix = directory + "/"
        return any(path.startswith(prefix) for path in used)

    base = "/" + _DP.rstrip("/")
    excludes = []
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if not os.path.isdir(path) or os.path.islink(path) or in_use(path):
            continue
        if not name.endswith((".dist-info", ".libs")) and directory_bytes(path) >= TRACE_MIN_BYTES:
            excludes.append(_DP + name)
    cache = os.path.join(base, "flashinfer_jit_cache", "jit_cache")
    if os.path.isdir(cache):
        excludes += [
            _DP + "flashinfer_jit_cache/jit_cache/" + name
            for name in sorted(os.listdir(cache))
            if os.path.isdir(os.path.join(cache, name)) and not in_use(os.path.join(cache, name))
        ]
    return excludes


def collect(excludes: list[str]) -> list[tuple[str, int]]:
    entries: list[tuple[str, int]] = []
    for root in ROOTS:
        if not os.path.isdir("/" + root):
            continue
        for directory, dirnames, filenames in os.walk("/" + root):
            relative = os.path.relpath(directory, "/")
            dirnames[:] = [
                d for d in dirnames
                if not any(os.path.join(relative, d) == ex or os.path.join(relative, d).startswith(ex + "/") for ex in excludes)
            ]
            entries.append((relative, 0))
            for name in dirnames:
                if os.path.islink(os.path.join(directory, name)):
                    entries.append((os.path.relpath(os.path.join(directory, name), "/"), 0))
            for name in filenames:
                path = os.path.join(directory, name)
                relative_path = os.path.relpath(path, "/")
                size = 0 if os.path.islink(path) else os.path.getsize(path)
                entries.append((relative_path, size))
    return entries


def main() -> None:
    container_url = os.environ["MODEL_BLOB_URL"].rstrip("/")
    client_id = os.environ["AZURE_CLIENT_ID"]
    version = sys.argv[1]
    shard_count = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    excludes = [item for item in os.environ.get("PACK_EXCLUDES", DEFAULT_EXCLUDES).split(",") if item]
    if os.environ.get("PACK_TRACE_FILE"):
        excludes += trace_excludes(os.environ["PACK_TRACE_FILE"])
        print(json.dumps({"trace_excludes": excludes}), flush=True)
    entries = collect(excludes)
    directories = [rel for rel, size in entries if os.path.isdir("/" + rel) and not os.path.islink("/" + rel)]
    # Volume mount points keep the platform's ownership and mode; tar must not try to change them.
    directories = [rel for rel in directories if rel not in MOUNT_POINTS]
    entries = [item for item in entries if item[0] not in MOUNT_POINTS]
    directory_set = set(directories)
    entries = [item for item in entries if item[0] not in directory_set]
    shards: list[list[str]] = [[] for _ in range(shard_count)]
    loads = [0] * shard_count
    for relative, size in sorted(entries, key=lambda item: -item[1]):
        target = loads.index(min(loads))
        shards[target].append(relative)
        loads[target] += max(size, 4096)
    prefix = f"runtime/{version}"
    workdir = tempfile.mkdtemp(prefix="pack-", dir="/tmp")
    jobs = [(f"shard-{i:02d}.tar", members, container_url, prefix, client_id, workdir) for i, members in enumerate(shards) if members]
    with ProcessPoolExecutor(min(len(jobs), 8)) as pool:
        results = list(pool.map(pack_shard, jobs))
    # Directories are extracted first so parallel shard extraction never races to create parents.
    dirs = pack_shard(("dirs.tar", directories, container_url, prefix, client_id, workdir))
    manifest = {"version": version, "dirs": dirs, "shards": results, "bytes": sum(item["size"] for item in results)}
    manifest_path = os.path.join(workdir, "manifest.json")
    Path(manifest_path).write_text(json.dumps(manifest))
    put_blob(container_url, f"{prefix}/manifest.json", manifest_path, client_id)
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
