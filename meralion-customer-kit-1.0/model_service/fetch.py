"""Fetch the Python/CUDA runtime and model weights from private Blob at line rate.

Only the standard library plus ``curl``, ``tar`` and ``dd`` are used so this
module can run before the large runtime exists. Data never passes through the
Python interpreter: each shard or byte range is a ``curl | tar`` or
``curl | dd`` pipeline, which removes the GIL and SDK overhead that limited the
previous staging path to a few hundred MB/s.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import time
from urllib.parse import quote, urlencode
import urllib.request

from .startup import startup_phase

logger = logging.getLogger("meralion_fetch")
API_VERSION = "2023-11-03"
RANGE_BYTES = 256 * 1024 * 1024
SMALL_FILE_BYTES = 64 * 1024 * 1024


def identity_token() -> str:
    query = {"resource": "https://storage.azure.com/", "api-version": "2019-08-01"}
    client_id = os.getenv("AZURE_CLIENT_ID")
    if client_id:
        query["client_id"] = client_id
    request = urllib.request.Request(
        f"{os.environ['IDENTITY_ENDPOINT']}?{urlencode(query)}",
        headers={"X-IDENTITY-HEADER": os.environ["IDENTITY_HEADER"]},
    )
    with urllib.request.urlopen(request, timeout=10) as response:  # nosec B310
        return json.load(response)["access_token"]


def _curl(url: str, token: str, byte_range: tuple[int, int] | None = None) -> subprocess.Popen:
    # Headers are passed on stdin so the bearer token never appears in argv.
    config = f'header = "Authorization: Bearer {token}"\nheader = "x-ms-version: {API_VERSION}"\n'
    if byte_range is not None:
        config += f'header = "Range: bytes={byte_range[0]}-{byte_range[1]}"\n'
    process = subprocess.Popen(  # nosec B603 B607
        ["curl", "--silent", "--show-error", "--fail", "--http1.1", "--config", "-", url],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    process.stdin.write(config.encode())
    process.stdin.close()
    return process


def _pipeline(url: str, token: str, consumer: list[str], byte_range=None) -> None:
    producer = _curl(url, token, byte_range)
    sink = subprocess.Popen(consumer, stdin=producer.stdout)  # nosec B603
    producer.stdout.close()
    sink_code = sink.wait()
    producer_code = producer.wait()
    if producer_code or sink_code:
        raise RuntimeError(f"transfer failed (curl={producer_code}, consumer={sink_code}): {url}")


def _with_retries(action, attempts: int = 3):
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except RuntimeError:
            if attempt == attempts:
                raise
            time.sleep(0.5 * attempt)


def blob_url(container_url: str, name: str) -> str:
    return f"{container_url.rstrip('/')}/{quote(name)}"


def read_json_blob(container_url: str, name: str, token: str) -> dict:
    process = _curl(blob_url(container_url, name), token)
    body = process.stdout.read()
    if process.wait():
        raise RuntimeError(f"could not read blob {name}")
    return json.loads(body)


def fetch_runtime(container_url: str, version: str, token: str, destination: str = "/", workers: int = 24) -> dict:
    prefix = f"runtime/{version}"
    manifest = read_json_blob(container_url, f"{prefix}/manifest.json", token)

    def shard(item):
        _with_retries(lambda: _pipeline(
            blob_url(container_url, f"{prefix}/{item['name']}"),
            token,
            ["tar", "--no-same-owner", "-x", "-f", "-", "-C", destination],
        ))

    # Create every directory first: parallel tar processes otherwise race to create parents.
    if manifest.get("dirs"):
        shard(manifest["dirs"])
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(shard, manifest["shards"]))
    return manifest


def start_weights(
    container_url: str, model_id: str, revision: str, target: Path, token: str, workers: int = 24
) -> "WeightTransfer":
    prefix = f"{model_id}/{revision}"
    manifest = read_json_blob(container_url, f"{prefix}/manifest.json", token)
    files = manifest["files"]
    if manifest.get("model_id") != model_id or manifest.get("revision") != revision:
        raise ValueError("model manifest does not match the pinned model revision")
    for item in files:
        destination = target / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            handle.truncate(item["size"])
    return WeightTransfer(container_url, prefix, target, token, files, workers)


class WeightTransfer:
    """Concurrent ranged download of every snapshot file, then checksum verification."""

    def __init__(self, container_url, prefix, target, token, files, workers):
        self._container_url = container_url
        self._prefix = prefix
        self._target = target
        self._token = token
        self.files = files
        self._pool = ThreadPoolExecutor(workers)
        self._futures = []
        self._small = []

    def begin(self) -> None:
        # Small files first so config and tokenizer are available early.
        for item in sorted(self.files, key=lambda f: f["size"]):
            url = blob_url(self._container_url, f"{self._prefix}/{item['path']}")
            path = str(self._target / item["path"])
            size = item["size"]
            for start in range(0, max(size, 1), RANGE_BYTES):
                end = min(start + RANGE_BYTES, size) - 1
                if size == 0:
                    break
                future = self._pool.submit(self._range, url, path, start, end)
                self._futures.append(future)
                if size <= SMALL_FILE_BYTES:
                    self._small.append(future)

    def _range(self, url: str, path: str, start: int, end: int) -> None:
        _with_retries(lambda: _pipeline(
            url,
            self._token,
            [
                "dd", f"of={path}", "bs=4M", f"seek={start}", "oflag=seek_bytes",
                "iflag=fullblock", "conv=notrunc", "status=none",
            ],
            (start, end),
        ))

    def wait_small_files(self) -> None:
        for future in self._small:
            future.result()

    def wait_downloaded(self) -> None:
        for future in self._futures:
            future.result()

    def verify(self) -> None:
        def check(item):
            path = self._target / item["path"]
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
                    digest.update(chunk)
            if path.stat().st_size != item["size"] or digest.hexdigest() != item["sha256"]:
                raise ValueError(f"model file integrity check failed: {item['path']}")

        with ThreadPoolExecutor(8) as pool:
            list(pool.map(check, self.files))

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def stage_runtime(container_url: str, version: str, token: str) -> None:
    with startup_phase("runtime_fetch", version=version):
        manifest = fetch_runtime(container_url, version, token)
    logger.info(json.dumps({"event": "runtime_fetched", "version": version, "bytes": manifest.get("bytes")}))
