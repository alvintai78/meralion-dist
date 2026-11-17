"""Stage a pinned model snapshot through private Blob Storage."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import time
from urllib.parse import urlparse

from model_service.startup import startup_phase

logger = logging.getLogger("meralion_weights")
MODEL_ID = "MERaLiON/MERaLiON-3-3B-ASR"
MODEL_REVISION = "9f226d738dee62c60a2f43b0ab7b9a2585a132c3"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(manifest: dict, model_id: str, revision: str) -> list[dict]:
    if manifest.get("model_id") != model_id or manifest.get("revision") != revision:
        raise ValueError("model manifest does not match the pinned model revision")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("model manifest has no files")
    seen = set()
    for item in files:
        name = item["path"]
        path = PurePosixPath(name)
        if (
            not name
            or path.is_absolute()
            or ".." in path.parts
            or str(path) != name
            or "\\" in name
            or name in seen
            or name == "manifest.json"
        ):
            raise ValueError(f"unsafe or duplicate model file: {name}")
        if type(item["size"]) is not int or item["size"] < 0:
            raise ValueError(f"invalid model file size: {name}")
        checksum = item["sha256"]
        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or any(c not in "0123456789abcdef" for c in checksum)
        ):
            raise ValueError(f"invalid model file checksum: {name}")
        seen.add(name)
    return files


def download_snapshot(container, target: Path, model_id: str, revision: str) -> Path:
    started = time.monotonic()
    prefix = f"{model_id}/{revision}"
    manifest = json.loads(container.download_blob(f"{prefix}/manifest.json").readall())
    files = validate_manifest(manifest, model_id, revision)
    target.mkdir(parents=True, exist_ok=True)

    def download(item):
        destination = target / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".partial")
        try:
            with startup_phase("weight_transfer", file=item["path"], bytes=item["size"]):
                data = container.download_blob(
                    f"{prefix}/{item['path']}", max_concurrency=8
                ).readall()
            with startup_phase("weight_checksum", file=item["path"]):
                if len(data) != item["size"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                    raise ValueError(f"model file integrity check failed: {item['path']}")
            with startup_phase("weight_write", file=item["path"]):
                with temporary.open("wb") as output:
                    output.write(data)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(download, files))
    logger.info(
        json.dumps({
            "event": "model_weights_staged",
            "revision": revision,
            "bytes": sum(item["size"] for item in files),
            "duration_seconds": time.monotonic() - started,
        })
    )
    return target


def seed_snapshot(container, model_id: str, revision: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(
        repo_id=model_id,
        revision=revision,
        allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt", "*.model", "*.tiktoken", "*.jinja"],
        max_workers=8,
    ))
    prefix = f"{model_id}/{revision}"
    paths = sorted(path for path in snapshot.rglob("*") if path.is_file())

    def upload(path):
        relative = path.relative_to(snapshot).as_posix()
        checksum = file_hash(path)
        with path.open("rb") as source:
            container.upload_blob(
                f"{prefix}/{relative}", source, overwrite=True, max_concurrency=8
            )
        return {"path": relative, "size": path.stat().st_size, "sha256": checksum}

    with ThreadPoolExecutor(max_workers=8) as pool:
        files = list(pool.map(upload, paths))
    manifest = {"model_id": model_id, "revision": revision, "files": files}
    validate_manifest(manifest, model_id, revision)
    # Publish only after every file is uploaded successfully.
    container.upload_blob(
        f"{prefix}/manifest.json", json.dumps(manifest), overwrite=True
    )
    logger.info("Published model manifest for %s at %s", model_id, revision)


def blob_container():
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import ContainerClient

    url = os.environ["MODEL_BLOB_URL"].rstrip("/")
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or not parsed.path.strip("/")
        or parsed.query
        or parsed.fragment
        or parsed.username
    ):
        raise ValueError("MODEL_BLOB_URL must be an HTTPS container URL without credentials")
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
    credential = DefaultAzureCredential(
        managed_identity_client_id=os.getenv("AZURE_CLIENT_ID")
    )
    return credential, ContainerClient.from_container_url(
        url,
        credential=credential,
        max_chunk_get_size=32 * 1024 * 1024,
        max_single_get_size=32 * 1024 * 1024,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    credential, container = blob_container()
    with credential, container:
        seed_snapshot(
            container,
            os.getenv("MODEL_ID", MODEL_ID),
            os.getenv("MODEL_REVISION", MODEL_REVISION),
        )


if __name__ == "__main__":
    main()
