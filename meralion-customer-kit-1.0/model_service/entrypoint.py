"""Supervise the official model sidecar and internal GPU API."""

from __future__ import annotations

import os
import logging
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from .startup import startup_event, startup_phase

MODEL_ID = os.getenv("MODEL_ID", "MERaLiON/MERaLiON-3-3B-ASR")
MODEL_REVISION = os.getenv(
    "MODEL_REVISION", "9f226d738dee62c60a2f43b0ab7b9a2585a132c3"
)
SIDECAR_PORT = 8001


def resolve_model_path() -> Path:
    configured_path = os.getenv("MODEL_LOCAL_PATH")
    if configured_path:
        model_path = Path(configured_path)
    elif os.getenv("MODEL_BLOB_URL"):
        from .weights import blob_container, download_snapshot

        model_path = Path(os.getenv("MODEL_CACHE_ROOT", "/app/model-cache")) / MODEL_REVISION
        credential, container = blob_container()
        with credential, container:
            return download_snapshot(container, model_path, MODEL_ID, MODEL_REVISION)
    else:
        cache_root = Path(os.getenv("MODEL_CACHE_ROOT", "/app/model-cache"))
        repository_directory = "models--" + MODEL_ID.replace("/", "--")
        model_path = cache_root / repository_directory / "snapshots" / MODEL_REVISION
    if not model_path.is_dir():
        raise FileNotFoundError(f"local model snapshot not found: {model_path}")
    return model_path


def listen_early(port: int = 8000) -> socket.socket:
    """Bind the service port at process start and hand the socket to uvicorn.

    The platform treats the replica as reachable as soon as the port accepts
    connections. Binding it before the runtime is fetched lets that registration
    overlap the model load; connections simply wait in the backlog until the API
    starts serving, and ``/health/ready`` still reports readiness.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("0.0.0.0", port))  # nosec B104
    listener.listen(1024)
    listener.set_inheritable(True)
    return listener


def fast_stage() -> tuple[Path, "object"]:
    """Fetch the runtime, then start (not await) the weights transfer."""
    from . import fetch

    container_url = os.environ["MODEL_BLOB_URL"]
    token = fetch.identity_token()
    fetch.stage_runtime(container_url, os.environ["RUNTIME_VERSION"], token)
    model_path = Path(os.getenv("MODEL_CACHE_ROOT", "/app/model-cache")) / MODEL_REVISION
    transfer = fetch.start_weights(container_url, MODEL_ID, MODEL_REVISION, model_path, token)
    transfer.begin()
    with startup_phase("small_model_files"):
        transfer.wait_small_files()
    return model_path, transfer


def finish_weights(transfer, verified_file: Path, terminate) -> None:
    try:
        with startup_phase("weight_transfer_complete"):
            transfer.wait_downloaded()
        verified_file.with_suffix(".downloaded").write_text("ok")
        with startup_phase("weight_verify"):
            transfer.verify()
        verified_file.write_text("ok")
    except Exception as exc:  # pylint: disable=broad-except
        logging.getLogger("meralion_entrypoint").error("model weights unusable: %s", exc)
        verified_file.with_suffix(".failed").write_text(str(exc))
        terminate()


def wait_for_model(process: subprocess.Popen, timeout_seconds: int = 900) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = f"http://127.0.0.1:{SIDECAR_PORT}/v1/models"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"model sidecar exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:  # nosec B310
                if response.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(2)
    raise TimeoutError("model sidecar did not become ready within 900 seconds")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    startup_event("gpu_entrypoint_started")
    mounts = Path("/proc/self/mountinfo")
    if mounts.is_file():
        startup_event(
            "runtime_root_mount",
            mount=next(line for line in mounts.read_text().splitlines() if line.split()[4] == "/"),
        )
    transfer = None
    listener = listen_early() if os.getenv("LISTEN_EARLY", "1") == "1" else None
    fast = bool(
        os.getenv("RUNTIME_VERSION")
        and os.getenv("MODEL_BLOB_URL")
        and not os.getenv("MODEL_LOCAL_PATH")
    )
    with startup_phase("model_snapshot"):
        if fast:
            model_path, transfer = fast_stage()
        else:
            model_path = resolve_model_path()
    os.environ["MODEL_LOCAL_PATH"] = str(model_path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sidecar = None
    if os.getenv("INFERENCE_ENGINE") == "vllm-sidecar":
        sidecar = subprocess.Popen(  # nosec B603
            [
                "meralion-3-asr",
                "serve",
                "--model",
                str(model_path),
                "--served-model-name",
                MODEL_ID,
                "--host",
                "127.0.0.1",
                "--port",
                str(SIDECAR_PORT),
                "--max-num-seqs",
                "1",
                "--enforce-eager",
                "--load-format",
                "safetensors",
                *shlex.split(os.getenv("SIDECAR_EXTRA_ARGS", "")),
            ]
        )
        startup_event("sidecar_spawned", child_pid=sidecar.pid)
    api = None

    def stop(*_args):
        for process in (api, sidecar):
            if process is not None and process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if transfer is not None:
        verified_file = Path(os.environ.setdefault("WEIGHTS_VERIFIED_FILE", "/tmp/meralion/weights.verified"))
        verified_file.parent.mkdir(parents=True, exist_ok=True)
        threading.Thread(
            target=finish_weights, args=(transfer, verified_file, stop), daemon=True
        ).start()
    try:
        bind = ["--host", "0.0.0.0", "--port", "8000"]
        popen_options = {}
        if listener is not None:
            bind = ["--fd", str(listener.fileno())]
            popen_options["pass_fds"] = (listener.fileno(),)
        api = subprocess.Popen(  # nosec B603
            [
                "uvicorn",
                "model_service.gpu_app:production_gpu_app",
                "--factory",
                *bind,
                "--workers",
                "1",
                "--no-access-log",
            ],
            **popen_options,
        )
        return api.wait()
    finally:
        stop()
        if sidecar is not None:
            sidecar.wait(timeout=30)


if __name__ == "__main__":
    sys.exit(main())