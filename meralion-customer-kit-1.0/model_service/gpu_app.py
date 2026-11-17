"""Internal GPU API that owns model inference and tokenization."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import io
import logging
import math
import os
from pathlib import Path
import struct
import wave

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from .app import _gpu_name
from .config import Settings
from .model import OfficialModelClient
from .startup import startup_event, startup_phase


logger = logging.getLogger("meralion_gpu")


def warmup_wav(seconds: float = 3.0, rate: int = 16000) -> bytes:
    """Speech-band synthetic audio that exercises the full encoder/decoder path."""
    frames = bytearray()
    for n in range(int(seconds * rate)):
        t = n / rate
        envelope = 0.5 + 0.5 * math.sin(2 * math.pi * 3 * t)
        sample = envelope * (
            math.sin(2 * math.pi * 220 * t) + 0.5 * math.sin(2 * math.pi * 880 * t)
        )
        frames += struct.pack("<h", int(sample * 6000))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(bytes(frames))
    return buffer.getvalue()


async def await_weights_verified() -> None:
    marker = os.getenv("WEIGHTS_VERIFIED_FILE")
    if not marker:
        return
    verified = Path(marker)
    failed = verified.with_suffix(".failed")
    while not verified.exists():
        if failed.exists():
            raise RuntimeError(f"model weights failed verification: {failed.read_text()}")
        await asyncio.sleep(0.1)


class InternalTranscriptionResponse(BaseModel):
    text: str
    output_tokens: int
    gpu_name: str
    cold: bool


def create_gpu_app(
    settings: Settings | None = None,
    model_client: OfficialModelClient | None = None,
) -> FastAPI:
    settings = settings or Settings.from_environment(require_blob=False)
    if not settings.model_local_path:
        raise RuntimeError("MODEL_LOCAL_PATH is required by the GPU service")
    if model_client is None:
        engine = os.getenv("INFERENCE_ENGINE", "vllm")
        if engine == "transformers":
            from .hf_model import TransformersModelClient

            model_client = TransformersModelClient(
                settings.model_id, settings.model_revision, settings.model_local_path
            )
        elif engine == "vllm":
            from .vllm_model import VllmModelClient

            model_client = VllmModelClient(
                settings.model_id, settings.model_revision, settings.model_local_path
            )
        elif engine == "vllm-sidecar":
            model_client = OfficialModelClient(
                "http://127.0.0.1:8001",
                settings.model_id,
                settings.model_revision,
                settings.model_local_path,
            )
        else:
            raise RuntimeError(f"unknown INFERENCE_ENGINE: {engine}")
    state = {
        "ready": False,
        "requests": 0,
        "gpu_name": "unavailable",
        "error": None,
    }

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        async def initialize_model():
            try:
                async def identify_gpu():
                    with startup_phase("gpu_metadata"):
                        return await asyncio.to_thread(_gpu_name)

                async with asyncio.TaskGroup() as tasks:
                    tasks.create_task(model_client.start())
                    gpu_metadata = tasks.create_task(identify_gpu())
                state["gpu_name"] = gpu_metadata.result()
                with startup_phase("weights_verified_wait"):
                    await await_weights_verified()
                if os.getenv("WARMUP_TRANSCRIPTION") == "1":
                    # Pays first-inference initialization before the replica is reported ready.
                    with startup_phase("warmup_transcription"):
                        try:
                            await model_client.transcribe(warmup_wav())
                        except Exception:  # pylint: disable=broad-except
                            logger.exception("warm-up transcription failed")
                if os.getenv("RUNTIME_TRACE_FILE"):
                    from .runtime_trace import write_trace

                    write_trace(os.environ["RUNTIME_TRACE_FILE"])
                state["ready"] = True
                startup_event("gpu_model_ready")
            except Exception as exc:  # pylint: disable=broad-except
                state["error"] = f"{type(exc).__name__}: {exc}"
                logger.exception("GPU model initialization failed")

        initialization = asyncio.create_task(initialize_model())
        await asyncio.sleep(0)
        try:
            yield
        finally:
            state["ready"] = False
            initialization.cancel()
            await model_client.close()

    app = FastAPI(title="MERaLiON internal GPU service", lifespan=lifespan)

    @app.get("/health/live")
    async def live():
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready():
        if state["error"]:
            raise HTTPException(status_code=500, detail=state["error"])
        if not state["ready"]:
            raise HTTPException(status_code=503, detail="model loading")
        return {"status": "ready", "model_loaded": True}

    @app.post(
        "/v1/internal/transcriptions", response_model=InternalTranscriptionResponse
    )
    async def transcribe(request: Request):
        if not state["ready"]:
            raise HTTPException(status_code=503, detail="model loading")
        cold = state["requests"] == 0
        state["requests"] += 1
        wav_bytes = await request.body()
        with startup_phase("transcription", cold=cold, bytes=len(wav_bytes)):
            text = await model_client.transcribe(wav_bytes)
        return InternalTranscriptionResponse(
            text=text,
            output_tokens=model_client.count_output_tokens(text),
            gpu_name=str(state["gpu_name"]),
            cold=cold,
        )

    return app


def production_gpu_app() -> FastAPI:
    return create_gpu_app()