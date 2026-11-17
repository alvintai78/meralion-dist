"""FastAPI benchmark gateway for the official MERaLiON ASR sidecar."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from .audio import normalize_audio
from .blob import BlobReader
from .config import Settings
from .metrics import PhaseMarks, build_phase_marks_ms, build_timing_metrics
from .model import ModelClient
from .schemas import BenchmarkRequest, BenchmarkResponse, HealthResponse


logger = logging.getLogger("meralion_benchmark")
logging.basicConfig(level=logging.INFO, format="%(message)s")

UNAVAILABLE_METRICS = {
    "model_queue_ms": (
        "The official meralion-3-asr 0.1.0 sidecar does not return per-request "
        "vLLM queue metrics."
    ),
    "prefill_to_first_token_ms": (
        "The official sidecar submits non-streaming chat completions and exposes "
        "only the completed transcript."
    ),
    "generation_ms": (
        "The completed sidecar response cannot separate audio encoding/prefill "
        "from token decode."
    ),
    "generation_tokens_per_second": (
        "Steady generation throughput requires a truthful first-token timestamp."
    ),
}
UI_PATH = Path(__file__).with_name("static") / "index.html"


def _gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:  # pragma: no cover - defensive metadata collection
        pass
    return "unavailable"


def _log(event: dict) -> None:
    logger.info(json.dumps(event, ensure_ascii=True, separators=(",", ":")))


def create_app(
    settings: Settings | None = None,
    blob_reader: BlobReader | None = None,
    model_client: ModelClient | None = None,
) -> FastAPI:
    settings = settings or Settings.from_environment()
    blob_reader = blob_reader or BlobReader(
        settings.audio_blob_url, settings.azure_client_id
    )
    model_client = model_client or ModelClient(
        settings.sidecar_url,
        settings.model_id,
        settings.model_revision,
    )
    state = {
        "ready": False,
        "gpu_status": "stopped",
        "gpu_error": None,
        "wake_task": None,
    }
    inference_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await blob_reader.start()
        try:
            await model_client.start()
            state["ready"] = True
            _log(
                {
                    "event": "model_ready",
                    "model_id": settings.model_id,
                    "model_revision": settings.model_revision,
                    "backend": settings.backend,
                    "gpu_name": "remote",
                }
            )
            yield
        finally:
            state["ready"] = False
            if state["wake_task"] is not None:
                state["wake_task"].cancel()
            await model_client.close()
            await blob_reader.close()

    app = FastAPI(
        title="MERaLiON ASR benchmark service",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_clock(request: Request, call_next):
        request.state.received_at = time.perf_counter()
        return await call_next(request)

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(UI_PATH, media_type="text/html")

    @app.get("/health/live")
    async def live():
        return {"status": "live"}

    @app.get(
        "/health/ready",
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse}},
    )
    async def ready():
        body = HealthResponse(
            status="ready" if state["ready"] else "loading",
            model_loaded=bool(state["ready"]),
            model_id=settings.model_id,
            model_revision=settings.model_revision,
            backend=settings.backend,
        )
        if not state["ready"]:
            return JSONResponse(status_code=503, content=body.model_dump())
        return body

    async def wake_model():
        state["gpu_status"] = "loading"
        state["gpu_error"] = None
        try:
            await model_client.wait_ready()
            state["gpu_status"] = "ready"
        except Exception as exc:  # pylint: disable=broad-except
            state["gpu_status"] = "error"
            state["gpu_error"] = f"{type(exc).__name__}: {exc}"

    @app.post("/v1/model/wake", status_code=202)
    async def wake():
        task = state["wake_task"]
        if task is None or task.done():
            state["wake_task"] = asyncio.create_task(wake_model())
        return {"status": state["gpu_status"]}

    @app.get("/v1/model/status")
    async def model_status():
        status = state["gpu_status"]
        body = {"status": status, "error": state["gpu_error"]}
        if status == "loading" or status == "stopped":
            return JSONResponse(status_code=503, content=body)
        if status == "error":
            return JSONResponse(status_code=502, content=body)
        return body

    @app.post("/v1/benchmark/transcriptions", response_model=BenchmarkResponse)
    async def benchmark(payload: BenchmarkRequest, request: Request):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request_start = request.state.received_at
        cold = False
        blob_name = "unresolved"
        duration = None
        chunk_count = None
        try:
            blob_name = blob_reader.resolve_name(
                blob_name=payload.blob_name, blob_url=payload.blob_url
            )
            download_start = time.perf_counter()
            file_bytes = await blob_reader.download(blob_name)
            download_end = time.perf_counter()
            normalized = await asyncio.to_thread(normalize_audio, file_bytes)
            preprocess_end = time.perf_counter()
            duration = normalized.duration_seconds
            chunk_count = normalized.chunk_count

            async with inference_lock:
                model_start = time.perf_counter()
                inference = await model_client.transcribe(normalized.wav_bytes)
                model_end = time.perf_counter()
            text = inference.text
            output_tokens = inference.output_tokens
            cold = inference.cold
            serialization_start = time.perf_counter()
            provisional_marks = PhaseMarks(
                request_start=request_start,
                download_start=download_start,
                download_end=download_end,
                preprocess_end=preprocess_end,
                model_start=model_start,
                model_end=model_end,
                serialization_start=serialization_start,
                response_end=serialization_start,
            )
            response = BenchmarkResponse(
                request_id=request_id,
                text=text,
                model_id=settings.model_id,
                model_revision=settings.model_revision,
                backend=settings.backend,
                gpu_name=inference.gpu_name,
                cold=cold,
                audio_duration_seconds=duration,
                chunk_count=chunk_count,
                output_tokens=output_tokens,
                timings=build_timing_metrics(provisional_marks, output_tokens),
                phase_marks_ms=build_phase_marks_ms(provisional_marks),
                unavailable_metrics=UNAVAILABLE_METRICS,
            )
            response.model_dump_json()
            response_end = time.perf_counter()
            marks = PhaseMarks(
                request_start=request_start,
                download_start=download_start,
                download_end=download_end,
                preprocess_end=preprocess_end,
                model_start=model_start,
                model_end=model_end,
                serialization_start=serialization_start,
                response_end=response_end,
            )
            timings = build_timing_metrics(marks, output_tokens)
            response.timings = type(response.timings)(**timings)
            response.phase_marks_ms = build_phase_marks_ms(marks)
            _log(
                {
                    "event": "transcription_completed",
                    "request_id": request_id,
                    "model_id": settings.model_id,
                    "model_revision": settings.model_revision,
                    "backend": settings.backend,
                    "gpu_name": inference.gpu_name,
                    "cold": cold,
                    "blob_name": blob_name,
                    "audio_duration_seconds": duration,
                    "chunk_count": chunk_count,
                    "output_tokens": output_tokens,
                    "timings": timings,
                    "errors": [],
                }
            )
            return response
        except ValueError as exc:
            status_code = 400
            error = str(exc)
        except Exception as exc:  # pylint: disable=broad-except
            status_code = 502
            error = f"{type(exc).__name__}: {exc}"
        _log(
            {
                "event": "transcription_failed",
                "request_id": request_id,
                "model_id": settings.model_id,
                "model_revision": settings.model_revision,
                "backend": settings.backend,
                "gpu_name": "unavailable",
                "cold": cold,
                "blob_name": blob_name,
                "audio_duration_seconds": duration,
                "chunk_count": chunk_count,
                "errors": [error],
            }
        )
        raise HTTPException(status_code=status_code, detail=error)

    return app


def production_app() -> FastAPI:
    return create_app()