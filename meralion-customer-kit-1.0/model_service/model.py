"""Clients for the internal GPU service and official vLLM sidecar."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import time

from .startup import startup_phase

@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    output_tokens: int
    gpu_name: str
    cold: bool


class ModelClient:
    def __init__(
        self,
        base_url: str,
        model_id: str,
        model_revision: str,
    ):
        self._base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.model_revision = model_revision
        self._http = None

    async def start(self) -> None:
        import httpx

        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(1800.0, connect=30.0),
        )

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()

    async def transcribe(self, wav_bytes: bytes) -> TranscriptionResult:
        if self._http is None:
            raise RuntimeError("ModelClient has not been started")
        await self.wait_ready()
        response = await self._http.post(
            "/v1/internal/transcriptions",
            content=wav_bytes,
            headers={"content-type": "audio/wav"},
        )
        response.raise_for_status()
        body = response.json()
        return TranscriptionResult(
            text=body["text"],
            output_tokens=body["output_tokens"],
            gpu_name=body["gpu_name"],
            cold=body["cold"],
        )

    async def wait_ready(self, timeout_seconds: int = 1800) -> None:
        import httpx

        if self._http is None:
            raise RuntimeError("ModelClient has not been started")
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                # A short per-request timeout abandons requests the ingress is holding for a
                # scaling replica, so each poll is a fresh request that can be routed directly.
                response = await self._http.get(
                    "/health/ready", timeout=httpx.Timeout(3.0, connect=3.0)
                )
                if response.status_code == 200 and response.json().get("model_loaded"):
                    return
                if response.status_code not in (408, 429, 502, 503, 504):
                    response.raise_for_status()
            except httpx.TransportError as exc:
                logging.getLogger("meralion_benchmark").debug("GPU readiness retry: %s", exc)
            await asyncio.sleep(1)
        raise TimeoutError("GPU model did not become ready within 1800 seconds")


class OfficialModelClient:
    def __init__(
        self,
        base_url: str,
        model_id: str,
        model_revision: str,
        model_local_path: str,
    ):
        self._base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.model_revision = model_revision
        self._model_local_path = model_local_path
        self._http = None
        self._tokenizer = None

    async def start(self) -> None:
        with startup_phase("tokenizer_import"):
            import httpx
            from transformers import AutoTokenizer

        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(900.0, connect=10.0),
        )
        async def load_tokenizer():
            with startup_phase("tokenizer_load"):
                self._tokenizer = await asyncio.to_thread(
                    AutoTokenizer.from_pretrained,
                    self._model_local_path,
                    trust_remote_code=True,
                    local_files_only=True,
                )

        async def await_sidecar():
            with startup_phase("sidecar_wait"):
                deadline = time.monotonic() + 900
                while time.monotonic() < deadline:
                    try:
                        await self.wait_ready()
                        return
                    except (httpx.TransportError, httpx.HTTPStatusError):
                        await asyncio.sleep(0.25)
                raise TimeoutError("official model sidecar did not become ready")

        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(load_tokenizer())
            tasks.create_task(await_sidecar())

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()

    async def wait_ready(self) -> None:
        if self._http is None:
            raise RuntimeError("OfficialModelClient has not been started")
        response = await self._http.get("/v1/models")
        response.raise_for_status()

    async def transcribe(self, wav_bytes: bytes) -> str:
        if self._http is None:
            raise RuntimeError("OfficialModelClient has not been started")
        response = await self._http.post(
            "/v1/audio/transcriptions",
            data={"model": self.model_id},
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
        )
        response.raise_for_status()
        text = response.json().get("text")
        if not isinstance(text, str):
            raise RuntimeError("official sidecar returned no transcription text")
        return text

    def count_output_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            raise RuntimeError("model tokenizer has not been loaded")
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        special_ids = set(self._tokenizer.all_special_ids)
        return sum(token_id not in special_ids for token_id in token_ids)