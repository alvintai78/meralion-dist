"""In-process Hugging Face Transformers client for MERaLiON-3-ASR.

This avoids the vLLM sidecar, whose process start-up, engine initialization and
first-request overhead dominate cold start for a single-sequence ASR service.
It reuses the official package's prompt, chunking, audio normalization and
post-processing so transcripts follow the same fixed-30s-window contract.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
from pathlib import Path
import threading

from .startup import startup_phase

logger = logging.getLogger("meralion_hf")


class TransformersModelClient:
    def __init__(
        self,
        model_id: str,
        model_revision: str,
        model_local_path: str,
        batch_size: int = 4,
    ):
        self.model_id = model_id
        self.model_revision = model_revision
        self._model_local_path = model_local_path
        self._batch_size = batch_size
        self._asr = None
        self._tokenizer = None
        self._lock = threading.Lock()

    async def start(self) -> None:
        await asyncio.to_thread(self._load)

    def _load(self) -> None:
        with startup_phase("hf_imports"):
            import torch
            import transformers
            from meralion_3_asr import Meralion3ASR
            from meralion_3_asr.backends.transformers_backend import TransformersBackend
            from meralion_3_asr.prompts import build_messages

        # Weight files may still be landing; the runtime imports above overlap that transfer.
        self._wait_for_weights()
        dtype = torch.bfloat16
        with startup_phase("hf_processor_load"):
            processor = transformers.AutoProcessor.from_pretrained(
                self._model_local_path, trust_remote_code=True, local_files_only=True
            )
        with startup_phase("hf_model_load"):
            model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
                self._model_local_path,
                trust_remote_code=True,
                local_files_only=True,
                dtype=dtype,
                # Gemma-2 logit soft-capping is not applied correctly under sdpa.
                attn_implementation=os.getenv("HF_ATTN_IMPLEMENTATION", "eager"),
                device_map={"": 0},
            ).eval()
        backend = TransformersBackend.__new__(TransformersBackend)
        backend.device = torch.device("cuda:0")
        backend.dtype = dtype
        backend.batch_size = self._batch_size
        backend.processor = processor
        backend.model = model
        backend._prompt = processor.tokenizer.apply_chat_template(  # pylint: disable=protected-access
            build_messages(), tokenize=False, add_generation_prompt=True
        )
        self._asr = Meralion3ASR(backend)
        self._tokenizer = processor.tokenizer

    def _wait_for_weights(self) -> None:
        import time

        marker = os.getenv("WEIGHTS_VERIFIED_FILE")
        if not marker:
            return
        downloaded = Path(marker).with_suffix(".downloaded")
        failed = Path(marker).with_suffix(".failed")
        with startup_phase("weights_downloaded_wait"):
            while not downloaded.exists():
                if failed.exists():
                    raise RuntimeError(f"model weights failed verification: {failed.read_text()}")
                time.sleep(0.05)

    async def close(self) -> None:
        return None

    async def wait_ready(self) -> None:
        if self._asr is None:
            raise RuntimeError("TransformersModelClient has not been started")

    async def transcribe(self, wav_bytes: bytes) -> str:
        if self._asr is None:
            raise RuntimeError("TransformersModelClient has not been started")
        return await asyncio.to_thread(self._transcribe, wav_bytes)

    def _transcribe(self, wav_bytes: bytes) -> str:
        import soundfile

        with io.BytesIO(wav_bytes) as source:
            samples, rate = soundfile.read(source, dtype="float32", always_2d=False)
        with self._lock:
            return self._asr.transcribe((samples, int(rate)))

    def count_output_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            raise RuntimeError("model tokenizer has not been loaded")
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        special_ids = set(self._tokenizer.all_special_ids)
        return sum(token_id not in special_ids for token_id in token_ids)
