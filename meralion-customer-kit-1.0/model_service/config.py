"""Environment-backed service configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_id: str = "MERaLiON/MERaLiON-3-3B-ASR"
    model_revision: str = "9f226d738dee62c60a2f43b0ab7b9a2585a132c3"
    model_local_path: str | None = None
    backend: str = "meralion-3-asr-0.1.0/vllm-0.16.0-cu130"
    sidecar_url: str = "http://127.0.0.1:8001"
    audio_blob_url: str = ""
    azure_client_id: str | None = None

    @classmethod
    def from_environment(cls, *, require_blob: bool = True) -> "Settings":
        return cls(
            model_id=os.getenv("MODEL_ID", cls.model_id),
            model_revision=os.getenv("MODEL_REVISION", cls.model_revision),
            model_local_path=os.getenv("MODEL_LOCAL_PATH"),
            backend=os.getenv("MODEL_BACKEND", cls.backend),
            sidecar_url=os.getenv("MODEL_SIDECAR_URL", cls.sidecar_url),
            audio_blob_url=(
                os.environ["AUDIO_BLOB_URL"]
                if require_blob
                else os.getenv("AUDIO_BLOB_URL", "")
            ),
            azure_client_id=os.getenv("AZURE_CLIENT_ID"),
        )