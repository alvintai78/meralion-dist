"""Public benchmark API models."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blob_name: Optional[str] = None
    blob_url: Optional[str] = None

    @model_validator(mode="after")
    def exactly_one_source(self):
        if bool(self.blob_name) == bool(self.blob_url):
            raise ValueError("provide exactly one of blob_name or blob_url")
        return self


class TimingMetrics(BaseModel):
    blob_download_ms: float = Field(ge=0)
    audio_preprocess_ms: float = Field(ge=0)
    model_request_ms: float = Field(ge=0)
    response_serialization_ms: float = Field(ge=0)
    model_queue_ms: Optional[float] = Field(default=None, ge=0)
    prefill_to_first_token_ms: Optional[float] = Field(default=None, ge=0)
    generation_ms: Optional[float] = Field(default=None, ge=0)
    server_request_ms: float = Field(ge=0)
    generation_tokens_per_second: Optional[float] = Field(default=None, ge=0)
    end_to_end_tokens_per_second: Optional[float] = Field(default=None, ge=0)


class BenchmarkResponse(BaseModel):
    request_id: str
    text: str
    model_id: str
    model_revision: str
    backend: str
    gpu_name: str
    cold: bool
    audio_duration_seconds: float
    chunk_count: int
    output_tokens: int
    timings: TimingMetrics
    phase_marks_ms: dict[str, float]
    unavailable_metrics: dict[str, str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_id: str
    model_revision: str
    backend: str