"""Truthful benchmark timing calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PhaseMarks:
    request_start: float
    download_start: float
    download_end: float
    preprocess_end: float
    model_start: float
    model_end: float
    serialization_start: float
    response_end: float

    def validate(self) -> None:
        marks = (
            self.request_start,
            self.download_start,
            self.download_end,
            self.preprocess_end,
            self.model_start,
            self.model_end,
            self.serialization_start,
            self.response_end,
        )
        if any(current < previous for previous, current in zip(marks, marks[1:])):
            raise ValueError("phase timing marks must be monotonic")


def milliseconds(start: float, end: float) -> float:
    return round((end - start) * 1000.0, 3)


def build_phase_marks_ms(marks: PhaseMarks) -> dict[str, float]:
    marks.validate()
    return {
        "request_received": 0.0,
        "blob_download_started": milliseconds(marks.request_start, marks.download_start),
        "blob_download_completed": milliseconds(marks.request_start, marks.download_end),
        "audio_preprocess_completed": milliseconds(
            marks.request_start, marks.preprocess_end
        ),
        "model_request_started": milliseconds(marks.request_start, marks.model_start),
        "model_response_completed": milliseconds(marks.request_start, marks.model_end),
        "response_serialization_started": milliseconds(
            marks.request_start, marks.serialization_start
        ),
        "response_serialization_completed": milliseconds(
            marks.request_start, marks.response_end
        ),
    }


def build_timing_metrics(
    marks: PhaseMarks,
    output_tokens: int,
    *,
    model_queue_ms: Optional[float] = None,
    prefill_to_first_token_ms: Optional[float] = None,
    generation_ms: Optional[float] = None,
) -> dict[str, Optional[float]]:
    marks.validate()
    server_request_ms = milliseconds(marks.request_start, marks.response_end)
    generation_tokens_per_second = None
    if generation_ms is not None and generation_ms > 0 and output_tokens > 1:
        generation_tokens_per_second = round(
            (output_tokens - 1) / (generation_ms / 1000.0), 3
        )
    end_to_end_tokens_per_second = (
        round(output_tokens / (server_request_ms / 1000.0), 3)
        if server_request_ms > 0
        else None
    )
    return {
        "blob_download_ms": milliseconds(marks.download_start, marks.download_end),
        "audio_preprocess_ms": milliseconds(marks.download_end, marks.preprocess_end),
        "model_request_ms": milliseconds(marks.model_start, marks.model_end),
        "response_serialization_ms": milliseconds(
            marks.serialization_start, marks.response_end
        ),
        "model_queue_ms": model_queue_ms,
        "prefill_to_first_token_ms": prefill_to_first_token_ms,
        "generation_ms": generation_ms,
        "server_request_ms": server_request_ms,
        "generation_tokens_per_second": generation_tokens_per_second,
        "end_to_end_tokens_per_second": end_to_end_tokens_per_second,
    }