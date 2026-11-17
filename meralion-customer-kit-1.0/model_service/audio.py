"""Audio normalization matching the official MERaLiON package."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass


SAMPLE_RATE = 16000
CHUNK_SECONDS = 30.0


@dataclass(frozen=True)
class NormalizedAudio:
    wav_bytes: bytes
    duration_seconds: float
    chunk_count: int


def normalize_audio(file_bytes: bytes) -> NormalizedAudio:
    import librosa
    import numpy as np
    import soundfile as sf

    with io.BytesIO(file_bytes) as source:
        audio, sample_rate = sf.read(source, dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1).astype(np.float32)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("audio must contain at least one mono-compatible sample")
    if int(sample_rate) != SAMPLE_RATE:
        audio = librosa.resample(
            audio, orig_sr=int(sample_rate), target_sr=SAMPLE_RATE
        ).astype(np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak
    audio = np.clip(audio, -1.0, 1.0)

    output = io.BytesIO()
    sf.write(output, audio, SAMPLE_RATE, format="WAV", subtype="FLOAT")
    duration_seconds = float(audio.shape[0]) / SAMPLE_RATE
    return NormalizedAudio(
        wav_bytes=output.getvalue(),
        duration_seconds=duration_seconds,
        chunk_count=max(1, math.ceil(duration_seconds / CHUNK_SECONDS)),
    )