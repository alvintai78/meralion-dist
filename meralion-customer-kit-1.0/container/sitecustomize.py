"""Hide first-use audio-library compilation behind vLLM's own startup.

numba JIT-compiles librosa's gufuncs on the first audio request, which costs
several seconds. Importing and exercising librosa on a background thread while
``vllm serve`` initializes moves that cost off the critical path.
"""

import os
import sys
import threading


def _prewarm() -> None:
    try:
        import io
        import wave

        import librosa
        import numpy as np

        for rate in (44100, 16000):
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(rate)
                output.writeframes(np.zeros(rate // 2, dtype="<i2").tobytes())
            buffer.seek(0)
            samples, source_rate = librosa.load(buffer, sr=None)
            if source_rate != 16000:
                librosa.resample(samples, orig_sr=source_rate, target_sr=16000)
    except Exception:  # pylint: disable=broad-except
        pass


# Both the gateway process and vLLM's API server decode audio with librosa.
_program = os.path.basename(sys.argv[0])
if (_program in ("vllm", "meralion-3-asr") and "serve" in sys.argv[1:]) or _program == "uvicorn":
    threading.Thread(target=_prewarm, name="audio-prewarm", daemon=True).start()
