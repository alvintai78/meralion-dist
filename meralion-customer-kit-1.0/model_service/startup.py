"""Structured, process-correlated startup phase measurements."""

from contextlib import contextmanager
import json
import logging
import os
import time


def startup_event(event: str, **fields) -> None:
    logging.getLogger("meralion_startup").info(json.dumps({
        "event": event,
        "pid": os.getpid(),
        "at_unix_ms": time.time_ns() // 1_000_000,
        **fields,
    }))


@contextmanager
def startup_phase(phase: str, **fields):
    started = time.monotonic()
    startup_event("startup_phase_started", phase=phase, **fields)
    try:
        yield
    except BaseException:
        startup_event(
            "startup_phase_failed", phase=phase,
            duration_seconds=time.monotonic() - started, **fields,
        )
        raise
    else:
        startup_event(
            "startup_phase_completed", phase=phase,
            duration_seconds=time.monotonic() - started, **fields,
        )
