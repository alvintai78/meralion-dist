"""In-process vLLM client for MERaLiON-3-ASR.

``meralion-3-asr serve`` starts a gateway, an API server and an engine-core process, each of
which imports vLLM again. Cold start pays for that (about 25 s on the A100 profile), so this
client runs the official package's vLLM backend (same plugin, prompt, sampling parameters and
30 s chunking) directly inside the GPU service process with a single-process engine.
"""

from __future__ import annotations

import json
import os
import threading

from .hf_model import TransformersModelClient
from .startup import startup_phase


# mode 0 = no torch.compile; capture one full decode graph for the single sequence.
DEFAULT_COMPILATION_CONFIG = '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1]}'


def engine_kwargs() -> dict:
    """Engine arguments tuned for one-sequence ASR and a short cold start."""
    kwargs = {
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "max_model_len": int(os.getenv("VLLM_MAX_MODEL_LEN", "1300")),
        "max_num_seqs": 1,
        "limit_mm_per_prompt": {"audio": 1},
        "attention_backend": os.getenv("VLLM_ATTENTION_BACKEND", "FLASHINFER"),
        # Eager decoding is CPU-launch bound (about 4x slower per call). Full CUDA graphs for the
        # one-sequence decode step fix that and cost about 1 s of capture; torch.compile is what
        # added 14 s, so it stays off (see DEFAULT_COMPILATION_CONFIG). VLLM_ENFORCE_EAGER=1
        # restores eager mode.
        "enforce_eager": os.getenv("VLLM_ENFORCE_EAGER", "0") == "1",
        "disable_log_stats": True,
    }
    compilation = os.getenv("VLLM_COMPILATION_CONFIG", DEFAULT_COMPILATION_CONFIG)
    if compilation and not kwargs["enforce_eager"]:
        kwargs["compilation_config"] = json.loads(compilation)
    if os.getenv("VLLM_SKIP_MM_PROFILING", "1") == "1":
        # The service only ever sees one 30 s audio chunk per request, so the worst-case
        # multimodal profiling pass is unnecessary.
        kwargs["skip_mm_profiling"] = True
    # A fixed KV-cache size skips the memory-profiling pass: 2 GiB holds about 20k tokens, far
    # more than one 1300-token sequence. Set it to 0 to size the cache from GPU memory instead.
    kv_cache_bytes = int(os.getenv("VLLM_KV_CACHE_MEMORY_BYTES", str(2 << 30)))
    if kv_cache_bytes:
        kwargs["kv_cache_memory_bytes"] = kv_cache_bytes
    else:
        kwargs["gpu_memory_utilization"] = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.85"))
    return kwargs


def prewarm_cuda(with_nccl: bool) -> None:
    """Create the CUDA context (and first NCCL group) while vLLM's Python imports run.

    Both are native and release the GIL, so they overlap the import-bound main thread instead
    of adding their 3-6 s to the engine start.
    """
    with startup_phase("cuda_prewarm", nccl=with_nccl):
        import torch

        torch.cuda.init()
        probe = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        (probe @ probe).sum().item()
        torch.nn.functional.conv1d(
            torch.randn(1, 128, 3000, device="cuda", dtype=torch.bfloat16),
            torch.randn(128, 128, 3, device="cuda", dtype=torch.bfloat16),
        ).sum().item()
        if with_nccl:
            import torch.distributed as dist

            dist.init_process_group(
                "nccl", init_method="tcp://127.0.0.1:%d" % free_port(), world_size=1, rank=0
            )


def free_port() -> int:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def speed_up_processor_loading() -> None:
    """Stop vLLM and the MERaLiON plugin from rebuilding the 256k-token tokenizer repeatedly.

    vLLM builds the multimodal processor several times (profiling, input processing, first
    request). Each build re-parses ``tokenizer.json`` (about 1 s) and then formats the processor
    with ``repr`` only to log it (another 1.5 s of deep copies). Sharing the parsed tokenizer
    and skipping the log formatting removes about 10 s from startup and the first request.
    """
    import copy

    from transformers import PreTrainedTokenizerBase
    from transformers.processing_utils import ProcessorMixin

    ProcessorMixin.__repr__ = lambda self: f"{type(self).__name__}()"
    load = PreTrainedTokenizerBase.from_pretrained.__func__
    parsed: dict = {}

    def cached_load(cls, name, *args, **kwargs):
        key = (cls, str(name), args, repr(sorted(kwargs.items())))
        if key not in parsed:
            parsed[key] = load(cls, name, *args, **kwargs)
        # A shallow copy shares the parsed Rust tokenizer but keeps Python attributes separate.
        return copy.copy(parsed[key])

    PreTrainedTokenizerBase.from_pretrained = classmethod(cached_load)


class VllmModelClient(TransformersModelClient):
    def _load(self) -> None:
        prewarm = None
        if os.getenv("VLLM_CUDA_PREWARM", "1") != "0":
            prewarm = threading.Thread(
                target=prewarm_cuda,
                args=(os.getenv("VLLM_NCCL_PREWARM", "1") == "1",),
                name="cuda-prewarm",
                daemon=True,
            )
            prewarm.start()
        # One process: no engine-core subprocess that would import vLLM a second time.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
        with startup_phase("vllm_imports"):
            from vllm import LLM, SamplingParams
            from meralion_3_asr import Meralion3ASR
            from meralion_3_asr._vllm_plugin import register
            from meralion_3_asr.backends.vllm_backend import VllmBackend
            from meralion_3_asr.prompts import VLLM_SAMPLING_PARAMS, build_prompt

            register()
            from transformers import AutoTokenizer

            if os.getenv("VLLM_FAST_PROCESSOR", "1") == "1":
                speed_up_processor_loading()

        with startup_phase("vllm_tokenizer_load"):
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_local_path, trust_remote_code=True, local_files_only=True
            )

        # Weight files may still be landing; the imports above overlap that transfer.
        self._wait_for_weights()
        if prewarm is not None:
            prewarm.join()
        with startup_phase("vllm_engine_init"):
            llm = LLM(model=self._model_local_path, tokenizer=self._model_local_path, **engine_kwargs())
        backend = VllmBackend.__new__(VllmBackend)
        backend.llm = llm
        backend.sampling_params = SamplingParams(**VLLM_SAMPLING_PARAMS)
        backend._prompt = build_prompt()  # pylint: disable=protected-access
        self._asr = Meralion3ASR(backend)
