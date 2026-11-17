# Source image for the packed runtime (see doc/cold-start-90s.md). It is not deployed: the
# runtime builder app runs it once, warms the caches and uploads /usr/local/lib/python3.12
# to Blob with scripts/pack_runtime.py. Production starts from Dockerfile.bootstrap.
FROM vllm/vllm-openai:v0.16.0-x86_64-cu130@sha256:e5cc1e1d812f33abc596779431b55e46a4dbd71cb606805d335f487d17fe123c

USER root
WORKDIR /app

ARG MODEL_ID=MERaLiON/MERaLiON-3-3B-ASR
ARG MODEL_REVISION=9f226d738dee62c60a2f43b0ab7b9a2585a132c3

COPY requirements.txt /app/requirements.txt
RUN uv pip install --system --no-cache-dir -r /app/requirements.txt
COPY scripts/prune_flashinfer_cache.py /app/prune_flashinfer_cache.py
# Pre-compiled bytecode is part of the packed runtime. Without it every cold start spends
# seconds compiling thousands of vLLM/torch/transformers modules, because the pack is
# extracted into a fresh filesystem.
RUN python3 /app/prune_flashinfer_cache.py && \
    python3 -m compileall -q -j 0 --invalidation-mode unchecked-hash \
        /usr/local/lib/python3.12/dist-packages /usr/lib/python3.12 && \
    python3 -c "import torch, vllm, transformers, meralion_3_asr, librosa; \
assert torch.__version__.startswith('2.9.1') and torch.version.cuda.startswith('13'), (torch.__version__, torch.version.cuda); \
assert vllm.__version__.startswith('0.16.0'), vllm.__version__"

RUN groupadd --gid 10001 app && \
    useradd --uid 10001 --gid app --create-home --home-dir /app/home app && \
    mkdir -p /app/model-cache /tmp/meralion && \
    chown -R app:app /app /tmp/meralion

USER app
ENV MODEL_ID=${MODEL_ID} \
    MODEL_REVISION=${MODEL_REVISION} \
    HF_HOME=/app/model-cache \
    HUGGINGFACE_HUB_CACHE=/app/model-cache/hub \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TMPDIR=/tmp/meralion \
    VLLM_WORKER_MULTIPROC_METHOD=spawn

COPY --chown=app:app model_service /app/model_service
COPY --chown=app:app container/sitecustomize.py /app/sitecustomize.py
COPY --chown=app:app container/vllm /app/bin/vllm

EXPOSE 8000
ENTRYPOINT ["python3", "-m", "model_service.entrypoint"]
