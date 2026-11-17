#!/usr/bin/env bash
# Runs inside the runtime builder container (infra/build-runtime.sh): start the real service,
# let it serve warm-up transcriptions so Triton/numba/vLLM caches exist, then pack the
# Python/CUDA runtime into Blob. Pack only after readiness so the caches are included.
set -euo pipefail

: "${PACK_RUNTIME_VERSION:?Set PACK_RUNTIME_VERSION}"
: "${MODEL_BLOB_URL:?Set MODEL_BLOB_URL}"

# With SOURCE_RUNTIME_VERSION the service fast-starts from that packed runtime and the new
# version is a pruned re-pack of it. Without it the service runs from the full runtime image
# (RUNTIME_VERSION must then stay unset: it would switch the service to the fast-start path).
if [[ -n "${SOURCE_RUNTIME_VERSION:-}" ]]; then
  export RUNTIME_VERSION="$SOURCE_RUNTIME_VERSION"
else
  unset RUNTIME_VERSION
fi
RUNTIME_TRACE_FILE=/tmp/meralion/runtime-trace.json python3 -m model_service.entrypoint &
service=$!
trap 'kill "$service" 2>/dev/null || true' EXIT

for _ in $(seq 1 900); do
  if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=2)" 2>/dev/null; then
    break
  fi
  kill -0 "$service" 2>/dev/null || { echo "service exited before becoming ready" >&2; exit 1; }
  sleep 1
done
python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=2)"

PACK_TRACE_FILE=/tmp/meralion/runtime-trace.json python3 /tmp/meralion/pack_runtime.py "$PACK_RUNTIME_VERSION" "${SHARDS:-24}"
echo "RUNTIME_PACK_DONE ${PACK_RUNTIME_VERSION}"
# Keep the app alive (no restart loop) until the infra script deletes it.
sleep infinity
