#!/usr/bin/env bash
# Deploy the fast-start GPU app: a ~1 GB bootstrap image that streams its Python/CUDA
# runtime and the model weights from private Blob in parallel (see doc/cold-start-90s.md).
#
# Prerequisites (one-off, not part of a cold start):
#   * Model snapshot seeded to  <models>/<model_id>/<revision>/   (infra/seed-model.sh)
#   * Runtime packed to         <models>/runtime/<RUNTIME_VERSION>/ (scripts/pack_runtime.py)
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
GPU_REPOSITORY="${GPU_REPOSITORY:-meralion-asr-boot}"
RUNTIME_VERSION="${RUNTIME_VERSION:?Set RUNTIME_VERSION to a packed runtime, e.g. v1-vllm-cu130}"
IMAGE_TAG="${IMAGE_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
BUILD_IMAGE="${BUILD_IMAGE:-true}"
# false: only build the image (e.g. to repack a runtime with infra/build-runtime.sh).
DEPLOY_APP="${DEPLOY_APP:-true}"
# Leave CPU/memory headroom for the platform's per-replica agents. With all 24 cores
# requested, platform readiness reporting lagged the app by 45-60 s under load.
GPU_CPU="${GPU_CPU:-16}"
GPU_MEMORY="${GPU_MEMORY:-128Gi}"

require_commands az jq
use_azure_context

restore_acr_privacy() { acr_restore; }

ENVIRONMENT_ID=$(az containerapp env show -g "$RESOURCE_GROUP" -n "$ENVIRONMENT_NAME" --query id -o tsv)
ENVIRONMENT_LOCATION=$(az containerapp env show -g "$RESOURCE_GROUP" -n "$ENVIRONMENT_NAME" --query location -o tsv)
IDENTITY_JSON=$(az identity show -g "$RESOURCE_GROUP" -n "$IDENTITY_NAME" -o json)
IDENTITY_ID=$(jq -r '.id' <<<"$IDENTITY_JSON")
CLIENT_ID=$(jq -r '.clientId' <<<"$IDENTITY_JSON")
ACR_SERVER=$(az acr show -g "$RESOURCE_GROUP" -n "$ACR_NAME" --query loginServer -o tsv)

if [[ "$BUILD_IMAGE" == "true" ]]; then
  trap restore_acr_privacy EXIT
  az acr update -g "$RESOURCE_GROUP" -n "$ACR_NAME" \
    --public-network-enabled true --default-action Allow --admin-enabled false --output none
  sleep "${ACR_OPEN_WAIT_SECONDS:-45}"
  az acr build --registry "$ACR_NAME" --image "${GPU_REPOSITORY}:${IMAGE_TAG}" \
    --file Dockerfile.bootstrap --build-arg "RUNTIME_VERSION=${RUNTIME_VERSION}" \
    --platform linux/amd64 --timeout 3600 "$SOURCE_ROOT"
  restore_acr_privacy
  trap - EXIT
fi

if [[ "$DEPLOY_APP" != "true" ]]; then
  echo "Built ${ACR_SERVER}/${GPU_REPOSITORY}:${IMAGE_TAG} (runtime ${RUNTIME_VERSION}); app not updated"
  exit 0
fi

# The runtime and weights volumes are EmptyDir: the node-local NVMe sustains about 1.85 GB/s of
# writes, while the container overlay root manages 0.5-0.8 GB/s, which made extraction disk-bound.
MANIFEST=$(mktemp "${TMPDIR:-/tmp}/meralion-fast-gpu.XXXXXX")
trap 'rm -f "$MANIFEST"' EXIT
# Readiness is reported by the app itself (/health/ready answers 503 while loading), so the
# manifest deliberately omits startup/readiness probes: with them, ingress learned about the
# replica long after it was ready.
jq -n \
  --arg environmentId "$ENVIRONMENT_ID" \
  --arg location "$ENVIRONMENT_LOCATION" \
  --arg profile "$GPU_PROFILE_NAME" \
  --arg identityId "$IDENTITY_ID" \
  --arg registry "$ACR_SERVER" \
  --arg image "${ACR_SERVER}/${GPU_REPOSITORY}:${IMAGE_TAG}" \
  --arg suffix "fast-$(date -u +%m%d%H%M%S)" \
  --arg clientId "$CLIENT_ID" \
  --arg runtimeVersion "$RUNTIME_VERSION" \
  --arg enforceEager "${VLLM_ENFORCE_EAGER:-0}" \
  --arg modelBlobUrl "https://${STORAGE_NAME}.blob.core.windows.net/models" \
  --argjson cpu "$GPU_CPU" \
  --arg memory "$GPU_MEMORY" \
  '{
    location: $location,
    identity: {type: "UserAssigned", userAssignedIdentities: {($identityId): {}}},
    properties: {
      environmentId: $environmentId,
      workloadProfileName: $profile,
      configuration: {
        activeRevisionsMode: "Single",
        ingress: {external: false, targetPort: 8000, transport: "auto", allowInsecure: false},
        registries: [{server: $registry, identity: $identityId}]
      },
      template: {
        revisionSuffix: $suffix,
        containers: [{
          name: "gpu-inference",
          image: $image,
          env: [
            {name: "AZURE_CLIENT_ID", value: $clientId},
            {name: "RUNTIME_VERSION", value: $runtimeVersion},
            {name: "VLLM_ENFORCE_EAGER", value: $enforceEager},
            {name: "MODEL_BLOB_URL", value: $modelBlobUrl}
          ],
          volumeMounts: [
            {volumeName: "runtime", mountPath: "/usr/local/lib/python3.12/dist-packages"},
            {volumeName: "weights", mountPath: "/app/model-cache"}
          ],
          resources: {cpu: $cpu, memory: $memory, gpu: 1}
        }],
        volumes: [
          {name: "runtime", storageType: "EmptyDir"},
          {name: "weights", storageType: "EmptyDir"}
        ],
        scale: {minReplicas: 0, maxReplicas: 1, rules: [{name: "http-one", http: {metadata: {concurrentRequests: "1"}}}]}
      }
    }
  }' >"$MANIFEST"

if az containerapp show -g "$RESOURCE_GROUP" -n "$GPU_APP_NAME" --output none 2>/dev/null; then
  az containerapp update -g "$RESOURCE_GROUP" -n "$GPU_APP_NAME" --yaml "$MANIFEST" --output none
else
  az containerapp create -g "$RESOURCE_GROUP" -n "$GPU_APP_NAME" --yaml "$MANIFEST" --output none
fi
echo "Deployed ${GPU_APP_NAME} with ${ACR_SERVER}/${GPU_REPOSITORY}:${IMAGE_TAG} (runtime ${RUNTIME_VERSION})"
echo "Measure with: TARGET_SECONDS=90 bash scripts/measure_cold_start.sh"
