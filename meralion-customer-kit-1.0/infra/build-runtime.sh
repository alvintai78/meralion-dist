#!/usr/bin/env bash
# One-off (per runtime version): build the full vLLM cu130 image, run it once on an A100 so the
# service warms its caches, and pack /usr/local/lib/python3.12 into
# <models>/runtime/<RUNTIME_VERSION>/ for the bootstrap image to stream at cold start.
# Packing must run in the VNet because the storage account has no public access.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
BUILDER_APP_NAME="${BUILDER_APP_NAME:-ca-runtime-builder}"
RUNTIME_VERSION="${RUNTIME_VERSION:?Set RUNTIME_VERSION, e.g. v1-vllm-cu130}"
IMAGE_TAG="${IMAGE_TAG:-runtime-${RUNTIME_VERSION}}"
BUILD_IMAGE="${BUILD_IMAGE:-true}"
# true: only build and push the runtime image, then stop. Use this first, because the model
# seeding job (infra/seed-model.sh) runs this image and the builder app needs the weights.
BUILD_ONLY="${BUILD_ONLY:-false}"
PACK_EXCLUDES="${PACK_EXCLUDES:-}"
# Re-pack mode: run an existing bootstrap image (BUILDER_IMAGE) that fast-starts from the packed
# runtime SOURCE_RUNTIME_VERSION, and pack the trace-pruned result as RUNTIME_VERSION.
BUILDER_IMAGE="${BUILDER_IMAGE:-}"
SOURCE_RUNTIME_VERSION="${SOURCE_RUNTIME_VERSION:-}"
BUILDER_SCRIPT="${BUILDER_SCRIPT:-$SOURCE_ROOT/scripts/build_runtime.sh}"
# true: mount the same NVMe EmptyDir volumes as production (diagnostics on an already packed runtime).
USE_RUNTIME_VOLUMES="${USE_RUNTIME_VOLUMES:-false}"

require_commands az jq
use_azure_context

restore_acr_privacy() { acr_restore; }

# The builder needs the GPU quota (one A100 replica at a time).
replicas=$(az containerapp replica list -g "$RESOURCE_GROUP" -n "$GPU_APP_NAME" --query 'length(@)' -o tsv 2>/dev/null || echo 0)
[[ "$replicas" == "0" ]] || { echo "$GPU_APP_NAME still has $replicas replica(s); wait for scale-to-zero" >&2; exit 1; }

ENVIRONMENT_ID=$(az containerapp env show -g "$RESOURCE_GROUP" -n "$ENVIRONMENT_NAME" --query id -o tsv)
IDENTITY_JSON=$(az identity show -g "$RESOURCE_GROUP" -n "$IDENTITY_NAME" -o json)
IDENTITY_ID=$(jq -r '.id' <<<"$IDENTITY_JSON")
CLIENT_ID=$(jq -r '.clientId' <<<"$IDENTITY_JSON")
ACR_SERVER=$(az acr show -g "$RESOURCE_GROUP" -n "$ACR_NAME" --query loginServer -o tsv)

if [[ -n "$BUILDER_IMAGE" ]]; then
  BUILD_IMAGE=false
  IMAGE_REF="$BUILDER_IMAGE"
else
  IMAGE_REF="${ACR_SERVER}/meralion-asr-runtime:${IMAGE_TAG}"
fi

if [[ "$BUILD_IMAGE" == "true" ]]; then
  trap restore_acr_privacy EXIT
  az acr update -g "$RESOURCE_GROUP" -n "$ACR_NAME" \
    --public-network-enabled true --default-action Allow --admin-enabled false --output none
  sleep "${ACR_OPEN_WAIT_SECONDS:-45}"
  az acr build --registry "$ACR_NAME" --image "meralion-asr-runtime:${IMAGE_TAG}" \
    --file Dockerfile --platform linux/amd64 --timeout 3600 "$SOURCE_ROOT"
  restore_acr_privacy
  trap - EXIT
fi

if [[ "$BUILD_ONLY" == "true" ]]; then
  echo "Built ${IMAGE_REF}"
  echo "Next: bash infra/seed-model.sh"
  exit 0
fi

MANIFEST=$(mktemp "${TMPDIR:-/tmp}/meralion-builder.XXXXXX")
trap 'rm -f "$MANIFEST"' EXIT
jq -n \
  --arg environmentId "$ENVIRONMENT_ID" \
  --arg location "$(az containerapp env show -g "$RESOURCE_GROUP" -n "$ENVIRONMENT_NAME" --query location -o tsv)" \
  --arg profile "$GPU_PROFILE_NAME" \
  --arg identityId "$IDENTITY_ID" \
  --arg registry "$ACR_SERVER" \
  --arg image "$IMAGE_REF" \
  --arg sourceVersion "$SOURCE_RUNTIME_VERSION" \
  --argjson volumes "$USE_RUNTIME_VOLUMES" \
  --arg clientId "$CLIENT_ID" \
  --arg modelBlobUrl "https://${STORAGE_NAME}.blob.core.windows.net/models" \
  --arg version "$RUNTIME_VERSION" \
  --arg excludes "$PACK_EXCLUDES" \
  --rawfile script "$BUILDER_SCRIPT" \
  --rawfile packer "$SOURCE_ROOT/scripts/pack_runtime.py" \
  '{
    location: $location,
    identity: {type: "UserAssigned", userAssignedIdentities: {($identityId): {}}},
    properties: {
      environmentId: $environmentId,
      workloadProfileName: $profile,
      configuration: {
        activeRevisionsMode: "Single",
        registries: [{server: $registry, identity: $identityId}]
      },
      template: {
        containers: [{
          name: "runtime-builder",
          image: $image,
          command: ["bash", "-c", ("mkdir -p /tmp/meralion\ncat > /tmp/meralion/pack_runtime.py <<\"PACK_EOF\"\n" + $packer + "PACK_EOF\n" + $script)],
          env: [
            {name: "AZURE_CLIENT_ID", value: $clientId},
            {name: "MODEL_BLOB_URL", value: $modelBlobUrl},
            {name: "PACK_RUNTIME_VERSION", value: $version},
            {name: "SOURCE_RUNTIME_VERSION", value: $sourceVersion},
            {name: "PACK_EXCLUDES", value: $excludes},
            {name: "INFERENCE_ENGINE", value: "vllm"},
            {name: "WARMUP_TRANSCRIPTION", value: "1"},
            {name: "LISTEN_EARLY", value: "0"}
          ],
          volumeMounts: (if $volumes then [
            {volumeName: "runtime", mountPath: "/usr/local/lib/python3.12/dist-packages"},
            {volumeName: "weights", mountPath: "/app/model-cache"}
          ] else [] end),
          resources: {cpu: 16, memory: "128Gi", gpu: 1}
        }],
        volumes: (if $volumes then [
          {name: "runtime", storageType: "EmptyDir"},
          {name: "weights", storageType: "EmptyDir"}
        ] else [] end),
        scale: {minReplicas: 1, maxReplicas: 1}
      }
    }
  }' >"$MANIFEST"

az containerapp delete -g "$RESOURCE_GROUP" -n "$BUILDER_APP_NAME" --yes --output none 2>/dev/null || true
az containerapp create -g "$RESOURCE_GROUP" -n "$BUILDER_APP_NAME" --yaml "$MANIFEST" --output none
echo "Builder started. Follow with:"
echo "  az containerapp logs show -g $RESOURCE_GROUP -n $BUILDER_APP_NAME --follow"
echo "When it logs RUNTIME_PACK_DONE, delete it:"
echo "  az containerapp delete -g $RESOURCE_GROUP -n $BUILDER_APP_NAME --yes"
