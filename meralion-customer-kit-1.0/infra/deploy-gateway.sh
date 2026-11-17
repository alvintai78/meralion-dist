#!/usr/bin/env bash
# Builds the CPU gateway/frontend image and deploys it as the public app, wired to the internal
# GPU app (which infra/deploy-fast-start.sh must have created first).
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
CPU_REPOSITORY="${CPU_REPOSITORY:-meralion-benchmark-cpu}"
IMAGE_TAG="${IMAGE_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
BUILD_IMAGE="${BUILD_IMAGE:-true}"

require_commands az jq curl
use_azure_context

ENVIRONMENT_ID=$(az containerapp env show -g "$RESOURCE_GROUP" -n "$ENVIRONMENT_NAME" --query id -o tsv)
ENVIRONMENT_LOCATION=$(az containerapp env show -g "$RESOURCE_GROUP" -n "$ENVIRONMENT_NAME" --query location -o tsv)
IDENTITY_JSON=$(az identity show -g "$RESOURCE_GROUP" -n "$IDENTITY_NAME" -o json)
IDENTITY_ID=$(jq -r '.id' <<<"$IDENTITY_JSON")
CLIENT_ID=$(jq -r '.clientId' <<<"$IDENTITY_JSON")
ACR_SERVER=$(az acr show -g "$RESOURCE_GROUP" -n "$ACR_NAME" --query loginServer -o tsv)
GPU_FQDN=$(az containerapp show -g "$RESOURCE_GROUP" -n "$GPU_APP_NAME" --query properties.configuration.ingress.fqdn -o tsv 2>/dev/null || true)
[[ -n "$GPU_FQDN" ]] || {
  echo "GPU app ${GPU_APP_NAME} not found; run infra/deploy-fast-start.sh first" >&2
  exit 1
}

if [[ "$BUILD_IMAGE" == "true" ]]; then
  trap acr_restore EXIT
  acr_open
  az acr build --registry "$ACR_NAME" --image "${CPU_REPOSITORY}:${IMAGE_TAG}" \
    --file Dockerfile.cpu --platform linux/amd64 --timeout 3600 "$SOURCE_ROOT"
  acr_restore
  trap - EXIT
fi

MANIFEST=$(mktemp "${TMPDIR:-/tmp}/meralion-gateway.XXXXXX")
trap 'rm -f "$MANIFEST"' EXIT
jq -n \
  --arg environmentId "$ENVIRONMENT_ID" \
  --arg location "$ENVIRONMENT_LOCATION" \
  --arg identityId "$IDENTITY_ID" \
  --arg registry "$ACR_SERVER" \
  --arg image "${ACR_SERVER}/${CPU_REPOSITORY}:${IMAGE_TAG}" \
  --arg suffix "cpu-$(date -u +%m%d%H%M%S)" \
  --arg clientId "$CLIENT_ID" \
  --arg blobUrl "https://${STORAGE_NAME}.blob.core.windows.net/${BLOB_CONTAINER_NAME}" \
  --arg modelUrl "https://${GPU_FQDN}" \
  '{
    location: $location,
    identity: {type: "UserAssigned", userAssignedIdentities: {($identityId): {}}},
    properties: {
      environmentId: $environmentId,
      workloadProfileName: "Consumption",
      configuration: {
        activeRevisionsMode: "Single",
        ingress: {external: true, targetPort: 8000, transport: "auto", allowInsecure: false},
        registries: [{server: $registry, identity: $identityId}]
      },
      template: {
        revisionSuffix: $suffix,
        containers: [{
          name: "benchmark-frontend",
          image: $image,
          env: [
            {name: "AUDIO_BLOB_URL", value: $blobUrl},
            {name: "AZURE_CLIENT_ID", value: $clientId},
            {name: "MODEL_SIDECAR_URL", value: $modelUrl}
          ],
          resources: {cpu: 2, memory: "4Gi"},
          probes: [
            {type: "Startup", httpGet: {path: "/health/live", port: 8000, scheme: "HTTP"}, periodSeconds: 5, timeoutSeconds: 3, failureThreshold: 30},
            {type: "Readiness", httpGet: {path: "/health/ready", port: 8000, scheme: "HTTP"}, periodSeconds: 10, timeoutSeconds: 3, failureThreshold: 3},
            {type: "Liveness", httpGet: {path: "/health/live", port: 8000, scheme: "HTTP"}, periodSeconds: 10, timeoutSeconds: 3, failureThreshold: 3}
          ]
        }],
        scale: {minReplicas: 1, maxReplicas: 1}
      }
    }
  }' >"$MANIFEST"

if az containerapp show -g "$RESOURCE_GROUP" -n "$CPU_APP_NAME" --output none 2>/dev/null; then
  az containerapp update -g "$RESOURCE_GROUP" -n "$CPU_APP_NAME" --yaml "$MANIFEST" --output none
else
  az containerapp create -g "$RESOURCE_GROUP" -n "$CPU_APP_NAME" --yaml "$MANIFEST" --output none
fi

CPU_FQDN=$(az containerapp show -g "$RESOURCE_GROUP" -n "$CPU_APP_NAME" --query properties.configuration.ingress.fqdn -o tsv)
curl --http1.1 --silent --show-error --fail --retry 30 --retry-delay 5 --retry-all-errors \
  --max-time 300 "https://${CPU_FQDN}/health/ready" >/dev/null
echo "Gateway ready: https://${CPU_FQDN}"
echo "Measure with: TARGET_SECONDS=90 bash scripts/measure_cold_start.sh"
