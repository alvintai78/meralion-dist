#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
JOB_NAME="${JOB_NAME:-meralion-model-seed}"
# GPU_IMAGE (optional) overrides the small seed image built below. The 9 GB runtime image
# fails to pull on a Consumption CPU node ("no space left on device").
GPU_IMAGE="${GPU_IMAGE:-}"
SEED_REPOSITORY="${SEED_REPOSITORY:-meralion-seed}"
IMAGE_TAG="${IMAGE_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"

require_commands az jq
use_azure_context

if [[ -z "$GPU_IMAGE" ]]; then
  trap acr_restore EXIT
  acr_open
  az acr build --registry "$ACR_NAME" --image "${SEED_REPOSITORY}:${IMAGE_TAG}" \
    --file Dockerfile.seed --platform linux/amd64 --timeout 1800 "$SOURCE_ROOT"
  acr_restore
  trap - EXIT
  GPU_IMAGE="${ACR_NAME}.azurecr.io/${SEED_REPOSITORY}:${IMAGE_TAG}"
fi

STORAGE_ID=$(az storage account show -g "$RESOURCE_GROUP" -n "$STORAGE_NAME" --query id -o tsv)
IDENTITY_JSON=$(az identity show -g "$RESOURCE_GROUP" -n "$IDENTITY_NAME" -o json)
IDENTITY_ID=$(jq -r '.id' <<<"$IDENTITY_JSON")
CLIENT_ID=$(jq -r '.clientId' <<<"$IDENTITY_JSON")
PRINCIPAL_ID=$(jq -r '.principalId' <<<"$IDENTITY_JSON")
CONTAINER_ID="${STORAGE_ID}/blobServices/default/containers/models"

# Management-plane creation does not require opening the private data plane.
az rest --method put \
  --url "https://management.azure.com${CONTAINER_ID}?api-version=2023-05-01" \
  --body '{"properties":{"publicAccess":"None"}}' --output none
az role assignment create \
  --assignee-object-id "$PRINCIPAL_ID" --assignee-principal-type ServicePrincipal \
  --role "Storage Blob Data Contributor" --scope "$CONTAINER_ID" --output none

if az containerapp job show -g "$RESOURCE_GROUP" -n "$JOB_NAME" --output none 2>/dev/null; then
  az containerapp job update -g "$RESOURCE_GROUP" -n "$JOB_NAME" --image "$GPU_IMAGE" --output none
else
  az containerapp job create -g "$RESOURCE_GROUP" -n "$JOB_NAME" \
    --environment "$ENVIRONMENT_NAME" --trigger-type Manual \
    --workload-profile-name Consumption --cpu 4 --memory 8Gi \
    --replica-timeout 3600 --replica-retry-limit 1 \
    --mi-user-assigned "$IDENTITY_ID" \
    --registry-server "${ACR_NAME}.azurecr.io" --registry-identity "$IDENTITY_ID" \
    --image "$GPU_IMAGE" \
    --env-vars "AZURE_CLIENT_ID=${CLIENT_ID}" \
      "MODEL_BLOB_URL=https://${STORAGE_NAME}.blob.core.windows.net/models" \
    --output none
fi

EXECUTION="${SEED_EXECUTION_NAME:-}"
if [[ -z "$EXECUTION" ]]; then
  EXECUTION=$(az containerapp job start -g "$RESOURCE_GROUP" -n "$JOB_NAME" --query name -o tsv)
fi
deadline=$(( $(date +%s) + 3900 ))
while true; do
  status=$(az containerapp job execution show -g "$RESOURCE_GROUP" -n "$JOB_NAME" \
    --job-execution-name "$EXECUTION" --query properties.status -o tsv)
  echo "Model seed execution ${EXECUTION}: ${status}" >&2
  case "$status" in
    Succeeded) break ;;
    Failed|Stopped|Degraded)
      echo "Model seeding failed; inspect ${JOB_NAME} execution logs" >&2
      exit 1 ;;
  esac
  if (( $(date +%s) >= deadline )); then
    echo "Model seeding exceeded its deadline" >&2
    exit 1
  fi
  sleep 10
done
