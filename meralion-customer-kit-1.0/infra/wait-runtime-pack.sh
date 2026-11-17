#!/usr/bin/env bash
# Waits for the runtime builder app to finish packing the runtime to Blob, then deletes it
# (it holds the only A100 replica). Run after: RUNTIME_VERSION=... bash infra/build-runtime.sh
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
BUILDER_APP_NAME="${BUILDER_APP_NAME:-ca-runtime-builder}"
RUNTIME_VERSION="${RUNTIME_VERSION:?Set RUNTIME_VERSION to the version being packed}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-2400}"

require_commands az
use_azure_context

deadline=$(( $(date +%s) + TIMEOUT_SECONDS ))
while true; do
  logs=$(az containerapp logs show -g "$RESOURCE_GROUP" -n "$BUILDER_APP_NAME" --type console --tail 300 --format text 2>/dev/null || true)
  if grep -q "RUNTIME_PACK_DONE ${RUNTIME_VERSION}" <<<"$logs"; then
    echo "Runtime ${RUNTIME_VERSION} packed; deleting ${BUILDER_APP_NAME}"
    az containerapp delete -g "$RESOURCE_GROUP" -n "$BUILDER_APP_NAME" --yes --output none
    exit 0
  fi
  if (( $(date +%s) >= deadline )); then
    echo "Timed out waiting for RUNTIME_PACK_DONE. Inspect:" >&2
    echo "  az containerapp logs show -g $RESOURCE_GROUP -n $BUILDER_APP_NAME --type console --tail 200" >&2
    exit 1
  fi
  echo "Waiting for RUNTIME_PACK_DONE ($(( deadline - $(date +%s) ))s left)" >&2
  sleep 20
done
