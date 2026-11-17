#!/usr/bin/env bash
# Shared configuration for infra/*.sh and scripts/measure_cold_start.sh. Source it; do not run it.
#
# Settings come from the environment first, then from customer.env (KEY=VALUE lines, parsed
# rather than sourced, so the file can never run code). Copy infra/customer.env.example to
# customer.env at the repository root, or point CUSTOMER_ENV at another file.

LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$LIB_DIR/.." && pwd)"
CUSTOMER_ENV="${CUSTOMER_ENV:-$REPO_ROOT/customer.env}"

load_customer_env() {
  local line key value
  [[ -f "$CUSTOMER_ENV" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    key="${BASH_REMATCH[1]}"
    value="${BASH_REMATCH[2]}"
    value="${value%\"}"; value="${value#\"}"
    value="${value%\'}"; value="${value#\'}"
    if ! printenv "$key" >/dev/null 2>&1; then
      export "$key=$value"
    fi
  done <"$CUSTOMER_ENV"
}
load_customer_env

: "${SUBSCRIPTION_ID:?Set SUBSCRIPTION_ID in customer.env (copy infra/customer.env.example)}"
: "${TENANT_ID:?Set TENANT_ID in customer.env (copy infra/customer.env.example)}"

LOCATION="${LOCATION:-westus3}"
RESOURCE_GROUP="${RESOURCE_GROUP:-rg-meralion}"
# ACR and storage names are globally unique; the first 8 hex characters of the subscription
# id keep them unique per customer. Override SUFFIX to run a second copy in one subscription.
SUFFIX="${SUFFIX:-$(printf '%s' "${SUBSCRIPTION_ID//-/}" | cut -c1-8)}"

VNET_NAME="${VNET_NAME:-vnet-meralion}"
ACA_SUBNET_NAME="${ACA_SUBNET_NAME:-snet-aca}"
PRIVATE_ENDPOINT_SUBNET_NAME="${PRIVATE_ENDPOINT_SUBNET_NAME:-snet-private-endpoints}"
IDENTITY_NAME="${IDENTITY_NAME:-id-meralion-aca}"
ENVIRONMENT_NAME="${ENVIRONMENT_NAME:-cae-meralion}"
CPU_APP_NAME="${CPU_APP_NAME:-ca-meralion}"
APP_NAME="${APP_NAME:-$CPU_APP_NAME}"
GPU_APP_NAME="${GPU_APP_NAME:-ca-meralion-gpu}"
GPU_PROFILE_NAME="${GPU_PROFILE_NAME:-A100GPU-Profile}"
GPU_PROFILE_TYPE="${GPU_PROFILE_TYPE:-Consumption-GPU-NC24-A100}"
ACR_NAME="${ACR_NAME:-acrmeralion${SUFFIX}}"
STORAGE_NAME="${STORAGE_NAME:-stmeralion${SUFFIX}}"
BLOB_CONTAINER_NAME="${BLOB_CONTAINER_NAME:-audio}"
SOURCE_ROOT="${SOURCE_ROOT:-$REPO_ROOT}"

require_commands() {
  local command_name
  for command_name in "$@"; do
    command -v "$command_name" >/dev/null 2>&1 || {
      echo "Required command not found: $command_name" >&2
      exit 1
    }
  done
}

use_azure_context() {
  az account set --subscription "$SUBSCRIPTION_ID" >/dev/null
  [[ "$(az account show --query tenantId -o tsv)" == "$TENANT_ID" ]] || {
    echo "Azure tenant mismatch (expected ${TENANT_ID}); run 'az login --tenant ${TENANT_ID}'" >&2
    exit 1
  }
}

# The ACR is private. Builds need it open for a short time; always restore it afterwards.
acr_open() {
  az acr update -g "$RESOURCE_GROUP" -n "$ACR_NAME" \
    --public-network-enabled true --default-action Allow --admin-enabled false --output none
  sleep "${ACR_OPEN_WAIT_SECONDS:-45}"
}

acr_restore() {
  az acr update -g "$RESOURCE_GROUP" -n "$ACR_NAME" \
    --public-network-enabled false --default-action Deny --admin-enabled false --output none
}
