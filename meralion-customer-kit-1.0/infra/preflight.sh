#!/usr/bin/env bash
# Read-only checks to run before deploying: tools, Azure context, resource providers, A100 profile
# availability in the region, name availability, and your permissions. Changes nothing.
set -uo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

failures=0
warnings=0
pass() { echo "  [ok]   $*"; }
fail() { echo "  [FAIL] $*"; failures=$((failures + 1)); }
warn() { echo "  [warn] $*"; warnings=$((warnings + 1)); }

echo "Tools"
for tool in az jq curl python3; do
  if command -v "$tool" >/dev/null 2>&1; then pass "$tool"; else fail "$tool not installed"; fi
done
command -v az >/dev/null 2>&1 || { echo "Install Azure CLI first."; exit 1; }
if az extension show --name containerapp >/dev/null 2>&1; then
  pass "az containerapp extension"
else
  fail "az extension add --name containerapp --upgrade"
fi
if [[ -x "$REPO_ROOT/.venv/bin/python" ]] && "$REPO_ROOT/.venv/bin/python" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
  pass ".venv Python >= 3.10 (used by scripts/measure_cold_start.sh)"
else
  warn "no .venv with Python >= 3.10; create one or set PYTHON for scripts/measure_cold_start.sh"
fi

echo "Azure context"
active_tenant=$(az account show --subscription "$SUBSCRIPTION_ID" --query tenantId -o tsv 2>/dev/null || true)
if [[ -z "$active_tenant" ]]; then
  fail "not logged in, or subscription ${SUBSCRIPTION_ID} not visible (az login --tenant ${TENANT_ID})"
  echo; echo "Stopping: fix the Azure login first."; exit 1
elif [[ "$active_tenant" != "$TENANT_ID" ]]; then
  fail "subscription belongs to tenant ${active_tenant}, not ${TENANT_ID}"
else
  pass "tenant ${TENANT_ID}, subscription ${SUBSCRIPTION_ID}"
fi
az account set --subscription "$SUBSCRIPTION_ID" >/dev/null 2>&1

echo "Resource providers"
for provider in Microsoft.App Microsoft.ContainerRegistry Microsoft.Storage Microsoft.Network \
  Microsoft.ManagedIdentity Microsoft.OperationalInsights; do
  state=$(az provider show --namespace "$provider" --query registrationState -o tsv 2>/dev/null || true)
  if [[ "$state" == "Registered" ]]; then
    pass "$provider"
  else
    warn "$provider is '${state:-unknown}' (az provider register --namespace $provider --wait; deploy.sh registers Microsoft.App)"
  fi
done

echo "A100 workload profile in ${LOCATION}"
profile=$(az rest --method get \
  --url "https://management.azure.com/subscriptions/${SUBSCRIPTION_ID}/providers/Microsoft.App/locations/${LOCATION}/availableManagedEnvironmentsWorkloadProfileTypes?api-version=2025-01-01" \
  --query "value[?name=='${GPU_PROFILE_TYPE}'] | [0].properties.gpus" -o tsv 2>/dev/null || true)
if [[ "$profile" == "1" ]]; then
  pass "${GPU_PROFILE_TYPE} offered (1 GPU)"
else
  fail "${GPU_PROFILE_TYPE} not offered in ${LOCATION}; pick another LOCATION"
fi
warn "A100 quota cannot be read from the CLI. Confirm in the portal (Quotas > Container Apps) that this subscription has consumption GPU A100 quota for 1 replica in ${LOCATION}."

echo "Names"
if az group show -n "$RESOURCE_GROUP" >/dev/null 2>&1; then
  pass "resource group ${RESOURCE_GROUP} exists (re-run is fine; deploy.sh is idempotent)"
else
  pass "resource group ${RESOURCE_GROUP} will be created"
fi
if az acr show -g "$RESOURCE_GROUP" -n "$ACR_NAME" >/dev/null 2>&1; then
  pass "ACR ${ACR_NAME} already exists in the resource group"
elif [[ "$(az acr check-name -n "$ACR_NAME" --query nameAvailable -o tsv 2>/dev/null)" == "true" ]]; then
  pass "ACR name ${ACR_NAME} is available"
else
  fail "ACR name ${ACR_NAME} is taken or invalid (set SUFFIX or ACR_NAME)"
fi
if az storage account show -g "$RESOURCE_GROUP" -n "$STORAGE_NAME" >/dev/null 2>&1; then
  pass "storage ${STORAGE_NAME} already exists in the resource group"
elif [[ "$(az storage account check-name -n "$STORAGE_NAME" --query nameAvailable -o tsv 2>/dev/null)" == "true" ]]; then
  pass "storage name ${STORAGE_NAME} is available"
else
  fail "storage name ${STORAGE_NAME} is taken or invalid (set SUFFIX or STORAGE_NAME)"
fi

echo "Permissions"
roles=$(az role assignment list --all --include-inherited --include-groups \
  --assignee "$(az ad signed-in-user show --query id -o tsv 2>/dev/null)" \
  --scope "/subscriptions/${SUBSCRIPTION_ID}" --query '[].roleDefinitionName' -o tsv 2>/dev/null || true)
if grep -qx "Owner" <<<"$roles"; then
  pass "Owner on the subscription"
elif grep -qx "Contributor" <<<"$roles" && grep -qE "^(User Access Administrator|Role Based Access Control Administrator)$" <<<"$roles"; then
  pass "Contributor plus a role-assignment role"
else
  warn "could not confirm Owner, or Contributor + User Access Administrator. deploy.sh creates role assignments and will fail without that right."
fi

echo
if (( failures > 0 )); then
  echo "Preflight FAILED: ${failures} problem(s), ${warnings} warning(s)."
  exit 1
fi
echo "Preflight passed with ${warnings} warning(s)."
