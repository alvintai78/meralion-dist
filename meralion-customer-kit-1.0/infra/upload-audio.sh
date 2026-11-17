#!/usr/bin/env bash
# Uploads the benchmark audio to the private storage account. The account has no public access, so
# this opens it to your public IP only for the duration of the upload, and always closes it again.
#   AUDIO_FILE=./samples/short.wav BLOB_NAME=compatibility/short.wav bash infra/upload-audio.sh
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
: "${AUDIO_FILE:?Set AUDIO_FILE to a local .wav file}"
: "${BLOB_NAME:?Set BLOB_NAME to the blob name to use, e.g. compatibility/short.wav}"
[[ -f "$AUDIO_FILE" ]] || { echo "AUDIO_FILE not found: $AUDIO_FILE" >&2; exit 1; }

require_commands az curl
use_azure_context

MY_IP="${MY_IP:-$(curl --silent --show-error --fail --max-time 15 https://api.ipify.org)}"
STORAGE_ID=$(az storage account show -g "$RESOURCE_GROUP" -n "$STORAGE_NAME" --query id -o tsv)

restore_storage_privacy() {
  az storage account network-rule remove -g "$RESOURCE_GROUP" --account-name "$STORAGE_NAME" --ip-address "$MY_IP" --output none 2>/dev/null || true
  az storage account update -g "$RESOURCE_GROUP" -n "$STORAGE_NAME" \
    --public-network-access Disabled --default-action Deny --output none
}
trap restore_storage_privacy EXIT

# Your own data-plane role; skipped quietly when you already have one or cannot assign it.
USER_OBJECT_ID=$(az ad signed-in-user show --query id -o tsv 2>/dev/null || true)
if [[ -n "$USER_OBJECT_ID" ]]; then
  az role assignment create --assignee-object-id "$USER_OBJECT_ID" --assignee-principal-type User \
    --role "Storage Blob Data Contributor" --scope "$STORAGE_ID" --output none 2>/dev/null || true
fi

az storage account update -g "$RESOURCE_GROUP" -n "$STORAGE_NAME" \
  --public-network-access Enabled --default-action Deny --output none
az storage account network-rule add -g "$RESOURCE_GROUP" --account-name "$STORAGE_NAME" --ip-address "$MY_IP" --output none

# Firewall and role changes take a few minutes to propagate.
for attempt in $(seq 1 12); do
  if az storage blob upload --auth-mode login --account-name "$STORAGE_NAME" \
    --container-name "$BLOB_CONTAINER_NAME" --name "$BLOB_NAME" --file "$AUDIO_FILE" \
    --overwrite --output none 2>/dev/null; then
    echo "Uploaded ${AUDIO_FILE} to ${BLOB_CONTAINER_NAME}/${BLOB_NAME}"
    exit 0
  fi
  echo "Upload attempt ${attempt}/12 not accepted yet; retrying in 30s" >&2
  sleep 30
done
echo "Upload failed. Check you have 'Storage Blob Data Contributor' on ${STORAGE_NAME}." >&2
exit 1
