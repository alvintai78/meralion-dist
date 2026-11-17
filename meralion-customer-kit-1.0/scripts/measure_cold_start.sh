#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../infra" && pwd)/lib.sh"
POLL_SECONDS="${POLL_SECONDS:-1}"
SCALE_ZERO_POLL_SECONDS="${SCALE_ZERO_POLL_SECONDS:-10}"
BLOB_NAME="${BLOB_NAME:?Set BLOB_NAME to an audio blob in the audio container (see infra/upload-audio.sh)}"
TARGET_SECONDS="${TARGET_SECONDS:-90}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
SCALE_ZERO_TIMEOUT_SECONDS="${SCALE_ZERO_TIMEOUT_SECONDS:-1800}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-1800}"

if [[ ! "$TARGET_SECONDS" =~ ^[0-9]+$ ]] || (( TARGET_SECONDS < 1 )); then
  echo "TARGET_SECONDS must be a positive integer" >&2
  exit 1
fi

for command_name in az curl jq "$PYTHON"; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "Required command not found: $command_name" >&2
    exit 1
  }
done

"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), "Python >=3.10 is required for a process-independent macOS monotonic clock"'

active_tenant=$(az account show --subscription "$SUBSCRIPTION_ID" --query tenantId -o tsv)
if [[ "$active_tenant" != "$TENANT_ID" ]]; then
  echo "Azure tenant mismatch; refusing to measure another deployment" >&2
  exit 1
fi

if [[ -z "${BASE_URL:-}" ]]; then
  gateway_fqdn=$(az containerapp show --subscription "$SUBSCRIPTION_ID" --resource-group "$RESOURCE_GROUP" \
    --name "$CPU_APP_NAME" --query properties.configuration.ingress.fqdn --output tsv)
  [[ -n "$gateway_fqdn" ]] || { echo "Could not find the gateway app ${CPU_APP_NAME}; set BASE_URL" >&2; exit 1; }
  BASE_URL="https://${gateway_fqdn}"
fi

now_epoch_ms() {
  "$PYTHON" -c 'import time; print(time.monotonic_ns() // 1_000_000)'
}

replica_count() {
  az containerapp replica list \
    --subscription "$SUBSCRIPTION_ID" \
    --resource-group "$RESOURCE_GROUP" \
    --name "$GPU_APP_NAME" \
    --query 'length(@)' \
    --output tsv
}

echo "Waiting for ${GPU_APP_NAME} to reach zero replicas..." >&2
scale_zero_deadline=$(( $(date +%s) + SCALE_ZERO_TIMEOUT_SECONDS ))
while true; do
  replicas=$(replica_count)
  if [[ "$replicas" == "0" ]]; then
    break
  fi
  if (( $(date +%s) >= scale_zero_deadline )); then
    echo "GPU app did not scale to zero within ${SCALE_ZERO_TIMEOUT_SECONDS} seconds" >&2
    exit 1
  fi
  echo "GPU replicas: ${replicas}; waiting ${SCALE_ZERO_POLL_SECONDS}s" >&2
  sleep "$SCALE_ZERO_POLL_SECONDS"
done

cold_confirmed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
started_wall_seconds=$(date +%s)
start_ms=$(now_epoch_ms)

curl --http1.1 --silent --show-error --fail \
  --request POST \
  --max-time 30 \
  "$BASE_URL/v1/model/wake" >/dev/null

ready_deadline=$(( $(date +%s) + READY_TIMEOUT_SECONDS ))
polls=0
response_file=$(mktemp "${TMPDIR:-/tmp}/meralion-cold-status.XXXXXX")
transcript_file=$(mktemp "${TMPDIR:-/tmp}/meralion-cold-transcript.XXXXXX")
trap 'rm -f "$response_file" "$transcript_file"' EXIT
while true; do
  polls=$((polls + 1))
  if ! http_status=$(curl --http1.1 --silent --show-error \
    --output "$response_file" \
    --write-out '%{http_code}' \
    --max-time 5 \
    "$BASE_URL/v1/model/status"); then
    echo "Status transport failed; retrying until readiness deadline" >&2
    http_status="000"
    status="unavailable"
    error=""
  elif jq -e 'type == "object" and (.status | type == "string")' "$response_file" >/dev/null 2>&1; then
    status=$(jq -r '.status' "$response_file")
    error=$(jq -r '.error // empty' "$response_file")
  else
    echo "Unexpected status response (HTTP ${http_status}); retrying until readiness deadline" >&2
    status="unavailable"
    error=""
  fi

  echo "poll=${polls} http=${http_status} status=${status}" >&2
  if [[ "$http_status" == "200" && "$status" == "ready" ]]; then
    break
  fi
  if [[ "$status" == "error" ]]; then
    echo "GPU wake failed: ${error:-unknown error}" >&2
    exit 1
  fi
  if (( $(date +%s) >= ready_deadline )); then
    echo "GPU model did not become ready within ${READY_TIMEOUT_SECONDS} seconds" >&2
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

end_ms=$(now_epoch_ms)
ready_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
wake_to_ready_ms=$((end_ms - start_ms))

payload=$(jq -n --arg blob "$BLOB_NAME" '{blob_name:$blob}')
curl --http1.1 --silent --show-error --fail \
  --max-time "$READY_TIMEOUT_SECONDS" \
  --header 'content-type: application/json' \
  --data "$payload" \
  "$BASE_URL/v1/benchmark/transcriptions" >"$transcript_file"
completed_ms=$(now_epoch_ms)
completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
if ! jq -e '.cold == true and .output_tokens > 0 and (.text | length > 0)' "$transcript_file" >/dev/null; then
  echo "Expected a non-empty first (cold) GPU transcription; trial is invalid" >&2
  exit 1
fi
cold_to_transcript_ms=$((completed_ms - start_ms))
wall_elapsed_ms=$(( ($(date +%s) - started_wall_seconds) * 1000 ))
clock_difference_ms=$((cold_to_transcript_ms - wall_elapsed_ms))
if (( cold_to_transcript_ms < 0 || clock_difference_ms < -5000 || clock_difference_ms > 5000 )); then
  echo "Clock consistency check failed; trial is invalid (monotonic=${cold_to_transcript_ms}ms, wall=${wall_elapsed_ms}ms)" >&2
  exit 1
fi

jq -n \
  --arg metric "gpu_cold_to_completed_transcription" \
  --arg app "$GPU_APP_NAME" \
  --arg endpoint "$BASE_URL" \
  --arg coldConfirmedAt "$cold_confirmed_at" \
  --arg startedAt "$started_at" \
  --arg readyAt "$ready_at" \
  --arg completedAt "$completed_at" \
  --arg blobName "$BLOB_NAME" \
  --argjson startingReplicas 0 \
  --argjson wakeToReadyMs "$wake_to_ready_ms" \
  --argjson coldToTranscriptMs "$cold_to_transcript_ms" \
  --argjson targetSeconds "$TARGET_SECONDS" \
  --slurpfile transcript "$transcript_file" \
  --argjson polls "$polls" \
  '{
    metric: $metric,
    gpu_app: $app,
    endpoint: $endpoint,
    cold_confirmed_at_utc: $coldConfirmedAt,
    started_at_utc: $startedAt,
    ready_at_utc: $readyAt,
    starting_replicas: $startingReplicas,
    wake_to_ready_ms: $wakeToReadyMs,
    wake_to_ready_seconds: ($wakeToReadyMs / 1000),
    completed_at_utc: $completedAt,
    blob_name: $blobName,
    cold_to_transcript_ms: $coldToTranscriptMs,
    cold_to_transcript_seconds: ($coldToTranscriptMs / 1000),
    target_seconds: $targetSeconds,
    target_met: ($coldToTranscriptMs <= ($targetSeconds * 1000)),
    transcription: $transcript[0],
    poll_count: $polls
  }'

if (( cold_to_transcript_ms > TARGET_SECONDS * 1000 )); then
  echo "Cold-to-transcript target missed: ${cold_to_transcript_ms}ms > ${TARGET_SECONDS}s" >&2
  exit 2
fi