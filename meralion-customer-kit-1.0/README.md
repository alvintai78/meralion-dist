# MERaLiON on Azure Container Apps A100: deployment and cold-start test guide

This kit deploys `MERaLiON/MERaLiON-3-3B-ASR` (vLLM 0.16.0, CUDA 13) into **your own Azure
subscription** and measures how long a scaled-to-zero A100 takes to return its first
transcript. Everything runs from your terminal; no VM and no local GPU or Docker are needed.

## What gets created

| Resource | Purpose |
| --- | --- |
| Resource group, VNet, two subnets | Network for the environment and private endpoints |
| Azure Container Registry (**Premium**, private) | Holds the container images |
| Storage account (private) | Blob containers `audio` (test clips) and `models` (weights + packed runtime) |
| Private endpoints and private DNS for ACR and Blob | Private access from the apps |
| User-assigned managed identity | Pulls images and reads Blob; no secrets or keys |
| Container Apps environment (Consumption + `Consumption-GPU-NC24-A100` profile) | Hosts the apps |
| `ca-meralion` (public gateway, 1 CPU replica always on) | Web UI and API |
| `ca-meralion-gpu` (internal, A100, scales 0 to 1) | Model inference |

```text
You -> ca-meralion (gateway) -> ca-meralion-gpu (A100, scale to zero)
              |                          |
              +-- private Blob: audio, model weights, packed runtime
```

## 1. Prerequisites

| Need | Notes |
| --- | --- |
| macOS or Linux shell with `bash`, `curl`, `jq`, Python 3.10+ | Azure CLI 2.60+ with `az extension add --name containerapp --upgrade` |
| Azure subscription with **A100 consumption quota for 1 replica** | Request under Quotas > Container Apps. Only one GPU replica can run at a time |
| A region offering `Consumption-GPU-NC24-A100` | Checked by the preflight script |
| **Owner** on the subscription, or Contributor + User Access Administrator | The deployment creates role assignments |
| Outbound internet from your laptop and from the environment | Hugging Face (model download) and Docker Hub (vLLM base image) |
| A short `.wav` file to transcribe | Any test clip you are allowed to use |

Accept the licence terms of the model, vLLM and the base images before use.

Costs: the CPU gateway runs all the time, the Premium ACR and storage are billed while they
exist, and the A100 is billed only while a replica runs. Delete the resource group when
finished (section 12).

## 2. Configure

```bash
az login --tenant <your-tenant-id>
cp infra/customer.env.example customer.env
python3 -m venv .venv          # used by the cold-start script for a process-independent clock
```

Edit `customer.env`: set `TENANT_ID`, `SUBSCRIPTION_ID`, `LOCATION`, `RESOURCE_GROUP`,
`RUNTIME_VERSION`, `AUDIO_FILE` and `BLOB_NAME`. Names of the ACR and storage account are
derived from the subscription id and must be globally unique; set `SUFFIX` if the preflight
reports a clash. All commands below read this file, so run them from the kit root.

## 3. Preflight (read-only)

```bash
bash infra/preflight.sh
```

Fix every `[FAIL]`. A100 quota cannot be read from the CLI: confirm it in the portal.

## 4. Create the infrastructure

```bash
bash infra/deploy.sh
```

Creates everything in the table above, including the A100 workload profile. Safe to re-run.

## 5. Build the runtime image (about 5-10 min)

```bash
BUILD_ONLY=true bash infra/build-runtime.sh
```

This builds the ~9 GB image in ACR (no local Docker). ACR is opened to the internet only
during the build and closed again afterwards. It prints the image reference for the next step.

## 6. Download the model weights into private Blob

```bash
bash infra/seed-model.sh
```

The script builds a small image (`Dockerfile.seed`) in ACR, then a short-lived CPU job downloads the pinned model revision from Hugging Face into the
`models` container and verifies checksums. The script waits and reports `Succeeded`.

## 7. Upload your test audio

```bash
bash infra/upload-audio.sh        # uses AUDIO_FILE and BLOB_NAME from customer.env
```

The storage account is opened to your public IP only for the upload and closed again.

## 8. Build and pack the runtime (once per runtime version)

The GPU app does not exist yet, so the A100 is free for the builder.

```bash
BUILD_IMAGE=false bash infra/build-runtime.sh     # starts a temporary A100 builder app
bash infra/wait-runtime-pack.sh                    # waits for RUNTIME_PACK_DONE, deletes the builder
```

The builder warms the service and uploads an 8.5 GB packed Python/CUDA runtime to
`models/runtime/<RUNTIME_VERSION>/`. The packed runtime is specific to Python 3.12, torch
2.9.1 and the A100 host CPU; repack with a new `RUNTIME_VERSION` if any of them changes.

## 9. Deploy the GPU app

```bash
bash infra/deploy-fast-start.sh
```

Builds the ~1 GB bootstrap image and creates `ca-meralion-gpu` (min 0 / max 1 replicas).
At cold start it streams the runtime and weights from Blob onto the node's NVMe.

## 10. Deploy the gateway

```bash
bash infra/deploy-gateway.sh
```

Prints the public URL (`Gateway ready: https://...`).

## 11. Smoke test, then the cold-start benchmark

Smoke test (wakes the GPU, so the first call takes a while):

Replace the two values on the first lines with the gateway URL printed in step 10 and the
`BLOB_NAME` from `customer.env`. Paste the block otherwise as is (it has no comments, so it
also works in zsh).

```bash
URL=https://your-gateway-fqdn
BLOB_NAME=your-audio-file.wav
curl -X POST "$URL/v1/model/wake"
until curl -s "$URL/v1/model/status" | grep -q '"status":"ready"'; do sleep 2; done
curl -H 'content-type: application/json' -d "{\"blob_name\":\"$BLOB_NAME\"}" \
  "$URL/v1/benchmark/transcriptions"
```

Check the returned `text` is a sensible transcription and `gpu_name` contains `A100`. A
`BlobNotFound` error means `BLOB_NAME` does not match a file uploaded in step 7.

Benchmark:

```bash
TARGET_SECONDS=90 bash scripts/measure_cold_start.sh > run1.json
```

The script first waits until the GPU app has **zero replicas** (scale-down takes about 5
minutes after the last request) and then starts the clock, sends the wake request, polls
until ready, and completes one transcription. The clock covers GPU scheduling, image pull,
runtime/weights fetch, model load and warm-up, and the first transcription. Setup and the
wait for scale-down are not timed.

Output fields: `wake_to_ready_seconds`, `cold_to_transcript_seconds`, `target_met`. Exit
code 2 means the target was missed. Run **at least 5 trials**, spaced out and at different
times of day (`run1.json` ... `run5.json`), and keep every file.

Reference result from the original environment: 67.9-77.4 s wake-to-ready in 5 of 5 trials, and
about 7.5 requests/s warm throughput on a 2.45 s clip. About 30-35 s of the cold start is Azure's
own scheduling and image pull and varies by region and time. Treat results as measurements, not a latency guarantee.

## 12. Clean up

```bash
az group delete -n <RESOURCE_GROUP> --yes --no-wait
```

To keep the environment instead: confirm `az containerapp list -g <RESOURCE_GROUP> -o table`
shows no `ca-runtime-builder`, and `az acr show -n <ACR> --query publicNetworkAccess` and
the same for the storage account report `Disabled`.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `denied: client with IP ... is not allowed access` during an image build | The ACR firewall change had not propagated. Re-run with `ACR_OPEN_WAIT_SECONDS=90` |
| `Temporary failure resolving 'archive.ubuntu.com'` or `failed to export image` | Transient ACR build error. Re-run the step |
| Builder: `ca-meralion-gpu still has N replica(s)` | Wait for scale-to-zero, or deactivate the old revision |
| Replica `ActivationFailed ... exceeded quota: consumption-gpu-a100` | Only one A100 replica is allowed. Delete `ca-runtime-builder` or any other GPU app/revision |
| `could not read blob runtime/<v>/manifest.json` | Step 8 did not finish for that `RUNTIME_VERSION` |
| `/health/ready` stays 503, then `weights failed verification` | Re-run step 6 |
| `tar: ... Cannot utime / Cannot change mode` | Repack with the kit's `scripts/pack_runtime.py` (step 8) |
| Upload keeps failing in step 7 | You need `Storage Blob Data Contributor` on the storage account; allow a few minutes for roles to propagate |
| Logs | `az containerapp logs show -g <rg> -n ca-meralion-gpu --type console --tail 200` (system events: `--type system`) |
