# Cold start: from about 300 s to about 90 s

## Why it is slow today

The A100 container app pulls a multi-GB image from ACR on every cold start (50+ s), then
loads the model. Measured result: 220-416 s.

## What changes

- **Small image:** ACA pulls a ~1 GB bootstrap image only.
- **Runtime and weights in private Blob:** streamed in parallel at start onto the node's NVMe
  (`EmptyDir` volumes). The runtime is packed once per version.
- **In-process vLLM** with `VLLM_ENFORCE_EAGER=0` (CUDA graphs on): full warm throughput.

## Expected results (A100, scale from zero, 2.45 s clip)

| | Result |
| --- | ---: |
| Wake to ready | 85-89 s (3 trials, West US 3) |
| Warm latency | ~0.31 s per call |
| Max throughput | ~8.4 req/s, one replica |

Set `VLLM_ENFORCE_EAGER=1` for a ~66-76 s cold start, at ~1.9 req/s and ~0.73 s per call.
These are measurements, not guarantees. About 33 s is Azure scheduling and varies.

## One-time setup (do once, before the steps)

1. **Tools** on your Mac/Linux shell: `bash`, `curl`, `jq`, Python 3.10+, Azure CLI 2.60+.
   No Docker or local GPU needed.
2. `az extension add --name containerapp --upgrade`
3. `az login --tenant <tenant-id>`
4. **Azure access:** Owner on the subscription, **A100 quota for 1 replica** in your region
   (portal: Quotas > Container Apps), and a region with `Consumption-GPU-NC24-A100`.
5. **Unzip the kit** and work from its root: `cd meralion-customer-kit-1.0`
6. **Config:** `cp infra/customer.env.example customer.env`, then set `TENANT_ID`,
   `SUBSCRIPTION_ID`, `LOCATION`, `RESOURCE_GROUP`, `RUNTIME_VERSION` (e.g. `v1-vllm-cu130`),
   `AUDIO_FILE` (a short local `.wav`) and `BLOB_NAME` (its name in Blob, e.g. `test/short.wav`).
7. **Python venv** for the timing script: `python3 -m venv .venv`
8. Accept the licences for the model, vLLM and the base images.

## Steps

Use `README.md` (the deployment guide) for detail. Run from the kit root. All steps are safe to re-run.

Re-running on an existing environment: before step 6, make sure no GPU replica is running (the packer needs the single A100). Wait about 5 minutes for `ca-meralion-gpu` to scale to zero, or delete it and let step 7 recreate it. If ACR build fails with `denied: client with IP`, re-run with `ACR_OPEN_WAIT_SECONDS=90`.

1. `bash infra/preflight.sh` and fix every `[FAIL]`.
2. `bash infra/deploy.sh` creates the network, ACR, Blob, identity and environment.
3. `BUILD_ONLY=true bash infra/build-runtime.sh` builds the source image in ACR.
4. `bash infra/seed-model.sh` builds a small image and downloads the weights to Blob (about 5-10 min).
5. `bash infra/upload-audio.sh` uploads a test clip.
6. `BUILD_IMAGE=false bash infra/build-runtime.sh`, then `bash infra/wait-runtime-pack.sh`.
   This packs the runtime to Blob once per `RUNTIME_VERSION`.
7. `bash infra/deploy-fast-start.sh` deploys the GPU app with `VLLM_ENFORCE_EAGER=0` by default.
8. `bash infra/deploy-gateway.sh` deploys the public gateway.
9. `TARGET_SECONDS=90 bash scripts/measure_cold_start.sh` (run 3+ times).

## Requirements

- A100 quota for **1 replica**. Only one GPU replica can run at a time, so delete the temporary
  `ca-runtime-builder` after step 6.
- Owner on the subscription (role assignments are created).
- Region with `Consumption-GPU-NC24-A100`.
- Outbound access to Hugging Face and Docker Hub.

## Caveats

- Cold start with CUDA graphs is only 1-4 s under 90 s. Use `VLLM_ENFORCE_EAGER=1` if you need margin.
- Repack with a new `RUNTIME_VERSION` if Python, torch, vLLM or the GPU host changes.
- The always-on CPU gateway, Premium ACR and storage are billed while idle. The A100 is billed only while running.
- Tested with one short clip. Longer audio takes longer per call.

## Logs Location
- Startup logs: az containerapp logs show -g <rg> -n ca-meralion-gpu --type console --tail 300 prints JSON lines. Each line has startup_phase_completed with a phase and duration_seconds.
- Platform events: --type system shows scheduling, image pull, container start, quota failures and restarts.
- Per-request timings: every /v1/benchmark/transcriptions response has timings (model_request_ms, blob_download_ms, audio_preprocess_ms, server_request_ms) and phase_marks_ms. Check the cold flag too.