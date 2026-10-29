#!/usr/bin/env bash
# Noise harness worker bootstrap; runs inside the GPU pod (worker/gpu_worker.yaml). No installation or download.
# Startup phases and errors go to artifacts/queue/bootstrap_logs/<pod>.log.
set -euo pipefail
LOUD_REPO="${LOUD_REPO:-/mnt/disk/loud}"
LOUD_QUEUE_DIR="${LOUD_QUEUE_DIR:-$LOUD_REPO/artifacts/queue}"
export LOUD_REPO LOUD_QUEUE_DIR
LOUD_BOOT_PHASE=initialization
mkdir -p "$LOUD_QUEUE_DIR/bootstrap_logs"
exec > >(tee -a "$LOUD_QUEUE_DIR/bootstrap_logs/$(hostname).log") 2>&1
phase(){ LOUD_BOOT_PHASE="$1"; printf '%s phase=%s\n' "$(date -u +%FT%TZ)" "$1"; }
trap 'LOUD_EXIT_CODE=$?; printf "%s startup exit=%s phase=%s\n" "$(date -u +%FT%TZ)" "$LOUD_EXIT_CODE" "$LOUD_BOOT_PHASE"' EXIT

LOUD_PYTHON="${LOUD_PYTHON:-$(command -v python3)}"
# Compilation caches stay on the pod's own disk; every job of this worker reuses them.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/loud-inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/loud-triton}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HUB_OFFLINE=1

phase runtime
printf 'Runtime: %s\n' "${LOUD_RUNTIME_IMAGE:-image not recorded}"
"$LOUD_PYTHON" - <<'PY'
import importlib.metadata as m
import platform
print('python', platform.python_version(), flush=True)
for name in ('torch', 'triton', 'numpy', 'PyYAML'):
    try:
        print(name, m.version(name), flush=True)
    except m.PackageNotFoundError:
        print(name, 'MISSING', flush=True)
import torch
print('cuda', torch.version.cuda, 'available', torch.cuda.is_available(), torch.cuda.get_device_name(0), flush=True)
PY
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,driver_version --format=csv,noheader
fi
df -h /tmp /mnt/disk /dev/shm

# Fail fast if the pinned external code, downloads or data splits are not exactly what the harness expects.
phase inputs
cd "$LOUD_REPO"
PYTHONPATH="$LOUD_REPO" "$LOUD_PYTHON" -m loud.refs --check

# Freeze the worker loop itself; each job runs from the snapshot recorded when it was submitted.
phase worker-snapshot
LOUD_WORKER_CODE="$(PYTHONPATH="$LOUD_REPO" "$LOUD_PYTHON" -c 'from loud.snapshot import snapshot; print(snapshot())')"
printf 'Worker code: %s\n' "$LOUD_WORKER_CODE"
export PYTHONPATH="$LOUD_WORKER_CODE" LOUD_CODE_DIR="$LOUD_WORKER_CODE"
cd "$LOUD_WORKER_CODE"

phase worker
LOUD_WORKERS_PER_POD="${LOUD_WORKERS_PER_POD:-1}"  # GPUs in the pod (gpu_worker_2.yaml: 2)
LOUD_JOBS_PER_GPU="${LOUD_JOBS_PER_GPU:-1}"        # workers sharing each GPU: the small jobs leave an H100 mostly idle
if [[ $LOUD_WORKERS_PER_POD == 1 && $LOUD_JOBS_PER_GPU == 1 ]]; then
  exec "$LOUD_PYTHON" -u -m loud.worker --queue "$LOUD_QUEUE_DIR"
fi
# LOUD_JOBS_PER_GPU workers per GPU, each pinned to its GPU, sharing its CPUs and claiming only jobs whose declared
# memory (gpu_memory_gb) fits beside the others' on that GPU. A pod deletion's SIGTERM is passed to every worker,
# which gives its job back; the pod ends when all have exited.
trap - EXIT
pids=()
threads=$(( ${OMP_NUM_THREADS:-8} / (LOUD_WORKERS_PER_POD * LOUD_JOBS_PER_GPU) ))
for ((gpu = 0; gpu < LOUD_WORKERS_PER_POD; gpu++)); do
  for ((slot = 0; slot < LOUD_JOBS_PER_GPU; slot++)); do
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=$(( threads > 0 ? threads : 1 )) LOUD_GPU_SLOT="$(hostname)-gpu$gpu" \
      "$LOUD_PYTHON" -u -m loud.worker --queue "$LOUD_QUEUE_DIR" --worker-id "$(hostname)-gpu$gpu-$slot" &
    pids+=($!)
  done
done
trap 'kill -TERM "${pids[@]}" 2>/dev/null || true' TERM INT
status=0
for pid in "${pids[@]}"; do
  until wait "$pid"; do  # a trapped signal interrupts the wait while the worker is still releasing its job
    code=$?
    kill -0 "$pid" 2>/dev/null || { status=$code; break; }
  done
done
exit "$status"
