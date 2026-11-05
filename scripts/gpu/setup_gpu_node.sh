#!/usr/bin/env bash
# scripts/gpu/setup_gpu_node.sh — prepare a GPU node for the v2 harness.
#   1. CUDA toolkit + driver must be present (nvcc builds the custom kernels:
#      AES, SHA-256, Monte Carlo, LBM, solver, bilateral). If missing: scripts/gpu/install_cuda.sh
#   2. v2 harness requirements (CUDA torch wheel, pynvml, ninja, ...)
#   3. Then:  make test_bench && make bench_smoke_cuda
#            python3 scripts/calibrate.py --device-config configs/devices/<DEVICE>.json
#            make bench_suite DEVICE=<DEVICE>
# NOTE: the CUDA paths have not yet been run on hardware -- the first session on every GPU
# type must be a validation (smoke) session, not a collection.
set -euo pipefail
cd "$(dirname "$0")/../.."
command -v nvidia-smi >/dev/null || { echo "nvidia-smi missing: install the driver (scripts/gpu/install_cuda.sh)"; exit 1; }
command -v nvcc >/dev/null || [ -x /usr/local/cuda/bin/nvcc ] || { echo "nvcc missing: install the CUDA toolkit (scripts/gpu/install_cuda.sh)"; exit 1; }
export PATH="/usr/local/cuda/bin:$PATH"
nvidia-smi --query-gpu=name,driver_version,memory.total,persistence_mode,ecc.mode.current --format=csv
bash scripts/setup_v2_harness.sh gpu
