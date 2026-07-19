#!/usr/bin/env bash
# run_sweep_saxpy.sh — full SAXPY sweep across size x precision x gpu-count.
# Emits one CSV under results/. Run after the driver is loaded (post-reboot).
#
#   ./scripts/run_sweep_saxpy.sh
#   SIZES=33554432,67108864 ./scripts/run_sweep_saxpy.sh
set -euo pipefail
cd "$(dirname "$0")/.."

command -v nvidia-smi >/dev/null || { echo "nvidia-smi missing — reboot to load the driver first"; exit 1; }
[ -x bin/gpu_saxpy_bench ] || make saxpy

NGPU=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1 | tr -d ' ')
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="results/sweep_saxpy_${STAMP}.csv"
SIZES="${SIZES:-33554432,67108864,134217728,268435456,536870912,1073741824}"
PRECS="${PRECS:-fp32,fp64}"
GPUS="${GPUS:-$(seq -s, 1 "$NGPU")}"
ITERS="${ITERS:-20}"

echo "==> persistence mode on (stable clocks)"
sudo nvidia-smi -pm 1 >/dev/null || echo "   (could not set persistence mode; continuing)"

echo "==> sweep: sizes=$SIZES precs=$PRECS gpus=$GPUS  -> $OUT"
echo "    Note: SAXPY has no bf16 support (skipped inside the benchmark)."

# 1) split mode — partitioned across GPUs
./bin/gpu_saxpy_bench --mode split    --sizes "$SIZES" --precisions "$PRECS" \
    --gpus "$GPUS" --iters "$ITERS" --validate --csv "$OUT" --tag split

# 2) replicas — perfect-scaling ceiling (each GPU runs the full problem)
./bin/gpu_saxpy_bench --mode replicas --sizes "$SIZES" --precisions "$PRECS" \
    --gpus "$GPUS" --iters "$ITERS" --csv "$OUT" --tag replicas

echo "==> done. CSV: $OUT"
echo "    columns: $(head -1 "$OUT")"
