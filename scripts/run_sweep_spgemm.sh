#!/usr/bin/env bash
# run_sweep_spgemm.sh — full SpGEMM sweep across size x precision x density x gpu-count.
# Emits one CSV under results/. Run after the driver is loaded (post-reboot).
#
#   ./scripts/run_sweep_spgemm.sh
#   SIZES=1024,2048 ./scripts/run_sweep_spgemm.sh
set -euo pipefail
cd "$(dirname "$0")/.."

command -v nvidia-smi >/dev/null || { echo "nvidia-smi missing — reboot to load the driver first"; exit 1; }
[ -x bin/gpu_spgemm_bench ] || make spgemm

NGPU=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1 | tr -d ' ')
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="results/sweep_spgemm_${STAMP}.csv"
SIZES="${SIZES:-512,1024,2048,4096,8192,16384}"
PRECS="${PRECS:-fp32,fp64}"
GPUS="${GPUS:-$(seq -s, 1 "$NGPU")}"
DENSITIES="${DENSITIES:-0.01,0.05,0.1,0.25,0.5}"
ITERS="${ITERS:-10}"

echo "==> persistence mode on (stable clocks)"
sudo nvidia-smi -pm 1 >/dev/null || echo "   (could not set persistence mode; continuing)"

echo "==> sweep: sizes=$SIZES precs=$PRECS gpus=$GPUS densities=$DENSITIES  -> $OUT"
echo "    Note: SpGEMM has no bf16 support."

# 1) split mode — partitioned across GPUs
./bin/gpu_spgemm_bench --mode split    --sizes "$SIZES" --precisions "$PRECS" \
    --density "$DENSITIES" --gpus "$GPUS" --iters "$ITERS" --validate --csv "$OUT" --tag split

# 2) replicas — perfect-scaling ceiling
./bin/gpu_spgemm_bench --mode replicas --sizes "$SIZES" --precisions "$PRECS" \
    --density "$DENSITIES" --gpus "$GPUS" --iters "$ITERS" --csv "$OUT" --tag replicas

echo "==> done. CSV: $OUT"
echo "    columns: $(head -1 "$OUT")"
