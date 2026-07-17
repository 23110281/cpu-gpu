#!/usr/bin/env bash
# run_sweep_gemv.sh — full GEMV/SpMV sweep across size x precision x gpu-count.
# Emits one CSV under results/. Run after the driver is loaded (post-reboot).
#
#   ./scripts/run_sweep_gemv.sh
#   SIZES=512,1024,2048 ./scripts/run_sweep_gemv.sh
set -euo pipefail
cd "$(dirname "$0")/.."

command -v nvidia-smi >/dev/null || { echo "nvidia-smi missing — reboot to load the driver first"; exit 1; }
[ -x bin/gpu_gemv_bench ] || make gemv

NGPU=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1 | tr -d ' ')
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="results/sweep_gemv_${STAMP}.csv"
SIZES="${SIZES:-512,1024,2048,4096,8192,16384,32768,65536}"
PRECS="${PRECS:-fp32,fp64,bf16}"
GPUS="${GPUS:-$(seq -s, 1 "$NGPU")}"
ITERS="${ITERS:-15}"

echo "==> persistence mode on (stable clocks)"
sudo nvidia-smi -pm 1 >/dev/null || echo "   (could not set persistence mode; continuing)"

echo "==> sweep: sizes=$SIZES precs=$PRECS gpus=$GPUS  -> $OUT"

# 1) dense GEMV, split mode — row-sliced across GPUs
./bin/gpu_gemv_bench --mode split    --sizes "$SIZES" --precisions "$PRECS" \
    --gpus "$GPUS" --iters "$ITERS" --validate --csv "$OUT" --tag split

# 2) dense GEMV, replicas — perfect-scaling ceiling (each GPU runs the full problem)
./bin/gpu_gemv_bench --mode replicas --sizes "$SIZES" --precisions "$PRECS" \
    --gpus "$GPUS" --iters "$ITERS" --csv "$OUT" --tag replicas

# 3) sparse SpMV density sweep (fp32/fp64 -- bf16 unsupported, see run_gemv_config)
# No --validate here: the sparse engine has no reference implementation
# (validate_gemv_slice is dense-only and would crash on sparse's NULL
# hA_pin), so spmv_worker never calls it -- --validate is a documented
# no-op for --engine sparse. Keep it on the dense stages above, which
# validate for real.
SPSIZES="${SPSIZES:-1024,2048,4096,8192,16384}"
DENS="${DENS:-0.01,0.05,0.1,0.25}"
./bin/gpu_gemv_bench --engine sparse --sizes "$SPSIZES" --precisions fp32,fp64 \
    --density "$DENS" --gpus "$GPUS" --iters "$ITERS" --csv "$OUT" --tag sparse

echo "==> done. CSV: $OUT"
echo "    columns: $(head -1 "$OUT")"
