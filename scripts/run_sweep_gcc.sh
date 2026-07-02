#!/usr/bin/env bash
# run_sweep_gcc.sh — full GEMM sweep using the GCC build (gpu_gemm_bench).
#
# Identical sweep logic to run_sweep.sh but uses ./bin/gpu_gemm_bench, so
# you can compare the GCC C11 build results with the NVCC build results.
#
#   ./scripts/run_sweep_gcc.sh
#   SIZES=256,512,1024,2048 ./scripts/run_sweep_gcc.sh
set -euo pipefail
cd "$(dirname "$0")/.."

command -v nvidia-smi >/dev/null || { echo "nvidia-smi missing — reboot to load the driver first"; exit 1; }
[ -x bin/gpu_gemm_bench ] || make bench_gcc

NGPU=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1 | tr -d ' ')
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="results/sweep_gcc_${STAMP}.csv"

# GCC build default starts at 512, but can run 256..65536.
SIZES="${SIZES:-256,512,1024,2048,4096,8192,16384,32768,65536}"
PRECS="${PRECS:-fp32,fp64,bf16}"
GPUS="${GPUS:-$(seq -s, 1 "$NGPU")}"
ITERS="${ITERS:-15}"

echo "==> persistence mode on (stable clocks)"
sudo nvidia-smi -pm 1 >/dev/null || echo "   (could not set persistence mode; continuing)"

echo "==> GCC sweep: sizes=$SIZES precs=$PRECS gpus=$GPUS  -> $OUT"

# 1) split mode — distributed GEMM across GPUs
./bin/gpu_gemm_bench --mode split    --sizes "$SIZES" --precisions "$PRECS" \
    --gpus "$GPUS" --iters "$ITERS" --validate --csv "$OUT" --tag split_gcc

# 2) replicas — perfect-scaling ceiling (each GPU runs the full problem)
./bin/gpu_gemm_bench --mode replicas --sizes "$SIZES" --precisions "$PRECS" \
    --gpus "$GPUS" --iters "$ITERS" --csv "$OUT" --tag replicas_gcc

# 3) FP64 DMMA tensor-core comparison
./bin/gpu_gemm_bench --mode split    --sizes "$SIZES" --precisions fp64 \
    --gpus "$GPUS" --iters "$ITERS" --fp64-tensor --csv "$OUT" --tag fp64_dmma_gcc

# 4) sparse SpMM density sweep (capped to avoid OOM from CSR overhead)
SPSIZES="${SPSIZES:-1024,2048,4096,8192,16384}"
DENS="${DENS:-0.01,0.05,0.1,0.25}"
./bin/gpu_gemm_bench --engine sparse --sizes "$SPSIZES" --precisions fp32,fp64 \
    --density "$DENS" --gpus "$GPUS" --iters "$ITERS" --validate --csv "$OUT" --tag sparse_gcc
# control: dense GEMM on sparse-patterned data
./bin/gpu_gemm_bench --engine dense  --sizes "$SPSIZES" --precisions fp32 \
    --fill sparse:0.1 --gpus "$GPUS" --iters "$ITERS" --csv "$OUT" --tag dense_sparsedata_gcc

echo "==> done. CSV: $OUT"
echo "    columns: $(head -1 "$OUT")"
