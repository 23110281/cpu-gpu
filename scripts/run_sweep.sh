#!/usr/bin/env bash
# run_sweep.sh — full GEMM sweep across size x precision x gpu-count.
# Emits one CSV under results/. Run after the driver is loaded (post-reboot).
#
#   ./scripts/run_sweep.sh                 # default full sweep
#   SIZES=512,1024,2048 ./scripts/run_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."

command -v nvidia-smi >/dev/null || { echo "nvidia-smi missing — reboot to load the driver first"; exit 1; }
[ -x bin/gpu_gemm_bench ] || make

NGPU=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1 | tr -d ' ')
STAMP=$(date +%Y%m%d_%H%M%S)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
mkdir -p "results/data/${GPU_NAME}"
OUT="results/data/${GPU_NAME}/sweep_${STAMP}.csv"
SIZES="${SIZES:-512,1024,2048,4096,8192,16384,32768,65536}"
PRECS="${PRECS:-fp32,fp64,bf16}"
GPUS="${GPUS:-$(seq -s, 1 "$NGPU")}"
ITERS="${ITERS:-15}"

echo "==> persistence mode on (stable clocks)"
sudo nvidia-smi -pm 1 >/dev/null || echo "   (could not set persistence mode; continuing)"

echo "==> sweep: sizes=$SIZES precs=$PRECS gpus=$GPUS  -> $OUT"

# 1) the headline: one big GEMM split across GPUs (your distributed-system analogue)
./bin/gpu_gemm_bench --mode split    --sizes "$SIZES" --precisions "$PRECS" \
    --gpus "$GPUS" --iters "$ITERS" --validate --csv "$OUT" --tag split

# 2) perfect-scaling ceiling: each GPU runs the full problem independently
./bin/gpu_gemm_bench --mode replicas --sizes "$SIZES" --precisions "$PRECS" \
    --gpus "$GPUS" --iters "$ITERS" --csv "$OUT" --tag replicas

# 3) FP64 DMMA tensor-core comparison point (off by default in the main sweep)
./bin/gpu_gemm_bench --mode split    --sizes "$SIZES" --precisions fp64 \
    --gpus "$GPUS" --iters "$ITERS" --fp64-tensor --csv "$OUT" --tag fp64_dmma

# 4) sparse-vs-dense "incident": cuSPARSE SpMM across a density sweep, alongside
#    the cheap control (dense cuBLAS on sparse-patterned data). Capped at sizes
#    that keep CSR memory sane.
SPSIZES="${SPSIZES:-1024,2048,4096,8192,16384}"
DENS="${DENS:-0.01,0.05,0.1,0.25}"
./bin/gpu_gemm_bench --engine sparse --sizes "$SPSIZES" --precisions fp32,fp64 \
    --density "$DENS" --gpus "$GPUS" --iters "$ITERS" --validate --csv "$OUT" --tag sparse
# control: dense GEMM fed 10%-nonzero data — should match the dense numbers above
./bin/gpu_gemm_bench --engine dense  --sizes "$SPSIZES" --precisions fp32 \
    --fill sparse:0.1 --gpus "$GPUS" --iters "$ITERS" --csv "$OUT" --tag dense_sparsedata

echo "==> done. CSV: $OUT"
echo "    columns: $(head -1 "$OUT")"
