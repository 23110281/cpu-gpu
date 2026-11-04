#!/usr/bin/env bash
# run_exploration_sweeps.sh — Executes all fixed benchmark sweeps for GH200 exploration
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH=/usr/local/cuda/bin:${PATH:-}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
STAMP=$(date +%Y%m%d_%H%M%S)

EXPLORE_DIR="results/data/${GPU_NAME}/exploration/full_fixed_sweeps"
mkdir -p "$EXPLORE_DIR"

echo "=================================================="
echo "Starting GH200 Exploration Sweeps"
echo "Timestamp: $STAMP"
echo "Target Directory: $EXPLORE_DIR"
echo "=================================================="

# 1. GEMV Fixed Sweep
OUT_GEMV="${EXPLORE_DIR}/sweep_gemv_fixed_${STAMP}.csv"
echo "==> Running GEMV Fixed Sweep -> $OUT_GEMV"
./bin/gpu_gemv_bench --mode split --sizes 512,1024,2048,4096,8192,16384,32768,65536 \
    --precisions fp32,fp64,bf16 --gpus 1 --iters 15 --validate --csv "$OUT_GEMV" --tag split
./bin/gpu_gemv_bench --mode replicas --sizes 512,1024,2048,4096,8192,16384,32768,65536 \
    --precisions fp32,fp64,bf16 --gpus 1 --iters 15 --csv "$OUT_GEMV" --tag replicas
./bin/gpu_gemv_bench --engine sparse --sizes 1024,2048,4096,8192,16384 \
    --precisions fp32,fp64 --density 0.01,0.05,0.1,0.25 --gpus 1 --iters 15 --csv "$OUT_GEMV" --tag sparse
echo "==> GEMV Fixed Sweep Complete."

# 2. SAXPY Fixed Sweep
OUT_SAXPY="${EXPLORE_DIR}/sweep_saxpy_fixed_${STAMP}.csv"
echo "==> Running SAXPY Fixed Sweep -> $OUT_SAXPY"
./bin/gpu_saxpy_bench --mode split --sizes 100000,500000,1000000,5000000,10000000,50000000,100000000,500000000,1000000000 \
    --precisions fp32,fp64 --gpus 1 --iters 15 --validate --csv "$OUT_SAXPY" --tag split
./bin/gpu_saxpy_bench --mode replicas --sizes 100000,500000,1000000,5000000,10000000,50000000,100000000,500000000,1000000000 \
    --precisions fp32,fp64 --gpus 1 --iters 15 --csv "$OUT_SAXPY" --tag replicas
echo "==> SAXPY Fixed Sweep Complete."

# 3. GEMM Fixed Sweep
OUT_GEMM="${EXPLORE_DIR}/sweep_gemm_fixed_${STAMP}.csv"
echo "==> Running GEMM Fixed Sweep -> $OUT_GEMM"
./bin/gpu_gemm_bench --mode split --sizes 512,1024,2048,4096,8192,16384,32768 \
    --precisions fp32,fp64,bf16 --gpus 1 --iters 15 --validate --csv "$OUT_GEMM" --tag split
./bin/gpu_gemm_bench --mode replicas --sizes 512,1024,2048,4096,8192,16384,32768 \
    --precisions fp32,fp64,bf16 --gpus 1 --iters 15 --csv "$OUT_GEMM" --tag replicas
./bin/gpu_gemm_bench --mode split --sizes 512,1024,2048,4096,8192,16384,32768 \
    --precisions fp64 --gpus 1 --iters 15 --fp64-tensor --csv "$OUT_GEMM" --tag fp64_dmma
./bin/gpu_gemm_bench --engine sparse --sizes 1024,2048,4096,8192,16384 \
    --precisions fp32,bf16 --density 0.01,0.05,0.1,0.25 --gpus 1 --iters 15 --validate --csv "$OUT_GEMM" --tag sparse
echo "==> GEMM Fixed Sweep Complete."

# 4. SpGEMM Fixed Sweep
OUT_SPGEMM="${EXPLORE_DIR}/sweep_spgemm_fixed_${STAMP}.csv"
echo "==> Running SpGEMM Fixed Sweep -> $OUT_SPGEMM"
./bin/gpu_spgemm_bench --mode split --sizes 512,1024,2048,4096,8192,16384 \
    --precisions fp32,fp64 --density 0.01,0.05,0.1,0.25,0.5 --gpus 1 --iters 15 --validate --csv "$OUT_SPGEMM" --tag split
echo "==> SpGEMM Fixed Sweep Complete."

echo "=================================================="
echo "All GH200 Exploration Sweeps Completed Successfully!"
echo "Files generated in $EXPLORE_DIR"
echo "=================================================="
