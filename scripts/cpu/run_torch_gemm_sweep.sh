#!/usr/bin/env bash
# scripts/cpu/run_torch_gemm_sweep.sh
# End-to-end automated PyTorch GEMM benchmark sweep on CPU nodes (Intel / AMD)
#
# Automatically:
#  1. Detects CPU model, physical core count, and maximum hardware threads.
#  2. Creates results/data/<CPU_MODEL>/ directory.
#  3. Sweeps sizes x precisions x thread counts.
#  4. Emits a clean, timestamped CSV matching existing schema.

set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python3}"

# ── 1. Hardware Auto-Detection ───────────────────────────────────────────────
CPU_MODEL=$(lscpu | grep "Model name:" | sed 's/Model name:[ \t]*//' | tr -s ' ' '_' | tr -d '()' | tr -d '@')
[ -n "$CPU_MODEL" ] || CPU_MODEL="Unknown_CPU"

PHYS_CORES=$(lscpu -p 2>/dev/null | grep -v '^#' | sort -u -t, -k 2,4 | wc -l || echo "8")
MAX_THREADS=$(nproc || echo "16")

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="results/data/cpu/${CPU_MODEL}"
mkdir -p "$OUT_DIR"
OUT="${OUT_DIR}/sweep_torch_gemm_${STAMP}.csv"

# ── 2. Parameter Configuration ───────────────────────────────────────────────
SIZES="${SIZES:-512,1024,2048,4096,8192}"
PRECS="${PRECS:-fp32,fp64,bf16}"

THREADS_ARR=()
t=1
while [ $t -le "$MAX_THREADS" ]; do
    THREADS_ARR+=("$t")
    t=$((t * 2))
done
if [[ ! " ${THREADS_ARR[*]} " =~ " ${PHYS_CORES} " ]] && [ "$PHYS_CORES" -le "$MAX_THREADS" ]; then
    THREADS_ARR+=("$PHYS_CORES")
fi
if [[ ! " ${THREADS_ARR[*]} " =~ " ${MAX_THREADS} " ]]; then
    THREADS_ARR+=("$MAX_THREADS")
fi

IFS=$'\n' SORTED_THREADS=($(sort -n -u <<<"${THREADS_ARR[*]}"))
unset IFS
THREADS=$(IFS=,; echo "${SORTED_THREADS[*]}")

ITERS="${ITERS:-15}"
WARMUP="${WARMUP:-3}"

echo "========================================================================"
echo " Starting End-to-End PyTorch CPU GEMM Sweep"
echo "  • Node CPU:        ${CPU_MODEL}"
echo "  • Physical Cores:  ${PHYS_CORES}"
echo "  • Max Threads:     ${MAX_THREADS}"
echo "  • Thread Sweep:    ${THREADS}"
echo "  • Precisions:      ${PRECS}"
echo "  • Matrix Sizes:    ${SIZES}"
echo "  • Output CSV:      ${OUT}"
echo "========================================================================"

# ── 3. Execute Sweep: Multi-threaded Split Mode ──────────────────────────────
echo "==> [Stage 1/2] Running multi-threaded PyTorch GEMM..."
$PYTHON src/torch/torch_gemm_bench.py \
    --device cpu \
    --mode split \
    --sizes "$SIZES" \
    --precisions "$PRECS" \
    --threads "$THREADS" \
    --iters "$ITERS" \
    --warmup "$WARMUP" \
    --validate \
    --csv "$OUT" \
    --tag torch_gemm

# ── 4. Execute Sweep: Single-Thread Baseline ─────────────────────────────────
echo "==> [Stage 2/2] Running single-core PyTorch calibration..."
$PYTHON src/torch/torch_gemm_bench.py \
    --device cpu \
    --mode split \
    --sizes "1024,2048,4096" \
    --precisions "$PRECS" \
    --threads 1 \
    --iters "$ITERS" \
    --warmup "$WARMUP" \
    --csv "$OUT" \
    --tag torch_gemm_single_core

echo "========================================================================"
echo " PyTorch GEMM Sweep Complete!"
echo " Results saved to: $OUT"
echo " Total rows logged: $(wc -l < "$OUT")"
echo "========================================================================"
