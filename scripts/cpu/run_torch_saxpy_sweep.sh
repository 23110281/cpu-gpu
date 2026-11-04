#!/usr/bin/env bash
# scripts/cpu/run_torch_saxpy_sweep.sh
# End-to-end automated PyTorch SAXPY / DAXPY benchmark sweep on CPU nodes (Intel / AMD)
#
# Automatically:
#  1. Detects CPU model, physical core count, and hardware threads.
#  2. Sweeps vector lengths (N) x precisions (fp32, fp64) x thread counts.
#  3. Emits a clean, timestamped CSV matching existing schema.

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
OUT="${OUT_DIR}/sweep_torch_saxpy_${STAMP}.csv"

# ── 2. Parameter Configuration ───────────────────────────────────────────────
# Vector sizes: 1M elements (4MB) up to 134M elements (~1GB)
SIZES="${SIZES:-1048576,4194304,16777216,33554432,67108864,134217728}"
PRECS="${PRECS:-fp32,fp64}"

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

ITERS="${ITERS:-20}"
WARMUP="${WARMUP:-3}"

echo "========================================================================"
echo " Starting End-to-End PyTorch CPU SAXPY Sweep"
echo "  • Node CPU:        ${CPU_MODEL}"
echo "  • Physical Cores:  ${PHYS_CORES}"
echo "  • Max Threads:     ${MAX_THREADS}"
echo "  • Thread Sweep:    ${THREADS}"
echo "  • Vector Sizes:    ${SIZES}"
echo "  • Output CSV:      ${OUT}"
echo "========================================================================"

# ── 3. Execute Sweep ─────────────────────────────────────────────────────────
$PYTHON src/torch/torch_saxpy_bench.py \
    --device cpu \
    --sizes "$SIZES" \
    --precisions "$PRECS" \
    --threads "$THREADS" \
    --iters "$ITERS" \
    --warmup "$WARMUP" \
    --validate \
    --csv "$OUT" \
    --tag torch_saxpy

echo "========================================================================"
echo " PyTorch SAXPY Sweep Complete!"
echo " Results saved to: $OUT"
echo " Total rows logged: $(wc -l < "$OUT")"
echo "========================================================================"
