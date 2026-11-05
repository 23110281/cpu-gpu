#!/usr/bin/env bash
# scripts/cpu/run_pca_sweep.sh
# End-to-end automated Principal Component Analysis (PCA) benchmark sweep on CPU nodes
#
# Automatically:
#  1. Detects CPU model, physical core count, and hardware threads.
#  2. Compiles bin/cpu_pca_bench if missing.
#  3. Sweeps sample sizes x precisions (fp32, fp64) x thread counts.
#  4. Emits a clean, timestamped CSV matching standard schema.

set -euo pipefail
cd "$(dirname "$0")/../.."

# ── 1. Ensure Environment & Build ────────────────────────────────────────────
if [ -f /opt/intel/oneapi/setvars.sh ] && [ -z "${MKLROOT:-}" ]; then
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
fi

[ -x bin/cpu_pca_bench ] || make cpu_pca

# ── 2. Hardware Auto-Detection ───────────────────────────────────────────────
CPU_MODEL=$(lscpu | grep "Model name:" | sed 's/Model name:[ \t]*//' | tr -s ' ' '_' | tr -d '()' | tr -d '@')
[ -n "$CPU_MODEL" ] || CPU_MODEL="Unknown_CPU"

PHYS_CORES=$(lscpu -p 2>/dev/null | grep -v '^#' | sort -u -t, -k 2,4 | wc -l || echo "8")
MAX_THREADS=$(nproc || echo "16")

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="results/data/native/cpu/${CPU_MODEL}"
mkdir -p "$OUT_DIR"
OUT="${OUT_DIR}/sweep_pca_${STAMP}.csv"

# ── 3. Parameter Configuration ───────────────────────────────────────────────
SIZES="${SIZES:-100000,500000,2000000}"
FEATURES="${FEATURES:-64}"
COMPONENTS="${COMPONENTS:-8}"
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

ITERS="${ITERS:-5}"
WARMUP="${WARMUP:-1}"

echo "========================================================================"
echo " Starting End-to-End CPU Principal Component Analysis (PCA) Sweep"
echo "  • Node CPU:        ${CPU_MODEL}"
echo "  • Physical Cores:  ${PHYS_CORES}"
echo "  • Max Threads:     ${MAX_THREADS}"
echo "  • Thread Sweep:    ${THREADS}"
echo "  • Precisions:      ${PRECS}"
echo "  • Sample Counts:   ${SIZES}"
echo "  • Features:        ${FEATURES}"
echo "  • Components:      ${COMPONENTS}"
echo "  • Output CSV:      ${OUT}"
echo "========================================================================"

# ── 4. Multi-Threaded Parameter Sweep ────────────────────────────────────────
echo "==> [Stage 1/2] Running multi-threaded PCA sweep..."
./bin/cpu_pca_bench \
    --sizes "$SIZES" \
    --features "$FEATURES" \
    --components "$COMPONENTS" \
    --precisions "$PRECS" \
    --threads "$THREADS" \
    --iters "$ITERS" \
    --warmup "$WARMUP" \
    --validate \
    --csv "$OUT" \
    --tag cpu_pca

# ── 5. Single-Core Calibration Baseline ─────────────────────────────────────
echo "==> [Stage 2/2] Running single-core calibration..."
./bin/cpu_pca_bench \
    --sizes "100000,500000" \
    --features "$FEATURES" \
    --components "$COMPONENTS" \
    --precisions "$PRECS" \
    --threads 1 \
    --iters "$ITERS" \
    --warmup "$WARMUP" \
    --csv "$OUT" \
    --tag cpu_pca_single_core

echo "========================================================================"
echo " PCA Sweep Complete!"
echo " Results saved to: $OUT"
echo " Total rows logged: $(wc -l < "$OUT")"
echo "========================================================================"
