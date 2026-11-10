#!/usr/bin/env bash
# scripts/cpu/run_gemv_sweep.sh
# End-to-end automated GEMV / SpMV benchmark sweep on Intel CPU nodes.
#
# Automatically:
#  1. Detects CPU model, physical core count, and hardware threads.
#  2. Compiles bin/cpu_gemv_bench if missing.
#  3. Sweeps sizes x precisions x thread counts x dense/sparse engines.
#  4. Emits a clean, timestamped CSV.

set -euo pipefail
cd "$(dirname "$0")/../.."

# ── 1. Ensure Environment & Build ────────────────────────────────────────────
if [ -f /opt/intel/oneapi/setvars.sh ] && [ -z "${MKLROOT:-}" ]; then
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
fi

[ -x bin/cpu_gemv_bench ] || make cpu_gemv

# ── 2. Hardware Auto-Detection ───────────────────────────────────────────────
CPU_MODEL=$(lscpu | grep "Model name:" | sed 's/Model name:[ \t]*//' | tr -s ' ' '_' | tr -d '()' | tr -d '@')
[ -n "$CPU_MODEL" ] || CPU_MODEL="Unknown_Intel_Xeon"

PHYS_CORES=$(lscpu -p | grep -v '^#' | sort -u -t, -k 2,4 | wc -l || echo "8")
MAX_THREADS=$(nproc || echo "16")

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="results/data/native/cpu/${CPU_MODEL}"
mkdir -p "$OUT_DIR"
OUT="${OUT_DIR}/sweep_gemv_${STAMP}.csv"

# ── 3. Parameter Configuration ───────────────────────────────────────────────
SIZES="${SIZES:-1024,2048,4096,8192,16384}"
PRECS="${PRECS:-fp32,fp64}"
DENSITIES="${DENSITIES:-0.01,0.05,0.10}"

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
echo " Starting End-to-End CPU GEMV/SpMV Sweep"
echo "  • Node CPU:        ${CPU_MODEL}"
echo "  • Physical Cores:  ${PHYS_CORES}"
echo "  • Max Threads:     ${MAX_THREADS}"
echo "  • Thread Sweep:    ${THREADS}"
echo "  • Matrix Sizes:    ${SIZES}"
echo "  • Output CSV:      ${OUT}"
echo "========================================================================"

# ── 4. Dense GEMV Sweep ──────────────────────────────────────────────────────
echo "==> [Stage 1/2] Running Dense GEMV..."
./bin/cpu_gemv_bench \
    --engine dense \
    --sizes "$SIZES" \
    --precisions "$PRECS" \
    --threads "$THREADS" \
    --iters "$ITERS" \
    --warmup "$WARMUP" \
    --validate \
    --csv "$OUT" \
    --tag dense_gemv

# ── 5. Sparse SpMV Sweep ─────────────────────────────────────────────────────
echo "==> [Stage 2/2] Running Sparse SpMV (CSR)..."
SPSIZES="${SPSIZES:-1024,2048,4096,8192}"
./bin/cpu_gemv_bench \
    --engine sparse \
    --sizes "$SPSIZES" \
    --density "$DENSITIES" \
    --precisions "$PRECS" \
    --threads "$THREADS" \
    --iters "$ITERS" \
    --warmup "$WARMUP" \
    --csv "$OUT" \
    --tag sparse_spmv

echo "========================================================================"
echo " GEMV/SpMV Sweep Complete!"
echo " Results saved to: $OUT"
echo " Total rows logged: $(wc -l < "$OUT")"
echo "========================================================================"
