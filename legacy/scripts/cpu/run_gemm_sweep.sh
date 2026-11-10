#!/usr/bin/env bash
# scripts/cpu/run_gemm_sweep.sh
# End-to-end automated GEMM benchmark sweep on Intel CPU nodes
#
# Automatically:
#  1. Detects CPU model, physical core count, and maximum hardware threads.
#  2. Compiles bin/cpu_gemm_bench if not already built.
#  3. Creates results/data/<CPU_MODEL>/ directory.
#  4. Sweeps sizes x precisions x thread counts.
#  5. Emits a clean, timestamped CSV.

set -euo pipefail
cd "$(dirname "$0")/../.."

# ── 1. Ensure Environment & Build ────────────────────────────────────────────
if [ -f /opt/intel/oneapi/setvars.sh ] && [ -z "${MKLROOT:-}" ]; then
    echo "==> Sourcing Intel oneAPI setvars.sh"
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
fi

[ -x bin/cpu_gemm_bench ] || make cpu_gemm

# ── 2. Hardware Auto-Detection ───────────────────────────────────────────────
CPU_MODEL=$(lscpu | grep "Model name:" | sed 's/Model name:[ \t]*//' | tr -s ' ' '_' | tr -d '()' | tr -d '@')
[ -n "$CPU_MODEL" ] || CPU_MODEL="Unknown_Intel_Xeon"

# Count online physical cores and total hardware threads
PHYS_CORES=$(lscpu -p | grep -v '^#' | sort -u -t, -k 2,4 | wc -l || echo "8")
MAX_THREADS=$(nproc || echo "16")

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="results/data/native/cpu/${CPU_MODEL}"
mkdir -p "$OUT_DIR"
OUT="${OUT_DIR}/sweep_gemm_${STAMP}.csv"

# ── 3. Parameter Configuration (Overridable via environment variables) ───────
# Default sizes: from 512 up to 8192 (fits comfortably in RAM on all machines)
SIZES="${SIZES:-512,1024,2048,4096,8192}"

# Precisions: fp32, fp64, and bf16 (unconditionally swept on all CPUs)
PRECS="${PRECS:-fp32,fp64,bf16}"

# Thread progression: Powers of 2, plus physical core peak and SMT peak
# On m510 (8c/16t), this produces: 1,2,4,8,16
THREADS_ARR=()
t=1
while [ $t -le "$MAX_THREADS" ]; do
    THREADS_ARR+=("$t")
    t=$((t * 2))
done
# Ensure physical core count is explicitly in the sweep
if [[ ! " ${THREADS_ARR[*]} " =~ " ${PHYS_CORES} " ]] && [ "$PHYS_CORES" -le "$MAX_THREADS" ]; then
    THREADS_ARR+=("$PHYS_CORES")
fi
# Ensure max threads is included
if [[ ! " ${THREADS_ARR[*]} " =~ " ${MAX_THREADS} " ]]; then
    THREADS_ARR+=("$MAX_THREADS")
fi

# Sort and join with commas
IFS=$'\n' SORTED_THREADS=($(sort -n -u <<<"${THREADS_ARR[*]}"))
unset IFS
THREADS=$(IFS=,; echo "${SORTED_THREADS[*]}")

ITERS="${ITERS:-15}"
WARMUP="${WARMUP:-3}"

echo "========================================================================"
echo " Starting End-to-End CPU GEMM Sweep"
echo "  • Node CPU:        ${CPU_MODEL}"
echo "  • Physical Cores:  ${PHYS_CORES}"
echo "  • Max Threads:     ${MAX_THREADS}"
echo "  • Thread Sweep:    ${THREADS}"
echo "  • Precisions:      ${PRECS}"
echo "  • Matrix Sizes:    ${SIZES}"
echo "  • Output CSV:      ${OUT}"
echo "========================================================================"

# ── 4. Execute Sweep: Split Mode (Multi-threaded MKL) ────────────────────────
echo "==> [Stage 1/2] Running standard multi-threaded GEMM (split mode)..."
./bin/cpu_gemm_bench \
    --mode split \
    --sizes "$SIZES" \
    --precisions "$PRECS" \
    --threads "$THREADS" \
    --iters "$ITERS" \
    --warmup "$WARMUP" \
    --validate \
    --csv "$OUT" \
    --tag split

# ── 5. Execute Sweep: Single-Thread Baseline (Independent check) ─────────────
echo "==> [Stage 2/2] Running single-core calibration..."
./bin/cpu_gemm_bench \
    --mode split \
    --sizes "1024,2048,4096" \
    --precisions "$PRECS" \
    --threads 1 \
    --iters "$ITERS" \
    --warmup "$WARMUP" \
    --csv "$OUT" \
    --tag single_core

echo "========================================================================"
echo " Sweep Complete!"
echo " Results saved to: $OUT"
echo " Total rows logged: $(wc -l < "$OUT")"
echo " Columns: $(head -1 "$OUT")"
echo "========================================================================"
