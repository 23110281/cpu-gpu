#!/usr/bin/env bash
# scripts/cpu/run_all_intel.sh
# Master runner for Intel Xeon CPU Benchmarks (GEMM, GEMV, SAXPY, SpGEMM) powered by Intel oneMKL.
# Automatically builds binaries, runs smoke tests, and executes all 4 sweeps.
# Output is saved simultaneously to terminal and to results/logs/suite_latest.log.

set -eo pipefail
trap '' HUP  # Ignore hangup signals so benchmarks persist if SSH/laptop disconnects
cd "$(dirname "$0")/../.."

# Enforce all precisions (fp32, fp64, and bf16) across sweeps
export PRECS="${PRECS:-fp32,fp64,bf16}"

mkdir -p results/logs

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="results/logs/suite_${STAMP}.log"
LATEST_LOG="results/logs/suite_latest.log"

# Tee all stdout and stderr to both terminal and log files
exec > >(tee -a "$LOG_FILE" | tee "$LATEST_LOG") 2>&1

echo "======================================================================"
echo "          MASTER INTEL XEON BENCHMARK RUNNER (oneMKL ENGINE)          "
echo "======================================================================"
echo "Started at:   $(date)"
echo "Host:         $(hostname)"
echo "Log file:     $LOG_FILE"
echo "PID:          $$"
echo "======================================================================"
START_TOTAL=$(date +%s)

# Ensure oneAPI MKL environment (safely without strict unset check)
if [ -f /opt/intel/oneapi/setvars.sh ] && [ -z "${MKLROOT:-}" ]; then
    echo "==> Sourcing Intel oneAPI setvars.sh"
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
fi

# 1. Build all binaries
echo ""
echo "─── [1/3] Building All CPU Binaries ───"
make cpu_all test_telemetry

# 2. Run quick smoke tests
echo ""
echo "─── [2/3] Running Quick Smoke Tests ───"
echo ">> Testing Telemetry (RAPL / MSR)..."
make test_telemetry

echo ""
echo ">> Smoke Test: GEMM..."
make run_cpu_gemm

echo ""
echo ">> Smoke Test: GEMV / SpMV..."
make run_cpu_gemv

echo ""
echo ">> Smoke Test: SAXPY..."
make run_cpu_saxpy

echo ""
echo ">> Smoke Test: SpGEMM..."
make run_cpu_spgemm

echo ""
echo ">> Smoke Test: Monte Carlo Option Pricing..."
make run_cpu_montecarlo




# 3. Execute full parameter sweeps
echo ""
echo "======================================================================"
echo "─── [3/3] Executing Full Parameter Sweeps (native cross-check benches) ───"
echo "======================================================================"

echo ""
echo ">>> [Sweep 1/5] Starting GEMM Sweep..."
make sweep_cpu_gemm

echo ""
echo ">>> [Sweep 2/5] Starting GEMV / SpMV Sweep..."
make sweep_cpu_gemv

echo ""
echo ">>> [Sweep 3/5] Starting SAXPY Sweep..."
make sweep_cpu_saxpy

echo ""
echo ">>> [Sweep 4/5] Starting SpGEMM Sweep..."
make sweep_cpu_spgemm

echo ""
echo ">>> [Sweep 5/5] Starting Monte Carlo Sweep..."
make sweep_cpu_montecarlo




END_TOTAL=$(date +%s)
ELAPSED=$((END_TOTAL - START_TOTAL))

echo ""
echo "======================================================================"
echo " [SUCCESS] ALL 8 INTEL WORKLOAD SWEEPS COMPLETED IN ${ELAPSED}s"
echo " Finished at: $(date)"
echo "======================================================================"
echo "Generated CSV results in results/data/native/cpu/:"
ls -lh results/data/native/cpu/*/*.csv 2>/dev/null || true
echo "======================================================================"

# 4. Autonomous offsite backup (pushes only this machine's CSVs to GitHub backup repo)
bash scripts/cpu/backup_results.sh || true
