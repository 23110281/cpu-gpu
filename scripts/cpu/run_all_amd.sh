#!/usr/bin/env bash
# scripts/cpu/run_all_amd.sh
# Master runner for AMD EPYC Benchmarks (GEMM, GEMV, SAXPY, SpGEMM) powered by OpenBLAS.
# Automatically builds OpenBLAS binaries, runs smoke tests, and executes all 4 sweeps.
# Output is saved simultaneously to terminal and to results/logs/suite_epyc_latest.log.

set -eo pipefail
trap '' HUP  # Ignore SIGHUP so benchmarks persist if SSH/laptop disconnects
cd "$(dirname "$0")/../.."

# Enforce OpenBLAS and all precisions across all sweeps
export BLAS=openblas
export PRECS="${PRECS:-fp32,fp64,bf16}"

mkdir -p results/logs

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="results/logs/suite_epyc_${STAMP}.log"
LATEST_LOG="results/logs/suite_epyc_latest.log"

# Tee all stdout and stderr to both terminal and persistent log files
exec > >(tee -a "$LOG_FILE" | tee "$LATEST_LOG") 2>&1

echo "======================================================================"
echo "          MASTER AMD EPYC BENCHMARK RUNNER (OPENBLAS ENGINE)          "
echo "======================================================================"
echo "Started at:   $(date)"
echo "Host:         $(hostname)"
echo "CPU Model:    $(lscpu | grep 'Model name:' | sed 's/Model name:[ \t]*//' || echo 'Unknown_AMD_CPU')"
echo "Log file:     $LOG_FILE"
echo "PID:          $$"
echo "BLAS Engine:  $BLAS"
echo "Precisions:   $PRECS"
echo "======================================================================"
START_TOTAL=$(date +%s)

# 1. Clean and build all binaries with OpenBLAS
echo ""
echo "─── [1/3] Building All CPU Binaries with OpenBLAS ───"
make clean
make BLAS=openblas cpu_all test_telemetry

# 2. Run quick smoke tests
echo ""
echo "─── [2/3] Running Quick Smoke Tests ───"
echo ">> Testing Hardware Telemetry..."
make test_telemetry

echo ""
echo ">> Smoke Test: GEMM (OpenBLAS)..."
make BLAS=openblas run_cpu_gemm

echo ""
echo ">> Smoke Test: GEMV / SpMV..."
make BLAS=openblas run_cpu_gemv

echo ""
echo ">> Smoke Test: SAXPY..."
make BLAS=openblas run_cpu_saxpy

echo ""
echo ">> Smoke Test: SpGEMM..."
make BLAS=openblas run_cpu_spgemm

echo ""
echo ">> Smoke Test: Monte Carlo Option Pricing..."
make BLAS=openblas run_cpu_montecarlo

echo ""
echo ">> Smoke Test: Cryptography (SHA-256 / SHA-512)..."
make BLAS=openblas run_cpu_cryptography

echo ""
echo ">> Smoke Test: Principal Component Analysis (PCA)..."
make BLAS=openblas run_cpu_pca

echo ""
echo ">> Smoke Test: K-Means Clustering..."
make BLAS=openblas run_cpu_kmeans

# 3. Execute full parameter sweeps
echo ""
echo "======================================================================"
echo "─── [3/3] Executing Full Parameter Sweeps (All 8 Workloads) ───"
echo "======================================================================"

echo ""
echo ">>> [Sweep 1/8] Starting GEMM Sweep..."
make BLAS=openblas sweep_cpu_gemm

echo ""
echo ">>> [Sweep 2/8] Starting GEMV / SpMV Sweep..."
make BLAS=openblas sweep_cpu_gemv

echo ""
echo ">>> [Sweep 3/8] Starting SAXPY Sweep..."
make BLAS=openblas sweep_cpu_saxpy

echo ""
echo ">>> [Sweep 4/8] Starting SpGEMM Sweep..."
make BLAS=openblas sweep_cpu_spgemm

echo ""
echo ">>> [Sweep 5/8] Starting Monte Carlo Sweep..."
make BLAS=openblas sweep_cpu_montecarlo

echo ""
echo ">>> [Sweep 6/8] Starting Cryptography Sweep..."
make BLAS=openblas sweep_cpu_cryptography

echo ""
echo ">>> [Sweep 7/8] Starting PCA Sweep..."
make BLAS=openblas sweep_cpu_pca

echo ""
echo ">>> [Sweep 8/8] Starting K-Means Sweep..."
make BLAS=openblas sweep_cpu_kmeans

END_TOTAL=$(date +%s)
ELAPSED=$((END_TOTAL - START_TOTAL))

echo ""
echo "======================================================================"
echo " [SUCCESS] ALL 8 AMD EPYC WORKLOAD SWEEPS COMPLETED IN ${ELAPSED}s"
echo " Finished at: $(date)"
echo "======================================================================"
echo "Generated CSV results in results/data/cpu/:"
ls -lh results/data/cpu/*/*.csv 2>/dev/null || true
echo "======================================================================"

# 4. Autonomous offsite backup (pushes only this machine's CSVs to GitHub backup repo)
bash scripts/cpu/backup_results.sh || true
