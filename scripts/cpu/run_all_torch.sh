#!/usr/bin/env bash
# scripts/cpu/run_all_torch.sh
# Master runner for PyTorch CPU Benchmarks (GEMM, GEMV/SpMV, SAXPY, SpGEMM).
# Compatible with both Intel Xeon and AMD EPYC architectures.
#
# Features:
#  1. Ignores hangup signals (SIGHUP) so benchmarks continue if SSH/terminal disconnects.
#  2. Executes quick smoke tests across all 4 workloads.
#  3. Executes all 4 full parameter sweeps.
#  4. Automatically triggers offsite GitHub backup via backup_results.sh upon completion.
#  5. Simultaneously logs to terminal and to results/logs/suite_torch_latest.log.

set -eo pipefail
trap '' HUP  # Ignore SIGHUP so benchmarks persist if SSH/laptop disconnects
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python3}"
export PRECS="${PRECS:-fp32,fp64,bf16}"

mkdir -p results/logs

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="results/logs/suite_torch_${STAMP}.log"
LATEST_LOG="results/logs/suite_torch_latest.log"

# Tee all stdout and stderr to both terminal and persistent log files
exec > >(tee -a "$LOG_FILE" | tee "$LATEST_LOG") 2>&1

CPU_MODEL=$(lscpu | grep "Model name:" | sed 's/Model name:[ \t]*//' || echo "Unknown_CPU")

echo "======================================================================"
echo "          MASTER PYTORCH CPU BENCHMARK RUNNER (ALL WORKLOADS)         "
echo "======================================================================"
echo "Started at:   $(date)"
echo "Host:         $(hostname)"
echo "CPU Model:    $CPU_MODEL"
echo "Python:       $($PYTHON --version 2>&1 || echo 'unknown')"
echo "PyTorch:      $($PYTHON -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
echo "Log file:     $LOG_FILE"
echo "PID:          $$"
echo "Precisions:   $PRECS"
echo "======================================================================"
START_TOTAL=$(date +%s)

# 1. Run quick smoke tests
echo ""
echo "─── [1/2] Running Quick PyTorch Smoke Tests ───"

echo ""
echo ">> Smoke Test: PyTorch GEMM..."
$PYTHON src/torch/torch_gemm_bench.py --sizes 512,1024 --threads 1,2 --precisions fp32,fp64,bf16 --validate --iters 5

echo ""
echo ">> Smoke Test: PyTorch GEMV / SpMV..."
$PYTHON src/torch/torch_gemv_bench.py --sizes 1024,2048 --threads 1,2 --engine all --density 0.05 --precisions fp32,fp64 --validate --iters 5

echo ""
echo ">> Smoke Test: PyTorch SAXPY..."
$PYTHON src/torch/torch_saxpy_bench.py --sizes 1048576,4194304 --threads 1,2 --precisions fp32,fp64 --validate --iters 5

echo ""
echo ">> Smoke Test: PyTorch SpGEMM..."
$PYTHON src/torch/torch_spgemm_bench.py --sizes 1024 --density 0.05 --threads 1,2 --precisions fp32,fp64 --validate --iters 3

# 2. Execute full parameter sweeps
echo ""
echo "======================================================================"
echo "─── [2/2] Executing Full Parameter Sweeps (All 4 Workloads) ───"
echo "======================================================================"

echo ""
echo ">>> [Sweep 1/4] Starting PyTorch GEMM Sweep..."
bash scripts/cpu/run_torch_gemm_sweep.sh

echo ""
echo ">>> [Sweep 2/4] Starting PyTorch GEMV / SpMV Sweep..."
bash scripts/cpu/run_torch_gemv_sweep.sh

echo ""
echo ">>> [Sweep 3/4] Starting PyTorch SAXPY Sweep..."
bash scripts/cpu/run_torch_saxpy_sweep.sh

echo ""
echo ">>> [Sweep 4/4] Starting PyTorch SpGEMM Sweep..."
bash scripts/cpu/run_torch_spgemm_sweep.sh

END_TOTAL=$(date +%s)
ELAPSED=$((END_TOTAL - START_TOTAL))

echo ""
echo "======================================================================"
echo " [SUCCESS] ALL 4 PYTORCH WORKLOAD SWEEPS COMPLETED IN ${ELAPSED}s"
echo " Finished at: $(date)"
echo "======================================================================"
echo "Generated CSV results in results/data/cpu/:"
ls -lh results/data/cpu/*/*.csv 2>/dev/null || true
echo "======================================================================"

# 3. Autonomous offsite backup (pushes completed CSVs to personal backup repo)
bash scripts/cpu/backup_results.sh || true
