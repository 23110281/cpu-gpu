#!/usr/bin/env bash
# profile_ncu.sh — Nsight Compute deep-dive on ONE representative GEMM kernel.
# Captures DRAM throughput, achieved SM/GPC frequency, tensor-core pipe
# utilization, compute throughput, and occupancy.
#
#   ./scripts/profile_ncu.sh <size> <fp32|fp64|bf16> [fp64-tensor]
#   ./scripts/profile_ncu.sh 8192 bf16
#   ./scripts/profile_ncu.sh 8192 fp64 fp64-tensor
#
# ncu replays each kernel many times to read all counters, so we profile a
# single steady-state launch (skip warmups) on ONE GPU. Needs profiling perms:
#   - NVreg_RestrictProfilingToAdminUsers=0 (set by install_cuda.sh + reboot), or
#   - run this script with sudo.
set -euo pipefail
cd "$(dirname "$0")/.."

SIZE="${1:-8192}"
PREC="${2:-bf16}"
EXTRA=""
[ "${3:-}" = "fp64-tensor" ] && EXTRA="--fp64-tensor"
NCU="$(command -v ncu || echo /usr/local/cuda/bin/ncu)"
[ -x bin/gpu_gemm_bench ] || make
mkdir -p results
OUT="results/ncu_${PREC}_${SIZE}"

# Metric set (A100 / Nsight Compute counter names):
METRICS=$(cat <<'EOF' | paste -sd, -
gpu__time_duration.sum
dram__throughput.avg.pct_of_peak_sustained_elapsed
dram__bytes.sum
dram__bytes_read.sum
dram__bytes_write.sum
dram__bytes.sum.per_second
sm__cycles_elapsed.avg.per_second
gpc__cycles_elapsed.avg.per_second
sm__throughput.avg.pct_of_peak_sustained_elapsed
sm__warps_active.avg.pct_of_peak_sustained_active
sm__pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active
sm__pipe_tensor_op_dmma.avg.pct_of_peak_sustained_active
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active
sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active
EOF
)

echo "==> ncu profiling: size=$SIZE prec=$PREC $EXTRA"
echo "    (sm__cycles_elapsed.avg.per_second = achieved SM clock in Hz)"
echo "    (sm__pipe_tensor_op_hmma = BF16/FP16 tensor; _dmma = FP64 tensor)"

# --launch-skip 9 --launch-count 1: skip 6 warmups + a few timed iters, grab one.
"$NCU" --target-processes all \
       --kernel-name-base demangled \
       --launch-skip 9 --launch-count 1 \
       --metrics "$METRICS" \
       --csv --log-file "${OUT}.csv" \
       ./bin/gpu_gemm_bench --sizes "$SIZE" --precisions "$PREC" --gpus 1 \
                            --no-nvml --warmup 6 --iters 5 $EXTRA \
  | tee "${OUT}.txt"

echo "==> wrote ${OUT}.csv and ${OUT}.txt"
