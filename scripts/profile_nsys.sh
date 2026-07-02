#!/usr/bin/env bash
# profile_nsys.sh — Nsight Systems timeline + sampled GPU metrics (clocks, DRAM,
# SM active %) across the whole run. Best for SEEING multi-GPU concurrency and
# frequency scaling over time (complements ncu's per-kernel detail and NVML's
# in-process sampling).
#
#   ./scripts/profile_nsys.sh <size> <fp32|fp64|bf16> <ngpus>
#   ./scripts/profile_nsys.sh 16384 bf16 4
set -euo pipefail
cd "$(dirname "$0")/.."

SIZE="${1:-16384}"; PREC="${2:-bf16}"; NG="${3:-4}"
NSYS="$(command -v nsys || echo /usr/local/cuda/bin/nsys)"
[ -x bin/gpu_gemm_bench ] || make
mkdir -p results
OUT="results/nsys_${PREC}_${SIZE}_${NG}gpu"

echo "==> nsys profiling: size=$SIZE prec=$PREC gpus=$NG -> ${OUT}.nsys-rep"
# --gpu-metrics-device=all samples HW counters (SM/MEM clock, DRAM BW, SM active)
# on every GPU at high frequency — the cleanest frequency-scaling timeline.
"$NSYS" profile \
    --gpu-metrics-device=all \
    --trace=cuda,cublas,osrt \
    --sample=cpu \
    --output "$OUT" --force-overwrite true \
    ./bin/gpu_gemm_bench --sizes "$SIZE" --precisions "$PREC" --gpus "$NG" \
                         --no-nvml --warmup 3 --iters 30

echo "==> wrote ${OUT}.nsys-rep"
echo "    summary:  nsys stats ${OUT}.nsys-rep"
echo "    open the .nsys-rep in the Nsight Systems GUI for the timeline."
