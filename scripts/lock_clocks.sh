#!/usr/bin/env bash
# lock_clocks.sh — pin GPU clocks for reproducible benchmarking (removes the
# frequency-scaling variable). A100-SXM4 max SM clock is 1410 MHz.
#
#   ./scripts/lock_clocks.sh lock      # persistence on + lock SM clock to max
#   ./scripts/lock_clocks.sh max       # show max supported clocks
#   ./scripts/lock_clocks.sh reset     # release locks, back to auto-boost
set -euo pipefail
SM_MAX="${SM_MAX:-1410}"   # A100-SXM4-40GB max graphics clock (MHz)

case "${1:-lock}" in
  lock)
    sudo nvidia-smi -pm 1
    sudo nvidia-smi --lock-gpu-clocks="${SM_MAX},${SM_MAX}"
    echo "locked SM clock to ${SM_MAX} MHz on all GPUs (persistence on)"
    nvidia-smi --query-gpu=index,clocks.sm,clocks.mem --format=csv
    ;;
  max)
    nvidia-smi --query-gpu=index,clocks.max.sm,clocks.max.mem --format=csv
    ;;
  reset)
    sudo nvidia-smi --reset-gpu-clocks
    echo "released clock locks (auto-boost restored)"
    ;;
  *) echo "usage: $0 {lock|max|reset}"; exit 2 ;;
esac
