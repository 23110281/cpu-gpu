#!/usr/bin/env bash
# scripts/setup_v2_harness.sh — node requirements of the v2 publication harness (src/torch/bench).
#
# Idempotent; called by scripts/cpu/setup_cloudlab_xeon.sh and scripts/gpu/setup_gpu_node.sh
# (setup_cloudlab_epyc.sh contains the same steps inline). Every step below exists because a
# measurement was wrong without it -- see docs/methodology.md "Platform pitfalls".
#
#   bash scripts/setup_v2_harness.sh [cpu|gpu]      (default: cpu)
set -euo pipefail
KIND="${1:-cpu}"
log()  { printf "\033[1;34m[v2]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[v2 WARN]\033[0m %s\n" "$*"; }
ARCH_DIR="/usr/lib/$(uname -m)-linux-gnu"

log "apt packages (build, NUMA, fastest-library candidates, crypto, extension headers)"
sudo apt-get update -y -q >/dev/null
sudo apt-get install -y -q --no-install-recommends build-essential cmake git numactl hwloc \
    linux-tools-common "linux-tools-$(uname -r)" python3-dev python3-pip \
    libopenblas-openmp-dev libblis3-openmp libtbb-dev libssl-dev libjemalloc2 >/dev/null \
    || warn "some packages failed; affected variants will report 'unsupported'"

if [ "$KIND" = cpu ]; then
    # pthread OpenBLAS + OpenMP binding = every BLAS thread on one core (80 vs 1760 GFLOP/s)
    for alt in libblas.so.3 liblapack.so.3 libopenblas.so.0; do
        t="$ARCH_DIR/openblas-openmp/$alt"
        [ -e "$t" ] && sudo update-alternatives --set "${alt}-$(uname -m)-linux-gnu" "$t" >/dev/null || true
    done
    log "OpenBLAS alternative -> $(readlink -f "$ARCH_DIR/libopenblas.so.0")"
fi

log "python packages"
if [ "$KIND" = cpu ]; then
    python3 -m pip install -q --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
        || warn "torch (CPU wheel) install failed"
else
    # The torch wheel must match the node's nvcc MAJOR version: torch.utils.cpp_extension raises
    # RuntimeError on a major mismatch, and every native CUDA kernel (B2 C2 C3 C5 C8 C9 S2) would
    # be recorded 'unsupported'. Plain `pip install torch` brings the newest CUDA (13.x) wheel,
    # while scripts/gpu/install_cuda.sh installs a 12.x toolkit.
    NVCC="$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)"
    NVCC_VER="$("$NVCC" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1.\2/p' || true)"
    case "${NVCC_VER%%.*}" in
        13) IDX="cu130" ;;
        12) IDX="cu129 cu128 cu126" ;;
        *)  IDX="" ; warn "nvcc version '${NVCC_VER}' not recognised: installing the default torch wheel" ;;
    esac
    ok_torch=0
    for i in $IDX; do
        python3 -m pip install -q --no-cache-dir torch --index-url "https://download.pytorch.org/whl/$i" \
            && { ok_torch=1; log "torch wheel from $i (nvcc ${NVCC_VER})"; break; }
    done
    [ "$ok_torch" = 1 ] || python3 -m pip install -q --no-cache-dir torch || warn "torch (CUDA wheel) install failed"
    TORCH_CUDA="$(python3 -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true)"
    [ "${TORCH_CUDA%%.*}" = "${NVCC_VER%%.*}" ] \
        || warn "torch CUDA ${TORCH_CUDA} vs nvcc ${NVCC_VER}: major versions differ -> native CUDA kernels will NOT build"
    python3 -m pip install -q --no-cache-dir nvidia-ml-py || warn "pynvml install failed (GPU power telemetry)"
fi
python3 -m pip install -q --no-cache-dir numpy scipy pytest ninja opencv-python-headless cryptography \
    || warn "python deps failed"

log "telemetry permissions (RAPL energy + APERF/MPERF MSRs)"
sudo modprobe msr 2>/dev/null || true
# powercap entries are symlinks: chmod the resolved files (root-only since CVE-2020-8694)
for f in /sys/class/powercap/intel-rapl*/energy_uj /sys/class/powercap/intel-rapl*/*/energy_uj; do
    [ -e "$f" ] && sudo chmod a+r "$(readlink -f "$f")" 2>/dev/null || true
done
PY_REAL="$(readlink -f "$(command -v python3)")"
sudo setcap cap_sys_rawio+ep "$PY_REAL" 2>/dev/null && log "CAP_SYS_RAWIO -> $PY_REAL" \
    || warn "setcap failed: CPU frequency falls back to cpufreq"
# CAP_SYS_RAWIO puts ld.so in secure mode: LD_PRELOAD needs a set-user-ID lib (opt-in jemalloc)
[ -e "$ARCH_DIR/libjemalloc.so.2" ] && sudo chmod u+s "$(readlink -f "$ARCH_DIR/libjemalloc.so.2")" || true

if command -v cpupower >/dev/null 2>&1; then
    sudo cpupower frequency-set -g performance >/dev/null 2>&1 || true
fi
if [ "$KIND" = gpu ] && command -v nvidia-smi >/dev/null 2>&1; then
    sudo nvidia-smi -pm 1 >/dev/null 2>&1 && log "GPU persistence mode on" || warn "persistence mode failed"
fi

log "verification"
PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src/torch" python3 - <<'EOF'
import torch, sys
print(f"  torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
try:
    open("/sys/class/powercap/intel-rapl:0/energy_uj").read(); print("  RAPL energy readable: yes")
except OSError:
    print("  RAPL energy readable: NO (power will be NaN)")
EOF
log "done -- next: make test_bench && make bench_smoke$( [ "$KIND" = gpu ] && echo _cuda )"
