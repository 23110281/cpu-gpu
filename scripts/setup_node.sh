#!/usr/bin/env bash
# scripts/setup_node.sh — everything a CloudLab node needs to run the benchmarks (CPU or GPU).
#
# Idempotent: safe to re-run. Every step below exists because a measurement was wrong without
# it -- see docs/methodology.md "Platform pitfalls". Ubuntu 22.04 (CloudLab default image).
#
#   bash scripts/setup_node.sh            auto-detect: GPU if the NVIDIA driver works, else CPU
#   bash scripts/setup_node.sh cpu|gpu    force one
#
# GPU nodes: install the driver + CUDA toolkit FIRST (sudo bash scripts/install_cuda.sh; reboot).
set -euo pipefail
log()  { printf "\033[1;34m[setup]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[setup WARN]\033[0m %s\n" "$*"; }
if [ -n "${1:-}" ]; then
    KIND="$1"
elif nvidia-smi -L >/dev/null 2>&1; then
    KIND=gpu
elif lspci 2>/dev/null | grep -qi "nvidia"; then
    echo "An NVIDIA GPU is present but its driver is not working."
    echo "Run:  sudo bash scripts/install_cuda.sh && sudo reboot   -- then run this script again."
    exit 1
else
    KIND=cpu
fi
log "node type: $KIND ($(lscpu | sed -n 's/^Model name: *//p' | head -1))"
ARCH_DIR="/usr/lib/$(uname -m)-linux-gnu"

log "apt packages (build, NUMA, fastest-library candidates, crypto, extension headers)"
sudo apt-get update -y -q >/dev/null
sudo apt-get install -y -q --no-install-recommends build-essential cmake git numactl hwloc \
    linux-tools-common "linux-tools-$(uname -r)" python3-dev python3-pip \
    libopenblas-dev libopenblas-openmp-dev libblis3-openmp libtbb-dev libssl-dev libjemalloc2 cpufrequtils >/dev/null \
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
        12) IDX="cu129 cu128 cu126 cu124" ;;
        *)  IDX="" ; warn "nvcc version '${NVCC_VER}' not recognised: installing the default torch wheel" ;;
    esac
    # The wheel must also contain kernels for THIS GPU: PyTorch's CUDA >= 12.8 builds dropped
    # Volta/Pascal/Maxwell (no sm_70 code -> "no kernel image is available"). So each candidate
    # is installed cleanly and checked against the GPU's compute capability; the first that fits
    # is kept. (Found on CloudLab c4130, 4x V100.)
    torch_fits_gpu() {
        python3 - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    print("  torch sees no GPU (driver not loaded?)"); sys.exit(2)
maj, mi = torch.cuda.get_device_capability(0); cc = 10 * maj + mi
archs = torch.cuda.get_arch_list()
sm = [int(a[3:]) for a in archs if a.startswith("sm_")]
ptx = [int(a[8:]) for a in archs if a.startswith("compute_")]
ok = any(s // 10 == maj and s <= cc for s in sm) or any(p <= cc for p in ptx)
print(f"  {torch.cuda.get_device_name(0)} sm_{cc}; torch {torch.__version__} archs {archs}: "
      + ("has kernels for this GPU" if ok else "NO kernels for this GPU"))
sys.exit(0 if ok else 1)
PY
    }
    ok_torch=0
    for i in $IDX; do
        python3 -m pip uninstall -y -q torch >/dev/null 2>&1 || true
        python3 -m pip install -q --no-cache-dir torch --index-url "https://download.pytorch.org/whl/$i" || continue
        rc=0; torch_fits_gpu || rc=$?
        if [ "$rc" -eq 0 ]; then
            ok_torch=1; log "torch wheel from $i (nvcc ${NVCC_VER})"; break
        elif [ "$rc" -eq 2 ]; then
            ok_torch=1; warn "torch wheel from $i installed, but no GPU visible to check it: re-run after the driver loads"; break
        fi
        warn "torch wheel from $i has no kernels for this GPU: trying the next CUDA build"
    done
    TORCH_CUDA="$(python3 -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true)"
    [ "${TORCH_CUDA%%.*}" = "${NVCC_VER%%.*}" ] \
        || warn "torch CUDA ${TORCH_CUDA} vs nvcc ${NVCC_VER}: major versions differ -> native CUDA kernels will NOT build"
    python3 -m pip install -q --no-cache-dir nvidia-ml-py || warn "pynvml install failed (GPU power telemetry)"
fi
python3 -m pip install -q --no-cache-dir numpy scipy pytest ninja opencv-python-headless cryptography \
    || warn "python deps failed"
# analysis (make analysis) on the node. Unpinned on purpose: requirements.txt pins versions for the
# GitHub CI dashboard (e.g. contourpy 1.3.3 needs Python >= 3.11; Ubuntu 22.04 nodes have 3.10).
python3 -m pip install -q --no-cache-dir pandas matplotlib || warn "analysis deps (pandas, matplotlib) failed"

log "telemetry permissions (RAPL energy + APERF/MPERF MSRs)"
sudo modprobe msr 2>/dev/null || true
grep -qx msr /etc/modules 2>/dev/null || echo msr | sudo tee -a /etc/modules >/dev/null || true   # survives reboots
# powercap entries are symlinks: chmod the resolved files (root-only since CVE-2020-8694)
for f in /sys/class/powercap/intel-rapl*/energy_uj /sys/class/powercap/intel-rapl*/*/energy_uj; do
    [ -e "$f" ] && sudo chmod a+r "$(readlink -f "$f")" 2>/dev/null || true
done
PY_REAL="$(readlink -f "$(command -v python3)")"
sudo setcap cap_sys_rawio+ep "$PY_REAL" 2>/dev/null && log "CAP_SYS_RAWIO -> $PY_REAL" \
    || warn "setcap failed: CPU frequency falls back to cpufreq"
# CAP_SYS_RAWIO puts ld.so in secure mode: LD_PRELOAD needs a set-user-ID lib (opt-in jemalloc)
[ -e "$ARCH_DIR/libjemalloc.so.2" ] && sudo chmod u+s "$(readlink -f "$ARCH_DIR/libjemalloc.so.2")" || true

# performance governor (no DVFS ramp-up inside timed regions); /sys fallback without cpupower
if command -v cpupower >/dev/null 2>&1 && sudo cpupower frequency-set -g performance >/dev/null 2>&1; then
    log "CPU governor: performance (cpupower)"
else
    for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        [ -f "$g" ] && echo performance | sudo tee "$g" >/dev/null 2>&1 || true
    done
    [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ] \
        && log "CPU governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) (sysfs)" \
        || warn "no cpufreq interface (governor not set; frequency is still recorded per row)"
fi
if [ "$KIND" = gpu ] && command -v nvidia-smi >/dev/null 2>&1; then
    sudo nvidia-smi -pm 1 >/dev/null 2>&1 && log "GPU persistence mode on" || warn "persistence mode failed"
fi

log "verification"
PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)" python3 - <<'EOF'
import torch, sys
print(f"  torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
try:
    open("/sys/class/powercap/intel-rapl:0/energy_uj").read(); print("  RAPL energy readable: yes")
except OSError:
    print("  RAPL energy readable: NO (power will be NaN)")
EOF
log "done -- next: bash scripts/run_campaign.sh start <DEVICE>   (or step by step: make test && make smoke)"
