#!/usr/bin/env bash
# install_cuda.sh — NVIDIA driver + CUDA toolkit + Nsight for the GPU GEMM bench.
#
# Target: Ubuntu 22.04, kernel 5.15, Secure Boot OFF, 4x A100-SXM4-40GB.
# Does EVERYTHING up to (but not including) the reboot. nouveau is unloaded
# only by the reboot, so the NVIDIA driver becomes active after you reboot.
#
# Usage:  bash install_cuda.sh           # full install (needs sudo; passwordless OK)
#         bash install_cuda.sh --probe   # only show what would be installed
#
# Idempotent: safe to re-run. Logs to third_party/gpu_bench/results/install.log
set -euo pipefail

CUDA_REPO="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64"
KEYRING_DEB="cuda-keyring_1.1-1_all.deb"
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a              # never interactively prompt for restarts
APT_OPTS=(-y --allow-change-held-packages \
          -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold)

log() { printf '\n\033[1;36m[install_cuda] %s\033[0m\n' "$*"; }

# ── 1. repo ────────────────────────────────────────────────────────────────
log "1/7 Adding NVIDIA CUDA apt repository"
cd /tmp
if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
    wget -q "${CUDA_REPO}/${KEYRING_DEB}"
    sudo dpkg -i "${KEYRING_DEB}"
fi
sudo apt-get update -y

# ── 2. pick latest available toolkit + driver metapackages ─────────────────
TOOLKIT_PKG="$(apt-cache search --names-only '^cuda-toolkit-12-[0-9]+$' \
              | awk '{print $1}' | sort -V | tail -1)"
[ -n "${TOOLKIT_PKG}" ] || { echo "no cuda-toolkit-12-* found in repo"; exit 1; }
DRIVER_PKG="cuda-drivers"              # pulls the latest DKMS data-center driver
log "Selected toolkit='${TOOLKIT_PKG}' driver='${DRIVER_PKG}'"

if [ "${1:-}" = "--probe" ]; then
    log "PROBE only — would install: ${TOOLKIT_PKG} ${DRIVER_PKG} nsight-compute nsight-systems"
    exit 0
fi

# ── 3. build prerequisites (DKMS needs headers for the running kernel) ─────
log "2/7 Installing build prerequisites"
sudo apt-get install "${APT_OPTS[@]}" build-essential dkms "linux-headers-$(uname -r)"

# ── 4. toolkit + driver + profilers ────────────────────────────────────────
log "3/7 Installing ${TOOLKIT_PKG} + ${DRIVER_PKG} (downloads several GB) ..."
sudo apt-get install "${APT_OPTS[@]}" "${TOOLKIT_PKG}" "${DRIVER_PKG}"
log "4/7 Installing Nsight Compute + Nsight Systems"
sudo apt-get install "${APT_OPTS[@]}" nsight-compute nsight-systems || \
    log "nsight meta-packages not separately available (likely bundled in toolkit)"

# ── 5. blacklist nouveau (reboot makes this take effect) ───────────────────
log "5/7 Blacklisting nouveau"
sudo tee /etc/modprobe.d/blacklist-nouveau.conf >/dev/null <<'EOF'
blacklist nouveau
options nouveau modeset=0
EOF
sudo update-initramfs -u

# ── 6. allow non-root GPU performance counters (ncu) ───────────────────────
log "6/7 Enabling non-root profiling counters for Nsight Compute"
sudo tee /etc/modprobe.d/nvidia-profiler.conf >/dev/null <<'EOF'
options nvidia NVreg_RestrictProfilingToAdminUsers=0
EOF

# ── 7. PATH / LD_LIBRARY_PATH for /usr/local/cuda ──────────────────────────
log "7/7 Adding CUDA to PATH (login shells)"
sudo tee /etc/profile.d/cuda.sh >/dev/null <<'EOF'
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
EOF

cat <<'EOF'

──────────────────────────────────────────────────────────────────────────────
  Installable steps complete.  A REBOOT is now required so that:
    • nouveau is unloaded and the NVIDIA driver loads,
    • NVreg_RestrictProfilingToAdminUsers=0 takes effect.

      sudo reboot

  After reboot, verify:
      nvidia-smi
      nvcc --version
      ncu --version
      sudo nvidia-smi -pm 1          # persistence mode (stable clocks)

  Then build + run the bench:
      cd third_party/gpu_bench && make
      ./scripts/run_sweep.sh
──────────────────────────────────────────────────────────────────────────────
EOF
