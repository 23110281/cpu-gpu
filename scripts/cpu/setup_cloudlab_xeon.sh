#!/usr/bin/env bash
# scripts/cpu/setup_cloudlab_xeon.sh
# Automated provisioning script for Intel Xeon nodes on CloudLab (Ubuntu 22.04 LTS).
#
# Idempotent: safe to re-run.
#
# Usage:
#   bash scripts/cpu/setup_cloudlab_xeon.sh          # Full install (needs sudo)
#   bash scripts/cpu/setup_cloudlab_xeon.sh --probe  # Probe system hardware & kernel only

set -euo pipefail

log() { printf '\n\033[1;34m[setup_xeon]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[setup_xeon:WARNING]\033[0m %s\n' "$*"; }
ok() { printf '\033[1;32m[setup_xeon:OK]\033[0m %s\n' "$*"; }

export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a

# ── 0. Hardware and Topology Probe ──────────────────────────────────────────
log "Step 0/5: Probing Node Hardware & Topology"

CPU_MODEL=$(lscpu | grep "Model name:" | sed 's/Model name:[ \t]*//' || echo "Unknown Xeon")
SOCKETS=$(lscpu | grep "Socket(s):" | awk '{print $2}' || echo "1")
CORES_PER_SOCKET=$(lscpu | grep "Core(s) per socket:" | awk '{print $4}' || echo "1")
TOTAL_THREADS=$(nproc || echo "1")
KERNEL_VER=$(uname -r)

echo "  • CPU Model:         ${CPU_MODEL}"
echo "  • Sockets:           ${SOCKETS}"
echo "  • Cores Per Socket:  ${CORES_PER_SOCKET}"
echo "  • Total Threads:     ${TOTAL_THREADS}"
echo "  • Linux Kernel:      ${KERNEL_VER}"

# Check for AVX-512 and AMX instructions
echo -n "  • AVX-512 Support:   "
if grep -q "avx512f" /proc/cpuinfo; then
    echo "YES (Found avx512f)"
else
    echo "NO"
fi

echo -n "  • Intel AMX Support: "
if grep -q "amx_tile" /proc/cpuinfo; then
    echo "YES (Found amx_tile, amx_bf16)"
else
    echo "NO (Standard AVX-512 mode will be used)"
fi

if [ "${1:-}" = "--probe" ]; then
    log "Probe complete. Exiting."
    exit 0
fi

# ── 1. Base Development Tools & OS Prerequisites ─────────────────────────────
log "Step 1/5: Installing base build tools and hardware utilities"
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    make \
    cmake \
    git \
    wget \
    gpg \
    pkg-config \
    numactl \
    hwloc \
    cpufrequtils \
    python3 \
    python3-pip \
    python3-venv \
    linux-tools-common \
    linux-tools-generic \
    linux-tools-"${KERNEL_VER}" || {
        warn "Could not install exact linux-tools for ${KERNEL_VER}; continuing with generic tools"
    }

log "Installing PyTorch CPU wheel (~180MB lightweight package)"
python3 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu || {
    warn "PyTorch installation via pip encountered an issue; check pip network access."
}


# ── 2. Add Intel oneAPI Official APT Repository ──────────────────────────────
log "Step 2/5: Configuring Intel oneAPI APT repository"
KEYRING_PATH="/usr/share/keyrings/oneapi-archive-keyring.gpg"
if [ ! -f "${KEYRING_PATH}" ]; then
    wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | gpg --dearmor | sudo tee "${KEYRING_PATH}" >/dev/null
fi

echo "deb [signed-by=${KEYRING_PATH}] https://apt.repos.intel.com/oneapi all main" \
    | sudo tee /etc/apt/sources.list.d/oneAPI.list >/dev/null

sudo apt-get update -y

# ── 3. Install Intel oneMKL and Compiler Runtime ─────────────────────────────
log "Step 3/5: Installing Intel oneMKL (CBLAS, LAPACK, Sparse BLAS)"
sudo apt-get install -y --no-install-recommends \
    intel-oneapi-mkl \
    intel-oneapi-mkl-devel \
    intel-oneapi-openmp

# ── 4. Set Environment Variables for all Login Shells ────────────────────────
log "Step 4/5: Creating environment profile hook in /etc/profile.d/oneapi_vars.sh"
ONEAPI_SETVARS="/opt/intel/oneapi/setvars.sh"

if [ -f "${ONEAPI_SETVARS}" ]; then
    sudo tee /etc/profile.d/oneapi_vars.sh >/dev/null <<'EOF'
# Source Intel oneAPI environment variables for non-interactive and interactive shells
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
fi
EOF
    sudo chmod +x /etc/profile.d/oneapi_vars.sh
    ok "Profile script /etc/profile.d/oneapi_vars.sh configured."
else
    warn "${ONEAPI_SETVARS} not found in default path; check installation."
fi

# ── 5. System Tuning: RAPL Power Telemetry & Performance Governor ────────────
log "Step 5/5: Tuning CPU Performance Governor and RAPL Power Permissions"

# A. Set performance governor across all cores to eliminate DVFS latency
if command -v cpupower >/dev/null 2>&1; then
    sudo cpupower frequency-set -g performance >/dev/null 2>&1 || true
    ok "CPU frequency governor set to 'performance'."
elif [ -d /sys/devices/system/cpu/cpu0/cpufreq ]; then
    for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        [ -f "$gov" ] && echo "performance" | sudo tee "$gov" >/dev/null 2>&1 || true
    done
    ok "Scaling governor set to 'performance' via /sys."
fi

# B. Allow non-root read access to Intel RAPL energy counters
if [ -d /sys/class/powercap/intel-rapl ]; then
    sudo chmod -R a+r /sys/class/powercap/intel-rapl/ 2>/dev/null || true
    ok "RAPL powercap read permissions enabled for non-root users."
else
    warn "Intel RAPL (/sys/class/powercap/intel-rapl) not found; power telemetry may require MSR drivers."
fi

cat <<'EOF'

──────────────────────────────────────────────────────────────────────────────
  Intel Xeon Provisioning Complete!
  
  To load oneMKL in your current terminal session immediately, run:
      source /opt/intel/oneapi/setvars.sh
  
  Verify installation:
      echo $MKLROOT
      numactl --hardware
──────────────────────────────────────────────────────────────────────────────
EOF
