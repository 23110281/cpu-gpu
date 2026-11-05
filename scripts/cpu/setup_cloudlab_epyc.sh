#!/usr/bin/env bash
# scripts/cpu/setup_cloudlab_epyc.sh
# Provisioning script for AMD EPYC bare-metal nodes (Ubuntu 20.04/22.04 on CloudLab)
#
# Configures:
#  1. Base build-essential and development utilities.
#  2. Open-source OpenBLAS (CBLAS & LAPACK with OpenMP support).
#  3. AMD Energy & Temperature kernel drivers (amd_energy, k10temp, msr).
#  4. CPU frequency scaling governor set to 'performance'.
#  5. Non-root read access to /sys/class/hwmon and /sys/class/powercap.

set -euo pipefail

log()  { printf "\033[1;34m[INFO]\033[0m %s\n" "$*"; }
ok()   { printf "\033[1;32m[OK]\033[0m   %s\n" "$*"; }
warn() { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[ERR]\033[0m  %s\n" "$*" >&2; exit 1; }

CPU_MODEL=$(lscpu | grep "Model name:" | sed 's/Model name:[ \t]*//' || echo "Unknown_AMD_CPU")
TOTAL_THREADS=$(nproc || echo "1")
KERNEL_VER=$(uname -r)

log "======================================================================"
log "      AMD EPYC CLOUDLAB PROVISIONING & OPENBLAS SETUP"
log "======================================================================"
log "CPU Model:      ${CPU_MODEL}"
log "Online Threads: ${TOTAL_THREADS}"
log "Linux Kernel:   ${KERNEL_VER}"
log "======================================================================"

if [ "${1:-}" = "--probe" ]; then
    log "Probe complete. Exiting."
    exit 0
fi

# ── 1. Base Development Tools & OS Prerequisites ─────────────────────────────
log "Step 1/4: Installing base build tools and hardware utilities"
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
    pkg-config \
    numactl \
    hwloc \
    cpufrequtils \
    python3 \
    python3-pip \
    python3-venv \
    linux-tools-common \
    linux-tools-generic || {
        warn "Could not install all generic linux-tools; continuing."
    }

# ── 2. Install OpenBLAS & PyTorch (CPU Wheel) ─────────────────────────────────
log "Step 2/4: Installing OpenBLAS and PyTorch CPU Runtime"
sudo apt-get install -y --no-install-recommends \
    libopenblas-dev \
    libopenblas-openmp-dev

ok "OpenBLAS installed (headers in /usr/include/x86_64-linux-gnu or /usr/include)."
# apt makes the *pthread* OpenBLAS the default. Its worker threads inherit the affinity of
# the calling thread, which OpenMP binding (OMP_PROC_BIND, used by the v2 harness) pins to
# ONE core -> all BLAS threads share a core (measured: 80 vs 1760 GFLOP/s, 24 cores).
# Select the OpenMP build so BLAS threads are OpenMP threads that honour OMP_PLACES.
for alt in libblas.so.3 liblapack.so.3 libopenblas.so.0; do
    target="/usr/lib/x86_64-linux-gnu/openblas-openmp/${alt}"
    [ -e "$target" ] && sudo update-alternatives --set "${alt}-x86_64-linux-gnu" "$target" >/dev/null \
        && ok "${alt} -> OpenMP OpenBLAS"
done

log "Installing PyTorch CPU wheel (~180MB lightweight package)"
python3 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu || {
    warn "PyTorch installation via pip encountered an issue; check pip network access."
}


# ── 3. Load AMD Hardware Telemetry & Energy Drivers ──────────────────────────
log "Step 3/4: Loading AMD hardware monitoring kernel modules (amd_energy, k10temp, msr)"
sudo modprobe msr 2>/dev/null || true
sudo modprobe amd_energy 2>/dev/null || warn "amd_energy kernel module not available in this kernel."
sudo modprobe k10temp 2>/dev/null || warn "k10temp kernel module not available in this kernel."

# Ensure modules reload on reboot
for mod in msr amd_energy k10temp; do
    if ! grep -q "^${mod}$" /etc/modules 2>/dev/null; then
        echo "$mod" | sudo tee -a /etc/modules >/dev/null || true
    fi
done

# Allow non-root read access to MSR and powercap/hwmon
sudo chmod a+r /dev/cpu/*/msr 2>/dev/null || true
if [ -d /sys/class/hwmon ]; then
    sudo chmod -R a+r /sys/class/hwmon/ 2>/dev/null || true
    ok "Read permissions enabled for /sys/class/hwmon."
fi
if [ -d /sys/class/powercap ]; then
    sudo chmod -R a+r /sys/class/powercap/ 2>/dev/null || true
    ok "Read permissions enabled for /sys/class/powercap."
fi

# ── 4. Set Performance Frequency Governor ───────────────────────────────────
# v2 harness telemetry (docs/methodology.md §11):
#  * powercap entries under /sys/class are symlinks, so `chmod -R` above does not reach
#    the real energy_uj files (root-only since CVE-2020-8694) -> chmod the resolved files.
#  * APERF/MPERF/TSC frequency and AMD RAPL MSRs need CAP_SYS_RAWIO, not just file mode:
#    grant it to the interpreter that runs the harness (disposable testbed node only).
for f in /sys/class/powercap/intel-rapl*/energy_uj /sys/class/powercap/intel-rapl*/*/energy_uj; do
    [ -e "$f" ] && sudo chmod a+r "$(readlink -f "$f")" 2>/dev/null || true
done
PY_REAL="$(readlink -f "$(command -v python3)")"
if command -v setcap >/dev/null 2>&1; then
    sudo setcap cap_sys_rawio+ep "$PY_REAL" && ok "CAP_SYS_RAWIO granted to $PY_REAL (MSR telemetry)." \
        || warn "setcap failed; CPU frequency telemetry will fall back to cpufreq."
fi
python3 -m pip install --no-cache-dir -q pytest scipy || warn "pytest/scipy install failed"
# v2 harness: jemalloc (CPU allocator reuse, methodology §4) and BLIS (fastest-BLAS candidate)
sudo apt-get install -y --no-install-recommends libjemalloc2 libblis3-openmp libtbb-dev python3-dev libssl-dev \
    || warn "jemalloc/BLIS/TBB/python3-dev install failed (affected variants report 'unsupported')"
# CAP_SYS_RAWIO (above) puts ld.so in secure mode, which ignores LD_PRELOAD unless the
# library is named without a slash, lives in a standard directory and is set-user-ID.
JEMALLOC_SO=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
[ -e "$JEMALLOC_SO" ] && sudo chmod u+s "$(readlink -f "$JEMALLOC_SO")" && ok "jemalloc preloadable under secure mode"
# ninja: builds the C++ extension for the parallel CPU sorts; OpenCV: fastest image filter
python3 -m pip install --no-cache-dir -q ninja opencv-python-headless cryptography || warn "ninja/opencv install failed"

log "Step 4/4: Setting CPU Frequency Governor to 'performance'"
if command -v cpupower >/dev/null 2>&1; then
    sudo cpupower frequency-set -g performance >/dev/null 2>&1 || true
    ok "CPU frequency governor set to 'performance' via cpupower."
elif [ -d /sys/devices/system/cpu/cpu0/cpufreq ]; then
    for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        [ -f "$gov" ] && echo "performance" | sudo tee "$gov" >/dev/null 2>&1 || true
    done
    ok "Scaling governor set to 'performance' via /sys."
fi

log "======================================================================"
ok  "AMD EPYC Provisioning Complete!"
log "To build all workloads with OpenBLAS, run: make BLAS=openblas cpu_all"
log "======================================================================"
