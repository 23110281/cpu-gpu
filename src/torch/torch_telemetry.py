"""src/torch/torch_telemetry.py

Hardware telemetry reader for CPU (Intel Xeon, AMD EPYC) and NVIDIA GPU architectures.
Mirrors src/cpu/cpu_telemetry.h in pure Python with non-crashing fallbacks.

Provides:
  1. CPU Socket Energy & Power (Watts):
     - Intel: /sys/class/powercap/intel-rapl/intel-rapl:X/energy_uj
     - AMD: /dev/cpu/0/msr (MSR 0xC0010299 & 0xC0010061) or /sys/class/hwmon
  2. CPU Core Frequency (avg active MHz across cores via sysfs or /proc/cpuinfo).
  3. CPU Temperature (peak degrees Celsius via thermal_zone or hwmon).
  4. GPU Telemetry (power, clock, temp via pynvml / ctypes NVML).
  5. Monotonic high-resolution wall-clock timing (time.perf_counter).
"""

import os
import sys
import glob
import time
import struct
import platform

# Constants for AMD & Intel MSR
AMD_MSR_PWR_UNIT = 0xC0010061
AMD_MSR_PKG_ENERGY = 0xC0010299
INTEL_MSR_PWR_UNIT = 0x00000606
INTEL_MSR_PKG_ENERGY = 0x00000611


class TelemetrySnapshot:
    def __init__(self):
        self.timestamp = time.perf_counter()
        self.rapl_energies_uj = []
        self.msr_energy_ticks = []
        self.gpu_energy_mj = None
        self.avg_freq_mhz = 0.0
        self.max_temp_c = 0.0


class HardwareTelemetry:
    def __init__(self, device_type="cpu", gpu_id=0):
        self.device_type = str(device_type).lower()
        self.gpu_id = gpu_id
        self.is_linux = platform.system() == "Linux"

        self.rapl_paths = []
        self.rapl_max_ranges = []
        self.use_msr = False
        self.msr_fd = -1
        self.msr_energy_unit_j = 0.0

        self.nvml_initialized = False
        self.nvml_handle = None

        if self.device_type.startswith("cuda") or self.device_type == "gpu":
            self._init_gpu_nvml()
        else:
            if self.is_linux:
                self._init_cpu_telemetry()

    def _init_gpu_nvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
            self.nvml_initialized = True
        except Exception:
            self.nvml_initialized = False

    def _init_cpu_telemetry(self):
        # 1. Probe Intel RAPL sysfs
        intel_rapl_dirs = sorted(glob.glob("/sys/class/powercap/intel-rapl/intel-rapl:*"))
        for d in intel_rapl_dirs:
            energy_f = os.path.join(d, "energy_uj")
            max_range_f = os.path.join(d, "max_energy_range_uj")
            if os.path.exists(energy_f):
                try:
                    with open(energy_f, "r") as f:
                        f.read().strip()
                    max_range = 0
                    if os.path.exists(max_range_f):
                        with open(max_range_f, "r") as f:
                            max_range = int(f.read().strip())
                    self.rapl_paths.append(energy_f)
                    self.rapl_max_ranges.append(max_range)
                except (IOError, PermissionError):
                    pass

        # 2. If no RAPL sysfs, probe AMD MSR
        if not self.rapl_paths:
            msr_path = f"/dev/cpu/{self.gpu_id}/msr"
            if os.path.exists(msr_path):
                try:
                    fd = os.open(msr_path, os.O_RDONLY)
                    buf = os.pread(fd, 8, AMD_MSR_PWR_UNIT)
                    if len(buf) == 8:
                        val = struct.unpack("<Q", buf)[0]
                        energy_unit_exp = (val >> 8) & 0x1F
                        self.msr_energy_unit_j = 0.5 ** energy_unit_exp
                        self.msr_fd = fd
                        self.use_msr = True
                except (OSError, PermissionError):
                    if self.msr_fd >= 0:
                        os.close(self.msr_fd)
                        self.msr_fd = -1

    def _read_cpu_freq_mhz(self):
        if not self.is_linux:
            return 0.0
        # Probe scaling_cur_freq
        cur_freqs = glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq")
        if cur_freqs:
            total_khz = 0.0
            count = 0
            for path in cur_freqs:
                try:
                    with open(path, "r") as f:
                        total_khz += float(f.read().strip())
                        count += 1
                except (IOError, ValueError):
                    pass
            if count > 0:
                return (total_khz / count) / 1000.0

        # Fallback to /proc/cpuinfo
        try:
            mhz_list = []
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("cpu MHz"):
                        parts = line.split(":")
                        if len(parts) == 2:
                            mhz_list.append(float(parts[1].strip()))
            if mhz_list:
                return sum(mhz_list) / len(mhz_list)
        except (IOError, ValueError):
            pass
        return 0.0

    def _read_cpu_temp_c(self):
        if not self.is_linux:
            return 0.0
        max_c = 0.0
        # Check thermal zones
        zones = glob.glob("/sys/class/thermal/thermal_zone*/temp")
        for path in zones:
            try:
                with open(path, "r") as f:
                    val = float(f.read().strip()) / 1000.0
                    if 0.0 < val < 150.0 and val > max_c:
                        max_c = val
            except (IOError, ValueError):
                pass
        # Check hwmon coretemp / k10temp
        hwmon_inputs = glob.glob("/sys/class/hwmon/hwmon*/temp*_input")
        for path in hwmon_inputs:
            try:
                with open(path, "r") as f:
                    val = float(f.read().strip()) / 1000.0
                    if 0.0 < val < 150.0 and val > max_c:
                        max_c = val
            except (IOError, ValueError):
                pass
        return max_c

    def _take_snapshot(self):
        snap = TelemetrySnapshot()
        snap.timestamp = time.perf_counter()

        if self.device_type.startswith("cuda") or self.device_type == "gpu":
            if self.nvml_initialized:
                try:
                    import pynvml
                    # Read instantaneous power (mW)
                    mw = pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle)
                    snap.gpu_energy_mj = mw  # instantaneous mW
                    snap.avg_freq_mhz = pynvml.nvmlDeviceGetClockInfo(
                        self.nvml_handle, pynvml.NVML_CLOCK_SM
                    )
                    snap.max_temp_c = pynvml.nvmlDeviceGetTemperature(
                        self.nvml_handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except Exception:
                    pass
            return snap

        # CPU Readings
        if self.rapl_paths:
            for p in self.rapl_paths:
                try:
                    with open(p, "r") as f:
                        snap.rapl_energies_uj.append(int(f.read().strip()))
                except (IOError, ValueError):
                    snap.rapl_energies_uj.append(0)

        elif self.use_msr and self.msr_fd >= 0:
            try:
                buf = os.pread(self.msr_fd, 8, AMD_MSR_PKG_ENERGY)
                if len(buf) == 8:
                    ticks = struct.unpack("<I", buf[:4])[0]  # 32-bit energy counter
                    snap.msr_energy_ticks.append(ticks)
            except OSError:
                pass

        snap.avg_freq_mhz = self._read_cpu_freq_mhz()
        snap.max_temp_c = self._read_cpu_temp_c()
        return snap

    def start(self):
        self._start_snap = self._take_snapshot()

    def stop(self):
        self._stop_snap = self._take_snapshot()

        wall_sec = self._stop_snap.timestamp - self._start_snap.timestamp
        if wall_sec <= 0:
            wall_sec = 1e-9

        self.elapsed_sec = wall_sec
        self.elapsed_ms = wall_sec * 1000.0

        # Calculate power
        self.avg_power_watts = 0.0
        if self.device_type.startswith("cuda") or self.device_type == "gpu":
            if self._stop_snap.gpu_energy_mj is not None:
                self.avg_power_watts = self._stop_snap.gpu_energy_mj / 1000.0  # mW -> W
            self.avg_freq_mhz = self._stop_snap.avg_freq_mhz
            self.max_temp_c = self._stop_snap.max_temp_c
            return

        # CPU Power
        total_energy_joules = 0.0
        if self.rapl_paths and len(self._start_snap.rapl_energies_uj) == len(self.rapl_paths):
            for i in range(len(self.rapl_paths)):
                s0 = self._start_snap.rapl_energies_uj[i]
                s1 = self._stop_snap.rapl_energies_uj[i]
                max_rng = self.rapl_max_ranges[i]
                delta_uj = (s1 - s0) if s1 >= s0 else (s1 + (max_rng - s0))
                total_energy_joules += delta_uj * 1e-6
            self.avg_power_watts = total_energy_joules / wall_sec

        elif self.use_msr and self._start_snap.msr_energy_ticks and self._stop_snap.msr_energy_ticks:
            t0 = self._start_snap.msr_energy_ticks[0]
            t1 = self._stop_snap.msr_energy_ticks[0]
            delta_ticks = (t1 - t0) if t1 >= t0 else (t1 + (0x100000000 - t0))
            total_energy_joules = delta_ticks * self.msr_energy_unit_j
            self.avg_power_watts = total_energy_joules / wall_sec

        self.avg_freq_mhz = (self._start_snap.avg_freq_mhz + self._stop_snap.avg_freq_mhz) / 2.0
        self.max_temp_c = max(self._start_snap.max_temp_c, self._stop_snap.max_temp_c)

    def close(self):
        if self.msr_fd >= 0:
            try:
                os.close(self.msr_fd)
            except OSError:
                pass
            self.msr_fd = -1
        if self.nvml_initialized:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self.nvml_initialized = False


def get_system_host_info():
    """Retrieve hostname and processor name cleanly."""
    host = platform.node() or "Unknown_Host"
    cpu_model = platform.processor() or "Unknown_CPU"
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        except IOError:
            pass
    return host.replace(" ", "_"), cpu_model.replace(" ", "_")
