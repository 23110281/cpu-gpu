"""Background power / clock / temperature sampling (docs/methodology.md §11).

CPU energy: Linux powercap RAPL (`/sys/class/powercap/intel-rapl:*`, also used by the
AMD RAPL driver), falling back to AMD MSRs (energy 0xC001029B, unit 0xC0010299) via
/dev/cpu/*/msr. Counters are differenced with wraparound handling at every sample, so
multiple wraps within a window are counted correctly.
CPU frequency: effective busy MHz from APERF/MPERF/TSC MSRs (turbostat's method),
falling back to cpufreq scaling_cur_freq when MSRs are not readable.
GPU: NVML (power, SM clock, temperature, throttle reasons; total-energy counter when
available). Anything unreadable is NaN and `power_valid` is 0 — never a silent 0 W.
"""
from __future__ import annotations

import math
import os
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

POWERCAP = Path("/sys/class/powercap")
MIN_VALID_WINDOW_S = 0.1
AMD_MSR_PWR_UNIT = 0xC0010299
AMD_MSR_PKG_ENERGY = 0xC001029B
NVML_THROTTLE_IDLE = 0x1


@dataclass
class _Counter:
    """A monotonically increasing energy counter in joules with a known wrap range."""
    name: str
    read_j: callable
    wrap_j: float
    last: float = math.nan
    total: float = 0.0

    def sample(self) -> None:
        v = self.read_j()
        if not math.isnan(self.last):
            d = v - self.last
            if d < 0:
                d += self.wrap_j
            self.total += d
        self.last = v


def _powercap_counters(sockets: set[int] | None) -> list[_Counter]:
    out = []
    for dom in sorted(POWERCAP.glob("intel-rapl:*")):
        parts = dom.name.split(":")
        try:
            pkg = int(parts[1])
            name = (dom / "name").read_text().strip()
            e = dom / "energy_uj"
            int(e.read_text())  # permission check
            wrap = int((dom / "max_energy_range_uj").read_text()) / 1e6
        except (OSError, ValueError, IndexError):
            continue
        if sockets is not None and pkg not in sockets:
            continue
        # top-level package zones and their DRAM sub-zones; skip core/uncore (inside package)
        if len(parts) == 2 or name == "dram":
            out.append(_Counter(f"pkg{pkg}:{name}", lambda e=e: int(e.read_text()) / 1e6, wrap))
    return out


def _amd_msr_counters(socket_cpus: dict[int, int]) -> list[_Counter]:
    out = []
    for sock, cpu in socket_cpus.items():
        path = f"/dev/cpu/{cpu}/msr"
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            unit_raw = struct.unpack("<Q", os.pread(fd, 8, AMD_MSR_PWR_UNIT))[0]
            esu = (unit_raw >> 8) & 0x1F
            joules_per_tick = 1.0 / (1 << esu)

            def rd(fd=fd, j=joules_per_tick):
                raw = struct.unpack("<Q", os.pread(fd, 8, AMD_MSR_PKG_ENERGY))[0] & 0xFFFFFFFF
                return raw * j
            rd()
            out.append(_Counter(f"pkg{sock}:msr", rd, (1 << 32) * joules_per_tick))
        except OSError:
            os.close(fd)
    return out


MSR_TSC, MSR_MPERF, MSR_APERF = 0x10, 0xE7, 0xE8


class _AperfMperf:
    """Effective (busy) core frequency from APERF/MPERF/TSC MSRs, the method used by
    turbostat:  MHz = dAPERF/dMPERF * dTSC/dt. Works without a cpufreq driver."""

    def __init__(self, cpus):
        self.fds = []
        for c in list(cpus)[:64]:
            try:
                self.fds.append(os.open(f"/dev/cpu/{c}/msr", os.O_RDONLY))
            except OSError:
                pass
        self.prev = None

    def _read(self):
        t = time.perf_counter()
        vals = []
        for fd in self.fds:
            r = [struct.unpack("<Q", os.pread(fd, 8, a))[0] for a in (MSR_TSC, MSR_MPERF, MSR_APERF)]
            vals.append(r)
        return t, vals

    def sample(self) -> float:
        if not self.fds:
            return math.nan
        try:
            cur = self._read()
        except OSError:          # e.g. no CAP_SYS_RAWIO: disable so cpufreq fallback is used
            self.close()
            self.fds = []
            return math.nan
        prev, self.prev = self.prev, cur
        if prev is None:
            return math.nan
        dt = cur[0] - prev[0]
        f = []
        for (tsc0, m0, a0), (tsc1, m1, a1) in zip(prev[1], cur[1]):
            dm, da, dtsc = m1 - m0, a1 - a0, tsc1 - tsc0
            if dm > 0 and dt > 0:
                f.append(da / dm * dtsc / dt / 1e6)
        return sum(f) / len(f) if f else math.nan

    def close(self):
        for fd in self.fds:
            os.close(fd)


def _cpu_freq_mhz(cpus) -> float:
    vals = []
    for c in cpus:
        try:
            vals.append(int(Path(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq").read_text()) / 1e3)
        except (OSError, ValueError):
            pass
    return sum(vals) / len(vals) if vals else math.nan


def _cpu_temp_c() -> float:
    for hw in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            if (hw / "name").read_text().strip() not in ("coretemp", "k10temp", "zenpower"):
                continue
            t = [int(f.read_text()) / 1e3 for f in hw.glob("temp*_input")]
            if t:
                return max(t)
        except (OSError, ValueError):
            continue
    return math.nan


class Sampler:
    """Context manager sampling during the measured window.

    with Sampler("cpu", cpus=[...], sockets={0}) as s: ... ; s.result(calls)
    """

    def __init__(self, device_class: str, cpus=(), sockets=None, gpu_index: int = 0,
                 period_s: float | None = None):
        self.device_class = device_class
        self.cpus = list(cpus)
        self.period = period_s or (0.01 if device_class == "gpu" else 0.05)
        self.gpu_index = gpu_index
        self.counters: list[_Counter] = []
        self.flags: list[str] = []
        self._freq, self._temp, self._power, self._throttle = [], [], [], 0
        self._stop = threading.Event()
        self._nvml = None
        self._am = None
        if device_class == "cpu":
            self._am = _AperfMperf(self.cpus)
            self.counters = _powercap_counters(sockets)
            if not self.counters:
                from . import devices
                first = {}
                for t in devices.topology():
                    if sockets is None or t.socket in sockets:
                        first.setdefault(t.socket, t.cpu)
                self.counters = _amd_msr_counters(first)
            if not self.counters:
                self.flags.append("no_rapl")
        else:
            try:
                import pynvml
                pynvml.nvmlInit()
                self._nvml = pynvml
                self._h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                try:
                    pynvml.nvmlDeviceGetTotalEnergyConsumption(self._h)
                    self.counters = [_Counter("gpu:energy",
                                              lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(self._h) / 1e3,
                                              float("inf"))]
                except pynvml.NVMLError:
                    pass
            except Exception:
                self.flags.append("no_nvml")

    def _tick(self):
        for c in self.counters:
            try:
                c.sample()
            except (OSError, ValueError):
                c.last = math.nan
                self.flags.append(f"read_fail:{c.name}")
        if self.device_class == "cpu":
            f = self._am.sample() if self._am and self._am.fds else math.nan
            if math.isnan(f) and not (self._am and self._am.fds):
                f = _cpu_freq_mhz(self.cpus)         # cpufreq fallback (no MSR access)
            self._freq.append(f)
            self._temp.append(_cpu_temp_c())
        elif self._nvml:
            n, h = self._nvml, self._h
            try:
                self._power.append(n.nvmlDeviceGetPowerUsage(h) / 1e3)
                self._freq.append(n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM))
                self._temp.append(n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU))
                self._throttle |= n.nvmlDeviceGetCurrentClocksThrottleReasons(h) & ~NVML_THROTTLE_IDLE
            except n.NVMLError:
                self.flags.append("nvml_read_fail")

    def _loop(self):
        while not self._stop.wait(self.period):
            self._tick()

    def __enter__(self):
        self._tick()
        self._t0 = time.perf_counter()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._th.join()
        self._tick()
        self.window_s = time.perf_counter() - self._t0
        if self._am:
            self._am.close()
        return False

    @staticmethod
    def _mean(v):
        v = [x for x in v if not math.isnan(x)]
        return sum(v) / len(v) if v else math.nan

    def result(self, calls: int) -> dict:
        energy = sum(c.total for c in self.counters) if self.counters else math.nan
        if self.counters:
            power = energy / self.window_s if self.window_s > 0 else math.nan
        elif self._power:
            power = self._mean(self._power)
            energy = power * self.window_s
        else:
            power = math.nan
        valid = int(self.window_s >= MIN_VALID_WINDOW_S and not math.isnan(power)
                    and not any(f.startswith("read_fail") for f in self.flags))
        return {
            "power_w": power,
            "energy_j_per_op": energy / calls if calls and not math.isnan(energy) else math.nan,
            "power_valid": valid,
            "freq_mhz": self._mean(self._freq),
            "temp_c": self._mean(self._temp),
            "throttle": hex(self._throttle) if self.device_class == "gpu" else "",
            "_flags": sorted(set(self.flags)),
        }
