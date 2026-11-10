"""Provenance record written once per process (docs/methodology.md §12)."""
from __future__ import annotations

import datetime as _dt
import importlib
import json
import os
import platform
import socket
import subprocess
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _read(p: str) -> str | None:
    try:
        return Path(p).read_text().strip()
    except OSError:
        return None


def _run(cmd: list[str]) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def new_run_id() -> str:
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{socket.gethostname().split('.')[0]}_{uuid.uuid4().hex[:6]}"


def git_state() -> tuple[str, bool]:
    """(HEAD sha, dirty). Dirty = any modified tracked file, or any untracked file under the
    code/config trees (uncommitted harness code would otherwise be invisible to the SHA)."""
    sha = _run(["git", "-C", str(REPO), "rev-parse", "HEAD"]) or "unknown"
    tracked = _run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"])
    untracked = _run(["git", "-C", str(REPO), "ls-files", "--others", "--exclude-standard",
                      "--", "bench", "analysis", "configs", "scripts", "crosscheck", "Makefile"])
    if tracked is None or untracked is None:
        return sha, True
    return sha, bool(tracked or untracked)


def _lib_versions() -> dict:
    out = {}
    for mod in ("torch", "numpy", "scipy", "cv2", "pyfftw", "lbmpy", "pystencils",
                "cryptography", "cupy", "pynvml"):
        try:
            m = importlib.import_module(mod)
            out[mod] = getattr(m, "__version__", "present")
        except Exception:
            pass
    return out


def collect(run_id: str, device_class: str, argv: list[str], extra: dict | None = None) -> dict:
    sha, dirty = git_state()
    cpu0 = "/sys/devices/system/cpu/cpu0/cpufreq"
    rec = {
        "run_id": run_id,
        "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_sha": sha,
        "git_dirty": dirty,
        "argv": argv,
        "python": sys.version,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "lscpu": _run(["lscpu"]),
        "microcode": _run(["sh", "-c", "grep -m1 microcode /proc/cpuinfo"]),
        "bios": _read("/sys/class/dmi/id/bios_version"),
        "board": _read("/sys/class/dmi/id/product_name"),
        "governor": _read(f"{cpu0}/scaling_governor"),
        "intel_no_turbo": _read("/sys/devices/system/cpu/intel_pstate/no_turbo"),
        "cpufreq_boost": _read("/sys/devices/system/cpu/cpufreq/boost"),
        "smt_active": _read("/sys/devices/system/cpu/smt/active"),
        "thp": _read("/sys/kernel/mm/transparent_hugepage/enabled"),
        "numa_balancing": _read("/proc/sys/kernel/numa_balancing"),
        "numactl_hardware": _run(["numactl", "-H"]),
        "meminfo_total": _run(["sh", "-c", "grep MemTotal /proc/meminfo"]),
        "env": {k: v for k, v in os.environ.items()
                if k.startswith(("OMP_", "MKL_", "KMP_", "OPENBLAS_", "GOMP_", "CUDA_", "BENCH_"))},
        "libs": _lib_versions(),
    }
    try:
        from . import devices
        rec["torch_backend"] = devices.torch_backend_info()
    except Exception as e:  # torch missing is fatal elsewhere; keep the record
        rec["torch_backend"] = {"error": repr(e)}
    if device_class == "gpu":
        rec["nvidia_smi"] = _run(["nvidia-smi", "--query-gpu=index,name,driver_version,pci.bus_id,"
                                  "persistence_mode,ecc.mode.current,power.limit,clocks.max.sm,"
                                  "clocks.max.mem,clocks.applications.graphics,vbios_version",
                                  "--format=csv"])
        try:
            import torch
            rec["cuda"] = {"torch_cuda": torch.version.cuda,
                           "cudnn": torch.backends.cudnn.version(),
                           "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
        except Exception as e:
            rec["cuda"] = {"error": repr(e)}
    if extra:
        rec.update(extra)
    return rec


HEAVY_KEYS = ("lscpu", "numactl_hardware", "nvidia_smi")


def light(rec: dict) -> dict:
    """Per-process record when a full campaign record exists: drop bulky host dumps."""
    out = {k: v for k, v in rec.items() if k not in HEAVY_KEYS}
    tb = out.get("torch_backend")
    if isinstance(tb, dict):
        out["torch_backend"] = {k: v for k, v in tb.items() if k not in ("config", "parallel_info")}
    return out


def write(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2, sort_keys=True, default=str))
