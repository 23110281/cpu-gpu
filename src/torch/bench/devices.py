"""CPU topology, thread placement, GPU selection (docs/methodology.md §1, §3, §6, §9).

`apply_cpu_placement` MUST run before `import torch` (OpenMP reads its environment
once, at first parallel region). Nothing here imports torch at module scope.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

SYS_CPU = Path("/sys/devices/system/cpu")


@dataclass(frozen=True)
class HwThread:
    cpu: int
    core: int
    socket: int
    node: int


@lru_cache(maxsize=1)
def topology() -> list[HwThread]:
    """Hardware threads from `lscpu -p` (falls back to one socket of os.cpu_count())."""
    try:
        out = subprocess.run(["lscpu", "-p=CPU,CORE,SOCKET,NODE"], capture_output=True,
                             text=True, check=True).stdout
        ts = []
        for line in out.splitlines():
            if not line or line.startswith("#"):
                continue
            cpu, core, sock, node = (int(v) if v else 0 for v in line.split(","))
            ts.append(HwThread(cpu, core, sock, node))
        if ts:
            return ts
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    return [HwThread(i, i, 0, 0) for i in range(os.cpu_count() or 1)]


def sockets() -> list[int]:
    return sorted({t.socket for t in topology()})


def _first_thread_per_core(threads):
    seen, out = set(), []
    for t in sorted(threads, key=lambda t: t.cpu):
        if (t.socket, t.core) not in seen:
            seen.add((t.socket, t.core))
            out.append(t)
    return out


@dataclass
class Placement:
    spec: str
    cpus: list[int]
    nodes: list[int]
    threads: int
    sockets: list[int] = field(default_factory=list)


PLACEMENT_RE = re.compile(r"^(socket(?P<s>\d+)|node)(?P<smt>-smt)?(:(?P<t>\d+))?$")


def resolve_placement(spec: str) -> Placement:
    """socket<S>[-smt][:T] | node[-smt][:T]  ->  concrete CPU list."""
    m = PLACEMENT_RE.match(spec)
    if not m:
        raise ValueError(f"bad placement {spec!r} (expected socketS[-smt][:T] or node[-smt][:T])")
    topo = topology()
    pool = [t for t in topo if m["s"] is None or t.socket == int(m["s"])]
    if not pool:
        raise ValueError(f"placement {spec!r}: no such socket (have {sockets()})")
    chosen = sorted(pool, key=lambda t: (t.cpu)) if m["smt"] else _first_thread_per_core(pool)
    if m["t"]:
        n = int(m["t"])
        if n > len(chosen):
            raise ValueError(f"placement {spec!r}: only {len(chosen)} threads available")
        chosen = chosen[:n]
    return Placement(spec=spec, cpus=[t.cpu for t in chosen],
                     nodes=sorted({t.node for t in chosen}), threads=len(chosen),
                     sockets=sorted({t.socket for t in chosen}))


def placement_env(p: Placement) -> dict:
    """Environment for the child process. OMP_PLACES lists the exact CPUs, one place each."""
    return {
        "OMP_NUM_THREADS": str(p.threads),
        "OMP_PLACES": ",".join("{%d}" % c for c in p.cpus),
        "OMP_PROC_BIND": "close",
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE",
        "MKL_NUM_THREADS": str(p.threads),
        "OPENBLAS_NUM_THREADS": str(p.threads),
        "BLIS_NUM_THREADS": str(p.threads),
        "KMP_AFFINITY": "disabled",  # let OMP_PLACES govern; avoid double binding
    }


JEMALLOC_PATHS = ("/usr/lib/x86_64-linux-gnu/libjemalloc.so.2", "/usr/lib/aarch64-linux-gnu/libjemalloc.so.2",
                  "/usr/lib64/libjemalloc.so.2")
# Never return freed memory to the OS and keep huge blocks in normal arenas: a freed output
# buffer is reused by the next call without page faults -- the CPU analogue of the CUDA
# caching allocator (methodology §4). Recommended for tuned PyTorch CPU runs.
JEMALLOC_CONF = "oversize_threshold:0,dirty_decay_ms:-1,muzzy_decay_ms:-1"


def cpu_allocator_env() -> dict:
    """Opt-in (BENCH_ALLOCATOR=jemalloc). Default is glibc: measured on EPYC 7402P, allocating
    1-3 GB outputs inside the call (conv2d 16K^2) costs nothing measurable with glibc and
    jemalloc was 0% / +2.7% slower (methodology, platform pitfalls).
    LD_PRELOAD by *name* (no slash): the interpreter carries CAP_SYS_RAWIO for MSR telemetry,
    which puts ld.so in secure mode, where only slash-free names from the standard library
    directories with the set-user-ID bit are preloaded (setup script sets it on jemalloc)."""
    if os.environ.get("BENCH_ALLOCATOR", "glibc") != "jemalloc":
        return {}
    for p in JEMALLOC_PATHS:
        if Path(p).exists():
            return {"LD_PRELOAD": Path(p).name, "MALLOC_CONF": JEMALLOC_CONF}
    return {}


def jemalloc_active() -> bool:
    """True only if jemalloc is actually mapped into this process (not just requested)."""
    try:
        return "libjemalloc" in Path("/proc/self/maps").read_text()
    except OSError:
        return False


def numactl_prefix(p: Placement) -> list[str]:
    """Memory binding prefix (empty if numactl is unavailable; recorded as a flag)."""
    if not shutil.which("numactl"):
        return []
    nodes = ",".join(map(str, p.nodes))
    cpus = ",".join(map(str, p.cpus))
    return ["numactl", f"--physcpubind={cpus}", f"--membind={nodes}"]


def apply_cpu_placement(p: Placement) -> list[str]:
    """In-process placement (must precede `import torch`). Returns flags describing gaps."""
    flags = []
    for k, v in placement_env(p).items():
        os.environ.setdefault(k, v)
    try:
        os.sched_setaffinity(0, set(p.cpus))
    except (AttributeError, OSError):
        flags.append("no_affinity")
    if os.environ.get("BENCH_MEMBIND") != ",".join(map(str, p.nodes)):
        flags.append("no_membind")
    return flags


# ── caches / working-set ─────────────────────────────────────────────────────
def _parse_size(s: str) -> int:
    s = s.strip().upper()
    mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    return int(s[:-1]) * mult[s[-1]] if s and s[-1] in mult else int(s)


@lru_cache(maxsize=None)
def llc_bytes_per_socket() -> int:
    """Total last-level cache of one socket (sum over distinct LLC instances)."""
    best_level, instances = 0, {}
    sock_of = {t.cpu: t.socket for t in topology()}
    for cpu_dir in SYS_CPU.glob("cpu[0-9]*"):
        cpu = int(cpu_dir.name[3:])
        if sock_of.get(cpu, 0) != sockets()[0]:
            continue
        for idx in (cpu_dir / "cache").glob("index*"):
            try:
                lvl = int((idx / "level").read_text())
                typ = (idx / "type").read_text().strip()
                if typ == "Instruction":
                    continue
                key = (idx / "shared_cpu_list").read_text().strip()
                size = _parse_size((idx / "size").read_text())
            except OSError:
                continue
            if lvl > best_level:
                best_level, instances = lvl, {}
            if lvl == best_level:
                instances[key] = size
    if not instances:
        return 32 << 20
    # A shared L3 never belongs to a single CPU on real hardware (Intel: one per socket,
    # AMD: one per CCX of 4-8 cores). Single-CPU "instances" are a virtualised topology
    # artefact that would multiply the LLC by the vCPU count -> count one instance.
    if best_level >= 3 and len(instances) > 1 and all(_ncpus(k) == 1 for k in instances):
        return next(iter(instances.values()))
    return sum(instances.values())


def _ncpus(cpu_list: str) -> int:
    n = 0
    for part in cpu_list.split(","):
        a, _, b = part.partition("-")
        n += (int(b) - int(a) + 1) if b else 1
    return n


def llc_bytes_for(placement: Placement | None) -> int:
    n = len(placement.sockets) if placement and placement.sockets else 1
    return llc_bytes_per_socket() * n


def gpu_llc_bytes(index: int) -> int:
    import torch
    props = torch.cuda.get_device_properties(index)
    return int(getattr(props, "L2_cache_size", 0) or 40 << 20)


# ── capability probes ────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def cpu_flags() -> set:
    try:
        txt = Path("/proc/cpuinfo").read_text()
    except OSError:
        return set()
    for key in ("flags", "Features"):
        m = re.search(rf"^{key}\s*:\s*(.*)$", txt, re.M)
        if m:
            return set(m.group(1).split())
    return set()


def native_bf16(device_class: str, gpu_index: int = 0) -> bool:
    """True only if bf16 math runs on native hardware (no emulation)."""
    if device_class == "cpu":
        f = cpu_flags()
        return bool({"avx512_bf16", "amx_bf16"} & f) or ("bf16" in f and platform.machine() == "aarch64")
    import torch
    return torch.cuda.get_device_capability(gpu_index) >= (8, 0)


def cpu_model() -> str:
    try:
        txt = Path("/proc/cpuinfo").read_text()
        m = re.search(r"^model name\s*:\s*(.*)$", txt, re.M)
        if m:
            return m.group(1).strip()
    except OSError:
        pass
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        m = re.search(r"^Model name:\s*(.*)$", out, re.M)
        if m:
            return m.group(1).strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def default_device_id(device_class: str, gpu_index: int = 0) -> str:
    if device_class == "cpu":
        name = cpu_model()
    else:
        import torch
        name = torch.cuda.get_device_name(gpu_index)
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def torch_backend_info() -> dict:
    """BLAS/LAPACK backend and parallel runtime as reported by torch itself."""
    import torch
    cfg = torch.__config__.show()
    blas = re.search(r"BLAS_INFO=([^,\s]+)", cfg)
    par = torch.__config__.parallel_info()
    return {
        "torch": torch.__version__,
        "blas": blas.group(1) if blas else "unknown",
        "mkldnn": bool(torch.backends.mkldnn.is_available()),
        "parallel_info": par,
        "config": cfg,
    }


def configure_torch_cpu(threads: int) -> None:
    """Thread counts are set exactly once per process (PyTorch forbids changing interop later)."""
    import torch
    torch.set_num_interop_threads(1)
    torch.set_num_threads(threads)


def configure_torch_cuda(gpu_index: int) -> None:
    import torch
    torch.cuda.set_device(gpu_index)
    # §8/§9: fp32 means fp32. TF32 only in explicit variant=ff rows.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
