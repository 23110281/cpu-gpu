"""Helpers shared by workload modules."""
from __future__ import annotations

import contextlib
import os

import torch

from ..core import Ctx, Unsupported

DTYPE = {"fp64": torch.float64, "fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
ELEM = {"fp64": 8, "fp32": 4, "bf16": 2, "fp16": 2}


def first_touch(n_or_shape, dtype, ctx: Ctx) -> torch.Tensor:
    """Allocate uninitialised and write once with a parallel ATen op, so that on CPU each page
    is first touched by the (pinned) thread that will later use it (methodology §3)."""
    t = torch.empty(n_or_shape, dtype=dtype, device=ctx.device)
    t.fill_(0)
    return t


def placement_cpus(ctx: Ctx) -> set | None:
    """The placement's CPU set (CPU device), or None."""
    if ctx.device_class != "cpu":
        return None
    from .. import devices
    return set(devices.resolve_placement(ctx.placement).cpus)


@contextlib.contextmanager
def widened_affinity(cpus):
    """Give the calling thread every CPU of the placement for the duration, then restore.
    torch's OpenMP binding (OMP_PROC_BIND) pins the main thread to ONE core, and pthread-based
    libraries (OpenCV, TBB) start their workers with the caller's mask -> all on one core
    (measured on EPYC 9354P: OpenCV filter2D 634 vs 6109 Mpx/s). Must wrap both the library's
    pool creation (setNumThreads / first call) and every call."""
    if not cpus:
        yield
        return
    saved = os.sched_getaffinity(0)
    os.sched_setaffinity(0, cpus)
    try:
        yield
    finally:
        os.sched_setaffinity(0, saved)


def cuda_declarations(cuda_src: str, functions) -> str:
    """C++ declarations of the functions a load_inline CUDA extension exposes.

    load_inline writes the Python bindings into main.cpp, compiled by the host C++ compiler, which
    never sees cuda_sources: every exposed function must be DECLARED in cpp_sources, or main.cpp
    fails with "'<fn>' was not declared in this scope" (all 7 CUDA kernels on the first V100 run).
    Parsing the definitions keeps declarations and code from drifting apart."""
    import re
    decls = []
    for fn in functions:
        m = re.search(r"^([A-Za-z_][\w:<>]*)\s+" + re.escape(fn) + r"\s*\(([^)]*)\)\s*\{", cuda_src, re.M)
        if not m:
            raise ValueError(f"cuda_declarations: no definition of {fn!r} in the CUDA source")
        args = " ".join(m.group(2).split())
        decls.append(f"{m.group(1)} {fn}({args});")
    return "#include <torch/extension.h>\n" + "\n".join(decls) + "\n"


def build_error_summary(e: Exception, limit: int = 300) -> str:
    """The compiler's own error lines from a failed load_inline build (the exception text starts
    with the full ninja command line, so a plain prefix cut hides the actual error)."""
    text = str(e).splitlines()
    compiler = [l.strip() for l in text if ": error" in l or "error:" in l]          # file:line: error: ...
    other = [l.strip() for l in text if "error" in l.lower() and l not in compiler
             and not l.startswith("Error building extension") and "ninja: build stopped" not in l]
    return (" | ".join(compiler or other) or str(e))[:limit]


def require_native(p: dict, ctx: Ctx) -> None:
    if p.get("precision") == "bf16" and not ctx.native_bf16:
        raise Unsupported("bf16 not native on this device")


def cpu_backend(kind: str) -> str:
    """The CPU library torch dispatches to, as far as torch itself reports it: MKL only when
    the wheel has MKL (x86 builds), otherwise ATen's own kernels / pocketfft (e.g. aarch64)."""
    mkl = bool(torch.backends.mkl.is_available())
    return {"sparse": "MKL" if mkl else "ATen", "fft": "MKL" if mkl else "pocketfft"}[kind]


def library_tag(ctx: Ctx, kernel: str) -> str:
    from .. import devices
    if ctx.device_class == "cpu":
        b = devices.torch_backend_info()
        return f"torch{b['torch']}/{kernel}/blas={b['blas']}"
    return f"torch{torch.__version__}/{kernel}/cuda{torch.version.cuda}"
