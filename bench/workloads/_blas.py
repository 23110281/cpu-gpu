"""CBLAS backends callable on torch CPU tensors (fastest-library rule, methodology §7).

torch's built-in CPU BLAS is MKL on x86 wheels — including on AMD, where MKL is not always
the fastest (EPYC 9354P / Zen 4, r6615: OpenBLAS wins fp32 3495 vs MKL 3066 GFLOP/s, MKL wins
fp64 1643 vs 1542; Ubuntu's BLIS 0.8.1 is ~3.5x slower there). For BLAS-backed
workloads the harness therefore times every well-known library available on the node on the
*same* tensors, and the analysis keeps the fastest per (device, precision).

Threading: OpenBLAS (OpenMP build) and BLIS (OpenMP build) follow OMP_NUM_THREADS /
BLIS_NUM_THREADS, both set by devices.placement_env from the pinned placement.
Library paths can be overridden with BENCH_LIB_<NAME>=/path/to/lib.so (e.g. AOCL-BLIS).

OpenBLAS CPU detection: a DYNAMIC_ARCH build older than the CPU can fail to recognise it and fall
back to a generic kernel (Ubuntu 22.04's OpenBLAS 0.3.20 on Xeon Gold 5512U / Emerald Rapids:
"Prescott", 893 vs 3381 GFLOP/s sgemm). `load` probes the detected core in a subprocess and, only
when it is a generic fallback on a CPU with AVX-512 / AVX2, forces the matching kernel family via
OPENBLAS_CORETYPE (SkylakeX / Haswell) before loading. The core actually used is part of
`version()` (and so of every row's `library` column); an explicit OPENBLAS_CORETYPE is respected.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from functools import lru_cache

from ..core import Unsupported

CBLAS_ROW_MAJOR, CBLAS_NO_TRANS = 101, 111

CANDIDATES = {
    "openblas": ["/usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so.0",
                 "/usr/lib/aarch64-linux-gnu/openblas-openmp/libopenblas.so.0",
                 "libopenblas.so.0"],
    "blis": ["/usr/lib/x86_64-linux-gnu/blis-openmp/libblis.so.4",
             "/usr/lib/x86_64-linux-gnu/blis-openmp/libblis.so.3",      # Ubuntu 22.04: libblis3-openmp
             "/usr/lib/aarch64-linux-gnu/blis-openmp/libblis.so.4",
             "/usr/lib/aarch64-linux-gnu/blis-openmp/libblis.so.3",
             "libblis-mt.so.4", "libblis.so.4", "libblis.so.3"],
}
CPU_LIBS = ("torch", "openblas", "blis")


# OpenBLAS core names that mean "CPU not recognised, generic x86 kernel"
_GENERIC_CORES = {"prescott", "katmai", "coppermine", "northwood", "banias", "core2", "penryn",
                  "dunnington", "nehalem", "atom", "unknown"}
_forced_core = None


def _openblas_corename(path: str, env: dict) -> str:
    code = ("import ctypes,sys; l=ctypes.CDLL(sys.argv[1]); f=l.openblas_get_corename; "
            "f.restype=ctypes.c_char_p; print(f().decode())")
    try:
        r = subprocess.run([sys.executable, "-c", code, path], env=env, capture_output=True, text=True, timeout=60)
        return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _fix_openblas_coretype(path: str):
    """Force OPENBLAS_CORETYPE only when auto-detection fell back to a generic core."""
    global _forced_core
    if "OPENBLAS_CORETYPE" in os.environ:
        return
    env = {k: v for k, v in os.environ.items() if k != "OPENBLAS_CORETYPE"}
    env["OMP_NUM_THREADS"] = "1"
    detected = _openblas_corename(path, env)
    if detected.lower() not in _GENERIC_CORES:
        return
    from ..devices import cpu_flags
    flags = cpu_flags()
    target = "SkylakeX" if "avx512f" in flags else "Haswell" if "avx2" in flags else None
    if target:
        os.environ["OPENBLAS_CORETYPE"] = target
        _forced_core = f"{target} (forced; auto-detected {detected})"


@lru_cache(maxsize=None)
def load(name: str):
    paths = [os.environ[f"BENCH_LIB_{name.upper()}"]] if os.environ.get(f"BENCH_LIB_{name.upper()}") \
        else CANDIDATES.get(name, [])
    for p in paths:
        if name == "openblas" and os.path.exists(p):
            _fix_openblas_coretype(p)
        try:
            lib = ctypes.CDLL(p, mode=ctypes.RTLD_LOCAL)
            lib.cblas_sgemm
            return lib, p
        except (OSError, AttributeError):
            continue
    raise Unsupported(f"{name} not installed")


def version(name: str) -> str:
    lib, path = load(name)
    try:
        if name == "openblas":
            f = lib.openblas_get_config
            f.restype = ctypes.c_char_p
            g = lib.openblas_get_corename
            g.restype = ctypes.c_char_p
            return f"{f().decode()} core={_forced_core or g().decode()}"
        if name == "blis":
            f = lib.bli_info_get_version_str
            f.restype = ctypes.c_char_p
            s = "BLIS " + f().decode()
            try:   # which kernel set BLIS picked for this CPU ('generic' = no tuned kernels, e.g. 0.8.1 on Zen 4)
                a, q = lib.bli_arch_string, lib.bli_arch_query_id
                a.restype, q.restype = ctypes.c_char_p, ctypes.c_int
                s += f" arch={a(q()).decode()}"
            except AttributeError:
                pass
            return s
    except AttributeError:
        pass
    return os.path.basename(path)


def gemm_fn(name: str, dtype_name: str):
    """Returns f(A, B, C) computing C = A @ B (square, contiguous, row-major torch tensors)."""
    lib, _ = load(name)
    if dtype_name == "fp32":
        fn, scal = lib.cblas_sgemm, ctypes.c_float
    elif dtype_name == "fp64":
        fn, scal = lib.cblas_dgemm, ctypes.c_double
    else:
        raise Unsupported(f"{name}: no {dtype_name} GEMM")
    i, p = ctypes.c_int, ctypes.c_void_p
    fn.argtypes = [i, i, i, i, i, i, scal, p, i, p, i, scal, p, i]
    fn.restype = None
    one, zero = scal(1.0), scal(0.0)

    def f(A, B, C):
        n, k = A.shape
        m = B.shape[1]
        fn(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS, n, m, k, one,
           A.data_ptr(), k, B.data_ptr(), m, zero, C.data_ptr(), m)
    return f


CBLAS_TRANS = 112


def linear_fn(name: str, dtype_name: str):
    """Returns f(X, W, Y) computing Y = X @ W.T (an nn.Linear without bias) for contiguous
    row-major X [M, K], W [N, K], Y [M, N]. M == 1 (LLM decode at batch 1) calls ?gemv
    directly -- not every library forwards a 1-row GEMM to its GEMV kernel."""
    lib, _ = load(name)
    if dtype_name == "fp32":
        mm, mv, scal = lib.cblas_sgemm, lib.cblas_sgemv, ctypes.c_float
    elif dtype_name == "fp64":
        mm, mv, scal = lib.cblas_dgemm, lib.cblas_dgemv, ctypes.c_double
    else:
        raise Unsupported(f"{name}: no {dtype_name} GEMM")
    i, p = ctypes.c_int, ctypes.c_void_p
    mm.argtypes = [i, i, i, i, i, i, scal, p, i, p, i, scal, p, i]
    mv.argtypes = [i, i, i, i, scal, p, i, p, i, scal, p, i]
    mm.restype = mv.restype = None
    one, zero = scal(1.0), scal(0.0)

    def f(X, W, Y):
        m, k = X.shape
        n = W.shape[0]
        if m == 1:
            mv(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, n, k, one, W.data_ptr(), k, X.data_ptr(), 1, zero, Y.data_ptr(), 1)
        else:
            mm(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS, m, n, k, one,
               X.data_ptr(), k, W.data_ptr(), k, zero, Y.data_ptr(), n)
    return f
