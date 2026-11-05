"""CBLAS backends callable on torch CPU tensors (fastest-library rule, methodology §7).

torch's built-in CPU BLAS is MKL on x86 wheels — including on AMD, where MKL is not always
the fastest (measured on EPYC 7402P: OpenBLAS +11% fp32, MKL +25% fp64). For BLAS-backed
workloads the harness therefore times every well-known library available on the node on the
*same* tensors, and the analysis keeps the fastest per (device, precision).

Threading: OpenBLAS (OpenMP build) and BLIS (OpenMP build) follow OMP_NUM_THREADS /
BLIS_NUM_THREADS, both set by devices.placement_env from the pinned placement.
Library paths can be overridden with BENCH_LIB_<NAME>=/path/to/lib.so (e.g. AOCL-BLIS).
"""
from __future__ import annotations

import ctypes
import os
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


@lru_cache(maxsize=None)
def load(name: str):
    paths = [os.environ[f"BENCH_LIB_{name.upper()}"]] if os.environ.get(f"BENCH_LIB_{name.upper()}") \
        else CANDIDATES.get(name, [])
    for p in paths:
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
            return f.decode()
        if name == "blis":
            f = lib.bli_info_get_version_str
            f.restype = ctypes.c_char_p
            return "BLIS " + f().decode()
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
