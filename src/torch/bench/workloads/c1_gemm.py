"""C1 — GEMM:  C <- A @ B, square N x N   (compute-bound; ISCA'10 §2 kernel 1)

Equal work: full N x N output materialised into a preallocated tensor (`out=`), same
seeded inputs on both devices.
Library: GPU `torch.mm` -> cuBLAS. CPU: every well-known BLAS on the node is timed on the
same tensors (config key `lib`): torch's own backend (MKL/oneDNN on x86 wheels), OpenBLAS,
BLIS (AMD AOCL's basis) via CBLAS (workloads/_blas.py); analysis keeps the fastest per
(device, precision). Measured on EPYC 7402P: no single library wins (MKL fp64, OpenBLAS fp32).
TF32 is disabled (devices.configure_torch_cuda), so fp32 is IEEE fp32 end to end.
Variant `ff` (bf16): tensor cores on GPU / AMX or native bf16 on CPU; only where native.
Validation (validate.gemm_check, full size, fp64): row-scaled Freivalds with tolerance
8*(u_out + u_acc*sqrt(N)) AND 2048 sampled entries recomputed exactly; accumulation is
assumed fp32 for fp32/bf16 and fp64 for fp64. max_err is in units of tolerance (tol=1).
"""
from __future__ import annotations

import torch

from ..core import Ctx, Unsupported, Workload
from ..validate import gemm_check
from . import _blas
from . import _common as C
from ._blas import CPU_LIBS

SIZES = [1024, 2048, 4096, 8192]   # 16384 dropped: ~10 s/call fp64, peak already reached at 8192


class Gemm(Workload):
    id, name, bound, field_ = "C1", "gemm", "C", "Dense LA"

    def configs(self, device_class=None):
        libs = CPU_LIBS if device_class in (None, "cpu") else ("torch",)
        out = [{"precision": p, "variant": "base", "n": n, "lib": lib}
               for lib in libs for p in ("fp32", "fp64") for n in SIZES]
        out += [{"precision": "bf16", "variant": "ff", "n": n, "lib": "torch"} for n in SIZES]
        return out

    def working_set_bytes(self, p):
        return 3 * p["n"] ** 2 * C.ELEM[p["precision"]]

    def work(self, p):
        n = p["n"]
        f = 2.0 * n ** 3
        return {"flops": f, "bytes": 3.0 * n * n * C.ELEM[p["precision"]],
                "metric_name": "GFLOP/s", "metric_units_per_call": f / 1e9}

    def library(self, ctx):
        return C.library_tag(ctx, "mm")

    def _library_for(self, p, ctx):
        lib = p.get("lib", "torch")
        if lib == "torch":
            return C.library_tag(ctx, "mm")
        if ctx.device_class != "cpu":
            raise Unsupported(f"{lib} is a CPU library")
        return f"{lib}/{_blas.version(lib)}"

    def setup(self, p, ctx: Ctx):
        C.require_native(p, ctx)
        lib = p.get("lib", "torch")
        if lib != "torch" and ctx.device_class != "cpu":
            raise Unsupported(f"{lib} is a CPU library")
        fn = None if lib == "torch" else _blas.gemm_fn(lib, p["precision"])
        n, dt = p["n"], C.DTYPE[p["precision"]]
        A = C.first_touch((n, n), dt, ctx)
        B = C.first_touch((n, n), dt, ctx)
        Cm = C.first_touch((n, n), dt, ctx)
        A.normal_(generator=ctx.generator(1))
        B.normal_(generator=ctx.generator(2))
        return {"A": A, "B": B, "C": Cm, "dtype": dt, "n": n, "fn": fn}

    def call(self, s):
        if s["fn"] is None:
            torch.mm(s["A"], s["B"], out=s["C"])
        else:
            s["fn"](s["A"], s["B"], s["C"])

    def validate(self, s, ctx):
        self.call(s)
        return gemm_check(s["A"], s["B"], s["C"], s["dtype"])

    def transfers(self, s):
        return [s["A"], s["B"]], [s["C"]]


WORKLOAD = Gemm()
