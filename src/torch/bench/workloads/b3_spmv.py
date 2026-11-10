"""B3 — SpMV  y <- A x, CSR, fp32 values, int32 indices   (bandwidth-bound; ISCA'10 kernel 8)

Equal work: same CSR matrix (same families/sizes, built deterministically) and same x on
both devices; y written into a preallocated vector (torch.mv(..., out=)).
Library: torch sparse CSR -> MKL sparse BLAS (x86 CPU) / cuSPARSE SpMV (GPU).
Matrices: poisson3d (7-point stencil, regular) and powerlaw (Zipf row lengths, random
columns: irregular gathers of x) — _sparse.py. SuiteSparse set added when downloadable.
Validation: normwise vs SciPy fp64 A @ x; tol scales with the longest row.
Work model: 2*nnz FLOPs; compulsory bytes = nnz*(val+idx) + (rows+1)*idx + x + y
(x counted once: a lower bound, re-reads of x are what irregular matrices cost).
"""
from __future__ import annotations

import numpy as np
import torch

from ..core import Ctx, Workload
from ..validate import normwise, tol_reduction
from . import _common as C
from . import _sparse

MATRICES = [("poisson3d", 64), ("poisson3d", 128), ("poisson3d", 256),
            ("powerlaw", 1 << 18), ("powerlaw", 1 << 21), ("powerlaw", 1 << 24)]
_EST_NNZ = {"poisson3d": lambda g: 7 * g ** 3, "powerlaw": lambda n: 16 * n}
_EST_ROWS = {"poisson3d": lambda g: g ** 3, "powerlaw": lambda n: n}


class SpMV(Workload):
    id, name, bound, field_ = "B3", "spmv", "B", "Sparse solvers"

    def configs(self, device_class=None):
        return [{"precision": "fp32", "variant": "base", "matrix": m, "size": s, "index": "int32"}
                for m, s in MATRICES]

    def _shape(self, p):
        return _EST_ROWS[p["matrix"]](p["size"]), _EST_NNZ[p["matrix"]](p["size"])

    def working_set_bytes(self, p):
        rows, nnz = self._shape(p)
        return nnz * 8 + (rows + 1) * 4 + 2 * rows * 4

    def work(self, p):
        # exact nnz is only known after build; setup() records it and run_one recomputes
        rows, nnz = self._shape(p)
        b = nnz * 8 + (rows + 1) * 4 + 2 * rows * 4
        return {"flops": 2.0 * nnz, "bytes": float(b), "metric_name": "GB/s",
                "metric_units_per_call": b / 1e9}

    def library(self, ctx):
        return C.library_tag(ctx, "spmv/" + (C.cpu_backend("sparse") if ctx.device_class == "cpu" else "cuSPARSE"))

    def setup(self, p, ctx: Ctx):
        a = _sparse.build(p["matrix"], p["size"])
        A = _sparse.to_torch(a, torch.float32, ctx.device, torch.int32)
        x = torch.empty(a.shape[1], device=ctx.device).uniform_(-1, 1, generator=ctx.generator(1))
        y = torch.empty(a.shape[0], device=ctx.device)
        rows, nnz = a.shape[0], a.nnz
        b = nnz * (4 + 4) + (rows + 1) * 4 + a.shape[1] * 4 + rows * 4
        work = {"flops": 2.0 * nnz, "bytes": float(b), "metric_name": "GB/s",
                "metric_units_per_call": b / 1e9, "working_set_bytes": b, "nnz": nnz}
        return {"A": A, "x": x, "y": y, "a": a, "max_row": int(np.diff(a.indptr).max()), "work": work}

    def call(self, s):
        torch.mv(s["A"], s["x"], out=s["y"])

    def validate(self, s, ctx):
        self.call(s)
        ref = torch.from_numpy(s["a"] @ s["x"].double().cpu().numpy())
        s.pop("a")                                   # free the SciPy copy before timing
        return normwise(s["y"].cpu(), ref, tol_reduction(torch.float32, s["max_row"]))

    def transfers(self, s):
        A = s["A"]
        return [A.crow_indices(), A.col_indices(), A.values(), s["x"]], [s["y"]]


WORKLOAD = SpMV()
