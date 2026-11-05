"""B4 — SpGEMM  C <- A A, CSR x CSR -> CSR, fp32   (bandwidth / irregular; dynamic sparse output)

Equal work: the full sparse product C (structure + values) materialised as a CSR tensor,
same inputs on both devices.
Library: torch.sparse.mm(csr, csr) -> MKL sparse (x86 CPU) / cuSPARSE SpGEMM (GPU).
The symbolic (structure) phase and the output allocation are part of every call in both
libraries (no separable API through torch) — documented, identical on both sides.
Validation: SciPy fp64 A @ A; nnz must match and the normwise value error is bounded by the
longest inner product.
Work model: FLOPs = 2 * (number of scalar multiplies) = 2 * sum_{(i,k) in A} nnz(A[k,:])
(exact, computed from the structure); compulsory bytes = A read twice + C written.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from ..core import Ctx, Workload
from ..validate import Check, tol_reduction
from . import _common as C
from . import _sparse

MATRICES = [("poisson3d", 32), ("poisson3d", 64), ("poisson3d", 128),
            ("powerlaw", 1 << 14), ("powerlaw", 1 << 16), ("powerlaw", 1 << 18)]


def _csr_bytes(nnz, rows):
    return nnz * 8 + (rows + 1) * 8


class SpGEMM(Workload):
    id, name, bound, field_ = "B4", "spgemm", "B", "Graph / sparse"

    def configs(self, device_class=None):
        return [{"precision": "fp32", "variant": "base", "matrix": m, "size": s} for m, s in MATRICES]

    def working_set_bytes(self, p):
        rows = p["size"] ** 3 if p["matrix"] == "poisson3d" else p["size"]
        return 40 * 16 * rows          # estimate (A + C); exact value set in setup()

    def work(self, p):
        return {"flops": float("nan"), "bytes": float("nan"), "metric_name": "GFLOP/s",
                "metric_units_per_call": float("nan")}

    def library(self, ctx):
        return C.library_tag(ctx, "spgemm/" + ("MKL" if ctx.device_class == "cpu" else "cuSPARSE"))

    def setup(self, p, ctx: Ctx):
        a = _sparse.build(p["matrix"], p["size"])
        A = _sparse.to_torch(a, torch.float32, ctx.device, torch.int64)
        rowlen = np.diff(a.indptr)
        mults = int(rowlen[a.indices].sum())
        ref = (a @ a).tocsr()
        ref.sort_indices()
        b = 2 * _csr_bytes(a.nnz, a.shape[0]) + _csr_bytes(ref.nnz, ref.shape[0])
        work = {"flops": 2.0 * mults, "bytes": float(b), "metric_name": "GFLOP/s",
                "metric_units_per_call": 2.0 * mults / 1e9, "working_set_bytes": b,
                "nnz_a": a.nnz, "nnz_c": ref.nnz}
        max_inner = int(rowlen.max())
        return {"A": A, "ref": ref, "work": work, "max_inner": max_inner, "C": None}

    def call(self, s):
        s["C"] = torch.sparse.mm(s["A"], s["A"])

    def validate(self, s, ctx):
        self.call(s)
        Ct = s["C"].to("cpu")
        got = sp.csr_matrix((Ct.values().double().numpy(), Ct.col_indices().numpy(), Ct.crow_indices().numpy()),
                            shape=tuple(Ct.shape))
        got.sum_duplicates()
        got.sort_indices()
        ref = s.pop("ref")
        tol = tol_reduction(torch.float32, s["max_inner"])
        if got.nnz != ref.nnz:
            return Check(float(abs(got.nnz - ref.nnz)), 0.0, False, "nnz+normwise")
        err = float(sp.linalg.norm(got - ref) / sp.linalg.norm(ref))
        return Check(err, tol, bool(err <= tol), "nnz+normwise")

    def transfers(self, s):
        A = s["A"]
        return [A.crow_indices(), A.col_indices(), A.values()], []


WORKLOAD = SpGEMM()
