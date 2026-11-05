"""B1 — SAXPY / DAXPY:  y <- a*x + y   (bandwidth-bound; ISCA'10 §2 kernel 5)

Equal work: y (length N) fully rewritten on both devices, same inputs (seeded).
Library: ATen's vectorised in-place `add_` on both sides (OpenMP-parallel on CPU,
vectorised elementwise kernel on CUDA) — a single fused streaming pass, which is the
fastest form available; cuBLAS/MKL axpy is what the native C/CUDA cross-check uses.
Work model: 2N FLOPs, 3N*e compulsory bytes (read x, read y, write y).
Inputs are first-touched in parallel under the run's thread placement (§3).
"""
from __future__ import annotations

import math

import torch

from ..core import Ctx, Workload
from ..validate import Check, tol_elementwise
from . import _common as C

SIZES = [1 << 20, 1 << 22, 1 << 24, 1 << 26, 1 << 28, 1 << 30]   # elements; 4 MB .. 4 GB per fp32 vector
ALPHA = 0.5
CHUNK = 1 << 24


class Saxpy(Workload):
    id, name, bound, field_ = "B1", "saxpy", "B", "Dense LA / BLAS-1"

    def configs(self, device_class=None):
        return [{"precision": p, "variant": "base", "n": n} for p in ("fp32", "fp64") for n in SIZES]

    def working_set_bytes(self, p):
        return 2 * p["n"] * C.ELEM[p["precision"]]

    def work(self, p):
        e = C.ELEM[p["precision"]]
        return {"flops": 2.0 * p["n"], "bytes": 3.0 * p["n"] * e,
                "metric_name": "GB/s", "metric_units_per_call": 3.0 * p["n"] * e / 1e9}

    def library(self, ctx):
        return C.library_tag(ctx, "aten-elementwise")

    def setup(self, p, ctx: Ctx):
        dt = C.DTYPE[p["precision"]]
        x = C.first_touch(p["n"], dt, ctx)
        y = C.first_touch(p["n"], dt, ctx)
        x.uniform_(-1.0, 1.0, generator=ctx.generator(1))
        y.uniform_(-1.0, 1.0, generator=ctx.generator(2))
        return {"x": x, "y": y, "dtype": dt}

    def call(self, s):
        s["y"].add_(s["x"], alpha=ALPHA)

    def validate(self, s, ctx):
        """One call on the real buffers, checked against an fp64 reference built chunk by
        chunk (full size, no full-length fp64 copy)."""
        x, y = s["x"], s["y"]
        y0 = y.clone()
        self.call(s)
        num = den = 0.0
        for i in range(0, y.numel(), CHUNK):
            r = y0[i:i + CHUNK].to(torch.float64) + ALPHA * x[i:i + CHUNK].to(torch.float64)
            d = y[i:i + CHUNK].to(torch.float64) - r
            num += float(torch.dot(d, d))
            den += float(torch.dot(r, r))
        del y0
        err = math.sqrt(num / den) if den > 0 else math.inf
        tol = tol_elementwise(s["dtype"])
        return Check(err, tol, bool(err <= tol), "normwise")

    def transfers(self, s):
        return [s["x"], s["y"]], [s["y"]]


WORKLOAD = Saxpy()
