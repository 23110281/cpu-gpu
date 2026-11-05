"""Correctness checks and the tolerance policy (docs/methodology.md §8).

Every check returns a `Check(err, tol, ok, kind)`. References are computed in fp64.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

TOL_C = 8.0  # safety factor on the unit roundoff

# unit roundoff of the compute dtype (bf16 has an 8-bit significand -> 2^-8)
_UNIT_ROUNDOFF = {
    torch.float64: 2.0 ** -53,
    torch.float32: 2.0 ** -24,
    torch.bfloat16: 2.0 ** -8,
    torch.float16: 2.0 ** -11,
}


@dataclass
class Check:
    err: float
    tol: float
    ok: bool
    kind: str


def unit_roundoff(dtype: torch.dtype) -> float:
    return _UNIT_ROUNDOFF[dtype]


def tol_elementwise(dtype) -> float:
    return TOL_C * unit_roundoff(dtype)


def tol_reduction(dtype, k: int) -> float:
    return TOL_C * unit_roundoff(dtype) * math.sqrt(max(k, 1))


def normwise(y: torch.Tensor, y_ref: torch.Tensor, tol: float, chunk: int = 1 << 24) -> Check:
    """||y - y_ref||_2 / ||y_ref||_2, accumulated in fp64 chunk by chunk (no full fp64 copy)."""
    y, y_ref = y.reshape(-1), y_ref.reshape(-1)
    num = den = 0.0
    for i in range(0, y.numel(), chunk):
        a = y[i:i + chunk].to(torch.float64)
        b = y_ref[i:i + chunk].to(torch.float64)
        num += float(torch.sum((a - b) ** 2))
        den += float(torch.sum(b ** 2))
    err = math.sqrt(num) / math.sqrt(den) if den > 0 else math.sqrt(num)
    return Check(err, tol, bool(err <= tol) and math.isfinite(err), "normwise")


def freivalds(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, tol: float,
              trials: int = 2, seed: int = 7) -> Check:
    """Randomised check of C = A @ B at full size in O(n^2), everything in fp64.

    Row-scaled residual: with random Gaussian r,
        err = max_trials max_i |(C r)_i - (A (B r))_i| / (||A_i,:||_2 * ||colnorm(B) * r||_2)
    The denominator is the natural rounding scale of row i (each C_ij has error of order
    u * ||A_i,:|| * ||B_:,j||), so `tol` has the usual form c*(u_out + u_acc*sqrt(k)).
    A global Frobenius normalisation would dilute a single wrong element by ~n^1.5 and
    miss it (see tests/torch/test_bench.py::test_freivalds_correct_and_wrong)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    A64, B64, C64 = (t.to(torch.float64) for t in (A, B, C))
    row_a = torch.linalg.vector_norm(A64, dim=1)            # ||A_i,:||
    col_b = torch.linalg.vector_norm(B64, dim=0)            # ||B_:,j||
    worst = 0.0
    for _ in range(trials):
        r = torch.randn(B.shape[1], dtype=torch.float64, generator=gen).to(C64.device)
        resid = (C64 @ r - A64 @ (B64 @ r)).abs()
        scale = row_a * float(torch.linalg.vector_norm(col_b * r))
        ok = scale > 0
        e = float((resid[ok] / scale[ok]).max()) if bool(ok.any()) else float(resid.max())
        worst = max(worst, e)
    return Check(worst, tol, bool(worst <= tol) and math.isfinite(worst), "freivalds")


def sampled_entries(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, u_out: float, u_acc: float,
                    c: float = TOL_C, m: int = 2048, seed: int = 11) -> Check:
    """m random entries C_ij recomputed exactly (fp64 dot products) and checked componentwise:
        |C_ij - ref_ij| <= c * (u_out*|ref_ij| + u_acc*sqrt(k)*||A_i,:||*||B_:,j||)
    Reported err is the worst ratio |dev|/bound, so tol = 1."""
    n, k = A.shape
    gen = torch.Generator(device="cpu").manual_seed(seed)
    i = torch.randint(0, n, (m,), generator=gen).to(A.device)
    j = torch.randint(0, B.shape[1], (m,), generator=gen).to(A.device)
    Ai = A[i].to(torch.float64)                    # (m, k)
    Bj = B[:, j].to(torch.float64).T               # (m, k)
    ref = (Ai * Bj).sum(dim=1)
    bound = c * (u_out * ref.abs() + u_acc * math.sqrt(k)
                 * torch.linalg.vector_norm(Ai, dim=1) * torch.linalg.vector_norm(Bj, dim=1))
    ratio = float(((C[i, j].to(torch.float64) - ref).abs() / bound.clamp_min(1e-300)).max())
    return Check(ratio, 1.0, bool(ratio <= 1.0) and math.isfinite(ratio), "sampled")


def gemm_check(A, B, C, dtype) -> Check:
    """Full-size GEMM validation = row-scaled Freivalds (global/structural errors: wrong
    kernel, missing or uninitialised tiles, transposes) AND sampled exact entries
    (local precision: silent TF32 / reduced-precision inputs). err is reported in units
    of the respective tolerance (tol = 1); the worse of the two is kept.
    Neither check can guarantee detection of one small corrupted element (no O(n^2)
    check can); see docs/methodology.md §8."""
    k = A.shape[1]
    u_out = unit_roundoff(dtype)
    u_acc = unit_roundoff(torch.float64 if dtype == torch.float64 else torch.float32)
    tol_f = TOL_C * (u_out + u_acc * math.sqrt(k))
    f = freivalds(A, B, C, tol_f)
    s = sampled_entries(A, B, C, u_out, u_acc)
    err = max(f.err / tol_f, s.err)
    return Check(err, 1.0, f.ok and s.ok, "freivalds+sampled")


def exact(equal: bool, mismatches: int = 0) -> Check:
    return Check(float(mismatches), 0.0, bool(equal), "exact")


def statistical(estimate: float, reference: float, std_err: float, k: float = 4.0) -> Check:
    err = abs(estimate - reference)
    tol = k * std_err
    return Check(err, tol, bool(err <= tol) and math.isfinite(err), f"{k:g}-sigma")
