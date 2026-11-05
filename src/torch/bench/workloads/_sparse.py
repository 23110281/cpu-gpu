"""Deterministic sparse-matrix families for B3 / B4 (SuiteSparse matrices are added when the
nodes can download them; these synthetic families are always available and reproducible).

  poisson3d : 7-point Laplacian on a g^3 grid (regular, banded: FEM/stencil-like)
  powerlaw  : row lengths ~ Zipf (heavy-tailed, web/graph-like), uniform random columns
Built with SciPy in CSR (fp64 values), sorted column indices, no duplicates.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def poisson3d(g: int) -> sp.csr_matrix:
    n1 = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(g, g))
    eye = sp.identity(g)
    a = sp.kron(sp.kron(n1, eye), eye) + sp.kron(sp.kron(eye, n1), eye) + sp.kron(sp.kron(eye, eye), n1)
    return sp.csr_matrix(a, dtype=np.float64)


def powerlaw(n: int, avg_nnz: int, seed: int = 7, s: float = 2.0) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    deg = rng.zipf(s, size=n).astype(np.int64)
    deg = np.minimum(deg, n // 4)
    deg = np.maximum(1, np.round(deg * (avg_nnz / deg.mean()))).astype(np.int64)
    deg = np.minimum(deg, n)
    rows = np.repeat(np.arange(n, dtype=np.int64), deg)
    cols = rng.integers(0, n, size=rows.size, dtype=np.int64)
    vals = rng.standard_normal(rows.size)
    a = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    a.sum_duplicates()
    a.sort_indices()
    return a


def build(family: str, size: int) -> sp.csr_matrix:
    if family == "poisson3d":
        return poisson3d(size)
    if family == "powerlaw":
        return powerlaw(size, avg_nnz=16)
    raise ValueError(family)


def to_torch(a: sp.csr_matrix, dtype, device, index_dtype):
    import torch
    crow = torch.from_numpy(a.indptr.astype(np.int64)).to(index_dtype)
    col = torch.from_numpy(a.indices.astype(np.int64)).to(index_dtype)
    val = torch.from_numpy(a.data).to(dtype)
    return torch.sparse_csr_tensor(crow, col, val, size=a.shape, device=device)
