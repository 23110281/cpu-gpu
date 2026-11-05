"""B2 — Lattice Boltzmann D3Q19, BGK (SRT)   (bandwidth-bound; ISCA'10 kernel 6)

One call = one time step on an n^3 periodic lattice: fused pull-stream + collide,
    f_i(x, t+1) = f_i*(x - c_i) - omega (f_i*(x - c_i) - f_i^eq(rho, u)),
SoA layout f[i][z][y][x], two lattices (src -> dst, swapped each step), tau = 0.6.
Equal work: identical lattice, initial state (density 1 + small perturbation, random
velocity field) and step count on both devices.
Implementations: native C++/OpenMP+SIMD (interior x vectorised, x-wrap peeled) and native
CUDA (one thread per cell) -- same algorithm (lbmpy-generated kernels are the alternative).
Validation: (1) mass and momentum conserved by the step (periodic BGK conserves both
exactly up to rounding: tol 1e-5 fp32 / 1e-12 fp64, relative); (2) 4 sampled z-planes
recomputed with an fp64 torch reference of the same step (normwise, tol 1e-5 / 1e-12).
Work model: MLUPS (million lattice-cell updates/s); compulsory bytes = 19 * 2 * elem per cell.
"""
from __future__ import annotations

from functools import lru_cache

import torch

from ..core import Ctx, Unsupported, Workload
from ..validate import Check, normwise
from . import _common as C

GRIDS = [64, 128, 256, 512]
TAU = 0.6

# D3Q19 velocity set and weights (rest, 6 faces, 12 edges)
CVEC = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
        (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0), (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
        (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1)]
WEIGHTS = [1 / 3] + [1 / 18] * 6 + [1 / 36] * 12

_C_TABLE = "\n".join([
    "static const int CX[19] = {" + ",".join(str(c[0]) for c in CVEC) + "};",
    "static const int CY[19] = {" + ",".join(str(c[1]) for c in CVEC) + "};",
    "static const int CZ[19] = {" + ",".join(str(c[2]) for c in CVEC) + "};",
    "static const double WQ[19] = {" + ",".join(repr(w) for w in WEIGHTS) + "};",
])

_CPU_SRC = r"""
#include <torch/extension.h>
#include <omp.h>
""" + _C_TABLE + r"""
template <typename T>
static inline void collide(T f[19], T omega) {
  T rho = 0, ux = 0, uy = 0, uz = 0;
  for (int i = 0; i < 19; ++i) { rho += f[i]; ux += f[i] * CX[i]; uy += f[i] * CY[i]; uz += f[i] * CZ[i]; }
  const T inv = T(1) / rho; ux *= inv; uy *= inv; uz *= inv;
  const T usq = T(1.5) * (ux * ux + uy * uy + uz * uz);
  for (int i = 0; i < 19; ++i) {
    const T cu = T(3) * (CX[i] * ux + CY[i] * uy + CZ[i] * uz);
    const T feq = T(WQ[i]) * rho * (T(1) + cu + T(0.5) * cu * cu - usq);
    f[i] -= omega * (f[i] - feq);
  }
}
template <typename T>
static void step(const T* __restrict src, T* __restrict dst, int64_t n, T omega) {
  const int64_t N3 = n * n * n;
  #pragma omp parallel for collapse(2) schedule(static)
  for (int64_t z = 0; z < n; ++z)
    for (int64_t y = 0; y < n; ++y) {
      const T* rows[19];
      for (int i = 0; i < 19; ++i) {
        const int64_t zz = (z - CZ[i] + n) % n, yy = (y - CY[i] + n) % n;
        rows[i] = src + i * N3 + (zz * n + yy) * n;
      }
      T* out = dst + (z * n + y) * n;
      auto cell = [&](int64_t x, int64_t xm, int64_t xp) {
        T f[19];
        for (int i = 0; i < 19; ++i) f[i] = rows[i][CX[i] > 0 ? xm : (CX[i] < 0 ? xp : x)];
        collide(f, omega);
        for (int i = 0; i < 19; ++i) out[i * N3 + x] = f[i];
      };
      cell(0, n - 1, 1);
      #pragma omp simd
      for (int64_t x = 1; x < n - 1; ++x) {
        T f[19];
        #pragma GCC unroll 19
        for (int i = 0; i < 19; ++i) f[i] = rows[i][x - CX[i]];
        collide(f, omega);
        #pragma GCC unroll 19
        for (int i = 0; i < 19; ++i) out[i * N3 + x] = f[i];
      }
      cell(n - 1, n - 2, 0);
    }
}
void lbm_step_cpu(torch::Tensor src, torch::Tensor dst, int64_t n, double omega) {
  if (src.scalar_type() == torch::kFloat64) step<double>(src.data_ptr<double>(), dst.data_ptr<double>(), n, omega);
  else step<float>(src.data_ptr<float>(), dst.data_ptr<float>(), n, (float)omega);
}
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
""" + _C_TABLE.replace("static const", "__constant__") + r"""
template <typename T>
__global__ void lbm_kernel(const T* __restrict__ src, T* __restrict__ dst, int n, T omega) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y, z = blockIdx.z;
  if (x >= n) return;
  const long long N3 = (long long)n * n * n;
  T f[19];
  #pragma unroll
  for (int i = 0; i < 19; ++i) {
    const int xx = (x - CX[i] + n) % n, yy = (y - CY[i] + n) % n, zz = (z - CZ[i] + n) % n;
    f[i] = src[i * N3 + ((long long)zz * n + yy) * n + xx];
  }
  T rho = 0, ux = 0, uy = 0, uz = 0;
  #pragma unroll
  for (int i = 0; i < 19; ++i) { rho += f[i]; ux += f[i] * CX[i]; uy += f[i] * CY[i]; uz += f[i] * CZ[i]; }
  const T inv = T(1) / rho; ux *= inv; uy *= inv; uz *= inv;
  const T usq = T(1.5) * (ux * ux + uy * uy + uz * uz);
  const long long o = ((long long)z * n + y) * n + x;
  #pragma unroll
  for (int i = 0; i < 19; ++i) {
    const T cu = T(3) * (CX[i] * ux + CY[i] * uy + CZ[i] * uz);
    const T feq = T(WQ[i]) * rho * (T(1) + cu + T(0.5) * cu * cu - usq);
    dst[i * N3 + o] = f[i] - omega * (f[i] - feq);
  }
}
void lbm_step_cuda(torch::Tensor src, torch::Tensor dst, int64_t n, double omega) {
  dim3 blk(128), grd((unsigned)((n + 127) / 128), (unsigned)n, (unsigned)n);
  auto st = at::cuda::getCurrentCUDAStream();
  if (src.scalar_type() == torch::kFloat64)
    lbm_kernel<double><<<grd, blk, 0, st>>>(src.data_ptr<double>(), dst.data_ptr<double>(), (int)n, omega);
  else
    lbm_kernel<float><<<grd, blk, 0, st>>>(src.data_ptr<float>(), dst.data_ptr<float>(), (int)n, (float)omega);
}
"""


@lru_cache(maxsize=1)
def _cpu_ext():
    from torch.utils.cpp_extension import load_inline
    from ._cpusort import _ensure_ninja_on_path
    _ensure_ninja_on_path()
    try:
        return load_inline(name="bench_lbm_cpu_v1", cpp_sources=_CPU_SRC, functions=["lbm_step_cpu"],
                           extra_cflags=["-O3", "-march=native", "-fopenmp"], extra_ldflags=["-fopenmp"])
    except Exception as e:
        raise Unsupported(f"LBM CPU extension build failed: {str(e)[:140]}") from e


@lru_cache(maxsize=1)
def _cuda_ext():
    from torch.utils.cpp_extension import load_inline
    from ._cpusort import _ensure_ninja_on_path
    _ensure_ninja_on_path()
    try:
        return load_inline(name="bench_lbm_cuda_v1", cpp_sources="",
                           cuda_sources="#include <ATen/cuda/CUDAContext.h>\n" + _CUDA_SRC,
                           functions=["lbm_step_cuda"], extra_cuda_cflags=["-O3"])
    except Exception as e:
        raise Unsupported(f"LBM CUDA extension build failed: {str(e)[:140]}") from e


def reference_step(f: torch.Tensor, omega: float) -> torch.Tensor:
    """fp64 torch reference of one pull-stream + BGK collide step on full planes (f: [19, z, y, x])."""
    f = f.double()
    pulled = torch.stack([torch.roll(f[i], shifts=(CVEC[i][2], CVEC[i][1], CVEC[i][0]), dims=(0, 1, 2))
                          for i in range(19)])
    c = torch.tensor(CVEC, dtype=torch.float64)
    w = torch.tensor(WEIGHTS, dtype=torch.float64)
    rho = pulled.sum(0)
    u = torch.einsum("izyx,ik->kzyx", pulled, c) / rho
    usq = 1.5 * (u ** 2).sum(0)
    cu = 3.0 * torch.einsum("ik,kzyx->izyx", c, u)
    feq = w[:, None, None, None] * rho * (1.0 + cu + 0.5 * cu ** 2 - usq)
    return pulled - omega * (pulled - feq)


def initial_state(n: int, dtype, device, gen) -> torch.Tensor:
    c = torch.tensor(CVEC, dtype=torch.float64, device=device)
    w = torch.tensor(WEIGHTS, dtype=torch.float64, device=device)
    rho = 1.0 + 0.01 * torch.rand(n, n, n, generator=gen, device=device, dtype=torch.float64)
    u = 0.05 * (torch.rand(3, n, n, n, generator=gen, device=device, dtype=torch.float64) - 0.5)
    usq = 1.5 * (u ** 2).sum(0)
    f = torch.empty(19, n, n, n, dtype=dtype, device=device)
    for i in range(19):                       # equilibrium init, plane by plane-free per direction
        cu = 3.0 * (c[i, 0] * u[0] + c[i, 1] * u[1] + c[i, 2] * u[2])
        f[i] = (w[i] * rho * (1.0 + cu + 0.5 * cu * cu - usq)).to(dtype)
    return f


class LBM(Workload):
    id, name, bound, field_ = "B2", "lbm_d3q19", "B", "CFD"

    def configs(self, device_class=None):
        return [{"precision": p, "variant": "base", "n": n} for p in ("fp32", "fp64") for n in GRIDS]

    def working_set_bytes(self, p):
        return 2 * 19 * p["n"] ** 3 * C.ELEM[p["precision"]]

    def work(self, p):
        cells = p["n"] ** 3
        return {"flops": 0.0, "bytes": 2.0 * 19 * cells * C.ELEM[p["precision"]], "metric_name": "MLUPS",
                "metric_units_per_call": cells / 1e6}

    def library(self, ctx):
        return "native C++/OpenMP+SIMD D3Q19 BGK" if ctx.device_class == "cpu" else "native CUDA D3Q19 BGK"

    def setup(self, p, ctx: Ctx):
        n, dt = p["n"], C.DTYPE[p["precision"]]
        f0 = initial_state(n, dt, ctx.device, ctx.generator(1))
        a = C.first_touch(f0.shape, dt, ctx)
        a.copy_(f0)
        del f0
        b = C.first_touch(a.shape, dt, ctx)
        ext = _cpu_ext() if ctx.device_class == "cpu" else _cuda_ext()
        fn = ext.lbm_step_cpu if ctx.device_class == "cpu" else ext.lbm_step_cuda
        return {"a": a, "b": b, "fn": fn, "n": n, "omega": 1.0 / TAU, "dtype": dt}

    def call(self, s):
        s["fn"](s["a"], s["b"], s["n"], s["omega"])
        s["a"], s["b"] = s["b"], s["a"]

    def validate(self, s, ctx):
        src = s["a"].clone()
        self.call(s)                                   # result now in s["a"]
        out = s["a"]
        tol = 1e-5 if s["dtype"] == torch.float32 else 1e-12
        c = torch.tensor(CVEC, dtype=torch.float64, device=out.device)
        mass0, mass1 = float(src.double().sum()), float(out.double().sum())
        mom0 = torch.einsum("izyx,ik->k", src.double(), c)
        mom1 = torch.einsum("izyx,ik->k", out.double(), c)
        cons = max(abs(mass1 - mass0) / abs(mass0), float((mom1 - mom0).abs().max()) / abs(mass0))
        if not cons <= tol:
            return Check(cons, tol, False, "conservation")
        n = s["n"]
        g = torch.Generator().manual_seed(9)
        planes = torch.randint(0, n, (4,), generator=g).tolist()
        worst = 0.0
        for z in planes:                              # reference on a 3-plane periodic slab
            zs = [(z - 1) % n, z, (z + 1) % n]
            slab = src[:, zs].cpu()
            ref = reference_step_slab(slab, s["omega"])
            chk = normwise(out[:, z].cpu(), ref, tol)
            worst = max(worst, chk.err)
        return Check(max(worst, cons), tol, worst <= tol, "conservation+reference")

    def transfers(self, s):
        return [s["a"]], [s["b"]]


def reference_step_slab(slab: torch.Tensor, omega: float) -> torch.Tensor:
    """Reference for the middle plane of a [19, 3, n, n] slab (z-1, z, z+1)."""
    f = slab.double()
    pulled = torch.stack([torch.roll(f[i], shifts=(CVEC[i][2], CVEC[i][1], CVEC[i][0]), dims=(0, 1, 2))[1]
                          for i in range(19)])
    c = torch.tensor(CVEC, dtype=torch.float64)
    w = torch.tensor(WEIGHTS, dtype=torch.float64)
    rho = pulled.sum(0)
    u = torch.einsum("iyx,ik->kyx", pulled, c) / rho
    usq = 1.5 * (u ** 2).sum(0)
    cu = 3.0 * torch.einsum("ik,kyx->iyx", c, u)
    feq = w[:, None, None] * rho * (1.0 + cu + 0.5 * cu ** 2 - usq)
    return pulled - omega * (pulled - feq)


WORKLOAD = LBM()
