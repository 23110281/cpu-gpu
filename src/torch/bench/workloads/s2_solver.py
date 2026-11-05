"""S2 — Constraint solver: batched projected Gauss-Seidel   (synchronization-bound; ISCA'10 kernel 7)

Problem: N point bodies (3-D velocity, inverse mass) and M contact constraints (i, j, unit
normal n, bias). One solver ITERATION visits every constraint once:
    vn  = (v_j - v_i) . n ;  dl = -(vn + bias) / (w_i + w_j)
    lam' = max(lam + dl, 0) ; dl = lam' - lam ; v_i -= w_i dl n ; v_j += w_j dl n
Constraints are generated as C colour classes, each a random matching of the bodies (no
body twice in a class), and executed as consecutive BATCHES of B constraints that never span
two colours -> conflict-free within a batch, so the result is independent of thread order.
A barrier separates batches: B is the paper's task-granularity knob (small B = sync-bound).
Equal work: identical constraints, initial state and batch sequence on both devices.
Implementations (config key `lib`):
  cpu native : one OpenMP parallel region per iteration, `omp for` (implicit barrier) per batch
  gpu graph  : one CUDA kernel per batch, the whole iteration captured in a CUDA Graph
  gpu coop   : one cooperative kernel, grid.sync() between batches
Validation: 2 iterations from a fresh state vs an fp64 torch reference of the same
algorithm (normwise on final velocities, tol 1e-5).  Metric: iterations/s; derived
us per batch (= barrier interval) in the analysis.
"""
from __future__ import annotations

from functools import lru_cache

import torch

from ..core import Ctx, Unsupported, Workload
from ..validate import normwise

BODIES = 1 << 17
COLORS = 8
BATCHES = [1 << 8, 1 << 10, 1 << 13, 1 << 16]

_CPU_SRC = r"""
#include <torch/extension.h>
#include <omp.h>
void pgs_iter_cpu(torch::Tensor v, torch::Tensor lam, torch::Tensor I, torch::Tensor J, torch::Tensor nrm,
                  torch::Tensor invm, torch::Tensor bias, torch::Tensor inv_meff, int64_t B) {
  float* V = v.data_ptr<float>(); float* L = lam.data_ptr<float>();
  const int32_t* Ii = I.data_ptr<int32_t>(); const int32_t* Jj = J.data_ptr<int32_t>();
  const float* Nr = nrm.data_ptr<float>(); const float* W = invm.data_ptr<float>();
  const float* Bs = bias.data_ptr<float>(); const float* Me = inv_meff.data_ptr<float>();
  const int64_t M = lam.numel();
  #pragma omp parallel
  for (int64_t b0 = 0; b0 < M; b0 += B) {
    const int64_t b1 = b0 + B < M ? b0 + B : M;
    #pragma omp for schedule(static)
    for (int64_t c = b0; c < b1; ++c) {            // implicit barrier after each batch
      const int32_t i = Ii[c], j = Jj[c];
      const float n0 = Nr[3*c], n1 = Nr[3*c+1], n2 = Nr[3*c+2];
      const float vn = (V[3*j]-V[3*i])*n0 + (V[3*j+1]-V[3*i+1])*n1 + (V[3*j+2]-V[3*i+2])*n2;
      const float ln = fmaxf(L[c] - (vn + Bs[c]) * Me[c], 0.0f);
      const float dl = ln - L[c]; L[c] = ln;
      const float a = W[i]*dl, bb = W[j]*dl;
      V[3*i] -= a*n0; V[3*i+1] -= a*n1; V[3*i+2] -= a*n2;
      V[3*j] += bb*n0; V[3*j+1] += bb*n1; V[3*j+2] += bb*n2;
    }
  }
}
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
__device__ __forceinline__ void solve_one(int64_t c, float* V, float* L, const int32_t* I, const int32_t* J,
    const float* N, const float* W, const float* Bs, const float* Me) {
  const int32_t i = I[c], j = J[c];
  const float n0 = N[3*c], n1 = N[3*c+1], n2 = N[3*c+2];
  const float vn = (V[3*j]-V[3*i])*n0 + (V[3*j+1]-V[3*i+1])*n1 + (V[3*j+2]-V[3*i+2])*n2;
  const float ln = fmaxf(L[c] - (vn + Bs[c]) * Me[c], 0.0f);
  const float dl = ln - L[c]; L[c] = ln;
  const float a = W[i]*dl, b = W[j]*dl;
  V[3*i] -= a*n0; V[3*i+1] -= a*n1; V[3*i+2] -= a*n2;
  V[3*j] += b*n0; V[3*j+1] += b*n1; V[3*j+2] += b*n2;
}
__global__ void batch_kernel(int64_t b0, int64_t b1, float* V, float* L, const int32_t* I, const int32_t* J,
    const float* N, const float* W, const float* Bs, const float* Me) {
  for (int64_t c = b0 + blockIdx.x * (int64_t)blockDim.x + threadIdx.x; c < b1; c += (int64_t)gridDim.x * blockDim.x)
    solve_one(c, V, L, I, J, N, W, Bs, Me);
}
__global__ void coop_kernel(int64_t M, int64_t B, float* V, float* L, const int32_t* I, const int32_t* J,
    const float* N, const float* W, const float* Bs, const float* Me) {
  cg::grid_group g = cg::this_grid();
  for (int64_t b0 = 0; b0 < M; b0 += B) {
    const int64_t b1 = b0 + B < M ? b0 + B : M;
    for (int64_t c = b0 + g.thread_rank(); c < b1; c += g.size()) solve_one(c, V, L, I, J, N, W, Bs, Me);
    g.sync();                                          // grid-wide barrier between batches
  }
}
#define ARGS v.data_ptr<float>(), lam.data_ptr<float>(), I.data_ptr<int32_t>(), J.data_ptr<int32_t>(), \
  nrm.data_ptr<float>(), invm.data_ptr<float>(), bias.data_ptr<float>(), inv_meff.data_ptr<float>()
void pgs_iter_launches(torch::Tensor v, torch::Tensor lam, torch::Tensor I, torch::Tensor J, torch::Tensor nrm,
                       torch::Tensor invm, torch::Tensor bias, torch::Tensor inv_meff, int64_t B, int64_t max_blocks) {
  const int64_t M = lam.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  for (int64_t b0 = 0; b0 < M; b0 += B) {
    const int64_t b1 = b0 + B < M ? b0 + B : M;
    int64_t blocks = (b1 - b0 + 255) / 256; if (blocks > max_blocks) blocks = max_blocks;
    batch_kernel<<<(int)blocks, 256, 0, stream>>>(b0, b1, ARGS);
  }
}
void pgs_iter_coop(torch::Tensor v, torch::Tensor lam, torch::Tensor I, torch::Tensor J, torch::Tensor nrm,
                   torch::Tensor invm, torch::Tensor bias, torch::Tensor inv_meff, int64_t B, int64_t blocks) {
  int64_t M = lam.numel();
  float *pv = v.data_ptr<float>(), *pl = lam.data_ptr<float>();
  const int32_t *pi = I.data_ptr<int32_t>(), *pj = J.data_ptr<int32_t>();
  const float *pn = nrm.data_ptr<float>(), *pw = invm.data_ptr<float>(), *pb = bias.data_ptr<float>(), *pm = inv_meff.data_ptr<float>();
  void* args[] = {&M, &B, &pv, &pl, &pi, &pj, &pn, &pw, &pb, &pm};
  cudaLaunchCooperativeKernel((void*)coop_kernel, dim3((unsigned)blocks), dim3(256), args, 0,
                              at::cuda::getCurrentCUDAStream());
}
int64_t coop_max_blocks() {
  int dev = 0, sms = 0, per_sm = 0; cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, coop_kernel, 256, 0);
  return (int64_t)sms * per_sm;
}
"""


@lru_cache(maxsize=1)
def _cpu_ext():
    from torch.utils.cpp_extension import load_inline
    from ._cpusort import _ensure_ninja_on_path
    _ensure_ninja_on_path()
    try:
        return load_inline(name="bench_pgs_cpu_v1", cpp_sources=_CPU_SRC, functions=["pgs_iter_cpu"],
                           extra_cflags=["-O3", "-march=native", "-fopenmp"], extra_ldflags=["-fopenmp"])
    except Exception as e:
        raise Unsupported(f"PGS CPU extension build failed: {str(e)[:140]}") from e


@lru_cache(maxsize=1)
def _cuda_ext():
    from torch.utils.cpp_extension import load_inline
    from ._cpusort import _ensure_ninja_on_path
    _ensure_ninja_on_path()
    try:
        return load_inline(name="bench_pgs_cuda_v1", cpp_sources="",
                           cuda_sources="#include <ATen/cuda/CUDAContext.h>\n" + _CUDA_SRC,
                           functions=["pgs_iter_launches", "pgs_iter_coop", "coop_max_blocks"],
                           extra_cuda_cflags=["-O3", "-rdc=true"], extra_ldflags=["-lcudadevrt"])
    except Exception as e:
        raise Unsupported(f"PGS CUDA extension build failed: {str(e)[:140]}") from e


def make_problem(n_bodies: int, colors: int, seed: int):
    """Deterministic constraints: `colors` random matchings, plus normals / masses / bias."""
    g = torch.Generator().manual_seed(seed)
    I, J = [], []
    for _ in range(colors):
        perm = torch.randperm(n_bodies, generator=g, dtype=torch.int64)
        I.append(perm[0::2])
        J.append(perm[1::2])
    I = torch.cat(I).to(torch.int32)
    J = torch.cat(J).to(torch.int32)
    m = I.numel()
    nrm = torch.randn(m, 3, generator=g)
    nrm /= nrm.norm(dim=1, keepdim=True)
    invm = torch.rand(n_bodies, generator=g) * 0.9 + 0.1
    bias = torch.randn(m, generator=g) * 0.05
    inv_meff = 1.0 / (invm[I.long()] + invm[J.long()])
    v0 = torch.randn(n_bodies, 3, generator=g)
    return {"I": I, "J": J, "nrm": nrm.contiguous(), "invm": invm, "bias": bias,
            "inv_meff": inv_meff.contiguous(), "v0": v0}


def reference_iterations(pr: dict, B: int, iters: int) -> torch.Tensor:
    """fp64 torch implementation of exactly the same batched algorithm (validation)."""
    v = pr["v0"].double().clone()
    lam = torch.zeros(pr["I"].numel(), dtype=torch.float64)
    I, J = pr["I"].long(), pr["J"].long()
    nrm, w, bias, me = pr["nrm"].double(), pr["invm"].double(), pr["bias"].double(), pr["inv_meff"].double()
    M = lam.numel()
    for _ in range(iters):
        for b0 in range(0, M, B):
            sl = slice(b0, min(M, b0 + B))
            i, j, n = I[sl], J[sl], nrm[sl]
            vn = ((v[j] - v[i]) * n).sum(1)
            ln = (lam[sl] - (vn + bias[sl]) * me[sl]).clamp_min(0.0)
            dl = ln - lam[sl]
            lam[sl] = ln
            v.index_add_(0, i, -(w[i] * dl)[:, None] * n)
            v.index_add_(0, j, (w[j] * dl)[:, None] * n)
    return v


class Solver(Workload):
    id, name, bound, field_ = "S2", "solver", "S", "Game physics"

    def configs(self, device_class=None):
        libs = ("native",) if device_class == "cpu" else ("graph", "coop") if device_class == "gpu" \
            else ("native", "graph", "coop")
        return [{"precision": "fp32", "variant": "base", "bodies": BODIES, "colors": COLORS, "batch": b, "lib": lib}
                for lib in libs for b in BATCHES]

    def working_set_bytes(self, p):
        m = p["bodies"] // 2 * p["colors"]
        return p["bodies"] * 16 + m * (4 + 4 + 12 + 4 + 4 + 4)

    def work(self, p):
        return {"flops": 0.0, "bytes": 0.0, "metric_name": "iterations/s", "metric_units_per_call": 1.0,
                "batches_per_iter": (p["bodies"] // 2 * p["colors"]) // p["batch"]}

    def library(self, ctx):
        return "native C++/OpenMP (omp for barrier per batch)" if ctx.device_class == "cpu" \
            else f"native CUDA (cuda{torch.version.cuda})"

    def _library_for(self, p, ctx):
        return {"native": "native C++/OpenMP, implicit omp-for barrier per batch",
                "graph": "native CUDA, kernel per batch replayed from a CUDA Graph",
                "coop": "native CUDA cooperative kernel, grid.sync() per batch"}[p["lib"]]

    def setup(self, p, ctx: Ctx):
        lib = p["lib"]
        if (lib == "native") != (ctx.device_class == "cpu"):
            raise Unsupported(f"{lib} variant is for the other device class")
        pr = make_problem(p["bodies"], p["colors"], ctx.seed)
        d = {k: t.to(ctx.device) for k, t in pr.items()}
        st = {"pr": pr, "d": d, "B": p["batch"], "lib": lib, "device": ctx.device}
        st["v"] = d["v0"].clone()
        st["lam"] = torch.zeros(pr["I"].numel(), device=ctx.device)
        args = lambda v, lam: (v, lam, d["I"], d["J"], d["nrm"], d["invm"], d["bias"], d["inv_meff"])  # noqa: E731
        if lib == "native":
            ext = _cpu_ext()
            st["step"] = lambda v, lam: ext.pgs_iter_cpu(*args(v, lam), p["batch"])
        elif lib == "graph":
            ext = _cuda_ext()
            blocks = torch.cuda.get_device_properties(ctx.gpu_index).multi_processor_count * 8
            st["step_raw"] = lambda v, lam: ext.pgs_iter_launches(*args(v, lam), p["batch"], blocks)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                st["step_raw"](st["v"], st["lam"])          # warm the kernel before capture
            torch.cuda.current_stream().wait_stream(s)
            st["graph"] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(st["graph"]):
                st["step_raw"](st["v"], st["lam"])
            st["step"] = None
        else:
            ext = _cuda_ext()
            blocks = int(ext.coop_max_blocks())
            st["step"] = lambda v, lam: ext.pgs_iter_coop(*args(v, lam), p["batch"], blocks)
        return st

    def call(self, s):
        if s["lib"] == "graph":
            s["graph"].replay()
        else:
            s["step"](s["v"], s["lam"])

    def validate(self, s, ctx):
        # fresh state, 2 iterations, compare with the fp64 reference of the same algorithm
        v = s["d"]["v0"].clone()
        lam = torch.zeros_like(s["lam"])
        step = s["step_raw"] if s["lib"] == "graph" else s["step"]
        for _ in range(2):
            step(v, lam)
        if ctx.device_class == "gpu":
            torch.cuda.synchronize()
        ref = reference_iterations(s["pr"], s["B"], 2)
        return normwise(v.cpu(), ref, 1e-5)


WORKLOAD = Solver()
