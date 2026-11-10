"""C5 — Bilateral filter (edge-preserving smoothing)   (compute-bound, transcendental; ISCA'10 kernel 14)

Definition (equal work, OpenCV semantics): grayscale fp32 H x W image, CIRCULAR window of
radius r (offsets with dx^2+dy^2 <= r^2), reflect-101 border,
    out(p) = sum_q w(p,q) I(q) / sum_q w(p,q),
    w(p,q) = exp(-|p-q|^2 / (2 s_s^2)) * exp(-(I(q)-I(p))^2 / (2 s_r^2)),  s_s = r/2, s_r = 0.1.
The padded (reflect-101) input is built in setup, identically on both devices.
Implementations (config key `lib`):
  cpu opencv : cv2.bilateralFilter (library reference; range weights from a lookup table ->
               approximate: validated with a documented looser tolerance 1e-2). Measured on
               EPYC 7402P 4096^2: OpenCV 65 / 6.4 Mpx/s vs native exact 452 / 37 Mpx/s (r=2/7).
  cpu native : exact algorithm, OpenMP over rows, SIMD over x, libmvec expf
  gpu native : exact algorithm, one thread per pixel, precise expf
  gpu fast   : same with __expf (SFU fast-math) -- isolates the paper's fixed-function effect
Validation: 64 sampled rows recomputed in fp64 (normwise; exact variants tol 1e-4,
OpenCV tol 1e-2).  Metric: Mpixels/s.
"""
from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn.functional as F

from ..core import Ctx, Unsupported, Workload
from ..validate import normwise
from . import _common as C

IMAGES = [1024, 4096, 8192]
RADII = [2, 4, 7]
SIGMA_R = 0.1

_CPU_SRC = r"""
#include <torch/extension.h>
#include <omp.h>
#include <cmath>
#include <vector>
// P: padded image (H+2r) x (W+2r); offsets (dy,dx) and spatial log-weights sw precomputed.
void bilateral_cpu(torch::Tensor P, torch::Tensor out, torch::Tensor dy, torch::Tensor dx, torch::Tensor sw,
                   int64_t r, double inv2sr2) {
  const int64_t H = out.size(0), W = out.size(1), Wp = W + 2 * r, K = dy.numel();
  const float* p = P.data_ptr<float>(); float* o = out.data_ptr<float>();
  const int32_t* DY = dy.data_ptr<int32_t>(); const int32_t* DX = dx.data_ptr<int32_t>();
  const float* SW = sw.data_ptr<float>(); const float kr = (float)inv2sr2;
  #pragma omp parallel
  {
    std::vector<float> num(W), den(W);
    #pragma omp for schedule(static)
    for (int64_t y = 0; y < H; ++y) {
      const float* c = p + (y + r) * Wp + r;
      for (int64_t x = 0; x < W; ++x) { num[x] = 0.f; den[x] = 0.f; }
      for (int64_t k = 0; k < K; ++k) {
        const float* q = p + (y + r + DY[k]) * Wp + r + DX[k];
        const float s = SW[k];
        float* nm = num.data(); float* dn = den.data();
        #pragma omp simd
        for (int64_t x = 0; x < W; ++x) {
          const float d = q[x] - c[x];
          const float w = expf(s - d * d * kr);
          nm[x] += w * q[x]; dn[x] += w;
        }
      }
      for (int64_t x = 0; x < W; ++x) o[y * W + x] = num[x] / den[x];
    }
  }
}
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
template <bool FAST>
__global__ void bilateral_kernel(const float* __restrict__ P, float* __restrict__ out, const int* __restrict__ DY,
    const int* __restrict__ DX, const float* __restrict__ SW, int K, int H, int W, int r, float kr) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;
  const int Wp = W + 2 * r;
  const float c = P[(y + r) * Wp + x + r];
  float num = 0.f, den = 0.f;
  for (int k = 0; k < K; ++k) {
    const float q = __ldg(&P[(y + r + DY[k]) * Wp + x + r + DX[k]]);
    const float d = q - c;
    const float a = SW[k] - d * d * kr;
    const float w = FAST ? __expf(a) : expf(a);
    num += w * q; den += w;
  }
  out[y * W + x] = num / den;
}
void bilateral_cuda(torch::Tensor P, torch::Tensor out, torch::Tensor dy, torch::Tensor dx, torch::Tensor sw,
                    int64_t r, double inv2sr2, bool fast) {
  const int H = out.size(0), W = out.size(1), K = dy.numel();
  dim3 blk(32, 8), grd((W + 31) / 32, (H + 7) / 8);
  auto st = at::cuda::getCurrentCUDAStream();
  if (fast) bilateral_kernel<true><<<grd, blk, 0, st>>>(P.data_ptr<float>(), out.data_ptr<float>(), dy.data_ptr<int>(),
      dx.data_ptr<int>(), sw.data_ptr<float>(), K, H, W, (int)r, (float)inv2sr2);
  else bilateral_kernel<false><<<grd, blk, 0, st>>>(P.data_ptr<float>(), out.data_ptr<float>(), dy.data_ptr<int>(),
      dx.data_ptr<int>(), sw.data_ptr<float>(), K, H, W, (int)r, (float)inv2sr2);
}
"""


@lru_cache(maxsize=1)
def _cpu_ext():
    from torch.utils.cpp_extension import load_inline
    from ._cpusort import _ensure_ninja_on_path
    _ensure_ninja_on_path()
    try:
        return load_inline(name="bench_bilat_cpu_v1", cpp_sources=_CPU_SRC, functions=["bilateral_cpu"],
                           extra_cflags=["-O3", "-march=native", "-fopenmp", "-ffast-math"], extra_ldflags=["-fopenmp"])
    except Exception as e:
        raise Unsupported(f"bilateral CPU extension build failed: {str(e)[:140]}") from e


@lru_cache(maxsize=1)
def _cuda_ext():
    from torch.utils.cpp_extension import load_inline
    from ._cpusort import _ensure_ninja_on_path
    _ensure_ninja_on_path()
    try:
        return load_inline(name="bench_bilat_cuda_v1", cpp_sources="",
                           cuda_sources="#include <ATen/cuda/CUDAContext.h>\n" + _CUDA_SRC,
                           functions=["bilateral_cuda"], extra_cuda_cflags=["-O3"])
    except Exception as e:
        raise Unsupported(f"bilateral CUDA extension build failed: {str(e)[:140]}") from e


def offsets(r: int):
    dy, dx, sw = [], [], []
    ss = r / 2.0
    for j in range(-r, r + 1):
        for i in range(-r, r + 1):
            if i * i + j * j <= r * r:
                dy.append(j)
                dx.append(i)
                sw.append(-(i * i + j * j) / (2 * ss * ss))
    return (torch.tensor(dy, dtype=torch.int32), torch.tensor(dx, dtype=torch.int32),
            torch.tensor(sw, dtype=torch.float32))


def test_image(n: int, device, gen) -> torch.Tensor:
    """Smooth structure + edges + noise in [0,1] (a pure-noise image is degenerate for a bilateral)."""
    y = torch.linspace(0, 1, n, device=device)[:, None]
    x = torch.linspace(0, 1, n, device=device)[None, :]
    img = 0.5 + 0.25 * torch.sin(12 * math.pi * x) * torch.cos(8 * math.pi * y) + 0.2 * ((x + y) > 1.0).float()
    img = img + 0.05 * torch.randn(n, n, generator=gen, device=device)
    return img.clamp_(0.0, 1.0).contiguous()


class Bilateral(Workload):
    id, name, bound, field_ = "C5", "bilateral", "C", "Image"

    def configs(self, device_class=None):
        libs = ("opencv", "native") if device_class == "cpu" else ("native", "fast") if device_class == "gpu" \
            else ("opencv", "native", "fast")
        return [{"precision": "fp32", "variant": "ff" if lib == "fast" else "base", "hw": n, "r": r, "lib": lib}
                for lib in libs for r in RADII for n in IMAGES]

    def working_set_bytes(self, p):
        return 2 * 4 * p["hw"] ** 2

    def work(self, p):
        k = int(offsets(p["r"])[0].numel())
        return {"flops": 0.0, "bytes": 8.0 * p["hw"] ** 2, "metric_name": "Mpixels/s",
                "metric_units_per_call": p["hw"] ** 2 / 1e6, "window_taps": k}

    def library(self, ctx):
        return "native"

    def _library_for(self, p, ctx):
        if p["lib"] == "opencv":
            import cv2
            return f"opencv{cv2.__version__}/bilateralFilter(LUT range weights)/threads={ctx.threads}(affinity widened)"
        if p["lib"] == "fast":
            return "native CUDA, __expf (SFU fast math)"
        return "native C++/OpenMP+SIMD, libmvec expf" if ctx.device_class == "cpu" else "native CUDA, precise expf"

    def setup(self, p, ctx: Ctx):
        lib, n, r = p["lib"], p["hw"], p["r"]
        if lib == "opencv" and ctx.device_class != "cpu":
            raise Unsupported("opencv is CPU-only")
        if lib == "fast" and ctx.device_class != "gpu":
            raise Unsupported("__expf variant is GPU-only")
        img = test_image(n, ctx.device, ctx.generator(1))
        padded = F.pad(img[None, None], (r, r, r, r), mode="reflect")[0, 0].contiguous()
        dy, dx, sw = (t.to(ctx.device) for t in offsets(r))
        out = torch.empty_like(img)
        kr = 1.0 / (2 * SIGMA_R * SIGMA_R)
        st = {"img": img, "padded": padded, "out": out, "r": r, "lib": lib}
        if lib == "opencv":
            import cv2
            cpus = C.placement_cpus(ctx)
            with C.widened_affinity(cpus):          # OpenCV's pthread pool inherits the caller's mask
                cv2.setNumThreads(ctx.threads)
            src, dst = img.numpy(), out.numpy()

            def fn():
                with C.widened_affinity(cpus):
                    cv2.bilateralFilter(src, 2 * r + 1, SIGMA_R, r / 2.0, dst=dst, borderType=cv2.BORDER_REFLECT_101)
            st["fn"] = fn
        elif ctx.device_class == "cpu":
            ext = _cpu_ext()
            st["fn"] = lambda: ext.bilateral_cpu(padded, out, dy, dx, sw, r, kr)
        else:
            ext = _cuda_ext()
            st["fn"] = lambda: ext.bilateral_cuda(padded, out, dy, dx, sw, r, kr, lib == "fast")
        return st

    def call(self, s):
        s["fn"]()

    def validate(self, s, ctx):
        self.call(s)
        if ctx.device_class == "gpu":
            torch.cuda.synchronize()
        r, img = s["r"], s["img"].cpu().double()
        P = F.pad(img[None, None], (r, r, r, r), mode="reflect")[0, 0]
        dy, dx, sw = offsets(r)
        g = torch.Generator().manual_seed(3)
        rows = torch.randint(0, img.shape[0], (64,), generator=g).tolist()
        kr = 1.0 / (2 * SIGMA_R * SIGMA_R)
        ref = []
        for y in rows:
            c = P[y + r, r:r + img.shape[1]]
            num = torch.zeros_like(c)
            den = torch.zeros_like(c)
            for k in range(dy.numel()):
                q = P[y + r + int(dy[k]), r + int(dx[k]): r + int(dx[k]) + img.shape[1]]
                w = torch.exp(float(sw[k]) - (q - c) ** 2 * kr)
                num += w * q
                den += w
            ref.append(num / den)
        got = s["out"].cpu()[rows]
        tol = 1e-2 if s["lib"] == "opencv" else 1e-4
        return normwise(got, torch.stack(ref), tol)

    def transfers(self, s):
        return ([s["padded"]], [s["out"]]) if s["lib"] != "opencv" else ([], [])


WORKLOAD = Bilateral()
