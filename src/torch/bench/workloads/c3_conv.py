"""C3 — 2-D image convolution (filtering)   (compute-bound; BW-bound for small filters; ISCA'10 kernel 3)

Equal work: "same"-size output (zero border) of ONE k x k filter applied to every channel
of an H x W image with C in {1 (grayscale), 3 (RGB)} -- the image-filtering definition
(cross-correlation, as both libraries compute); identical seeded image and filter.
Libraries (config key `lib`):
  torch  : F.conv2d, groups=C, filter replicated per channel -> torch's CPU conv / cuDNN (GPU);
           cuDNN autotuning (cudnn.benchmark) happens during warmup, never in samples.
  opencv : cv2.filter2D on the HxWxC image (CPU only), inside _common.widened_affinity (OpenCV
           rows before commit d51fa7e ran on one core). NN-oriented conv kernels are not built
           for one small filter, so all three are timed and the fastest is kept.
  native : direct convolution, same algorithm on both devices, K in {3,5,7} a compile-time
           constant (all K*K taps unrolled). CPU: OpenMP over (channel, row), x loop vectorised,
           rows outside the image read a zero row prepared in setup (no allocation in the
           timed call). CUDA: 32x32 output tile per block, (32+K-1)^2 input tile in shared
           memory with a zero halo, 4 outputs per thread. Added because the library paths
           reached ~2% of peak on EPYC; it wins every size on EPYC 9354P and on V100.
  The torch CPU path is tagged `aten-cpu`: which kernel torch dispatches to (oneDNN or
  ATen's own) is not verified yet (check with ONEDNN_VERBOSE=1 on the next CPU node).
Validation: normwise vs an fp64 convolution of the same inputs; k*k-term reductions.
Work model: 2*k*k*C*H*W FLOPs; compulsory bytes = input + output (+ filter).
"""
from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F

from ..core import Ctx, Unsupported, Workload
from ..validate import normwise, tol_reduction
from . import _common as C

IMAGES = [1024, 4096, 8192]        # 8192^2 = 64x LLC on EPYC; 16384 added hours without new information
KERNELS = [3, 5, 7]
CHANNELS = [1, 3]
NATIVE_K = (3, 5, 7)

_CPU_CORE = r"""
#include <cstdint>
#include <omp.h>
template <int K>
static void conv_direct(const float* __restrict in, float* __restrict out, const float* __restrict wt,
                        const float* __restrict zrow, int64_t C, int64_t H, int64_t W) {
  constexpr int P = K / 2;
  float w[K * K];
  for (int i = 0; i < K * K; ++i) w[i] = wt[i];
  #pragma omp parallel for collapse(2) schedule(static)
  for (int64_t c = 0; c < C; ++c)
    for (int64_t y = 0; y < H; ++y) {
      const float* plane = in + c * H * W;
      const float* r[K];
      for (int ky = 0; ky < K; ++ky) {
        const int64_t yy = y + ky - P;
        r[ky] = (yy >= 0 && yy < H) ? plane + yy * W : zrow;     // zero border rows
      }
      float* o = out + (c * H + y) * W;
      auto edge = [&](int64_t x) {                                // zero border columns
        float acc = 0.f;
        for (int ky = 0; ky < K; ++ky)
          for (int kx = 0; kx < K; ++kx) {
            const int64_t xx = x + kx - P;
            if (xx >= 0 && xx < W) acc += w[ky * K + kx] * r[ky][xx];
          }
        o[x] = acc;
      };
      for (int64_t x = 0; x < P && x < W; ++x) edge(x);
      #pragma omp simd
      for (int64_t x = P; x < W - P; ++x) {
        float acc = 0.f;
        #pragma GCC unroll 7
        for (int ky = 0; ky < K; ++ky) {
          #pragma GCC unroll 7
          for (int kx = 0; kx < K; ++kx) acc += w[ky * K + kx] * r[ky][x + kx - P];
        }
        o[x] = acc;
      }
      for (int64_t x = (W - P > P ? W - P : P); x < W; ++x) edge(x);
    }
}
static void conv_direct_any(int64_t k, const float* in, float* out, const float* w, const float* z,
                            int64_t C, int64_t H, int64_t W) {
  switch (k) {
    case 3: conv_direct<3>(in, out, w, z, C, H, W); break;
    case 5: conv_direct<5>(in, out, w, z, C, H, W); break;
    case 7: conv_direct<7>(in, out, w, z, C, H, W); break;
    default: throw std::runtime_error("native conv: K must be 3, 5 or 7");
  }
}
"""

CPU_CORE = "#include <stdexcept>\n" + _CPU_CORE
_CPU_SRC = "#include <torch/extension.h>\n" + CPU_CORE + r"""
void conv2d_native_cpu(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor zrow, int64_t k) {
  conv_direct_any(k, x.data_ptr<float>(), out.data_ptr<float>(), w.data_ptr<float>(), zrow.data_ptr<float>(),
                  x.size(1), x.size(2), x.size(3));
}
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#define TW 32          // output tile 32 x 32 per block; block = 32 x 8 threads, 4 rows per thread
#define TR 8
template <int K>
__global__ void conv_direct_k(const float* __restrict__ in, float* __restrict__ out, const float* __restrict__ wt,
                              int H, int W) {
  constexpr int P = K / 2, SW = TW + K - 1;
  __shared__ float tile[SW][SW + 1];
  float w[K * K];
  #pragma unroll
  for (int i = 0; i < K * K; ++i) w[i] = __ldg(wt + i);
  const int c = blockIdx.z, x0 = blockIdx.x * TW, y0 = blockIdx.y * TW;
  const float* plane = in + (long long)c * H * W;
  for (int i = threadIdx.y * TW + threadIdx.x; i < SW * SW; i += TW * TR) {
    const int ty = i / SW, tx = i % SW, yy = y0 + ty - P, xx = x0 + tx - P;
    tile[ty][tx] = (yy >= 0 && yy < H && xx >= 0 && xx < W) ? plane[(long long)yy * W + xx] : 0.f;
  }
  __syncthreads();
  const int x = x0 + threadIdx.x;
  #pragma unroll
  for (int j = 0; j < TW / TR; ++j) {
    const int ly = threadIdx.y + j * TR, y = y0 + ly;
    float acc = 0.f;
    #pragma unroll
    for (int ky = 0; ky < K; ++ky)
      #pragma unroll
      for (int kx = 0; kx < K; ++kx) acc += w[ky * K + kx] * tile[ly + ky][threadIdx.x + kx];
    if (x < W && y < H) out[((long long)c * H + y) * W + x] = acc;
  }
}
void conv2d_native_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t k) {
  const int C = x.size(1), H = x.size(2), W = x.size(3);
  dim3 blk(TW, TR), grd((W + TW - 1) / TW, (H + TW - 1) / TW, C);
  auto st = at::cuda::getCurrentCUDAStream();
  const float *xp = x.data_ptr<float>(), *wp = w.data_ptr<float>();
  float* op = out.data_ptr<float>();
  switch (k) {
    case 3: conv_direct_k<3><<<grd, blk, 0, st>>>(xp, op, wp, H, W); break;
    case 5: conv_direct_k<5><<<grd, blk, 0, st>>>(xp, op, wp, H, W); break;
    case 7: conv_direct_k<7><<<grd, blk, 0, st>>>(xp, op, wp, H, W); break;
    default: TORCH_CHECK(false, "native conv: K must be 3, 5 or 7");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


@lru_cache(maxsize=1)
def _cpu_ext():
    from torch.utils.cpp_extension import load_inline
    from ._cpusort import _ensure_ninja_on_path
    _ensure_ninja_on_path()
    try:
        return load_inline(name="bench_conv_cpu_v1", cpp_sources=_CPU_SRC, functions=["conv2d_native_cpu"],
                           extra_cflags=["-O3", "-march=native", "-fopenmp"], extra_ldflags=["-fopenmp"])
    except Exception as e:
        raise Unsupported(f"conv CPU extension build failed: {C.build_error_summary(e)}") from e


@lru_cache(maxsize=1)
def _cuda_ext():
    from torch.utils.cpp_extension import load_inline
    from ._cpusort import _ensure_ninja_on_path
    _ensure_ninja_on_path()
    try:
        src = "#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAException.h>\n" + _CUDA_SRC
        return load_inline(name="bench_conv_cuda_v2", cpp_sources=C.cuda_declarations(src, ["conv2d_native_cuda"]),
                           cuda_sources=src, functions=["conv2d_native_cuda"], extra_cuda_cflags=["-O3"])
    except Exception as e:
        raise Unsupported(f"conv CUDA extension build failed: {C.build_error_summary(e)}") from e


def _cv2():
    try:
        import cv2
    except ImportError as e:
        raise Unsupported("opencv-python not installed") from e
    return cv2


class Conv2d(Workload):
    id, name, bound, field_ = "C3", "conv2d", "C", "Image"

    def configs(self, device_class=None):
        libs = ("torch", "opencv", "native") if device_class in (None, "cpu") else ("torch", "native")
        return [{"precision": "fp32", "variant": "base", "hw": hw, "k": k, "c": c, "lib": lib}
                for lib in libs for c in CHANNELS for k in KERNELS for hw in IMAGES]

    def working_set_bytes(self, p):
        return 2 * p["c"] * p["hw"] ** 2 * 4

    def work(self, p):
        f = 2.0 * p["k"] ** 2 * p["c"] * p["hw"] ** 2
        return {"flops": f, "bytes": 2.0 * p["c"] * p["hw"] ** 2 * 4,
                "metric_name": "Mpixels/s", "metric_units_per_call": p["hw"] ** 2 / 1e6}

    def library(self, ctx):
        return C.library_tag(ctx, "conv2d/" + ("aten-cpu" if ctx.device_class == "cpu" else "cuDNN"))

    def _library_for(self, p, ctx):
        if p.get("lib") == "native":
            return ("native C++/OpenMP+SIMD direct conv (K unrolled)" if ctx.device_class == "cpu"
                    else "native CUDA direct conv (smem tile, K unrolled)")
        if p.get("lib", "torch") == "opencv":
            cv2 = _cv2()
            return f"opencv{cv2.__version__}/filter2D/threads={ctx.threads}(affinity widened)"
        return self.library(ctx)

    def setup(self, p, ctx: Ctx):
        lib = p.get("lib", "torch")
        if lib == "opencv" and ctx.device_class != "cpu":
            raise Unsupported("opencv variant is CPU-only")
        if ctx.device_class == "gpu":
            torch.backends.cudnn.benchmark = True
        c, hw, k = p["c"], p["hw"], p["k"]
        x = torch.empty(1, c, hw, hw, device=ctx.device)
        x.fill_(0).uniform_(0.0, 1.0, generator=ctx.generator(1))
        filt = torch.randn(k, k, generator=ctx.generator(2), device=ctx.device) / k
        s = {"x": x, "filt": filt, "pad": k // 2, "c": c, "k": k, "lib": lib}
        if lib == "torch":
            s["w"] = filt.expand(c, 1, k, k).contiguous()
            s["y"] = torch.empty_like(x)
        elif lib == "native":
            if k not in NATIVE_K:
                raise Unsupported(f"native conv compiled for K in {NATIVE_K}")
            s["w"] = filt.contiguous()
            s["y"] = C.first_touch(x.shape, torch.float32, ctx)
            if ctx.device_class == "cpu":
                s["zrow"] = torch.zeros(hw, dtype=torch.float32)
                s["fn"] = _cpu_ext().conv2d_native_cpu
            else:
                s["fn"] = _cuda_ext().conv2d_native_cuda
        else:
            cv2 = _cv2()
            s["cpus"] = C.placement_cpus(ctx)
            with C.widened_affinity(s["cpus"]):      # OpenCV's pthread pool inherits the caller's mask
                cv2.setNumThreads(ctx.threads)
            # OpenCV's native layouts: HxW for one channel, HxWxC otherwise
            img = x[0, 0].contiguous() if c == 1 else x[0].permute(1, 2, 0).contiguous()
            s["img"], s["dst"] = img.numpy(), torch.empty_like(img).numpy()
            s["kern"] = filt.numpy()
            s["cv2"] = cv2
        return s

    def call(self, s):
        if s["lib"] == "native":
            if "zrow" in s:
                s["fn"](s["x"], s["w"], s["y"], s["zrow"], s["k"])
            else:
                s["fn"](s["x"], s["w"], s["y"], s["k"])
        elif s["lib"] == "torch":
            # F.conv2d has no out= variant: each call allocates its output. On GPU the caching
            # allocator reuses the block; on CPU the harness runs under jemalloc with decay
            # disabled (sweep.py, methodology §4) so the block is reused without page faults.
            s["y"] = F.conv2d(s["x"], s["w"], padding=s["pad"], groups=s["c"])
        else:
            with C.widened_affinity(s["cpus"]):
                s["cv2"].filter2D(s["img"], -1, s["kern"], dst=s["dst"], borderType=s["cv2"].BORDER_CONSTANT)

    def _output_nchw(self, s):
        if s["lib"] in ("torch", "native"):
            return s["y"]
        d = torch.from_numpy(s["dst"])
        return (d.unsqueeze(-1) if d.dim() == 2 else d).permute(2, 0, 1).unsqueeze(0)

    def validate(self, s, ctx):
        self.call(s)
        w64 = s["filt"].double().expand(s["c"], 1, s["k"], s["k"])
        ref = F.conv2d(s["x"].double(), w64, padding=s["pad"], groups=s["c"])
        return normwise(self._output_nchw(s), ref, tol_reduction(torch.float32, s["k"] ** 2))

    def transfers(self, s):
        return ([s["x"]], [s["y"]]) if s["lib"] in ("torch", "native") else ([], [])


WORKLOAD = Conv2d()
