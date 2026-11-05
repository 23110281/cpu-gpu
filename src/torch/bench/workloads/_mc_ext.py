"""Native Monte Carlo kernels for C2 (CPU: C++/OpenMP/SIMD; GPU: CUDA), same algorithm.

RNG: Philox4x32-10 (Salmon et al., SC'11 "Parallel random numbers: as easy as 1, 2, 3";
the Random123 / cuRAND counter-based generator). Each Philox block (counter = path-pair
index, key = seed) yields 4 x uint32 -> 2 fp64 uniforms with 52 random bits -> 2 standard
normals (Box-Muller) -> 2 independent paths. Results are therefore bit-identical in the
random stream regardless of thread count or device, and CPU and GPU estimate the SAME
price from the SAME normals (differences only from libm rounding).
Validated against Random123 known-answer vectors (philox4x32_kat) in tests.
Each call prices a European call under GBM with a single exact terminal step (as the
ISCA'10 kernel) and returns (sum payoff, sum payoff^2) for the standard error.
Math libraries: CPU vectorises exp/log/sin/cos through glibc libmvec (needs -ffast-math;
<= 4 ULP); GPU uses CUDA double-precision libm (<= 2 ULP). Both are each platform's
standard vector math; the statistical validation bounds any effect on the result.
"""
from __future__ import annotations

from functools import lru_cache

from ..core import Unsupported

_COMMON = r"""
#include <cstdint>
#include <cmath>
#ifndef HD
#define HD
#endif
HD inline void mulhilo32(uint32_t a, uint32_t b, uint32_t& hi, uint32_t& lo) {
  uint64_t p = (uint64_t)a * (uint64_t)b; hi = (uint32_t)(p >> 32); lo = (uint32_t)p;
}
// Philox4x32-10 on four scalars (no local array: arrays stay in memory and block SIMD
// vectorisation of the calling loop -- measured with -fopt-info-vec-missed).
HD inline void philox4x32_10(uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3, uint32_t k0, uint32_t k1) {
  const uint32_t M0 = 0xD2511F53u, M1 = 0xCD9E8D57u, W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;
#ifdef __CUDACC__
  #pragma unroll
#else
  #pragma GCC unroll 10
#endif
  for (int r = 0; r < 10; ++r) {
    uint32_t hi0, lo0, hi1, lo1;
    mulhilo32(M0, c0, hi0, lo0);
    mulhilo32(M1, c2, hi1, lo1);
    const uint32_t n0 = hi1 ^ c1 ^ k0, n2 = hi0 ^ c3 ^ k1;
    c0 = n0; c1 = lo1; c2 = n2; c3 = lo0;
    k0 += W0; k1 += W1;
  }
}
#ifdef __CUDA_ARCH__
#define BITS_TO_DOUBLE(x) __longlong_as_double((long long)(x))
#else
#define BITS_TO_DOUBLE(x) __builtin_bit_cast(double, (uint64_t)(x))
#endif
// two fp64 uniforms in (0,1) with 52 random bits each, from one Philox block.
// Bit-pattern construction (mantissa of a double in [1,2), minus 1, plus 2^-53) instead of an
// integer->double conversion: AVX2 has no packed 64-bit int->double convert.
HD inline double bits_to_unit(uint64_t m52) {
  return (BITS_TO_DOUBLE(0x3FF0000000000000ull | m52) - 1.0) + 1.1102230246251565e-16;
}
HD inline void two_uniforms(uint64_t ctr, uint32_t k0, uint32_t k1, double& u0, double& u1) {
  uint32_t c0 = (uint32_t)ctr, c1 = (uint32_t)(ctr >> 32), c2 = 0u, c3 = 0u;
  philox4x32_10(c0, c1, c2, c3, k0, k1);
  u0 = bits_to_unit((((uint64_t)c0 << 32) | c1) >> 12);
  u1 = bits_to_unit((((uint64_t)c2 << 32) | c3) >> 12);
}
"""

_CPU_SRC = _COMMON + r"""
#include <torch/extension.h>
#include <omp.h>
std::vector<double> mc_european_cpu(int64_t n_pairs, int64_t seed, double S0, double K, double r,
                                    double sigma, double T) {
  const uint32_t k0 = (uint32_t)seed, k1 = (uint32_t)((uint64_t)seed >> 32);
  const double drift = (r - 0.5 * sigma * sigma) * T, vol = sigma * std::sqrt(T);
  const double two_pi = 6.283185307179586476925286766559;
  // Vectorisation (verified with -fopt-info-vec / libmvec symbols on EPYC, GCC 11):
  //  * sin and cos of the same angle are fused by GCC into a scalar sincos that libmvec
  //    does not vectorise -> the second normal uses cos(theta - pi/2) (== sin, to rounding);
  //  * parallel-for over blocks + inner `omp simd` reduction (the combined construct's
  //    per-lane arrays defeated the vectoriser).
  const double half_pi = 1.5707963267948966192313216916398;
  const int64_t BS = 4096;
  double sum = 0.0, sum2 = 0.0;
  #pragma omp parallel for reduction(+:sum, sum2) schedule(static)
  for (int64_t blk = 0; blk < (n_pairs + BS - 1) / BS; ++blk) {
    const int64_t lo = blk * BS, hi = lo + BS < n_pairs ? lo + BS : n_pairs;
    double s = 0.0, s2 = 0.0;
    #pragma omp simd reduction(+:s, s2)
    for (int64_t i = lo; i < hi; ++i) {
      double u0, u1;
      two_uniforms((uint64_t)i, k0, k1, u0, u1);
      const double rad = std::sqrt(-2.0 * std::log(u0));
      const double z0 = rad * std::cos(two_pi * u1), z1 = rad * std::cos(two_pi * u1 - half_pi);
      const double p0 = std::fmax(S0 * std::exp(drift + vol * z0) - K, 0.0);
      const double p1 = std::fmax(S0 * std::exp(drift + vol * z1) - K, 0.0);
      s += p0 + p1; s2 += p0 * p0 + p1 * p1;
    }
    sum += s; sum2 += s2;
  }
  return {sum, sum2};
}
std::vector<int64_t> philox_kat(int64_t c0, int64_t c1, int64_t c2, int64_t c3, int64_t k0, int64_t k1) {
  uint32_t a0 = (uint32_t)c0, a1 = (uint32_t)c1, a2 = (uint32_t)c2, a3 = (uint32_t)c3;
  philox4x32_10(a0, a1, a2, a3, (uint32_t)k0, (uint32_t)k1);
  return {(int64_t)a0, (int64_t)a1, (int64_t)a2, (int64_t)a3};
}
"""

_CUDA_SRC = "#define HD __host__ __device__\n" + _COMMON + r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
__global__ void mc_kernel(int64_t n_pairs, uint32_t k0, uint32_t k1, double S0, double K, double drift,
                          double vol, double* acc) {
  const double two_pi = 6.283185307179586476925286766559;
  double s = 0.0, s2 = 0.0;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n_pairs;
       i += (int64_t)gridDim.x * blockDim.x) {
    double u0, u1;
    two_uniforms((uint64_t)i, k0, k1, u0, u1);
    double rad = sqrt(-2.0 * log(u0)), sn, cs;
    sincos(two_pi * u1, &sn, &cs);
    double p0 = fmax(S0 * exp(drift + vol * rad * cs) - K, 0.0);
    double p1 = fmax(S0 * exp(drift + vol * rad * sn) - K, 0.0);
    s += p0 + p1; s2 += p0 * p0 + p1 * p1;
  }
  for (int o = 16; o > 0; o >>= 1) { s += __shfl_down_sync(0xffffffff, s, o); s2 += __shfl_down_sync(0xffffffff, s2, o); }
  __shared__ double ws[2][32];
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) { ws[0][wid] = s; ws[1][wid] = s2; }
  __syncthreads();
  if (wid == 0) {
    int nw = blockDim.x >> 5;
    s = lane < nw ? ws[0][lane] : 0.0; s2 = lane < nw ? ws[1][lane] : 0.0;
    for (int o = 16; o > 0; o >>= 1) { s += __shfl_down_sync(0xffffffff, s, o); s2 += __shfl_down_sync(0xffffffff, s2, o); }
    if (lane == 0) { atomicAdd(&acc[0], s); atomicAdd(&acc[1], s2); }
  }
}
void mc_european_cuda(int64_t n_pairs, int64_t seed, double S0, double K, double r, double sigma, double T,
                      torch::Tensor acc, int64_t blocks) {
  acc.zero_();
  const double drift = (r - 0.5 * sigma * sigma) * T, vol = sigma * sqrt(T);
  mc_kernel<<<(int)blocks, 256>>>(n_pairs, (uint32_t)seed, (uint32_t)((uint64_t)seed >> 32), S0, K, drift, vol,
                                  acc.data_ptr<double>());
}
"""


@lru_cache(maxsize=1)
def cpu():
    from torch.utils.cpp_extension import load_inline
    from ._cpusort import _ensure_ninja_on_path
    _ensure_ninja_on_path()
    try:
        return load_inline(name="bench_mc_cpu_v4", cpp_sources=_CPU_SRC,
                           functions=["mc_european_cpu", "philox_kat"],
                           extra_cflags=["-O3", "-march=native", "-fopenmp", "-ffast-math"],
                           extra_ldflags=["-fopenmp"], verbose=False)
    except Exception as e:
        raise Unsupported(f"MC CPU extension build failed: {str(e)[:160]}") from e


@lru_cache(maxsize=1)
def cuda():
    from torch.utils.cpp_extension import load_inline
    from ._cpusort import _ensure_ninja_on_path
    _ensure_ninja_on_path()
    try:
        return load_inline(name="bench_mc_cuda_v4", cpp_sources="", cuda_sources=_CUDA_SRC,
                           functions=["mc_european_cuda"], extra_cuda_cflags=["-O3"], verbose=False)
    except Exception as e:
        raise Unsupported(f"MC CUDA extension build failed: {str(e)[:160]}") from e
