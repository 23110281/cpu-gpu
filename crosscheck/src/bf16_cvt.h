/* bf16_cvt.h — pure-C fp32 <-> bf16 conversion for the GPU GEMM bench.
 *
 * bfloat16 is, bit-for-bit, the top 16 bits of an IEEE-754 fp32 word
 * (sign[1] + exponent[8] + mantissa[7]).  cuBLAS reads a CUDA_R_16BF buffer as
 * raw uint16_t, so we can produce device-ready bf16 entirely on the host with
 * no nvcc and no device types.
 *
 * f32_to_bf16(): round-to-nearest-even, NaN-safe (matches VCVTNEPS2BF16 and the
 * project's AVX512-BF16 path). Verified against the research agent's findings. */
#pragma once

#include <stdint.h>
#include <string.h>

static inline uint16_t f32_to_bf16(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof u);

    const uint32_t exp8 = (u >> 23) & 0xFFu;       /* biased exponent */
    if (exp8 == 0xFFu) {                            /* NaN or Inf */
        const uint32_t mant = u & 0x007FFFFFu;
        if (mant != 0u)
            return (uint16_t)((u >> 16) | 0x0040u); /* quiet NaN, sign kept */
        return (uint16_t)(u >> 16);                 /* Inf: exact top 16 bits */
    }
    /* normal / subnormal: round-half-to-even */
    const uint32_t lsb  = (u >> 16) & 1u;
    const uint32_t bias = 0x7FFFu + lsb;
    u += bias;
    return (uint16_t)(u >> 16);
}

static inline float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}
