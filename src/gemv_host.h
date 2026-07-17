/* gemv_host.h -- pure-C host-side math for the GEMV/SpMV benchmark.
 * No CUDA, no GPU, no cuBLAS/cuSPARSE headers -- deliberately standalone so
 * this logic can be unit-tested with plain gcc (see ../tests/test_gemv_host.c). */
#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* -- deterministic host RNG (mirrors gpu_gemm_bench.cu's hashf/fill_rand) -- */
static inline float gemv_hashf(uint64_t i, uint64_t seed) {
    uint64_t x = i * 0x9E3779B97F4A7C15ULL + seed;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL; x ^= x >> 31;
    return (float)((double)(uint32_t)(x >> 32) / (double)UINT32_MAX) * 2.0f - 1.0f;
}

static inline void gemv_fill_rand(float *p, size_t n, uint64_t seed) {
    for (size_t i = 0; i < n; i++) p[i] = gemv_hashf(i, seed);
}

static inline uint64_t gemv_hashu_seed(uint64_t i, uint64_t seed) {
    uint64_t x = i ^ seed;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL; x ^= x >> 31;
    return x;
}

/* Build a square SxS sparse matrix A in CSR format, ~density nonzeros/row,
 * sorted distinct columns per row. Load-bearing here (unlike GEMM's now-
 * vestigial gen_csr()): GEMV's sparse engine uses regular cuSPARSE CSR
 * SpMV, where density is a real, meaningful sweep dimension. */
static inline int gemv_gen_csr(long S, double density,
                                int **rowOff_o, int **colInd_o,
                                float **vals_o, long *nnz_o, uint64_t seed) {
    int npr = (int)llround(density * (double)S);
    if (npr < 1) npr = 1;
    if (npr > S) npr = (int)S;
    long nnz = (long)npr * S;
    if (nnz >= (long)2147483647) return -2;
    int   *rowOff = (int *)  malloc((size_t)(S + 1) * sizeof(int));
    int   *colInd = (int *)  malloc((size_t)nnz     * sizeof(int));
    float *vals   = (float *)malloc((size_t)nnz     * sizeof(float));
    if (!rowOff || !colInd || !vals) { free(rowOff); free(colInd); free(vals); return -1; }
    long stride = S / npr; if (stride < 1) stride = 1;
    for (long r = 0; r < S; r++) {
        rowOff[r] = (int)(r * (long)npr);
        for (int jj = 0; jj < npr; jj++) {
            uint64_t base = ((uint64_t)r * 1469598103934665603ULL)
                           ^ ((uint64_t)jj * 1099511628211ULL);
            uint64_t x = gemv_hashu_seed(base, seed);
            long jit = stride > 1 ? (long)(x % (uint64_t)stride) : 0;
            long col = (long)jj * stride + jit;
            if (col >= S) col = S - 1;
            size_t idx = (size_t)r * (size_t)npr + (size_t)jj;
            colInd[idx] = (int)col;
            uint64_t y = x * 0x9E3779B97F4A7C15ULL; y ^= y >> 29;
            vals[idx] = (float)((double)(uint32_t)(y >> 32) / (double)UINT32_MAX) * 2.0f - 1.0f;
        }
    }
    rowOff[S] = (int)nnz;
    *rowOff_o = rowOff; *colInd_o = colInd; *vals_o = vals; *nnz_o = nnz;
    return 0;
}

/* A100-SXM4-40GB HBM2 peak memory bandwidth, GB/s. Precision-independent --
 * unlike GEMM's peak TFLOPS (which varies by precision/tensor-core
 * routing), the memory bus doesn't care about dtype. */
static inline double gemv_peak_gbps(void) { return 1555.0; }

/* Per-GPU memory footprint (bytes) for one GEMV job under the given
 * multi-GPU mode. mode_replicas=1 for MODE_REPLICAS, 0 for MODE_SPLIT.
 * *max_mrows_o receives this GPU's row-slice size (S for replicas, ceil(S/G)
 * for split) -- mirrors gpu_gemm_bench.cu's mem_footprint(). Dominant terms:
 * A row-slice (mrows x S) + x (full S, replicated on every GPU) + y row-slice. */
static inline long gemv_mem_footprint_bytes(size_t elem_in, size_t elem_out,
                                              int mode_replicas, long S, int G,
                                              long *max_mrows_o) {
    long base = S / G, rem = S % G;
    long mrows = mode_replicas ? S : (base + (rem ? 1 : 0));
    *max_mrows_o = mrows;
    return (long)((size_t)mrows * (size_t)S * elem_in
                 + (size_t)S * elem_in
                 + (size_t)mrows * elem_out);
}

/* Comma-separated long parser (identical pattern to gpu_gemm_bench.cu's
 * parse_longs()). */
static inline int gemv_parse_longs(const char *s, long *out, int max) {
    int n = 0; char buf[512]; snprintf(buf, sizeof buf, "%s", s);
    for (char *t = strtok(buf, ","); t && n < max; t = strtok(NULL, ","))
        out[n++] = atol(t);
    return n;
}
