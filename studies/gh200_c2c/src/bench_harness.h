#ifndef BENCH_HARNESS_H
#define BENCH_HARNESS_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    REGIME_ARITHMETIC_COMPUTE  = 1,
    REGIME_STREAMING_BANDWIDTH = 2,
    REGIME_IRREGULAR_LATENCY   = 3,
    REGIME_DIVERGENCE_ATOMIC   = 4
} workload_regime_t;

typedef enum {
    RESIDENCY_CPU_LOCAL         = 0,
    RESIDENCY_GPU_EXPLICIT_COPY = 1,
    RESIDENCY_GPU_ZERO_COPY     = 2,
    RESIDENCY_GPU_UM_MIGRATE    = 3
} residency_mode_t;

static inline const char* residency_to_str(residency_mode_t r) {
    switch (r) {
        case RESIDENCY_CPU_LOCAL:         return "cpu_local";
        case RESIDENCY_GPU_EXPLICIT_COPY: return "gpu_explicit";
        case RESIDENCY_GPU_ZERO_COPY:     return "gpu_zero_copy";
        case RESIDENCY_GPU_UM_MIGRATE:    return "gpu_um_migrate";
        default:                          return "unknown";
    }
}

static inline residency_mode_t str_to_residency(const char *s) {
    if (!s) return RESIDENCY_GPU_EXPLICIT_COPY;
    if (strcmp(s, "cpu_local") == 0 || strcmp(s, "cpu") == 0)
        return RESIDENCY_CPU_LOCAL;
    if (strcmp(s, "gpu_zero_copy") == 0 || strcmp(s, "zero_copy") == 0 || strcmp(s, "zerocopy") == 0)
        return RESIDENCY_GPU_ZERO_COPY;
    if (strcmp(s, "gpu_um_migrate") == 0 || strcmp(s, "um_migrate") == 0 || strcmp(s, "uvm") == 0 || strcmp(s, "managed") == 0)
        return RESIDENCY_GPU_UM_MIGRATE;
    return RESIDENCY_GPU_EXPLICIT_COPY;
}

static inline const char* regime_to_str(workload_regime_t reg) {
    switch (reg) {
        case REGIME_ARITHMETIC_COMPUTE:  return "1_compute";
        case REGIME_STREAMING_BANDWIDTH: return "2_streaming_bw";
        case REGIME_IRREGULAR_LATENCY:   return "3_irregular_latency";
        case REGIME_DIVERGENCE_ATOMIC:   return "4_divergence_atomic";
        default:                         return "unknown";
    }
}

/* Dual-pointer memory allocation structure */
typedef struct {
    void *host_ptr;          /* Accessible on Grace CPU (LPDDR5X) */
    void *dev_ptr;           /* Accessible on Hopper GPU (HBM3 or mapped LPDDR5X) */
    size_t bytes;
    residency_mode_t mode;
} bench_buf_t;

static inline bench_buf_t bench_alloc_buf(size_t bytes, residency_mode_t mode) {
    bench_buf_t b;
    b.bytes = bytes;
    b.mode = mode;
    b.host_ptr = NULL;
    b.dev_ptr = NULL;

#ifdef __CUDACC__
    switch (mode) {
        case RESIDENCY_CPU_LOCAL:
            /* Pinned to Grace LPDDR5X host memory */
            b.host_ptr = aligned_alloc(64, bytes);
            b.dev_ptr = NULL;
            break;
        case RESIDENCY_GPU_EXPLICIT_COPY:
            /* Pinned host memory in LPDDR5X + Device memory in HBM3 */
            if (cudaMallocHost(&b.host_ptr, bytes) != cudaSuccess) {
                b.host_ptr = NULL;
            }
            if (cudaMalloc(&b.dev_ptr, bytes) != cudaSuccess) {
                if (b.host_ptr) { cudaFreeHost(b.host_ptr); b.host_ptr = NULL; }
                b.dev_ptr = NULL;
            }
            break;
        case RESIDENCY_GPU_ZERO_COPY:
            /* Pinned host memory in LPDDR5X, mapped to Hopper virtual address space */
            if (cudaHostAlloc(&b.host_ptr, bytes, cudaHostAllocMapped) == cudaSuccess) {
                if (cudaHostGetDevicePointer(&b.dev_ptr, b.host_ptr, 0) != cudaSuccess) {
                    b.dev_ptr = b.host_ptr; /* On GH200 unified addressing, dev_ptr == host_ptr */
                }
            }
            break;
        case RESIDENCY_GPU_UM_MIGRATE:
            /* Unified memory managed by NVIDIA UVM driver over NVLink-C2C */
            if (cudaMallocManaged(&b.host_ptr, bytes, cudaMemAttachGlobal) == cudaSuccess) {
                b.dev_ptr = b.host_ptr;
            }
            break;
    }
#else
    b.host_ptr = aligned_alloc(64, bytes);
    b.dev_ptr = NULL;
#endif

    return b;
}

static inline void bench_free_buf(bench_buf_t *b) {
    if (!b) return;
#ifdef __CUDACC__
    switch (b->mode) {
        case RESIDENCY_CPU_LOCAL:
            if (b->host_ptr) free(b->host_ptr);
            break;
        case RESIDENCY_GPU_EXPLICIT_COPY:
            if (b->dev_ptr) cudaFree(b->dev_ptr);
            if (b->host_ptr) cudaFreeHost(b->host_ptr);
            break;
        case RESIDENCY_GPU_ZERO_COPY:
            if (b->host_ptr) cudaFreeHost(b->host_ptr);
            break;
        case RESIDENCY_GPU_UM_MIGRATE:
            if (b->host_ptr) cudaFree(b->host_ptr);
            break;
    }
#else
    if (b->host_ptr) free(b->host_ptr);
#endif
    b->host_ptr = NULL;
    b->dev_ptr = NULL;
    b->bytes = 0;
}

/* High-resolution CPU timer helper */
static inline double bench_get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* Unified Benchmark Record */
typedef struct {
    char workload[32];          /* "saxpy", "gemv", "gemm", "llama32_sdpa", etc. */
    int regime;                 /* 1, 2, 3, or 4 */
    char precision[16];         /* "fp32", "fp64", "bf16" */
    residency_mode_t residency; /* RESIDENCY_* */
    size_t working_set_bytes;   /* Total bytes across inputs and outputs */
    char problem_size[64];      /* "N=100000000" or "M=16384,N=16384,K=16384" */
    int threads_or_sms;         /* 72 (CPU cores) or 132 (Hopper SMs) */

    /* Measured Timings (in milliseconds) */
    double t_alloc_ms;          /* Allocation + setup time */
    double t_h2d_ms;            /* Explicit H2D transfer time (0 if CPU or ZERO_COPY) */
    double t_kernel_cold_ms;    /* 1st iteration time (cold cache, ATS faulting, migration) */
    double t_kernel_warm_ms;    /* Average warm iteration compute/streaming time */
    double t_d2h_ms;            /* Explicit D2H transfer time (0 if CPU or ZERO_COPY) */
    double t_e2e_ms;            /* Total end-to-end wall time (h2d + warm_kernel + d2h) */

    /* Derived Metrics */
    double eff_bw_gbps;         /* Algorithmic data volume / t_e2e */
    double c2c_bw_gbps;         /* NVLink-C2C bus traffic rate */
    double gflops;              /* Giga-FLOP/s */
    double pct_peak;            /* % of target hardware theoretical peak */
    char status[16];            /* "ok", "oom-skip", "l2-bound", etc. */
} bench_record_t;

static inline void print_bench_csv_header(FILE *f) {
    fprintf(f, "workload,regime,precision,residency_mode,working_set_bytes,problem_size,threads_or_sms,"
               "t_alloc_ms,t_h2d_ms,t_kernel_cold_ms,t_kernel_warm_ms,t_d2h_ms,t_e2e_ms,"
               "eff_bw_gbps,c2c_bw_gbps,gflops,pct_peak,status\n");
    fflush(f);
}

static inline void log_bench_record(FILE *f, const bench_record_t *r) {
    fprintf(f, "%s,%d,%s,%s,%zu,\"%s\",%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%.1f,%s\n",
            r->workload,
            r->regime,
            r->precision,
            residency_to_str(r->residency),
            r->working_set_bytes,
            r->problem_size,
            r->threads_or_sms,
            r->t_alloc_ms,
            r->t_h2d_ms,
            r->t_kernel_cold_ms,
            r->t_kernel_warm_ms,
            r->t_d2h_ms,
            r->t_e2e_ms,
            r->eff_bw_gbps,
            r->c2c_bw_gbps,
            r->gflops,
            r->pct_peak,
            r->status);
    fflush(f);
}

#ifdef __cplusplus
}
#endif

#endif /* BENCH_HARNESS_H */
