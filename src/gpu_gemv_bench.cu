/* gpu_gemv_bench.cu -- multi-GPU cuBLAS GEMV / cuSPARSE SpMV benchmark.
 *
 * Companion to gpu_gemm_bench.cu. y = alpha*A*x + beta*y, A is SxS, x and y
 * are S-vectors. Memory-bandwidth-bound at every problem size (constant
 * arithmetic intensity ~2/elemsize FLOPs/byte) -- unlike GEMM, this never
 * becomes compute-bound, so the headline metric here is achieved GB/s and
 * % of peak HBM bandwidth, not TFLOPS.
 *
 * Storage convention: A is ROW-MAJOR (row stride = S) so that --mode split's
 * row-slices are contiguous. cuBLAS/cuSPARSE calls that expect column-major
 * data receive this same buffer with transa=CUBLAS_OP_T and lda=S -- a
 * row-major MxN matrix has identical memory layout to a column-major NxM
 * matrix, i.e. exactly A^T as cuBLAS would see it; op(A)=A^T with OP_T
 * recovers A. See Task 4 of the implementation plan for the full derivation.
 *
 * Sparse engine uses regular cusparseSpMV (CSR) -- NOT cuSPARSELt.
 * Structured 2:4 sparsity targets Sparse Tensor Cores (matrix-matrix
 * tiles); a single vector has no tile shape to speak of. This also
 * reuses the same CSR pattern already proven quiet, with no corruption
 * history, in gpu_gemm_bench.c's legacy spmm_worker().
 *
 * Safety (see memory/gpu_bench_sparse_corruption.md for the incident this
 * proactively avoids): no cudaDeviceReset() anywhere; a device that raises
 * an async fault is poisoned (skipped for the rest of the process) rather
 * than reset-then-reused; one-time per-job cuSPARSE setup is mutexed across
 * the per-GPU worker threads; the NVML sampler is stopped before any fault
 * is inspected.
 *
 * Compile:  make gpu_gemv_bench
 * Run:      ./bin/gpu_gemv_bench --help
 */

/* -- includes -------------------------------------------------------------- */
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <nvml.h>

#include <cassert>
#include <cerrno>
#include <cmath>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <pthread.h>
#include <atomic>
#include <mutex>
#include <unistd.h>
#include "bf16_cvt.h"
#include "gemv_host.h"

/* -- error-checking macros (identical pattern to gpu_gemm_bench.cu) -------- */
#define CUDA_CHECK_J(expr, lbl) do {                                         \
    cudaError_t _e = (expr);                                                 \
    if (_e != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA %s:%d %s -> %s\n",                             \
                __FILE__, __LINE__, #expr, cudaGetErrorString(_e));          \
        j->rc = -1; goto lbl;                                                \
    }                                                                         \
} while (0)

#define CUBLAS_CHECK_J(expr, lbl) do {                                       \
    cublasStatus_t _s = (expr);                                               \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                        \
        fprintf(stderr, "cuBLAS %s:%d %s -> %d\n",                           \
                __FILE__, __LINE__, #expr, (int)_s);                         \
        j->rc = -1; goto lbl;                                                \
    }                                                                         \
} while (0)

#define SPARSE_CHECK_J(expr, lbl) do {                                       \
    cusparseStatus_t _s = (expr);                                             \
    if (_s != CUSPARSE_STATUS_SUCCESS) {                                      \
        fprintf(stderr, "cuSPARSE %s:%d %s -> %d\n",                         \
                __FILE__, __LINE__, #expr, (int)_s);                         \
        j->rc = -1; goto lbl;                                                \
    }                                                                         \
} while (0)

#define CUDA_CHECK(expr) do {                                                 \
    cudaError_t _e = (expr);                                                  \
    if (_e != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA %s:%d %s -> %s\n",                             \
                __FILE__, __LINE__, #expr, cudaGetErrorString(_e));          \
        exit(1);                                                              \
    }                                                                         \
} while (0)

/* -- precision / mode / engine types --------------------------------------- */
typedef enum { PREC_FP32, PREC_FP64, PREC_BF16 } prec_t;
typedef enum { MODE_SPLIT, MODE_REPLICAS }         bench_mode_t;
typedef enum { ENG_DENSE,  ENG_SPARSE  }           engine_t;

static const char *eng_name(engine_t e) { return e == ENG_DENSE ? "dense" : "sparse"; }
static const char *prec_name(prec_t p) {
    return p == PREC_FP32 ? "fp32" : p == PREC_FP64 ? "fp64" : "bf16";
}
static size_t in_bytes(prec_t p)  { return p == PREC_FP32 ? 4 : p == PREC_FP64 ? 8 : 2; }
static size_t out_bytes(prec_t p) { return p == PREC_FP32 ? 4 : p == PREC_FP64 ? 8 : 4; }

static cudaDataType cuda_type(prec_t p) {
    return p == PREC_FP32 ? CUDA_R_32F : p == PREC_FP64 ? CUDA_R_64F : CUDA_R_16BF;
}

/* -- NVML live sampler (identical pattern to gpu_gemm_bench.cu) ------------ */
typedef struct {
    nvmlDevice_t dev;
    int          valid;
    std::atomic<int> stop;
    unsigned long long sm_sum, mem_sum, pow_sum, util_sum;
    unsigned     sm_max, mem_max, pow_max, temp_max;
    unsigned long long throttle_or;
    long         n;
} sampler_t;

static void *sampler_run(void *arg) {
    sampler_t *s = (sampler_t *)arg;
    if (!s->valid) return NULL;
    while (!s->stop.load(std::memory_order_relaxed)) {
        unsigned sm = 0, mem = 0, mw = 0, t = 0;
        unsigned long long tr = 0;
        nvmlUtilization_t u = {0, 0};
        if (nvmlDeviceGetClockInfo(s->dev, NVML_CLOCK_SM,  &sm)  == NVML_SUCCESS &&
            nvmlDeviceGetClockInfo(s->dev, NVML_CLOCK_MEM, &mem) == NVML_SUCCESS) {
            nvmlDeviceGetPowerUsage(s->dev, &mw);
            nvmlDeviceGetTemperature(s->dev, NVML_TEMPERATURE_GPU, &t);
            nvmlDeviceGetUtilizationRates(s->dev, &u);
            nvmlDeviceGetCurrentClocksThrottleReasons(s->dev, &tr);
            s->sm_sum   += sm;  s->mem_sum  += mem;
            s->pow_sum  += mw;  s->util_sum += u.gpu;
            if (sm  > s->sm_max)   s->sm_max   = sm;
            if (mem > s->mem_max)  s->mem_max  = mem;
            if (mw  > s->pow_max)  s->pow_max  = mw;
            if (t   > s->temp_max) s->temp_max = t;
            s->throttle_or |= tr; s->n++;
        }
        struct timespec ts = {0, 2*1000*1000};
        nanosleep(&ts, NULL);
    }
    return NULL;
}

static void throttle_str(unsigned long long tr, char *buf, size_t n) {
    if (tr == 0ULL || tr == nvmlClocksThrottleReasonNone) {
        snprintf(buf, n, "none"); return;
    }
    buf[0] = '\0'; size_t l = 0;
#define ADD(bit,nm) do { if (tr & (bit)) \
    l += (size_t)snprintf(buf+l, l<n?n-l:0, "%s%s", l?"|":"", nm); } while (0)
    ADD(nvmlClocksThrottleReasonGpuIdle,                  "idle");
    ADD(nvmlClocksThrottleReasonApplicationsClocksSetting,"app_clk");
    ADD(nvmlClocksThrottleReasonSwPowerCap,               "sw_pwr");
    ADD(nvmlClocksThrottleReasonHwSlowdown,               "hw_slow");
    ADD(nvmlClocksThrottleReasonSwThermalSlowdown,        "sw_therm");
    ADD(nvmlClocksThrottleReasonHwThermalSlowdown,        "hw_therm");
    ADD(nvmlClocksThrottleReasonHwPowerBrakeSlowdown,     "hw_pbrake");
#undef ADD
    if (l == 0) snprintf(buf, n, "0x%llx", tr);
}

/* -- per-GPU job descriptor ------------------------------------------------ */
typedef struct {
    /* inputs */
    int      dev;
    long     S;              /* full problem size, A is logically SxS */
    long     mrows;           /* this GPU's row-slice count (S for replicas) */
    long     row0;            /* starting row offset for this GPU's slice */
    prec_t   prec;
    engine_t engine;
    double   density; long nnz;
    /* sparse (CSR) fields -- full arrays + this GPU's slice descriptors */
    const int   *hRowOff, *hColInd;  /* FULL SxS CSR, host, unsliced */
    const float *hVals;
    const int   *hRowOff_local;      /* mrows+1 ints, rebased to start at 0 */
    long         colval_offset;      /* offset into hColInd/hVals for this GPU's rows */
    long         local_nnz;
    int      warmup, iters;
    /* PINNED host operands, full fp32 (validation reference) */
    const float *hA_pin;             /* full S*S, ROW-MAJOR */
    const float *hx_pin;             /* full S */
    /* native-precision operands actually used for compute */
    const void  *hA_nat;             /* this GPU's row-slice, native precision */
    const void  *hx_nat;             /* full S, native precision (replicated) */
    int      nvml_on, validate, valid_max, measure_xfer;
    pthread_barrier_t *bar;
    /* outputs */
    int      rc;
    double   batch_ms, h2d_ms, d2h_ms, max_rel_err, mem_gb;
    sampler_t smp;
} job_t;

/* -- sparse-engine device poisoning + one-time-setup mutex ----------------- */
/* An async (context-level) CUDA fault during a sparse job leaves that
 * physical GPU's driver-visible state unconfirmed safe for reuse. Skip it
 * for the rest of this process rather than reusing it (see the design's
 * Global Constraints: no cudaDeviceReset() anywhere). */
static std::atomic<int> g_gemv_sparse_dev_poisoned[8] = {};

/* Serializes the one-time per-job cuSPARSE handle/descriptor setup across
 * the per-GPU worker threads -- never the timed SpMV loop itself. */
static std::mutex g_gemv_cusparse_setup_mutex;

/* -- dense GEMV dispatch ---------------------------------------------------
 * y(mrows) = A(mrows x S, row-major) * x(S)
 *
 * fp32/fp64: native cuBLAS Level-2 gemv, routed via the row-major-as-
 * transposed-column-major trick (transa=OP_T, m=S, n=mrows, lda=S).
 * bf16: no native Level-2 bf16 gemv exists in cuBLAS -- routed through
 * cublasGemmEx with x as an Sx1 column-major "matrix" (trivial, no
 * transpose needed for a vector) and A via the same OP_T/lda=S trick,
 * k=S, m=mrows, n=1.                                                       */
static cublasStatus_t do_gemv(cublasHandle_t h, prec_t p, long S, long mrows,
                               const void *dA, const void *dx, void *dy) {
    static const float  f1 = 1.0f, f0 = 0.0f;
    static const double d1 = 1.0,  d0 = 0.0;
    int iS = (int)S, iM = (int)mrows;
    switch (p) {
    case PREC_FP32:
        return cublasSgemv(h, CUBLAS_OP_T, iS, iM, &f1,
                            (const float *)dA, iS,
                            (const float *)dx, 1, &f0,
                            (float *)dy, 1);
    case PREC_FP64:
        return cublasDgemv(h, CUBLAS_OP_T, iS, iM, &d1,
                            (const double *)dA, iS,
                            (const double *)dx, 1, &d0,
                            (double *)dy, 1);
    case PREC_BF16:
        return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, iM, 1, iS,
                             &f1, dA, CUDA_R_16BF, iS,
                             dx, CUDA_R_16BF, iS,
                             &f0, dy, CUDA_R_32F, iM,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

/* -- sampled correctness check (dense engine) ------------------------------
 * Compares a handful of computed y[i] (this GPU's row-slice) against a
 * host-computed fp64 reference dot product of row (row0+i) against x,
 * with precision-appropriate truncation applied to the reference. */
static double validate_gemv_slice(job_t *j, const void *hy) {
    const long S = j->S, mrows = j->mrows;
    double worst = 0.0;
    uint64_t rng = 0xC0FFEEull ^ (uint64_t)j->dev;
    int nsamp = mrows < 128 ? (int)mrows : 128;
    for (int s = 0; s < nsamp; s++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        long i = (long)(rng % (uint64_t)mrows);
        double ref = 0.0;
        for (long k = 0; k < S; k++) {
            double a = (double)j->hA_pin[(j->row0 + i) * S + k];
            double x = (double)j->hx_pin[k];
            if (j->prec == PREC_BF16) {
                a = (double)bf16_to_f32(f32_to_bf16((float)a));
                x = (double)bf16_to_f32(f32_to_bf16((float)x));
            }
            ref += a * x;
        }
        double got = (j->prec == PREC_FP64) ? ((const double *)hy)[i]
                                             : (double)((const float *)hy)[i];
        double denom = fabs(ref) > 1e-12 ? fabs(ref) : 1e-12;
        double rel = fabs(got - ref) / denom;
        if (rel > worst) worst = rel;
    }
    return worst;
}

/* -- dense GEMV worker ------------------------------------------------------
 * One thread per GPU. Allocates this GPU's row-slice of A (mrows x S), the
 * FULL x vector (replicated on every GPU -- small, cheap), and this GPU's
 * slice of y (mrows). No cross-GPU communication: split mode's row-slicing
 * is embarrassingly parallel, same as GEMM's column-split. */
static void *gemv_worker(void *arg) {
    job_t *j = (job_t *)arg;
    j->rc = 0; j->max_rel_err = -1.0;
    j->batch_ms = j->h2d_ms = j->d2h_ms = 0.0;
    const long S = j->S, mrows = j->mrows;
    size_t eb = in_bytes(j->prec), ob = out_bytes(j->prec);

    cublasHandle_t h = NULL; cudaStream_t st = NULL;
    void *dA = NULL, *dx = NULL, *dy = NULL;
    cudaEvent_t e_h2d0 = NULL, e_h2d1 = NULL, e0 = NULL, e1 = NULL, e_d2h0 = NULL, e_d2h1 = NULL;
    pthread_t smp_th; int smp_started = 0;
    void *hy_pin = NULL;

    if (cudaSetDevice(j->dev) != cudaSuccess) { j->rc = -1; goto barrier_out; }
    CUDA_CHECK_J(cudaStreamCreate(&st), barrier_out);
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { j->rc = -1; goto barrier_out; }
    cublasSetStream(h, st);
    cublasSetMathMode(h, j->prec == PREC_BF16 ? CUBLAS_DEFAULT_MATH : CUBLAS_PEDANTIC_MATH);

    CUDA_CHECK_J(cudaMalloc(&dA, (size_t)mrows * S * eb), barrier_out);
    CUDA_CHECK_J(cudaMalloc(&dx, (size_t)S * eb), barrier_out);
    CUDA_CHECK_J(cudaMalloc(&dy, (size_t)mrows * ob), barrier_out);
    CUDA_CHECK_J(cudaMemset(dy, 0, (size_t)mrows * ob), barrier_out);
    CUDA_CHECK_J(cudaStreamSynchronize(st), barrier_out);

    CUDA_CHECK_J(cudaEventCreate(&e_h2d0), barrier_out);
    CUDA_CHECK_J(cudaEventCreate(&e_h2d1), barrier_out);
    CUDA_CHECK_J(cudaEventCreate(&e0),     barrier_out);
    CUDA_CHECK_J(cudaEventCreate(&e1),     barrier_out);
    CUDA_CHECK_J(cudaEventCreate(&e_d2h0), barrier_out);
    CUDA_CHECK_J(cudaEventCreate(&e_d2h1), barrier_out);

    cudaEventRecord(e_h2d0, st);
    CUDA_CHECK_J(cudaMemcpyAsync(dA, j->hA_nat, (size_t)mrows * S * eb, cudaMemcpyHostToDevice, st), barrier_out);
    CUDA_CHECK_J(cudaMemcpyAsync(dx, j->hx_nat, (size_t)S * eb, cudaMemcpyHostToDevice, st), barrier_out);
    cudaEventRecord(e_h2d1, st);
    CUDA_CHECK_J(cudaStreamSynchronize(st), barrier_out);
    {
        float ms = 0.0f; cudaEventElapsedTime(&ms, e_h2d0, e_h2d1);
        j->h2d_ms = (double)ms;
    }

    for (int w = 0; w < j->warmup; w++) {
        if (do_gemv(h, j->prec, S, mrows, dA, dx, dy) != CUBLAS_STATUS_SUCCESS) j->rc = -1;
    }
    CUDA_CHECK_J(cudaStreamSynchronize(st), barrier_out);
    if (j->rc) goto barrier_out;

    pthread_barrier_wait(j->bar);

    if (j->nvml_on && j->smp.valid) {
        j->smp.stop.store(0, std::memory_order_relaxed);
        if (pthread_create(&smp_th, NULL, sampler_run, &j->smp) == 0)
            smp_started = 1;
    }

    {
        int loop_err = 0;
        cudaEventRecord(e0, st);
        for (int it = 0; it < j->iters; it++) {
            if (do_gemv(h, j->prec, S, mrows, dA, dx, dy) != CUBLAS_STATUS_SUCCESS)
                loop_err = 1;
        }
        cudaEventRecord(e1, st);
        cudaEventSynchronize(e1);
        if (smp_started) { j->smp.stop.store(1, std::memory_order_relaxed); pthread_join(smp_th, NULL); }
        if (loop_err) { j->rc = -1; goto done; }
        float ms = 0.0f; cudaEventElapsedTime(&ms, e0, e1);
        j->batch_ms = (double)ms;
    }

    if (j->measure_xfer || (j->validate && S <= j->valid_max)) {
        size_t yb = (size_t)mrows * ob;
        if (cudaMallocHost(&hy_pin, yb) == cudaSuccess) {
            cudaEventRecord(e_d2h0, st);
            cudaMemcpyAsync(hy_pin, dy, yb, cudaMemcpyDeviceToHost, st);
            cudaEventRecord(e_d2h1, st);
            cudaStreamSynchronize(st);
            float ms = 0.0f; cudaEventElapsedTime(&ms, e_d2h0, e_d2h1);
            j->d2h_ms = (double)ms;
            if (j->validate && S <= j->valid_max)
                j->max_rel_err = validate_gemv_slice(j, hy_pin);
            cudaFreeHost(hy_pin); hy_pin = NULL;
        }
    }

done:
    if (e_h2d0) cudaEventDestroy(e_h2d0);
    if (e_h2d1) cudaEventDestroy(e_h2d1);
    if (e0)     cudaEventDestroy(e0);
    if (e1)     cudaEventDestroy(e1);
    if (e_d2h0) cudaEventDestroy(e_d2h0);
    if (e_d2h1) cudaEventDestroy(e_d2h1);
    if (dA) cudaFree(dA); if (dx) cudaFree(dx); if (dy) cudaFree(dy);
    if (h)  cublasDestroy(h);
    if (st) cudaStreamDestroy(st);
    return NULL;

barrier_out:
    if (e_h2d0) cudaEventDestroy(e_h2d0);
    if (e_h2d1) cudaEventDestroy(e_h2d1);
    if (e0)     cudaEventDestroy(e0);
    if (e1)     cudaEventDestroy(e1);
    if (e_d2h0) cudaEventDestroy(e_d2h0);
    if (e_d2h1) cudaEventDestroy(e_d2h1);
    if (dA) cudaFree(dA); if (dx) cudaFree(dx); if (dy) cudaFree(dy);
    if (h)  cublasDestroy(h);
    if (st) cudaStreamDestroy(st);
    pthread_barrier_wait(j->bar);
    return NULL;
}

/* -- sparse SpMV worker ------------------------------------------------------
 * y(mrows) = A_sparse(mrows x S, CSR, this GPU's row-slice) * x(S)
 *
 * Uses regular cusparseSpMV (CSR) -- NOT cuSPARSELt (see the file header
 * comment and design spec §5.5 for why). Safety measures below mirror
 * memory/gpu_bench_sparse_corruption.md's fix, applied from first commit:
 *   1. No cudaDeviceReset() anywhere.
 *   2. Poison-and-skip: check g_gemv_sparse_dev_poisoned[dev] first; on an
 *      async fault, set it and let the *next* job on this device skip
 *      instead of blindly reusing unconfirmed-safe state.
 *   3. One-time setup (cusparseCreate through SpMV_bufferSize) is mutexed
 *      across the per-GPU worker threads -- never the timed loop.
 *   4. NVML sampler stopped before any fault is inspected.
 *
 * Single cleanup label + pre_barrier flag => the barrier is hit exactly once
 * per thread (same pattern as gpu_gemm_bench.c/.cu's spmm_worker). Calling
 * pthread_barrier_wait(j->bar) twice on a success/late-failure path while a
 * poisoned/early-failing sibling thread on another GPU only calls it once
 * would desync the round count across the harness's fixed-count barrier and
 * hang the whole process in pthread_join -- the same "needs a power cycle"
 * failure class this file exists to avoid, just via deadlock instead of
 * driver corruption.                                                       */
static void *spmv_worker(void *arg) {
    job_t *j = (job_t *)arg;
    j->rc = 0; j->max_rel_err = -1.0;
    j->batch_ms = j->h2d_ms = j->d2h_ms = 0.0;
    const long S = j->S, mrows = j->mrows;
    int smp_started = 0;
    int pre_barrier = 0;

    if (j->dev >= 0 && j->dev < 8 &&
        g_gemv_sparse_dev_poisoned[j->dev].load(std::memory_order_relaxed)) {
        fprintf(stderr,
            "[dev %d] skipping sparse job: device was marked unsafe after an "
            "async CUDA fault earlier this run.\n", j->dev);
        /* Reached before any resource is allocated and before the mid-job
         * barrier below -- a direct return (not `goto sout`, which would
         * collide with the label inside the resource block below: labels
         * have function scope in C/C++, so two `sout:` labels in one
         * function is always a redefinition error, not compiler-dependent)
         * doing exactly what that label's cleanup would do here (nothing
         * to free yet) keeps the barrier call count at exactly one. */
        j->rc = -1;
        pthread_barrier_wait(j->bar);
        return NULL;
    }

    if (j->prec == PREC_FP64) {
        /* regular cuSPARSE SpMV supports fp64 fine, unlike GEMM's
         * cuSPARSELt path -- included for completeness, no rejection here. */
    }

    {
        cusparseHandle_t sph = NULL;
        cudaStream_t st = NULL;
        void *dRowOff = NULL, *dColInd = NULL, *dVals = NULL;
        void *dx = NULL, *dy = NULL, *dBuf = NULL;
        cusparseSpMatDescr_t matA = NULL;
        cusparseDnVecDescr_t vecX = NULL, vecY = NULL;
        cudaEvent_t e_h2d0 = NULL, e_h2d1 = NULL, e0 = NULL, e1 = NULL, e_d2h0 = NULL, e_d2h1 = NULL;
        pthread_t smp_th;
        void *hy_pin = NULL;
        size_t eb = in_bytes(j->prec), ob = out_bytes(j->prec);

        if (cudaSetDevice(j->dev) != cudaSuccess) { j->rc = -1; goto sout; }
        CUDA_CHECK_J(cudaStreamCreate(&st), sout);

        CUDA_CHECK_J(cudaMalloc(&dRowOff, (size_t)(mrows + 1) * sizeof(int)), sout);
        CUDA_CHECK_J(cudaMalloc(&dColInd, (size_t)j->local_nnz * sizeof(int)), sout);
        CUDA_CHECK_J(cudaMalloc(&dVals,   (size_t)j->local_nnz * eb), sout);
        CUDA_CHECK_J(cudaMalloc(&dx, (size_t)S * eb), sout);
        CUDA_CHECK_J(cudaMalloc(&dy, (size_t)mrows * ob), sout);
        CUDA_CHECK_J(cudaMemset(dy, 0, (size_t)mrows * ob), sout);
        CUDA_CHECK_J(cudaStreamSynchronize(st), sout);

        CUDA_CHECK_J(cudaEventCreate(&e_h2d0), sout);
        CUDA_CHECK_J(cudaEventCreate(&e_h2d1), sout);
        cudaEventRecord(e_h2d0, st);
        CUDA_CHECK_J(cudaMemcpyAsync(dRowOff, j->hRowOff_local, (size_t)(mrows + 1) * sizeof(int),
                                      cudaMemcpyHostToDevice, st), sout);
        CUDA_CHECK_J(cudaMemcpyAsync(dColInd, j->hColInd + j->colval_offset,
                                      (size_t)j->local_nnz * sizeof(int),
                                      cudaMemcpyHostToDevice, st), sout);
        CUDA_CHECK_J(cudaMemcpyAsync(dVals, (const char *)j->hVals + (size_t)j->colval_offset * eb,
                                      (size_t)j->local_nnz * eb, cudaMemcpyHostToDevice, st), sout);
        CUDA_CHECK_J(cudaMemcpyAsync(dx, j->hx_nat, (size_t)S * eb, cudaMemcpyHostToDevice, st), sout);
        cudaEventRecord(e_h2d1, st);
        CUDA_CHECK_J(cudaStreamSynchronize(st), sout);
        {
            float ms = 0.0f; cudaEventElapsedTime(&ms, e_h2d0, e_h2d1);
            j->h2d_ms = (double)ms;
        }
        cudaEventDestroy(e_h2d0); cudaEventDestroy(e_h2d1);
        e_h2d0 = e_h2d1 = NULL;

        /* No initializer (unlike a plain `= 0`): every `goto sout` above this
         * point jumps into this variable's scope, and C++ forbids that for a
         * variable with automatic storage duration UNLESS it has scalar type
         * and no initializer (a "vacuous" declaration) -- the exact reason
         * cusparseLtMatmulGetWorkspace's output vars in gpu_gemm_bench.cu's
         * spmm_worker are declared the same bare way. bufSize is only ever
         * read after cusparseSpMV_bufferSize() below has written to it (any
         * earlier failure jumps to sout before the read), so leaving it
         * uninitialized here is safe. */
        size_t bufSize;
        {
            /* Mutexed one-time setup -- released automatically (RAII) on
             * every exit path, including the CHECK_J `goto sout` jumps
             * below (C++ runs the lock_guard's destructor when the
             * enclosing scope is exited via goto, same as normal exit). */
            std::lock_guard<std::mutex> setup_lock(g_gemv_cusparse_setup_mutex);

            if (cusparseCreate(&sph) != CUSPARSE_STATUS_SUCCESS) { j->rc = -1; goto sout; }
            cusparseSetStream(sph, st);

            SPARSE_CHECK_J(cusparseCreateCsr(&matA, mrows, S, j->local_nnz,
                dRowOff, dColInd, dVals,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                cuda_type(j->prec)), sout);
            SPARSE_CHECK_J(cusparseCreateDnVec(&vecX, S, dx, cuda_type(j->prec)), sout);
            SPARSE_CHECK_J(cusparseCreateDnVec(&vecY, mrows, dy, cuda_type(j->prec)), sout);

            static const float  f1 = 1.0f, f0 = 0.0f;
            static const double d1 = 1.0,  d0 = 0.0;
            const void *alpha = (j->prec == PREC_FP64) ? (const void *)&d1 : (const void *)&f1;
            const void *beta  = (j->prec == PREC_FP64) ? (const void *)&d0 : (const void *)&f0;
            cudaDataType compute = (j->prec == PREC_FP64) ? CUDA_R_64F : CUDA_R_32F;

            SPARSE_CHECK_J(cusparseSpMV_bufferSize(sph, CUSPARSE_OPERATION_NON_TRANSPOSE,
                alpha, matA, vecX, beta, vecY, compute,
                CUSPARSE_SPMV_ALG_DEFAULT, &bufSize), sout);
            if (bufSize) CUDA_CHECK_J(cudaMalloc(&dBuf, bufSize), sout);
        }

        {
            static const float  f1 = 1.0f, f0 = 0.0f;
            static const double d1 = 1.0,  d0 = 0.0;
            const void *alpha = (j->prec == PREC_FP64) ? (const void *)&d1 : (const void *)&f1;
            const void *beta  = (j->prec == PREC_FP64) ? (const void *)&d0 : (const void *)&f0;
            cudaDataType compute = (j->prec == PREC_FP64) ? CUDA_R_64F : CUDA_R_32F;

            for (int w = 0; w < j->warmup; w++) {
                if (cusparseSpMV(sph, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, matA, vecX,
                        beta, vecY, compute, CUSPARSE_SPMV_ALG_DEFAULT, dBuf)
                        != CUSPARSE_STATUS_SUCCESS) j->rc = -1;
            }
            CUDA_CHECK_J(cudaStreamSynchronize(st), sout);
            if (j->rc) goto sout;

            pthread_barrier_wait(j->bar); pre_barrier = 1;

            if (j->nvml_on && j->smp.valid) {
                j->smp.stop.store(0, std::memory_order_relaxed);
                if (pthread_create(&smp_th, NULL, sampler_run, &j->smp) == 0)
                    smp_started = 1;
            }

            int loop_err = 0;
            CUDA_CHECK_J(cudaEventCreate(&e0), sout);
            CUDA_CHECK_J(cudaEventCreate(&e1), sout);
            cudaEventRecord(e0, st);
            for (int it = 0; it < j->iters; it++) {
                if (cusparseSpMV(sph, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, matA, vecX,
                        beta, vecY, compute, CUSPARSE_SPMV_ALG_DEFAULT, dBuf)
                        != CUSPARSE_STATUS_SUCCESS) loop_err = 1;
            }
            cudaEventRecord(e1, st);
            cudaEventSynchronize(e1);

            /* Stop the sampler BEFORE inspecting/handling any fault. */
            if (smp_started) {
                j->smp.stop.store(1, std::memory_order_relaxed);
                pthread_join(smp_th, NULL);
                smp_started = 0;
            }

            /* Async-fault check: poison, do NOT reset. */
            cudaError_t aerr = cudaGetLastError();
            if (aerr != cudaSuccess) {
                fprintf(stderr, "[dev %d] async CUDA error after SpMV loop: %s\n",
                        j->dev, cudaGetErrorString(aerr));
                loop_err = 1;
                if (j->dev >= 0 && j->dev < 8)
                    g_gemv_sparse_dev_poisoned[j->dev].store(1, std::memory_order_relaxed);
            }

            float ms = 0.0f; cudaEventElapsedTime(&ms, e0, e1);
            j->batch_ms = (double)ms;
            cudaEventDestroy(e0); cudaEventDestroy(e1);
            e0 = e1 = NULL;

            if (loop_err) { j->rc = -1; goto sout; }
        }

        /* Sparse has no reference implementation to validate against: unlike
         * GEMM's cuSPARSELt path (which has validate_structured_spmm),
         * validate_gemv_slice is a dense-only full-row dot-product
         * reference that unconditionally dereferences j->hA_pin -- and
         * hA_pin is NULL for every sparse job (main() never materializes a
         * dense A for the CSR engine). Calling it here would be a
         * guaranteed NULL-pointer crash on any --engine sparse --validate
         * run. A CSR-based reference is real kernel-authorship work that's
         * out of scope for this benchmark, so the sparse engine
         * intentionally never validates: j->max_rel_err stays at its -1.0
         * "not validated" sentinel set at the top of this function, which
         * the CSV writer and the ok/ok-val/ok-HIERR status string already
         * treat correctly via `rel >= 0` checks. The D2H transfer below is
         * still performed (when --no-transfers wasn't passed) purely to
         * measure d2h_ms -- it no longer feeds a validation call. */
        if (j->measure_xfer) {
            size_t yb = (size_t)mrows * ob;
            if (cudaMallocHost(&hy_pin, yb) == cudaSuccess) {
                CUDA_CHECK_J(cudaEventCreate(&e_d2h0), sout);
                CUDA_CHECK_J(cudaEventCreate(&e_d2h1), sout);
                cudaEventRecord(e_d2h0, st);
                cudaMemcpyAsync(hy_pin, dy, yb, cudaMemcpyDeviceToHost, st);
                cudaEventRecord(e_d2h1, st);
                cudaStreamSynchronize(st);
                float ms = 0.0f; cudaEventElapsedTime(&ms, e_d2h0, e_d2h1);
                j->d2h_ms = (double)ms;
                cudaEventDestroy(e_d2h0); cudaEventDestroy(e_d2h1);
                e_d2h0 = e_d2h1 = NULL;
                cudaFreeHost(hy_pin); hy_pin = NULL;
            }
        }

sout:
        if (smp_started) {
            j->smp.stop.store(1, std::memory_order_relaxed);
            pthread_join(smp_th, NULL);
        }
        if (vecY) cusparseDestroyDnVec(vecY);
        if (vecX) cusparseDestroyDnVec(vecX);
        if (matA) cusparseDestroySpMat(matA);
        if (sph)  cusparseDestroy(sph);
        if (e_h2d0) cudaEventDestroy(e_h2d0);
        if (e_h2d1) cudaEventDestroy(e_h2d1);
        if (e0)     cudaEventDestroy(e0);
        if (e1)     cudaEventDestroy(e1);
        if (e_d2h0) cudaEventDestroy(e_d2h0);
        if (e_d2h1) cudaEventDestroy(e_d2h1);
        if (dRowOff) cudaFree(dRowOff);
        if (dColInd) cudaFree(dColInd);
        if (dVals)   cudaFree(dVals);
        if (dx) cudaFree(dx);
        if (dy) cudaFree(dy);
        if (dBuf) cudaFree(dBuf);
        if (st) cudaStreamDestroy(st);
        if (!pre_barrier) pthread_barrier_wait(j->bar);  /* errored before timed region */
        return NULL;
    }
}

/* -- config ---------------------------------------------------------------- */
typedef struct {
    long   sizes[32]; int n_sizes;
    prec_t precs[3];  int n_precs;
    int    gpus[8];   int n_gpus;
    bench_mode_t mode;
    engine_t engines[2]; int n_engines;
    double densities[8]; int n_dens;
    int    iters, warmup;
    int    nvml_on, validate, valid_max, measure_xfer;
    double mem_frac;
    const char *csv, *tag;
} cfg_t;

static void usage(const char *p) {
    printf(
"Usage: %s [options]\n"
"  --sizes a,b,..     matrix sizes (A is SxS)        [256..65536 pow2]\n"
"  --precisions p,..  fp32,fp64,bf16                 [all]\n"
"  --gpus n,..        gpu counts to test              [1..4]\n"
"  --mode m           split | replicas                [split]\n"
"  --engine e,..      dense | sparse                  [dense]\n"
"  --density d,..     sparse A nonzero fraction        [0.01,0.05,0.1,0.25]\n"
"  --iters n          timed iterations                 [10]\n"
"  --warmup n         warmup iterations                [3]\n"
"  --no-nvml          disable NVML telemetry\n"
"  --validate         sampled correctness check\n"
"  --validate-max n   max size to validate              [2048]\n"
"  --no-transfers     skip timed H2D/D2H\n"
"  --mem-frac f       usable GPU mem fraction           [0.90]\n"
"  --csv path         append CSV results\n"
"  --tag s            CSV label\n"
"  --help\n"
"\n"
"GEMV is memory-bandwidth-bound at every size (constant arithmetic intensity)\n"
"-- unlike GEMM, it never becomes compute-bound. Headline metric is achieved\n"
"GB/s and %% of A100-SXM4-40GB's ~1555 GB/s HBM peak, not TFLOPS. TFLOPS is\n"
"kept as a secondary column for cross-reference against the GEMM benchmark.\n",
    p);
}

/* -- one benchmark point ---------------------------------------------------- */
static void run_gemv_config(const cfg_t *c, FILE *csv, const char *host, int cudart,
                             long cap, int nvml_ready, long S, prec_t p, int G,
                             engine_t engine, double density, long nnz,
                             const float *hA_pin, const float *hx_pin,
                             const int *hRowOff, const int *hColInd, const float *hVals)
{
    long ts = (long)time(NULL);
    const char *mode_s = (c->mode == MODE_SPLIT) ? "split" : "replicas";
    const char *eng_s  = eng_name(engine);
    const char *pn = prec_name(p);
    size_t eb = in_bytes(p), ob = out_bytes(p);

    long max_mrows = 0;
    long foot = gemv_mem_footprint_bytes(eb, ob, c->mode == MODE_REPLICAS, S, G, &max_mrows);
    double mem_gb = (double)foot / 1e9;

    if (foot > cap) {
        printf("%-6s %-5s %-3d %6ld %9s %9s %6s %8s %8s %5s %7s %7s %s\n",
               eng_s, pn, G, S, "-", "-", "-", "-", "-", "-", "-", "-", "oom-skip");
        if (csv) fprintf(csv,
            "%s,%ld,%s,%d,%s,%s,%d,%ld,%d,%d,0,0,0,0,0,0,0,0,0,0,none,0,%.3f,0,-1,"
            "0,0,0,0,0,0,%s,%.4g,%ld,oom-skip\n",
            c->tag, ts, host, cudart, mode_s, pn, G, S, c->iters, c->warmup,
            mem_gb, eng_s, density, nnz);
        return;
    }

    void *hA_nat_full = NULL, *hx_nat = NULL, *hVals_nat = NULL;
    if (engine == ENG_DENSE) {
        if (p == PREC_FP32) {
            hA_nat_full = (void *)hA_pin;
            hx_nat = (void *)hx_pin;
        } else {
            if (cudaMallocHost(&hA_nat_full, (size_t)S * S * eb) != cudaSuccess) {
                fprintf(stderr, "Host OOM for pre-converted A\n"); return;
            }
            if (cudaMallocHost(&hx_nat, (size_t)S * eb) != cudaSuccess) {
                fprintf(stderr, "Host OOM for pre-converted x\n");
                cudaFreeHost(hA_nat_full); return;
            }
            if (p == PREC_FP64) {
                for (size_t i = 0; i < (size_t)S * S; i++) ((double *)hA_nat_full)[i] = (double)hA_pin[i];
                for (size_t i = 0; i < (size_t)S; i++) ((double *)hx_nat)[i] = (double)hx_pin[i];
            } else {
                for (size_t i = 0; i < (size_t)S * S; i++) ((uint16_t *)hA_nat_full)[i] = f32_to_bf16(hA_pin[i]);
                for (size_t i = 0; i < (size_t)S; i++) ((uint16_t *)hx_nat)[i] = f32_to_bf16(hx_pin[i]);
            }
        }
    } else {
        if (p == PREC_FP32) {
            hVals_nat = (void *)hVals;
            hx_nat = (void *)hx_pin;
        } else if (p == PREC_FP64) {
            if (cudaMallocHost(&hVals_nat, (size_t)nnz * eb) != cudaSuccess) {
                fprintf(stderr, "Host OOM for pre-converted sparse vals\n"); return;
            }
            if (cudaMallocHost(&hx_nat, (size_t)S * eb) != cudaSuccess) {
                fprintf(stderr, "Host OOM for pre-converted sparse x\n");
                cudaFreeHost(hVals_nat); return;
            }
            for (long i = 0; i < nnz; i++) ((double *)hVals_nat)[i] = (double)hVals[i];
            for (long i = 0; i < S; i++) ((double *)hx_nat)[i] = (double)hx_pin[i];
        } else {
            fprintf(stderr, "bf16 not supported for sparse SpMV engine; skipping\n");
            return;
        }
    }

    job_t job[8]; pthread_barrier_t bar;
    pthread_barrier_init(&bar, NULL, (unsigned)G);
    long base = S / G, rem = S % G, row0 = 0;
    int *local_rowoffs[8] = {};
    for (int g = 0; g < G; g++) {
        long mrows = (c->mode == MODE_REPLICAS) ? S : (base + (g < rem ? 1 : 0));
        memset(&job[g], 0, sizeof job[g]);
        job[g].dev = g; job[g].S = S; job[g].mrows = mrows; job[g].row0 = row0;
        job[g].prec = p; job[g].engine = engine;
        job[g].density = density; job[g].nnz = nnz;
        job[g].warmup = c->warmup; job[g].iters = c->iters;
        job[g].hA_pin = hA_pin; job[g].hx_pin = hx_pin;
        job[g].nvml_on = nvml_ready;
        job[g].validate = c->validate; job[g].valid_max = c->valid_max;
        job[g].measure_xfer = c->measure_xfer;
        job[g].bar = &bar; job[g].mem_gb = mem_gb;

        if (engine == ENG_DENSE) {
            job[g].hA_nat = (const char *)hA_nat_full + (size_t)row0 * S * eb;
            job[g].hx_nat = hx_nat;
        } else {
            job[g].hRowOff = hRowOff; job[g].hColInd = hColInd;
            job[g].hVals = (const float *)hVals_nat;
            job[g].hx_nat = hx_nat;
            job[g].colval_offset = hRowOff[row0];
            job[g].local_nnz = hRowOff[row0 + mrows] - hRowOff[row0];
            int *local = (int *)malloc((size_t)(mrows + 1) * sizeof(int));
            for (long i = 0; i <= mrows; i++) local[i] = hRowOff[row0 + i] - hRowOff[row0];
            job[g].hRowOff_local = local;
            local_rowoffs[g] = local;
        }

        if (nvml_ready) {
            char bus[32];
            if (cudaDeviceGetPCIBusId(bus, sizeof bus, g) == cudaSuccess)
                if (nvmlDeviceGetHandleByPciBusId_v2(bus, &job[g].smp.dev) == NVML_SUCCESS)
                    job[g].smp.valid = 1;
            job[g].smp.stop.store(0, std::memory_order_relaxed);
        }
        if (c->mode != MODE_REPLICAS) row0 += mrows;
    }

    pthread_t th[8];
    for (int g = 0; g < G; g++) {
        int prc = pthread_create(&th[g], NULL,
                                  engine == ENG_SPARSE ? spmv_worker : gemv_worker, &job[g]);
        if (prc != 0) {
            /* Unrecoverable for this config: bar was pthread_barrier_init'd
             * with count G, and any already-started sibling threads may
             * already be past their own pthread_barrier_wait(j->bar) call,
             * committed to waiting for G arrivals that can now never come.
             * Continuing (or trying to shrink the barrier count after the
             * fact) risks exactly the "process needs a power cycle" failure
             * class this file's design otherwise avoids -- fail loudly and
             * immediately instead, same philosophy as CUDA_CHECK's exit(1). */
            fprintf(stderr,
                "FATAL: pthread_create failed for GPU index %d (of %d requested): %s\n"
                "One or more sibling worker threads for this config may already be running "
                "and blocked on a barrier that can now never reach its target count of %d -- "
                "not attempting to continue or unwind. Exiting.\n",
                g, G, strerror(prc), G);
            exit(1);
        }
    }
    for (int g = 0; g < G; g++) pthread_join(th[g], NULL);
    pthread_barrier_destroy(&bar);

    for (int g = 0; g < G; g++) if (local_rowoffs[g]) free(local_rowoffs[g]);
    if (engine == ENG_DENSE && p != PREC_FP32) {
        if (hA_nat_full) cudaFreeHost(hA_nat_full);
        if (hx_nat) cudaFreeHost(hx_nat);
    } else if (engine == ENG_SPARSE && p == PREC_FP64) {
        if (hVals_nat) cudaFreeHost(hVals_nat);
        if (hx_nat) cudaFreeHost(hx_nat);
    }

    int err = 0; double wall = 0, sum_bytes = 0, rel = -1.0;
    unsigned long long sm_sum = 0, mem_sum = 0, pow_sum = 0, util_sum = 0;
    long nsamp = 0;
    unsigned sm_max = 0, mem_max = 0, pow_max = 0, temp_max = 0;
    unsigned long long tr = 0;
    for (int g = 0; g < G; g++) {
        if (job[g].rc) { err = 1; continue; }
        double tg = job[g].batch_ms / 1000.0; if (tg > wall) wall = tg;
        sum_bytes += (engine == ENG_SPARSE)
            ? (double)c->iters * (double)job[g].local_nnz * (double)(eb + sizeof(int))
            : (double)c->iters * (double)job[g].mrows * (double)S * (double)eb;
        if (job[g].max_rel_err > rel) rel = job[g].max_rel_err;
        sampler_t *s = &job[g].smp;
        if (s->n > 0) {
            sm_sum += s->sm_sum; mem_sum += s->mem_sum; pow_sum += s->pow_sum; util_sum += s->util_sum;
            nsamp += s->n;
            if (s->sm_max > sm_max) sm_max = s->sm_max;
            if (s->mem_max > mem_max) mem_max = s->mem_max;
            if (s->pow_max > pow_max) pow_max = s->pow_max;
            if (s->temp_max > temp_max) temp_max = s->temp_max;
            tr |= s->throttle_or;
        }
    }

    if (err || wall <= 0.0) {
        const char *status = err ? "error" : "time-zero";
        printf("%-6s %-5s %-3d %6ld %9s %9s %6s %8s %8s %5s %7s %7s %s\n",
               eng_s, pn, G, S, "-", "-", "-", "-", "-", "-", "-", "-", status);
        if (csv) fprintf(csv,
            "%s,%ld,%s,%d,%s,%s,%d,%ld,%d,%d,0,0,0,0,0,0,0,0,0,0,none,0,%.3f,0,-1,"
            "0,0,0,0,0,0,%s,%.4g,%ld,%s\n",
            c->tag, ts, host, cudart, mode_s, pn, G, S, c->iters, c->warmup,
            mem_gb, eng_s, density, nnz, status);
        return;
    }

    double agg_gbps = sum_bytes / wall / 1e9;
    double per_gpu = agg_gbps / (double)G;
    double pct = 100.0 * agg_gbps / ((double)G * gemv_peak_gbps());
    double replicas_mult = (c->mode == MODE_REPLICAS) ? (double)G : 1.0;
    double agg_tflops = (engine == ENG_SPARSE)
        ? replicas_mult * (double)nnz * 2.0 * (double)c->iters / wall / 1e12
        : replicas_mult * (double)S * S * 2.0 * (double)c->iters / wall / 1e12;

    int bn = 0; double max_e2e = 0;
    for (int g = 0; g < G; g++) {
        double comp = job[g].batch_ms / (double)c->iters;
        double e2e = job[g].h2d_ms + comp + job[g].d2h_ms;
        if (e2e > max_e2e) { max_e2e = e2e; bn = g; }
    }
    double b_h2d = job[bn].h2d_ms, b_d2h = job[bn].d2h_ms;
    double b_comp = job[bn].batch_ms / (double)c->iters;
    double e2e_ms = max_e2e;
    double xfer_pct = e2e_ms > 0 ? 100.0 * (b_h2d + b_d2h) / e2e_ms : 0;

    double sm_avg = nsamp ? (double)sm_sum / (double)nsamp : 0;
    double mem_avg = nsamp ? (double)mem_sum / (double)nsamp : 0;
    double pow_avg = nsamp ? (double)pow_sum / (double)nsamp / 1000.0 : 0;
    double util_avg = nsamp ? (double)util_sum / (double)nsamp : 0;
    char trbuf[128]; throttle_str(tr, trbuf, sizeof trbuf);

    double tol = (p == PREC_BF16) ? 5e-2 : (p == PREC_FP32) ? 1e-3 : 1e-9;
    const char *vstr = rel >= 0 ? (rel < tol ? "ok/val" : "ok/HIERR") : "ok";

    printf("%-6s %-5s %-3d %6ld %9.1f %9.1f %5.1f%% %8.2f %8.2f %5.1f%% %7.2f %7.0f %s\n",
           eng_s, pn, G, S, agg_gbps, per_gpu, pct, b_comp, e2e_ms, xfer_pct, agg_tflops, sm_avg, vstr);

    if (csv) {
        fprintf(csv,
            "%s,%ld,%s,%d,%s,%s,%d,%ld,%d,%d,"
            "%.2f,%.2f,%.2f,%.3f,"
            "%.0f,%u,%.0f,%.1f,%.1f,%u,%s,%.1f,"
            "%.3f,%d,%.3g,"
            "%.3f,%.3f,%.3f,%.3f,%.2f,%.4g,"
            "%s,%.4g,%ld,ok\n",
            c->tag, ts, host, cudart, mode_s, pn, G, S, c->iters, c->warmup,
            agg_gbps, per_gpu, pct, wall * 1000.0,
            sm_avg, sm_max, mem_avg, pow_avg, (double)pow_max / 1000.0, temp_max, trbuf, util_avg,
            /* Sparse never validates (spmv_worker never calls
             * validate_gemv_slice -- see its comment): don't claim
             * validated=1 for a sparse row when nothing was actually
             * checked, even if --validate/--validate-max would have
             * qualified this size for the dense engine. */
            mem_gb, (c->validate && S <= c->valid_max && engine == ENG_DENSE), rel,
            b_h2d, b_d2h, b_comp, e2e_ms, xfer_pct, agg_tflops,
            eng_s, density, nnz);
        fflush(csv);
    }
}

/* -- main ------------------------------------------------------------------ */
int main(int argc, char **argv) {
    cfg_t c = {};
    c.mode = MODE_SPLIT; c.iters = 10; c.warmup = 3; c.nvml_on = 1;
    c.valid_max = 2048; c.mem_frac = 0.90; c.tag = ""; c.measure_xfer = 1;
    int set_sizes = 0, set_precs = 0, set_gpus = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--sizes") && i + 1 < argc) {
            c.n_sizes = gemv_parse_longs(argv[++i], c.sizes, 32); set_sizes = 1;
        } else if (!strcmp(a, "--precisions") && i + 1 < argc) {
            char b[128]; snprintf(b, sizeof b, "%s", argv[++i]); c.n_precs = 0;
            for (char *t = strtok(b, ","); t; t = strtok(NULL, ",")) {
                if      (!strcmp(t, "fp32")) c.precs[c.n_precs++] = PREC_FP32;
                else if (!strcmp(t, "fp64")) c.precs[c.n_precs++] = PREC_FP64;
                else if (!strcmp(t, "bf16")) c.precs[c.n_precs++] = PREC_BF16;
            } set_precs = 1;
        } else if (!strcmp(a, "--gpus") && i + 1 < argc) {
            long g[8]; int n = gemv_parse_longs(argv[++i], g, 8);
            for (int k = 0; k < n; k++) c.gpus[k] = (int)g[k];
            c.n_gpus = n; set_gpus = 1;
        } else if (!strcmp(a, "--mode") && i + 1 < argc) {
            c.mode = strcmp(argv[++i], "replicas") ? MODE_SPLIT : MODE_REPLICAS;
        } else if (!strcmp(a, "--engine") && i + 1 < argc) {
            char b[64]; snprintf(b, sizeof b, "%s", argv[++i]); c.n_engines = 0;
            for (char *t = strtok(b, ","); t && c.n_engines < 2; t = strtok(NULL, ",")) {
                if      (!strcmp(t, "dense"))  c.engines[c.n_engines++] = ENG_DENSE;
                else if (!strcmp(t, "sparse")) c.engines[c.n_engines++] = ENG_SPARSE;
            }
        } else if (!strcmp(a, "--density") && i + 1 < argc) {
            char b[128]; snprintf(b, sizeof b, "%s", argv[++i]); c.n_dens = 0;
            for (char *t = strtok(b, ","); t && c.n_dens < 8; t = strtok(NULL, ","))
                c.densities[c.n_dens++] = atof(t);
        } else if (!strcmp(a, "--iters") && i + 1 < argc) c.iters = atoi(argv[++i]);
        else if   (!strcmp(a, "--warmup") && i + 1 < argc) c.warmup = atoi(argv[++i]);
        else if   (!strcmp(a, "--no-nvml"))                c.nvml_on = 0;
        else if   (!strcmp(a, "--validate"))               c.validate = 1;
        else if   (!strcmp(a, "--validate-max") && i + 1 < argc) c.valid_max = atoi(argv[++i]);
        else if   (!strcmp(a, "--no-transfers"))           c.measure_xfer = 0;
        else if   (!strcmp(a, "--mem-frac") && i + 1 < argc) c.mem_frac = atof(argv[++i]);
        else if   (!strcmp(a, "--csv") && i + 1 < argc)    c.csv = argv[++i];
        else if   (!strcmp(a, "--tag") && i + 1 < argc)    c.tag = argv[++i];
        else if   (!strcmp(a, "--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown arg: %s\n", a); usage(argv[0]); return 2; }
    }

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
        fprintf(stderr, "no CUDA devices (run nvidia-smi to check driver)\n"); return 1;
    }
    if (!set_sizes) {
        long d[] = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
        c.n_sizes = 9; memcpy(c.sizes, d, sizeof d);
    }
    if (!set_precs) {
        c.precs[0] = PREC_FP32; c.precs[1] = PREC_FP64; c.precs[2] = PREC_BF16; c.n_precs = 3;
    }
    if (!set_gpus) {
        c.n_gpus = 0;
        for (int g = 1; g <= ndev && g <= 4; g++) c.gpus[c.n_gpus++] = g;
    }
    if (c.n_engines == 0) { c.engines[0] = ENG_DENSE; c.n_engines = 1; }
    if (c.n_dens == 0) {
        double dd[] = {0.01, 0.05, 0.1, 0.25}; c.n_dens = 4;
        memcpy(c.densities, dd, sizeof dd);
    }

    CUDA_CHECK(cudaSetDevice(0));
    size_t mfree = 0, mtotal = 0; cudaMemGetInfo(&mfree, &mtotal);
    long cap = (long)((double)mtotal * c.mem_frac);

    int nvml_ready = 0;
    if (c.nvml_on && nvmlInit_v2() == NVML_SUCCESS) nvml_ready = 1;
    else if (c.nvml_on) fprintf(stderr, "warning: nvmlInit failed; telemetry off\n");

    char host[128] = "Unknown";
    struct cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        int i = 0;
        for (; prop.name[i] && i < 127; i++) {
            host[i] = (prop.name[i] == ' ') ? '_' : prop.name[i];
        }
        host[i] = '\0';
    } else {
        gethostname(host, sizeof(host));
    }
    int cudart = 0; cudaRuntimeGetVersion(&cudart);

    FILE *csv = NULL; int new_file = 1;
    if (c.csv) {
        FILE *t = fopen(c.csv, "r");
        if (t) { fseek(t, 0, SEEK_END); if (ftell(t) > 0) new_file = 0; fclose(t); }
        csv = fopen(c.csv, "a"); if (!csv) perror("csv");
    }
    const char *HDR =
        "tag,ts,host,cudart,mode,precision,gpus,S,iters,warmup,"
        "agg_gbps,per_gpu_gbps,pct_peak_bw,wall_ms,"
        "sm_mhz_avg,sm_mhz_max,mem_mhz_avg,power_w_avg,power_w_max,temp_c_max,throttle,util_pct_avg,"
        "mem_gb_per_gpu,validated,max_rel_err,"
        "h2d_ms,d2h_ms,compute_ms,e2e_ms,transfer_pct,agg_tflops,"
        "engine,density,nnz,status\n";
    if (csv && new_file) fputs(HDR, csv);

    printf("# GPU GEMV/SpMV bench  host=%s devices=%d mode=%s cudart=%d cap=%.1fGB\n",
           host, ndev, c.mode == MODE_SPLIT ? "split" : "replicas", cudart, (double)cap / 1e9);
    printf("# Memory-bandwidth-bound at every size -- peak ref ~%.0f GB/s (A100-SXM4-40GB HBM2)\n",
           gemv_peak_gbps());
    printf("%-6s %-5s %-3s %6s %9s %9s %6s %8s %8s %5s %7s %7s %s\n",
           "eng", "prec", "G", "S", "aggGBps", "perGPU", "pk%", "comp_ms", "e2e_ms", "xfer%", "TFLOPs", "SMclk", "status");

    for (int si = 0; si < c.n_sizes; si++) {
        long S = c.sizes[si];
        float *hx_pin = NULL;
        if (cudaMallocHost((void **)&hx_pin, (size_t)S * sizeof(float)) != cudaSuccess) {
            fprintf(stderr, "pinned host OOM (x) at S=%ld\n", S); continue;
        }
        gemv_fill_rand(hx_pin, (size_t)S, 2);

        for (int ei = 0; ei < c.n_engines; ei++) {
            engine_t eng = c.engines[ei];
            if (eng == ENG_DENSE) {
                float *hA_pin = NULL;
                if (cudaMallocHost((void **)&hA_pin, (size_t)S * S * sizeof(float)) != cudaSuccess) {
                    fprintf(stderr, "pinned host OOM (A) at S=%ld\n", S); continue;
                }
                gemv_fill_rand(hA_pin, (size_t)S * S, 1);
                for (int pi = 0; pi < c.n_precs; pi++)
                    for (int gi = 0; gi < c.n_gpus; gi++) {
                        int G = c.gpus[gi]; if (G > ndev) continue;
                        run_gemv_config(&c, csv, host, cudart, cap, nvml_ready,
                                        S, c.precs[pi], G, ENG_DENSE, 1.0, 0,
                                        hA_pin, hx_pin, NULL, NULL, NULL);
                    }
                cudaFreeHost(hA_pin);
            } else {
                for (int di = 0; di < c.n_dens; di++) {
                    double dens = c.densities[di];
                    int *rowOff, *colInd; float *vals; long nnz;
                    if (gemv_gen_csr(S, dens, &rowOff, &colInd, &vals, &nnz, 3) != 0) {
                        fprintf(stderr, "host OOM generating CSR at S=%ld density=%g\n", S, dens);
                        continue;
                    }
                    for (int pi = 0; pi < c.n_precs; pi++) {
                        if (c.precs[pi] == PREC_BF16) continue; /* not supported, see run_gemv_config */
                        for (int gi = 0; gi < c.n_gpus; gi++) {
                            int G = c.gpus[gi]; if (G > ndev) continue;
                            run_gemv_config(&c, csv, host, cudart, cap, nvml_ready,
                                            S, c.precs[pi], G, ENG_SPARSE, dens, nnz,
                                            NULL, hx_pin, rowOff, colInd, vals);
                        }
                    }
                    free(rowOff); free(colInd); free(vals);
                }
            }
        }
        cudaFreeHost(hx_pin);
    }

    if (csv) fclose(csv);
    if (nvml_ready) nvmlShutdown();
    return 0;
}
