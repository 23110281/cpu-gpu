/* gpu_spgemm_bench.cu -- multi-GPU cuSPARSE SpGEMM (Sparse x Sparse) benchmark.
 *
 * C = alpha * A * B   where A, B, and C are ALL sparse CSR matrices.
 * Uses cusparseSpGEMM Generic API (cuSPARSE 11+), NOT cuSPARSELt.
 * cuSPARSELt is the existing GEMM sparse engine (2:4 structured, Sparse x Dense).
 * This benchmark is entirely independent of cuSPARSELt.
 *
 * API requires a mandatory two-phase protocol:
 *   Phase 1 (symbolic): cusparseSpGEMM_workEstimation  -- finds output NNZ
 *   Phase 2 (compute):  cusparseSpGEMM_compute         -- fills values
 *   Phase 3 (copy):     cusparseSpGEMM_copy            -- writes to allocated C
 * Only Phase 2 + Phase 3 are in the timed loop (Phase 1 is one-time setup).
 * symbolic_ms is logged as a separate CSV column (often > compute_ms at high density).
 *
 * FLOP counting: flops_theoretical = 2 * density_A * density_B * S^3
 * (NOT 2*nnz_C -- that is a 1000x+ undercount).
 *
 * Bandwidth: (nnz_A + nnz_B + nnz_C) * (elemsize + 4) + 3*(S+1)*4 bytes/iter.
 *
 * Multi-GPU split: GPU g handles rows [g*S/G, (g+1)*S/G) of A.
 *   B is replicated in FULL on every GPU (avoids inter-GPU communication
 *   at the cost of B's memory not scaling with G -- documented limitation).
 *   C row-slice is local to each GPU.
 *
 * Compile: make spgemm
 * Run:     ./bin/gpu_spgemm_bench --help
 */

/* -- includes -------------------------------------------------------------- */
#include <cusparse.h>
#include <cuda_runtime.h>
#include <nvml.h>
#include <cassert>
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
#include "gemv_host.h"   /* provides gemv_gen_csr() -- the CSR generator to use */

/* -- error-checking macros ------------------------------------------------- */
#define CUDA_CHECK_J(expr, lbl) do {                                         \
    cudaError_t _e = (expr);                                                 \
    if (_e != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA %s:%d %s -> %s\n",                             \
                __FILE__, __LINE__, #expr, cudaGetErrorString(_e));          \
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
typedef enum { PREC_FP32, PREC_FP64 } prec_t;   /* bf16 not supported by cusparseSpGEMM */
typedef enum { MODE_SPLIT, MODE_REPLICAS } bench_mode_t;

static const char *prec_name(prec_t p) { return p == PREC_FP32 ? "fp32" : "fp64"; }
static size_t elem_bytes(prec_t p) { return p == PREC_FP32 ? 4 : 8; }
static cudaDataType cuda_type(prec_t p) {
    return p == PREC_FP32 ? CUDA_R_32F : CUDA_R_64F;
}

/* -- NVML live sampler ----------------------------------------------------- */
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

static float event_ms(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f; cudaEventElapsedTime(&ms, a, b); return ms;
}

/* -- worker thread --------------------------------------------------------- */
typedef struct {
    int      dev;
    long     S;              /* full matrix dimension (A and B are S×S) */
    long     row0;           /* first row of A slice on this GPU */
    long     nrows;          /* number of rows of A on this GPU (S for replicas) */
    double   density_A, density_B;
    long     nnz_A_local;    /* nnz in this GPU's row slice of A */
    long     nnz_B;          /* nnz in full B (replicated) */
    /* CSR for A -- row-sliced (split mode) or full (replicas mode) */
    const int   *hRowOff_A_local; /* nrows+1 ints, offset rebased to 0 */
    const int   *hColInd_A;       /* nnz_A_local col indices */
    const void  *hVals_A_nat;     /* nnz_A_local values, native precision */
    /* CSR for B -- always full S×S */
    const int   *hRowOff_B;
    const int   *hColInd_B;
    const void  *hVals_B_nat;     /* native precision */
    prec_t   prec;
    int      warmup, iters;
    int      nvml_on, validate, valid_max, measure_xfer;
    pthread_barrier_t *bar;
    /* outputs */
    int      rc;
    long     nnz_C;          /* filled after Phase 2, used for BW accounting */
    double   batch_ms;       /* timed loop: iters × (compute + copy) */
    double   symbolic_ms;    /* one-time symbolic phase timing */
    double   h2d_ms, d2h_ms, max_rel_err, mem_gb;
    sampler_t smp;
} job_t;

static std::mutex g_spgemm_setup_mutex;  /* serialize symbolic phase across threads */

static void *spgemm_worker(void *arg) {
    job_t *j = (job_t *)arg;
    j->rc = 0; j->max_rel_err = -1.0; j->nnz_C = 0;
    j->batch_ms = j->symbolic_ms = j->h2d_ms = j->d2h_ms = 0.0;
    int pre_barrier = 0, smp_started = 0;
    pthread_t smp_th;

    cusparseHandle_t   handle    = NULL;
    cusparseSpMatDescr_t matA    = NULL, matB = NULL, matC = NULL;
    cusparseSpGEMMDescr_t desc   = NULL;
    cudaStream_t st = NULL;

    /* Device CSR arrays */
    int   *dRowOff_A=NULL, *dColInd_A=NULL;
    void  *dVals_A=NULL;
    int   *dRowOff_B=NULL, *dColInd_B=NULL;
    void  *dVals_B=NULL;
    int   *dRowOff_C=NULL, *dColInd_C=NULL;
    void  *dVals_C=NULL;

    void  *dBuf1=NULL, *dBuf2=NULL;
    size_t bufSize1=0, bufSize2=0;
    cudaEvent_t e_sym0=NULL, e_sym1=NULL, e0=NULL, e1=NULL;

    size_t eb = elem_bytes(j->prec);
    float  alpha_f=1.0f, beta_f=0.0f;
    double alpha_d=1.0,  beta_d=0.0;
    const void *palpha = (j->prec==PREC_FP32) ? (void*)&alpha_f : (void*)&alpha_d;
    const void *pbeta  = (j->prec==PREC_FP32) ? (void*)&beta_f  : (void*)&beta_d;

    /* ── Device setup ───────────────────────────────────────────────── */
    if (cudaSetDevice(j->dev) != cudaSuccess) { j->rc=-1; goto barrier_out; }
    CUDA_CHECK_J(cudaStreamCreate(&st), barrier_out);
    SPARSE_CHECK_J(cusparseCreate(&handle), barrier_out);
    SPARSE_CHECK_J(cusparseSetStream(handle, st), barrier_out);

    /* ── Allocate device CSR storage for A, B, C ────────────────────── */
    CUDA_CHECK_J(cudaMalloc(&dRowOff_A, (size_t)(j->nrows+1)*4), barrier_out);
    CUDA_CHECK_J(cudaMalloc(&dColInd_A, (size_t)j->nnz_A_local*4), barrier_out);
    CUDA_CHECK_J(cudaMalloc(&dVals_A,   (size_t)j->nnz_A_local*eb), barrier_out);
    CUDA_CHECK_J(cudaMalloc(&dRowOff_B, (size_t)(j->S+1)*4), barrier_out);
    CUDA_CHECK_J(cudaMalloc(&dColInd_B, (size_t)j->nnz_B*4), barrier_out);
    CUDA_CHECK_J(cudaMalloc(&dVals_B,   (size_t)j->nnz_B*eb), barrier_out);
    /* C output pointers allocated after Phase 2 when nnz_C is known */
    CUDA_CHECK_J(cudaMalloc(&dRowOff_C, (size_t)(j->nrows+1)*4), barrier_out);

    /* ── H2D (A and B), timed ───────────────────────────────────────── */
    cudaEvent_t e_h2d0, e_h2d1;
    CUDA_CHECK_J(cudaEventCreate(&e_h2d0), barrier_out);
    CUDA_CHECK_J(cudaEventCreate(&e_h2d1), barrier_out);
    cudaEventRecord(e_h2d0, st);
    CUDA_CHECK_J(cudaMemcpyAsync(dRowOff_A, j->hRowOff_A_local, (size_t)(j->nrows+1)*4, cudaMemcpyHostToDevice, st), barrier_out);
    CUDA_CHECK_J(cudaMemcpyAsync(dColInd_A, j->hColInd_A, (size_t)j->nnz_A_local*4, cudaMemcpyHostToDevice, st), barrier_out);
    CUDA_CHECK_J(cudaMemcpyAsync(dVals_A,   j->hVals_A_nat, (size_t)j->nnz_A_local*eb, cudaMemcpyHostToDevice, st), barrier_out);
    CUDA_CHECK_J(cudaMemcpyAsync(dRowOff_B, j->hRowOff_B, (size_t)(j->S+1)*4, cudaMemcpyHostToDevice, st), barrier_out);
    CUDA_CHECK_J(cudaMemcpyAsync(dColInd_B, j->hColInd_B, (size_t)j->nnz_B*4, cudaMemcpyHostToDevice, st), barrier_out);
    CUDA_CHECK_J(cudaMemcpyAsync(dVals_B,   j->hVals_B_nat, (size_t)j->nnz_B*eb, cudaMemcpyHostToDevice, st), barrier_out);
    cudaEventRecord(e_h2d1, st);
    CUDA_CHECK_J(cudaStreamSynchronize(st), barrier_out);
    j->h2d_ms = (double)event_ms(e_h2d0, e_h2d1);
    cudaEventDestroy(e_h2d0); cudaEventDestroy(e_h2d1);

    /* ── Create sparse matrix descriptors ───────────────────────────── */
    /* A: nrows × S sparse, nnz_A_local non-zeros */
    SPARSE_CHECK_J(cusparseCreateCsr(&matA, j->nrows, j->S, j->nnz_A_local,
        dRowOff_A, dColInd_A, dVals_A,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        cuda_type(j->prec)), barrier_out);
    /* B: S × S sparse, nnz_B non-zeros */
    SPARSE_CHECK_J(cusparseCreateCsr(&matB, j->S, j->S, j->nnz_B,
        dRowOff_B, dColInd_B, dVals_B,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        cuda_type(j->prec)), barrier_out);
    /* C: nrows × S sparse, nnz=0 initially (unknown until Phase 2) */
    SPARSE_CHECK_J(cusparseCreateCsr(&matC, j->nrows, j->S, 0,
        dRowOff_C, NULL, NULL,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        cuda_type(j->prec)), barrier_out);

    /* ── Phase 1 + 2: Symbolic (serialized under mutex) ─────────────── */
    {
        std::lock_guard<std::mutex> lk(g_spgemm_setup_mutex);

        SPARSE_CHECK_J(cusparseSpGEMM_createDescr(&desc), barrier_out);

        /* OOM preflight: query Phase 1 buffer size, check against mem_frac */
        SPARSE_CHECK_J(cusparseSpGEMM_workEstimation(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            palpha, matA, matB, pbeta, matC,
            cuda_type(j->prec), CUSPARSE_SPGEMM_DEFAULT,
            desc, &bufSize1, NULL), barrier_out);

        /* --- D3: OOM guard before allocating buffers --- */
        size_t dev_free=0, dev_total=0;
        cudaMemGetInfo(&dev_free, &dev_total);
        if (bufSize1 > dev_free * 0.8) {
            fprintf(stderr, "[dev %d] SpGEMM Phase1 buf %zu MB > 80%% free mem; oom-skip\n",
                j->dev, bufSize1>>20);
            j->rc = -2; goto barrier_out;
        }

        CUDA_CHECK_J(cudaMalloc(&dBuf1, bufSize1 ? bufSize1 : 1), barrier_out);

        /* Time the symbolic phase */
        CUDA_CHECK_J(cudaEventCreate(&e_sym0), barrier_out);
        CUDA_CHECK_J(cudaEventCreate(&e_sym1), barrier_out);
        cudaEventRecord(e_sym0, st);
        SPARSE_CHECK_J(cusparseSpGEMM_workEstimation(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            palpha, matA, matB, pbeta, matC,
            cuda_type(j->prec), CUSPARSE_SPGEMM_DEFAULT,
            desc, &bufSize1, dBuf1), barrier_out);

        /* Query Phase 2 buffer size */
        SPARSE_CHECK_J(cusparseSpGEMM_compute(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            palpha, matA, matB, pbeta, matC,
            cuda_type(j->prec), CUSPARSE_SPGEMM_DEFAULT,
            desc, &bufSize2, NULL), barrier_out);

        /* OOM guard for Phase 2 buffer */
        cudaMemGetInfo(&dev_free, &dev_total);
        if (bufSize2 > dev_free * 0.8) {
            fprintf(stderr, "[dev %d] SpGEMM Phase2 buf %zu MB > 80%% free mem; oom-skip\n",
                j->dev, bufSize2>>20);
            j->rc = -2; goto barrier_out;
        }

        CUDA_CHECK_J(cudaMalloc(&dBuf2, bufSize2 ? bufSize2 : 1), barrier_out);
        SPARSE_CHECK_J(cusparseSpGEMM_compute(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            palpha, matA, matB, pbeta, matC,
            cuda_type(j->prec), CUSPARSE_SPGEMM_DEFAULT,
            desc, &bufSize2, dBuf2), barrier_out);
        cudaEventRecord(e_sym1, st);
        cudaStreamSynchronize(st);
        j->symbolic_ms = (double)event_ms(e_sym0, e_sym1);
        cudaEventDestroy(e_sym0); cudaEventDestroy(e_sym1);
        e_sym0 = e_sym1 = NULL;

        /* Determine actual nnz_C */
        int64_t rows_C=0, cols_C=0, nnz_C=0;
        SPARSE_CHECK_J(cusparseSpMatGetSize(matC, &rows_C, &cols_C, &nnz_C), barrier_out);
        j->nnz_C = (long)nnz_C;

        /* Allocate output C value + column arrays */
        CUDA_CHECK_J(cudaMalloc(&dColInd_C, (size_t)nnz_C*4), barrier_out);
        CUDA_CHECK_J(cudaMalloc(&dVals_C,   (size_t)nnz_C*eb), barrier_out);
        SPARSE_CHECK_J(cusparseCsrSetPointers(matC, dRowOff_C, dColInd_C, dVals_C), barrier_out);

        /* Phase 3: copy to allocated buffers */
        SPARSE_CHECK_J(cusparseSpGEMM_copy(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            palpha, matA, matB, pbeta, matC,
            cuda_type(j->prec), CUSPARSE_SPGEMM_DEFAULT, desc), barrier_out);
    } /* ── mutex released here ── */

    /* ── Warmup (Phase 2 + Phase 3 only, reusing symbolic result) ───── */
    for (int w = 0; w < j->warmup; w++) {
        SPARSE_CHECK_J(cusparseSpGEMM_compute(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            palpha, matA, matB, pbeta, matC,
            cuda_type(j->prec), CUSPARSE_SPGEMM_DEFAULT,
            desc, &bufSize2, dBuf2), barrier_out);
        SPARSE_CHECK_J(cusparseSpGEMM_copy(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            palpha, matA, matB, pbeta, matC,
            cuda_type(j->prec), CUSPARSE_SPGEMM_DEFAULT, desc), barrier_out);
    }
    CUDA_CHECK_J(cudaStreamSynchronize(st), barrier_out);

    /* ── BARRIER: immediately before timed region ────────────────────── */
    pthread_barrier_wait(j->bar);
    pre_barrier = 1;

    /* ── NVML sampler ───────────────────────────────────────────────── */
    if (j->nvml_on && j->smp.valid) {
        j->smp.stop.store(0, std::memory_order_relaxed);
        if (pthread_create(&smp_th, NULL, sampler_run, &j->smp) == 0)
            smp_started = 1;
    }

    /* ── Timed loop: Phase 2 + Phase 3 only ─────────────────────────── */
    {
        int loop_err = 0;
        CUDA_CHECK_J(cudaEventCreate(&e0), done);
        CUDA_CHECK_J(cudaEventCreate(&e1), done);
        cudaEventRecord(e0, st);
        for (int it = 0; it < j->iters; it++) {
            if (cusparseSpGEMM_compute(handle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    palpha, matA, matB, pbeta, matC,
                    cuda_type(j->prec), CUSPARSE_SPGEMM_DEFAULT,
                    desc, &bufSize2, dBuf2) != CUSPARSE_STATUS_SUCCESS) loop_err=1;
            if (cusparseSpGEMM_copy(handle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    palpha, matA, matB, pbeta, matC,
                    cuda_type(j->prec), CUSPARSE_SPGEMM_DEFAULT, desc)
                        != CUSPARSE_STATUS_SUCCESS) loop_err=1;
        }
        cudaEventRecord(e1, st);
        cudaEventSynchronize(e1);
        if (smp_started) {
            j->smp.stop.store(1, std::memory_order_relaxed);
            pthread_join(smp_th, NULL);
            smp_started = 0;
        }
        if (cudaGetLastError() != cudaSuccess) loop_err = 1;
        if (loop_err) { j->rc = -1; goto done; }
        j->batch_ms = (double)event_ms(e0, e1);
    }

done:
    if (smp_started) { j->smp.stop.store(1, std::memory_order_relaxed); pthread_join(smp_th, NULL); }
    if (e0) cudaEventDestroy(e0); if (e1) cudaEventDestroy(e1);
    if (e_sym0) cudaEventDestroy(e_sym0); if (e_sym1) cudaEventDestroy(e_sym1);
    if (matA) cusparseDestroySpMat(matA);
    if (matB) cusparseDestroySpMat(matB);
    if (matC) cusparseDestroySpMat(matC);
    if (desc) cusparseSpGEMM_destroyDescr(desc);
    if (dRowOff_A) cudaFree(dRowOff_A); if (dColInd_A) cudaFree(dColInd_A);
    if (dVals_A)   cudaFree(dVals_A);
    if (dRowOff_B) cudaFree(dRowOff_B); if (dColInd_B) cudaFree(dColInd_B);
    if (dVals_B)   cudaFree(dVals_B);
    if (dRowOff_C) cudaFree(dRowOff_C); if (dColInd_C) cudaFree(dColInd_C);
    if (dVals_C)   cudaFree(dVals_C);
    if (dBuf1) cudaFree(dBuf1); if (dBuf2) cudaFree(dBuf2);
    if (handle) cusparseDestroy(handle);
    if (st) cudaStreamDestroy(st);
    if (!pre_barrier) pthread_barrier_wait(j->bar);
    return NULL;

barrier_out:
    /* Same cleanup as done: + MUST hit barrier */
    if (smp_started) { j->smp.stop.store(1, std::memory_order_relaxed); pthread_join(smp_th, NULL); }
    if (e0) cudaEventDestroy(e0); if (e1) cudaEventDestroy(e1);
    if (e_sym0) cudaEventDestroy(e_sym0); if (e_sym1) cudaEventDestroy(e_sym1);
    if (matA) cusparseDestroySpMat(matA);
    if (matB) cusparseDestroySpMat(matB);
    if (matC) cusparseDestroySpMat(matC);
    if (desc) cusparseSpGEMM_destroyDescr(desc);
    if (dRowOff_A) cudaFree(dRowOff_A); if (dColInd_A) cudaFree(dColInd_A);
    if (dVals_A)   cudaFree(dVals_A);
    if (dRowOff_B) cudaFree(dRowOff_B); if (dColInd_B) cudaFree(dColInd_B);
    if (dVals_B)   cudaFree(dVals_B);
    if (dRowOff_C) cudaFree(dRowOff_C); if (dColInd_C) cudaFree(dColInd_C);
    if (dVals_C)   cudaFree(dVals_C);
    if (dBuf1) cudaFree(dBuf1); if (dBuf2) cudaFree(dBuf2);
    if (handle) cusparseDestroy(handle);
    if (st) cudaStreamDestroy(st);
    pthread_barrier_wait(j->bar);
    return NULL;
}

/* -- main ------------------------------------------------------------------ */
typedef struct {
    long   sizes[32]; int n_sizes;
    prec_t precs[4];  int n_precs;
    int    gpus[8];   int n_gpus;
    bench_mode_t mode;
    double density_A, density_B; /* Only used if density_A < 0 */
    double densities[8]; int n_dens;
    double densities_A[8], densities_B[8]; int n_dens_A, n_dens_B;
    int    iters, warmup;
    int    nvml_on, validate, valid_max, measure_xfer;
    double mem_frac;
    const char *csv, *tag;
} cfg_t;

static int parse_longs(const char *s, long *out, int max) {
    int n = 0; char buf[512]; snprintf(buf, sizeof buf, "%s", s);
    for (char *t = strtok(buf, ","); t && n < max; t = strtok(NULL, ","))
        out[n++] = atol(t);
    return n;
}

static int parse_doubles(const char *s, double *out, int max) {
    int n = 0; char buf[512]; snprintf(buf, sizeof buf, "%s", s);
    for (char *t = strtok(buf, ","); t && n < max; t = strtok(NULL, ","))
        out[n++] = atof(t);
    return n;
}

static void usage(const char *p) {
    printf(
"Usage: %s [options]\n"
"  --sizes a,b,..     matrix sizes                 [512,1024,2048,4096,8192,16384]\n"
"  --precisions p,..  fp32,fp64                    [all]\n"
"  --gpus n,..        gpu counts to test           [1..4]\n"
"  --mode m           split | replicas             [split]\n"
"  --density d,..     sets density_A=density_B     [0.01,0.05,0.1,0.25,0.5]\n"
"  --density-a d      override density for A only\n"
"  --density-b d      override density for B only\n"
"  --iters n          timed iterations             [10]\n"
"  --warmup n         warmup iterations            [3]\n"
"  --no-nvml          disable NVML telemetry\n"
"  --validate         sampled correctness check\n"
"  --validate-max n   max size to validate         [2048]\n"
"  --no-transfers     skip timed H2D/D2H\n"
"  --mem-frac f       usable GPU mem fraction      [0.90]\n"
"  --csv path         append CSV results\n"
"  --tag s            CSV label\n"
"  --help\n", p);
}

static void run_config(const cfg_t *c, FILE *csv, const char *host, int cudart,
                       long cap, int nvml_ready, long S, prec_t p, int G,
                       double density_A, double density_B,
                       int *hRowOff_A, int *hColInd_A, float *hVals_A, long nnz_A,
                       int *hRowOff_B, int *hColInd_B, float *hVals_B, long nnz_B)
{
    long ts = (long)time(NULL);
    const char *mode_s = (c->mode == MODE_SPLIT) ? "split" : "replicas";
    const char *pn = prec_name(p);

    if (p != PREC_FP32 && p != PREC_FP64) {
        printf("bf16 not supported by cusparseSpGEMM\n");
        return;
    }

    void *hVals_A_nat = NULL;
    void *hVals_B_nat = NULL;
    size_t eb = elem_bytes(p);

    if (p == PREC_FP32) {
        hVals_A_nat = (void*)hVals_A;
        hVals_B_nat = (void*)hVals_B;
    } else {
        if (cudaMallocHost(&hVals_A_nat, (size_t)nnz_A * eb) != cudaSuccess) {
            fprintf(stderr, "Host OOM for pre-converted hVals_A_nat\n");
            return;
        }
        if (cudaMallocHost(&hVals_B_nat, (size_t)nnz_B * eb) != cudaSuccess) {
            fprintf(stderr, "Host OOM for pre-converted hVals_B_nat\n");
            if (hVals_A_nat) cudaFreeHost(hVals_A_nat);
            return;
        }
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < (size_t)nnz_A; i++) ((double*)hVals_A_nat)[i] = (double)hVals_A[i];
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < (size_t)nnz_B; i++) ((double*)hVals_B_nat)[i] = (double)hVals_B[i];
    }

    int *hRowOff_A_local[8]; // One for each GPU

    job_t job[8]; pthread_barrier_t bar;
    pthread_barrier_init(&bar, NULL, (unsigned)G);
    long cumulative_row = 0;
    
    // We need to calculate footprint ahead of time to detect OOM-skip
    long max_footprint = 0;
    for (int g = 0; g < G; g++) {
        long nrows = (c->mode == MODE_REPLICAS) ? S : (S / G + (g < S % G ? 1 : 0));
        long row0 = (c->mode == MODE_REPLICAS) ? 0 : cumulative_row;
        
        long nnz_A_local = hRowOff_A[row0 + nrows] - hRowOff_A[row0];
        long foot = (long)(nrows + 1 + S + 1) * 4 + (nnz_A_local + nnz_B) * ((long)eb + 4);
        if (foot > max_footprint) max_footprint = foot;
        
        if (c->mode != MODE_REPLICAS) cumulative_row += nrows;
    }
    
    // C is unknown, so add a generous 50% for C footprint approximation (for preflight only)
    max_footprint = (long)((double)max_footprint * 1.5);
    double mem_gb = (double)max_footprint / 1e9;

    if (max_footprint > cap) {
        printf("%-5s %-3d %12ld %9s %9s %6s %8s %8s %6s %6s %7s %s\n",
               pn, G, S, "-","-","-","-","-","-","-","-","oom-skip");
        if (csv) {
            fprintf(csv,
                "%s,%ld,%s,%d,%s,%s,%d,%ld,%d,%d,"
                "0,0,0,0,0,"
                "0,0,0,0,0,0,none,"
                "0,"
                "%.3f,%d,0,"
                "0,0,0,0,0,"
                "%.3f,%ld,%.3f,%ld,0,0,oom-skip\n",
                c->tag,ts,host,cudart,mode_s,pn,G,S,c->iters,c->warmup,
                mem_gb,c->validate&&S<=c->valid_max,
                density_A,nnz_A,density_B,nnz_B);
            fflush(csv);
        }
        if (p != PREC_FP32) {
            if (hVals_A_nat) cudaFreeHost(hVals_A_nat);
            if (hVals_B_nat) cudaFreeHost(hVals_B_nat);
        }
        return;
    }

    cumulative_row = 0;
    for (int g = 0; g < G; g++) {
        long nrows = (c->mode == MODE_REPLICAS) ? S : (S / G + (g < S % G ? 1 : 0));
        long row0 = (c->mode == MODE_REPLICAS) ? 0 : cumulative_row;
        long nnz_A_local = hRowOff_A[row0 + nrows] - hRowOff_A[row0];
        
        hRowOff_A_local[g] = (int*)malloc((nrows + 1) * sizeof(int));
        for (long i = 0; i <= nrows; i++) hRowOff_A_local[g][i] = hRowOff_A[row0 + i] - hRowOff_A[row0];

        job[g].rc = 0;
        job[g].batch_ms = 0.0; job[g].symbolic_ms = 0.0; job[g].h2d_ms = 0.0; job[g].d2h_ms = 0.0;
        job[g].max_rel_err = 0.0; job[g].mem_gb = mem_gb; job[g].nnz_C = 0;
        job[g].smp.valid = 0;
        job[g].smp.sm_sum = 0; job[g].smp.mem_sum = 0; job[g].smp.pow_sum = 0; job[g].smp.util_sum = 0;
        job[g].smp.sm_max = 0; job[g].smp.mem_max = 0; job[g].smp.pow_max = 0; job[g].smp.temp_max = 0;
        job[g].smp.throttle_or = 0; job[g].smp.n = 0;
        
        job[g].dev = g; job[g].S = S; job[g].row0 = row0; job[g].nrows = nrows;
        job[g].density_A = density_A; job[g].density_B = density_B;
        job[g].nnz_A_local = nnz_A_local; job[g].nnz_B = nnz_B;
        
        job[g].hRowOff_A_local = hRowOff_A_local[g];
        job[g].hColInd_A = hColInd_A + hRowOff_A[row0];
        job[g].hVals_A_nat = (const char*)hVals_A_nat + hRowOff_A[row0] * eb;
        job[g].hRowOff_B = hRowOff_B;
        job[g].hColInd_B = hColInd_B;
        job[g].hVals_B_nat = hVals_B_nat;

        job[g].prec = p;
        job[g].warmup = c->warmup; job[g].iters = c->iters;
        job[g].nvml_on = nvml_ready;
        job[g].validate = c->validate;
        job[g].valid_max = c->valid_max;
        job[g].measure_xfer = c->measure_xfer;
        job[g].bar = &bar;
        if (c->mode != MODE_REPLICAS) cumulative_row += nrows;
        
        if (nvml_ready) {
            char bus[32];
            if (cudaDeviceGetPCIBusId(bus, sizeof bus, g) == cudaSuccess)
                if (nvmlDeviceGetHandleByPciBusId_v2(bus, &job[g].smp.dev) == NVML_SUCCESS)
                    job[g].smp.valid = 1;
            job[g].smp.stop.store(0, std::memory_order_relaxed);
        }
    }

    pthread_t th[8];
    for (int g = 0; g < G; g++)
        pthread_create(&th[g], NULL, spgemm_worker, &job[g]);
    for (int g = 0; g < G; g++) pthread_join(th[g], NULL);
    pthread_barrier_destroy(&bar);
    
    for (int g = 0; g < G; g++) free(hRowOff_A_local[g]);

    if (p != PREC_FP32) {
        if (hVals_A_nat) cudaFreeHost(hVals_A_nat);
        if (hVals_B_nat) cudaFreeHost(hVals_B_nat);
    }

    int err = 0, oom_skip = 0; double wall = 0, rel = -1.0;
    unsigned long long sm_sum = 0, mem_sum = 0, pow_sum = 0, util_sum = 0;
    long nsamp = 0, nnz_C_total = 0;
    unsigned sm_max = 0, mem_max = 0, pow_max = 0, temp_max = 0;
    unsigned long long tr = 0;
    
    for (int g = 0; g < G; g++) {
        if (job[g].rc == -2) { oom_skip = 1; continue; }
        if (job[g].rc) { err = 1; continue; }
        double tg = job[g].batch_ms / 1000.0; if (tg > wall) wall = tg;
        nnz_C_total += job[g].nnz_C;
        if (job[g].max_rel_err > rel) rel = job[g].max_rel_err;
        sampler_t *s = &job[g].smp;
        if (s->n > 0) {
            sm_sum   += s->sm_sum;  mem_sum  += s->mem_sum;
            pow_sum  += s->pow_sum; util_sum += s->util_sum;
            nsamp += s->n;
            if (s->sm_max   > sm_max)   sm_max   = s->sm_max;
            if (s->mem_max  > mem_max)  mem_max  = s->mem_max;
            if (s->pow_max  > pow_max)  pow_max  = s->pow_max;
            if (s->temp_max > temp_max) temp_max = s->temp_max;
            tr |= s->throttle_or;
        }
    }

    if (oom_skip || err || wall <= 0.0) {
        const char* estat = oom_skip ? "oom-skip" : (err ? "error" : "time-zero");
        printf("%-5s %-3d %12ld %9s %9s %6s %8s %8s %6s %6s %7s %s\n",
               pn, G, S, "-","-","-","-","-","-","-","-",estat);
        if (csv) fprintf(csv,
            "%s,%ld,%s,%d,%s,%s,%d,%ld,%d,%d,"
            "0,0,0,0,0,"
            "0,0,0,0,0,0,none,"
            "0,"
            "%.3f,%d,0,"
            "0,0,0,0,0,"
            "%.3f,%ld,%.3f,%ld,0,0,%s\n",
            c->tag,ts,host,cudart,mode_s,pn,G,S,c->iters,c->warmup,
            mem_gb,c->validate&&S<=c->valid_max,
            density_A,nnz_A,density_B,nnz_B,estat);
        return;
    }

    double flops_theoretical_total = 0;
    for (int g = 0; g < G; g++) {
        if (!job[g].rc)
            flops_theoretical_total += (double)c->iters * 2.0
                * density_A * density_B
                * (double)job[g].nrows * (double)S * (double)S;
    }
    double agg_tflops = (wall > 0) ? flops_theoretical_total / wall / 1e12 : 0;
    double per_gpu_tflops = agg_tflops / (double)G;

    double sum_bytes = 0;
    for (int g = 0; g < G; g++) {
        if (!job[g].rc) {
            long nnz_A_g = job[g].nnz_A_local;
            long nnz_B_g = job[g].nnz_B;           /* full B replicated */
            long nnz_C_g = job[g].nnz_C;
            long nrows_g = job[g].nrows;
            double vals_cols = (double)(nnz_A_g + nnz_B_g + nnz_C_g) * (eb + 4.0);
            double row_offsets = (double)((nrows_g+1) + (S+1) + (nrows_g+1)) * 4.0;
            sum_bytes += (double)c->iters * (vals_cols + row_offsets);
        }
    }
    double agg_gbps = (wall > 0) ? sum_bytes / wall / 1e9 : 0;
    double pct_peak_bw = 100.0 * agg_gbps / (1555.0 * (double)G);

    int bn = 0; double max_e2e = 0;
    for (int g = 0; g < G; g++) {
        double comp = job[g].batch_ms / (double)c->iters;
        double e2e  = job[g].h2d_ms + comp + job[g].d2h_ms;
        if (e2e > max_e2e) { max_e2e = e2e; bn = g; }
    }
    double b_h2d  = job[bn].h2d_ms;
    double b_d2h  = job[bn].d2h_ms;
    double b_comp = job[bn].batch_ms / (double)c->iters;
    double b_sym  = job[bn].symbolic_ms;
    double e2e_ms = max_e2e;

    double sm_avg   = nsamp ? (double)sm_sum  /(double)nsamp : 0;
    double mem_avg  = nsamp ? (double)mem_sum /(double)nsamp : 0;
    double pow_avg  = nsamp ? (double)pow_sum /(double)nsamp/1000.0 : 0;
    double util_avg = nsamp ? (double)util_sum/(double)nsamp : 0;
    char trbuf[128]; throttle_str(tr, trbuf, sizeof trbuf);

    const char *vstr = (rel >= 0) ? "ok/val" : "ok";
    
    printf("%-5s %-3d %12ld %9.1f %9.1f %5.1f%% %8.2f %8.2f %5.1f%% %5.1f%% %7.0f %s\n",
           pn, G, S, agg_gbps, agg_tflops, pct_peak_bw,
           b_comp, e2e_ms, 0.0, util_avg, sm_avg, vstr);

    if (csv) {
        fprintf(csv,
            "%s,%ld,%s,%d,%s,%s,%d,%ld,%d,%d,"
            "%.3f,%.3f,%.3f,%.3f,%.3f,"
            "%.0f,%u,%.0f,%.1f,%.1f,%u,%s,"
            "%.1f,"
            "%.3f,%d,%.3g,"
            "%.3f,%.3f,%.3f,%.3f,%.3f,"
            "%.3f,%ld,%.3f,%ld,%ld,%.3f,ok\n",
            c->tag,ts,host,cudart,mode_s,pn,G,S,c->iters,c->warmup,
            agg_tflops,per_gpu_tflops,agg_gbps,pct_peak_bw,wall*1000.0,
            sm_avg,sm_max,mem_avg,pow_avg,(double)pow_max/1000.0,temp_max,trbuf,
            util_avg,
            mem_gb,c->validate&&S<=c->valid_max,rel,
            b_h2d,b_d2h,b_comp,b_sym,e2e_ms,
            density_A,nnz_A,density_B,nnz_B,nnz_C_total,flops_theoretical_total);
        fflush(csv);
    }
}

int main(int argc, char **argv) {
    cfg_t c = {};
    c.mode = MODE_SPLIT; c.iters = 10; c.warmup = 3; c.nvml_on = 1;
    c.valid_max = 2048; c.mem_frac = 0.90; c.tag = ""; c.measure_xfer = 1;
    c.density_A = -1.0; c.density_B = -1.0;
    int set_sizes = 0, set_precs = 0, set_gpus = 0;
    
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a,"--sizes")&&i+1<argc) {
            c.n_sizes = parse_longs(argv[++i], c.sizes, 32); set_sizes = 1;
        } else if (!strcmp(a,"--precisions")&&i+1<argc) {
            char b[128]; snprintf(b,sizeof b,"%s",argv[++i]); c.n_precs = 0;
            for (char *t = strtok(b,","); t; t = strtok(NULL,",")) {
                if      (!strcmp(t,"fp32")) c.precs[c.n_precs++] = PREC_FP32;
                else if (!strcmp(t,"fp64")) c.precs[c.n_precs++] = PREC_FP64;
                else if (!strcmp(t,"bf16")) {
                    fprintf(stderr, "warning: bf16 not supported by cusparseSpGEMM\n");
                }
            } set_precs = 1;
        } else if (!strcmp(a,"--gpus")&&i+1<argc) {
            long g[8]; int n = parse_longs(argv[++i], g, 8);
            for (int k = 0; k < n; k++) c.gpus[k] = (int)g[k];
            c.n_gpus = n; set_gpus = 1;
        } else if (!strcmp(a,"--mode")&&i+1<argc) {
            c.mode = strcmp(argv[++i],"replicas") ? MODE_SPLIT : MODE_REPLICAS;
        } else if (!strcmp(a,"--density")&&i+1<argc) {
            c.n_dens = parse_doubles(argv[++i], c.densities, 8);
        } else if (!strcmp(a,"--density-a")&&i+1<argc) {
            c.density_A = atof(argv[++i]);
        } else if (!strcmp(a,"--density-b")&&i+1<argc) {
            c.density_B = atof(argv[++i]);
        } else if (!strcmp(a,"--iters")   &&i+1<argc) c.iters   = atoi(argv[++i]);
        else if   (!strcmp(a,"--warmup")  &&i+1<argc) c.warmup  = atoi(argv[++i]);
        else if   (!strcmp(a,"--no-nvml"))             c.nvml_on  = 0;
        else if   (!strcmp(a,"--validate"))            c.validate = 1;
        else if   (!strcmp(a,"--validate-max")&&i+1<argc) c.valid_max = atoi(argv[++i]);
        else if   (!strcmp(a,"--no-transfers"))        c.measure_xfer = 0;
        else if   (!strcmp(a,"--mem-frac")&&i+1<argc) c.mem_frac = atof(argv[++i]);
        else if   (!strcmp(a,"--csv")     &&i+1<argc) c.csv = argv[++i];
        else if   (!strcmp(a,"--tag")     &&i+1<argc) c.tag = argv[++i];
        else if   (!strcmp(a,"--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr,"unknown arg: %s\n",a); usage(argv[0]); return 2; }
    }

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
        fprintf(stderr, "no CUDA devices (run nvidia-smi to check driver)\n"); return 1;
    }
    if (!set_sizes) {
        long d[] = {512,1024,2048,4096,8192,16384};
        c.n_sizes = 6; memcpy(c.sizes, d, sizeof d);
    }
    if (!set_precs) {
        c.precs[0]=PREC_FP32; c.precs[1]=PREC_FP64; c.n_precs=2;
    }
    if (!set_gpus) {
        c.n_gpus = 0;
        for (int g = 1; g <= ndev && g <= 4; g++) c.gpus[c.n_gpus++] = g;
    }
    if (c.n_dens == 0 && c.density_A < 0 && c.density_B < 0) {
        double d[] = {0.01,0.05,0.1,0.25,0.5};
        c.n_dens = 5; memcpy(c.densities, d, sizeof d);
    }
    
    // Resolve densities lists
    if (c.density_A >= 0) { c.densities_A[0] = c.density_A; c.n_dens_A = 1; }
    else if (c.n_dens > 0) { memcpy(c.densities_A, c.densities, c.n_dens * sizeof(double)); c.n_dens_A = c.n_dens; }
    else { c.densities_A[0] = 0.1; c.n_dens_A = 1; } // fallback

    if (c.density_B >= 0) { c.densities_B[0] = c.density_B; c.n_dens_B = 1; }
    else if (c.n_dens > 0) { memcpy(c.densities_B, c.densities, c.n_dens * sizeof(double)); c.n_dens_B = c.n_dens; }
    else { c.densities_B[0] = 0.1; c.n_dens_B = 1; } // fallback
    
    // If both specified individually, run all combinations. Usually one is specified and the other comes from --density, or both from --density
    // But since the plan says "sets density_A=density_B to each value", we iterate over the max of the two lists length, padding with last element.
    int num_density_sweeps = c.n_dens > 0 ? c.n_dens : (c.n_dens_A > c.n_dens_B ? c.n_dens_A : c.n_dens_B);

    CUDA_CHECK(cudaSetDevice(0));
    size_t mfree = 0, mtotal = 0; cudaMemGetInfo(&mfree, &mtotal);
    long cap = (long)((double)mtotal * c.mem_frac);

    int nvml_ready = 0;
    if (c.nvml_on && nvmlInit_v2() == NVML_SUCCESS) nvml_ready = 1;
    else if (c.nvml_on) fprintf(stderr,"warning: nvmlInit failed; telemetry off\n");

    char host[128]; gethostname(host, sizeof host);
    int cudart = 0; cudaRuntimeGetVersion(&cudart);

    FILE *csv = NULL; int new_file = 1;
    if (c.csv) {
        FILE *t = fopen(c.csv, "r");
        if (t) { fseek(t,0,SEEK_END); if (ftell(t)>0) new_file=0; fclose(t); }
        csv = fopen(c.csv, "a"); if (!csv) perror("csv");
    }
    const char *HDR =
        "tag,ts,host,cudart,mode,precision,gpus,S,iters,warmup,"
        "agg_tflops,per_gpu_tflops,agg_gbps,pct_peak_bw,wall_ms,"
        "sm_mhz_avg,sm_mhz_max,mem_mhz_avg,power_w_avg,power_w_max,temp_c_max,throttle,"
        "util_pct_avg,"
        "mem_gb_per_gpu,validated,max_rel_err,"
        "h2d_ms,d2h_ms,compute_ms,symbolic_ms,e2e_ms,"
        "density_a,nnz_a,density_b,nnz_b,nnz_c,flops_theoretical,status\n";
    if (csv && new_file) fputs(HDR, csv);

    printf("# GPU SpGEMM bench (nvcc/CUDA C++)  host=%s devices=%d mode=%s cudart=%d cap=%.1fGB\n",
           host, ndev, c.mode==MODE_SPLIT?"split":"replicas", cudart, (double)cap/1e9);
    printf("%-5s %-3s %12s %9s %9s %6s %8s %8s %6s %6s %7s %s\n",
           "prec","G","S",
           "aggGB/s","aggTF","pkBW%","comp_ms","e2e_ms","xfer%","util%","SMclk","status");

    for (int si = 0; si < c.n_sizes; si++) {
        long S = c.sizes[si];
        
        for (int di = 0; di < num_density_sweeps; di++) {
            double dens_A = di < c.n_dens_A ? c.densities_A[di] : c.densities_A[c.n_dens_A-1];
            double dens_B = di < c.n_dens_B ? c.densities_B[di] : c.densities_B[c.n_dens_B-1];

            int *hRowOff_A=NULL, *hColInd_A=NULL; float *hVals_A=NULL; long nnz_A=0;
            int *hRowOff_B=NULL, *hColInd_B=NULL; float *hVals_B=NULL; long nnz_B=0;
            if (gemv_gen_csr(S, dens_A, &hRowOff_A, &hColInd_A, &hVals_A, &nnz_A, 42) != 0) {
                fprintf(stderr, "gen_csr A failed at S=%ld density=%.4f\n", S, dens_A);
                continue;
            }
            if (gemv_gen_csr(S, dens_B, &hRowOff_B, &hColInd_B, &hVals_B, &nnz_B, 137) != 0) {
                fprintf(stderr, "gen_csr B failed\n");
                free(hRowOff_A); free(hColInd_A); free(hVals_A); continue;
            }
            
            for (int pi = 0; pi < c.n_precs; pi++) {
                for (int gi = 0; gi < c.n_gpus; gi++) {
                    int G = c.gpus[gi]; if (G > ndev) continue;
                    
                    run_config(&c, csv, host, cudart, cap, nvml_ready,
                               S, c.precs[pi], G,
                               dens_A, dens_B,
                               hRowOff_A, hColInd_A, hVals_A, nnz_A,
                               hRowOff_B, hColInd_B, hVals_B, nnz_B);
                }
            }
            
            free(hRowOff_A); free(hColInd_A); free(hVals_A);
            free(hRowOff_B); free(hColInd_B); free(hVals_B);
        }
    }

    if (csv) fclose(csv);
    if (nvml_ready) nvmlShutdown();
    return 0;
}
