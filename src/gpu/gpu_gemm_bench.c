/* gpu_gemm_bench.c — multi-GPU cuBLAS GEMM benchmark for the dgemm project.
 *
 * Pure C11 (no nvcc): links cudart + cublas + nvidia-ml.  Mirrors the CPU
 * benchmarking story (tests/unit/test_compute.c + bench/) for A100 GPUs.
 *
 * Sweeps:   size {512..65536} x precision {fp32,fp64,bf16} x gpus {1..N}.
 * Precision routing (the whole point):
 *   FP32 -> CUDA cores      : cublasGemmEx + CUBLAS_COMPUTE_32F_PEDANTIC
 *   FP64 -> CUDA cores      : cublasGemmEx + CUBLAS_COMPUTE_64F_PEDANTIC
 *          (--fp64-tensor    : CUBLAS_COMPUTE_64F + *_TENSOR_OP -> FP64 DMMA)
 *   BF16 -> Tensor cores    : CUDA_R_16BF in / CUDA_R_32F accum,
 *                             CUBLAS_COMPUTE_32F + CUBLAS_GEMM_DEFAULT_TENSOR_OP
 * Multi-GPU: column-split of one NxN GEMM (A replicated, B/C sliced; no P2P,
 *            no NCCL).  Aggregate TFLOPS = sum(work)/max(per-GPU time).
 * Telemetry: NVML samples SM/MEM clock, power, temp, throttle reasons live
 *            during the timed region (the "frequency scaling" signal).
 * OOM:       configs exceeding mem-frac * device memory are logged "oom-skip".
 *
 * Compile:  make            (see Makefile)
 * Run:      ./gpu_gemm_bench --help
 */
#define _GNU_SOURCE
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <nvml.h>

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "bf16_cvt.h"

static inline void *my_malloc(size_t size) {
    void *ptr = NULL;
    if (cudaMallocHost(&ptr, size) != cudaSuccess) {
        return NULL;
    }
    return ptr;
}

static inline void my_free(void *ptr) {
    if (ptr) {
        cudaFreeHost(ptr);
    }
}


/* ── error checking ──────────────────────────────────────────────────────── */
#define CUDA_CHECK(x) do { cudaError_t e_=(x); if (e_!=cudaSuccess) { \
    fprintf(stderr,"CUDA %s:%d %s -> %s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e_)); \
    return -1; } } while (0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s_=(x); if (s_!=CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr,"cuBLAS %s:%d %s -> %d\n",__FILE__,__LINE__,#x,(int)s_); \
    return -1; } } while (0)
#define SPARSE_OK(x) do { cusparseStatus_t s_=(x); if (s_!=CUSPARSE_STATUS_SUCCESS) { \
    fprintf(stderr,"cuSPARSE %s:%d %s -> %d\n",__FILE__,__LINE__,#x,(int)s_); \
    return -1; } } while (0)

/* ── precision / mode / engine ───────────────────────────────────────────── */
typedef enum { PREC_FP32, PREC_FP64, PREC_BF16 } prec_t;
typedef enum { MODE_SPLIT, MODE_REPLICAS } bench_mode_t;
typedef enum { ENG_DENSE, ENG_SPARSE } engine_t;   /* cuBLAS GEMM vs cuSPARSE SpMM */

static const char *eng_name(engine_t e) { return e==ENG_DENSE?"dense":"sparse"; }
static cudaDataType cuda_type(prec_t p) {
    return p==PREC_FP32?CUDA_R_32F:p==PREC_FP64?CUDA_R_64F:CUDA_R_16BF;
}
/* C/accumulate type: bf16 accumulates in fp32, fp32->fp32, fp64->fp64 */
static cudaDataType out_type(prec_t p) { return p==PREC_FP64?CUDA_R_64F:CUDA_R_32F; }

static const char *prec_name(prec_t p) {
    return p==PREC_FP32?"fp32":p==PREC_FP64?"fp64":"bf16";
}
static size_t in_bytes(prec_t p)  { return p==PREC_FP32?4:p==PREC_FP64?8:2; }
static size_t out_bytes(prec_t p) { return p==PREC_FP32?4:p==PREC_FP64?8:4; }

/* A100-SXM4-40GB dense peaks (TFLOPS), per GPU. cuBLAS DGEMM on A100 dispatches
 * to the FP64 DMMA tensor cores (19.5 TFLOPS) even with COMPUTE_64F_PEDANTIC
 * (DMMA is bit-identical IEEE-754). The 9.7 TFLOPS FP64 CUDA-core path is only
 * reachable by forcing a non-tensor algo via cuBLASLt (--fp64-cudacore). */
static double peak_tflops(prec_t p, int fp64_cudacore) {
    switch (p) {
    case PREC_FP32: return 19.5;                                /* FP32 CUDA cores   */
    case PREC_FP64: return fp64_cudacore ? 9.7 : 19.5;          /* CUDA-core vs DMMA  */
    case PREC_BF16: return 312.0;                       /* BF16 tensor cores */
    }
    return 1.0;
}

/* ── stateless deterministic matrix fill (OpenMP-friendly) ───────────────── */
static inline float hashf(uint64_t i, uint64_t seed) {
    uint64_t x = i * 0x9E3779B97F4A7C15ull + seed;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27; x *= 0x94D049BB133111EBull; x ^= x >> 31;
    return (float)((double)(uint32_t)(x >> 32) / (double)UINT32_MAX) * 2.0f - 1.0f;
}
static void fill_rand(float *p, size_t n, uint64_t seed) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) p[i] = hashf(i, seed);
}

/* uniform [0,1) from a hashed index — for deterministic sparsity decisions */
static inline double hashu(uint64_t i, uint64_t seed) {
    uint64_t x = (i * 0x9E3779B97F4A7C15ull) ^ seed;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27; x *= 0x94D049BB133111EBull; x ^= x >> 31;
    return (double)(uint32_t)(x >> 32) / ((double)UINT32_MAX + 1.0);
}

/* --fill sparse control: zero out (1-density) of an already-filled dense buffer.
 * The dense cuBLAS GEMM still does full S^3 work — this just proves throughput
 * is independent of how many entries are zero. */
static void sparsify_inplace(float *p, size_t n, double density, uint64_t seed) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++)
        if (hashu(i, seed) >= density) p[i] = 0.0f;
}

/* Build a sparse SxS matrix A in CSR (32-bit indices) with ~density nonzeros,
 * fixed nnz-per-row, columns sorted+distinct, values in [-1,1]. */
static int gen_csr(long S, double density, int **rowOff_o, int **colInd_o,
                   float **vals_o, long *nnz_o) {
    int npr = (int)llround(density * (double)S);
    if (npr < 1) npr = 1;
    if (npr > S) npr = (int)S;
    long nnz = (long)npr * S;
    if (nnz >= (long)2147483647) return -2;   /* exceeds 32-bit CSR indexing */
    int   *rowOff = (int *)  my_malloc((size_t)(S+1) * sizeof(int));
    int   *colInd = (int *)  my_malloc((size_t)nnz   * sizeof(int));
    float *vals   = (float *)my_malloc((size_t)nnz   * sizeof(float));
    if (!rowOff || !colInd || !vals) { my_free(rowOff); my_free(colInd); my_free(vals); return -1; }
    long stride = S / npr; if (stride < 1) stride = 1;
#pragma omp parallel for schedule(static)
    for (long r = 0; r < S; r++) {
        rowOff[r] = (int)(r * (long)npr);
        for (int j = 0; j < npr; j++) {
            uint64_t x = ((uint64_t)r * 1469598103934665603ull) ^ ((uint64_t)j * 1099511628211ull);
            x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33;
            long jit = stride > 1 ? (long)(x % (uint64_t)stride) : 0;
            long col = (long)j * stride + jit;        /* disjoint, increasing ranges */
            /* col is always in [j*stride, (j+1)*stride); ranges are disjoint so
               columns are sorted+distinct. When S%npr!=0 the top S%npr columns
               are structurally absent (fine — A stays a valid CSR). Clamp is a
               belt-and-suspenders guard, normally unreachable. */
            if (col >= S) col = S - 1;
            size_t idx = (size_t)r * (size_t)npr + (size_t)j;
            colInd[idx] = (int)col;
            uint64_t y = x * 0x9E3779B97F4A7C15ull; y ^= y >> 29;
            vals[idx] = (float)((double)(uint32_t)(y >> 32) / (double)UINT32_MAX) * 2.0f - 1.0f;
        }
    }
    rowOff[S] = (int)nnz;
    *rowOff_o = rowOff; *colInd_o = colInd; *vals_o = vals; *nnz_o = nnz;
    return 0;
}

/* ── NVML live sampler ───────────────────────────────────────────────────── */
typedef struct {
    nvmlDevice_t dev;
    int          valid;
    atomic_int   stop;
    /* aggregates */
    unsigned long long sm_sum, mem_sum, pow_sum, util_sum;  /* MHz, MHz, mW, % */
    unsigned     sm_max, mem_max, pow_max, temp_max;
    unsigned long long throttle_or;
    long         n;
} sampler_t;

static void *sampler_run(void *arg) {
    sampler_t *s = (sampler_t *)arg;
    if (!s->valid) return NULL;
    while (!atomic_load(&s->stop)) {
        unsigned sm = 0, mem = 0, mw = 0, t = 0;
        unsigned long long tr = 0;
        nvmlUtilization_t u = {0,0};
        if (nvmlDeviceGetClockInfo(s->dev, NVML_CLOCK_SM,  &sm)  == NVML_SUCCESS &&
            nvmlDeviceGetClockInfo(s->dev, NVML_CLOCK_MEM, &mem) == NVML_SUCCESS) {
            nvmlDeviceGetPowerUsage(s->dev, &mw);
            nvmlDeviceGetTemperature(s->dev, NVML_TEMPERATURE_GPU, &t);
            nvmlDeviceGetUtilizationRates(s->dev, &u);
            nvmlDeviceGetCurrentClocksThrottleReasons(s->dev, &tr);
            s->sm_sum += sm; s->mem_sum += mem; s->pow_sum += mw; s->util_sum += u.gpu;
            if (sm  > s->sm_max)  s->sm_max  = sm;
            if (mem > s->mem_max) s->mem_max = mem;
            if (mw  > s->pow_max) s->pow_max = mw;
            if (t   > s->temp_max) s->temp_max = t;
            s->throttle_or |= tr;
            s->n++;
        }
        struct timespec ts = {0, 2*1000*1000};  /* 2 ms */
        nanosleep(&ts, NULL);
    }
    return NULL;
}

static void throttle_str(unsigned long long tr, char *buf, size_t n) {
    if (tr == 0ULL || tr == nvmlClocksThrottleReasonNone) { snprintf(buf,n,"none"); return; }
    buf[0] = '\0';
    size_t l = 0;
    #define ADD(bit,name) do { if (tr & (bit)) l += (size_t)snprintf(buf+l, l<n?n-l:0, \
        "%s%s", l?"|":"", name); } while (0)
    ADD(nvmlClocksThrottleReasonGpuIdle,                 "idle");
    ADD(nvmlClocksThrottleReasonApplicationsClocksSetting,"app_clk");
    ADD(nvmlClocksThrottleReasonSwPowerCap,              "sw_pwr");
    ADD(nvmlClocksThrottleReasonHwSlowdown,              "hw_slow");
    ADD(nvmlClocksThrottleReasonSwThermalSlowdown,       "sw_therm");
    ADD(nvmlClocksThrottleReasonHwThermalSlowdown,       "hw_therm");
    ADD(nvmlClocksThrottleReasonHwPowerBrakeSlowdown,    "hw_pbrake");
    #undef ADD
    if (l == 0) snprintf(buf, n, "0x%llx", tr);
}

/* ── per-GPU job ─────────────────────────────────────────────────────────── */
typedef struct {
    int     dev;            /* CUDA ordinal                     */
    long    S;              /* M = N = K                        */
    long    col0, ncols;    /* this GPU's column block of C     */
    prec_t  prec;
    engine_t engine;        /* dense (cuBLAS) | sparse (cuSPARSE)*/
    double  density;        /* sparse: nonzero fraction of A     */
    long    nnz;            /* sparse: total nonzeros in full A  */
    const int   *hRowOff;   /* sparse: host CSR row offsets (S+1)*/
    const int   *hColInd;   /* sparse: host CSR column indices   */
    const float *hVals;     /* sparse: host CSR values (fp32)    */
    int     fp64_tensor;
    int     fp64_cudacore;  /* force non-tensor FP64 via cuBLASLt */
    int     warmup, iters;
    const float *hA, *hB;   /* full host A,B (col-major fp32)   */
    int     nvml_on, validate, valid_max, measure_xfer;
    pthread_barrier_t *bar;
    /* outputs */
    int     rc;             /* 0 ok, <0 error/oom               */
    double  batch_ms;       /* time for `iters` gemms (compute) */
    double  h2d_ms;         /* PCIe upload of A (full) + B slice */
    double  d2h_ms;         /* PCIe download of the C slice      */
    double  max_rel_err;    /* -1 if not validated              */
    double  mem_gb;         /* device footprint                 */
    sampler_t smp;
} job_t;

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec*1e3 + (double)t.tv_nsec*1e-6;
}

/* convert contiguous host fp32 slice -> device buffer of precision p.
 * *h2d_ms accumulates ONLY the PCIe H2D copy time (synchronous cudaMemcpy ->
 * wall-clock = transfer); the host fp32->fp64/bf16 conversion is NOT counted. */
static int alloc_upload(prec_t p, const float *src, size_t n, void **dout, double *h2d_ms) {
    size_t eb = in_bytes(p);
    CUDA_CHECK(cudaMalloc(dout, n * eb));
    double t0;
    if (p == PREC_FP32) {
        t0 = now_ms();
        CUDA_CHECK(cudaMemcpy(*dout, src, n*4, cudaMemcpyHostToDevice));
        *h2d_ms += now_ms() - t0;
    } else if (p == PREC_FP64) {
        double *t = (double *)my_malloc(n*8);
        if (!t) return -1;
#pragma omp parallel for schedule(static)
        for (size_t i=0;i<n;i++) t[i] = (double)src[i];
        t0 = now_ms();
        cudaError_t e = cudaMemcpy(*dout, t, n*8, cudaMemcpyHostToDevice);
        *h2d_ms += now_ms() - t0;
        my_free(t);
        if (e != cudaSuccess) { fprintf(stderr,"H2D fp64: %s\n",cudaGetErrorString(e)); return -1; }
    } else {
        uint16_t *t = (uint16_t *)my_malloc(n*2);
        if (!t) return -1;
#pragma omp parallel for schedule(static)
        for (size_t i=0;i<n;i++) t[i] = f32_to_bf16(src[i]);
        t0 = now_ms();
        cudaError_t e = cudaMemcpy(*dout, t, n*2, cudaMemcpyHostToDevice);
        *h2d_ms += now_ms() - t0;
        my_free(t);
        if (e != cudaSuccess) { fprintf(stderr,"H2D bf16: %s\n",cudaGetErrorString(e)); return -1; }
    }
    return 0;
}

static cublasStatus_t do_gemm(cublasHandle_t h, prec_t p, int fp64_tensor,
                              int M, int N, int K,
                              const void *dA, const void *dB, void *dC) {
    static const float  f1 = 1.0f, f0 = 0.0f;
    static const double d1 = 1.0,  d0 = 0.0;
    switch (p) {
    case PREC_FP32:
        return cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
            &f1, dA, CUDA_R_32F, M, dB, CUDA_R_32F, K, &f0, dC, CUDA_R_32F, M,
            CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT);
    case PREC_FP64:
        return cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
            &d1, dA, CUDA_R_64F, M, dB, CUDA_R_64F, K, &d0, dC, CUDA_R_64F, M,
            fp64_tensor ? CUBLAS_COMPUTE_64F : CUBLAS_COMPUTE_64F_PEDANTIC,
            fp64_tensor ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT);
    case PREC_BF16:
        return cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
            &f1, dA, CUDA_R_16BF, M, dB, CUDA_R_16BF, K, &f0, dC, CUDA_R_32F, M,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

/* ── FP64 CUDA-core path via cuBLASLt (--fp64-cudacore) ───────────────────────
 * cuBLAS DGEMM always picks the FP64 DMMA tensor cores on A100. To measure the
 * non-tensor (CUDA-core, ~9.7 TFLOPS) path we enumerate cuBLASLt algorithms and
 * select one whose NUMERICAL_IMPL flags carry NO tensor-op bit (pure FMA), then
 * run it with cublasLtMatmul. If no such algo validates, setup returns nonzero
 * and the caller reports the config as an error (honest: forcing failed). */
typedef struct {
    cublasLtHandle_t       lt;
    cublasLtMatmulDesc_t   desc;
    cublasLtMatrixLayout_t la, lb, lc;
    cublasLtMatmulAlgo_t   algo;
    int                    have_algo;
    void                  *ws;
    size_t                 ws_size;
} lt_fp64_t;

static int lt_fp64_setup(lt_fp64_t *x, int M, int N, int K) {
    memset(x, 0, sizeof *x);
    if (cublasLtCreate(&x->lt)!=CUBLAS_STATUS_SUCCESS) return -1;
    if (cublasLtMatmulDescCreate(&x->desc, CUBLAS_COMPUTE_64F, CUDA_R_64F)!=CUBLAS_STATUS_SUCCESS) return -1;
    cublasOperation_t opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(x->desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof opN);
    cublasLtMatmulDescSetAttribute(x->desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof opN);
    /* A: MxK (ld=M), B: KxN (ld=K), C=D: MxN (ld=M), all col-major fp64 */
    if (cublasLtMatrixLayoutCreate(&x->la, CUDA_R_64F, (uint64_t)M, (uint64_t)K, M)!=CUBLAS_STATUS_SUCCESS) return -1;
    if (cublasLtMatrixLayoutCreate(&x->lb, CUDA_R_64F, (uint64_t)K, (uint64_t)N, K)!=CUBLAS_STATUS_SUCCESS) return -1;
    if (cublasLtMatrixLayoutCreate(&x->lc, CUDA_R_64F, (uint64_t)M, (uint64_t)N, M)!=CUBLAS_STATUS_SUCCESS) return -1;
    x->ws_size = (size_t)64*1024*1024;
    if (cudaMalloc(&x->ws, x->ws_size)!=cudaSuccess){ x->ws=NULL; x->ws_size=0; }

    int ids[256]; int nids=0;
    if (cublasLtMatmulAlgoGetIds(x->lt, CUBLAS_COMPUTE_64F, CUDA_R_64F,
            CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, 256, ids, &nids)!=CUBLAS_STATUS_SUCCESS)
        return -1;
    for (int i=0;i<nids;i++){
        cublasLtMatmulAlgo_t algo;
        if (cublasLtMatmulAlgoInit(x->lt, CUBLAS_COMPUTE_64F, CUDA_R_64F,
                CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, ids[i], &algo)!=CUBLAS_STATUS_SUCCESS)
            continue;
        uint64_t flags=0; size_t got=0;
        if (cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
                &flags, sizeof flags, &got)!=CUBLAS_STATUS_SUCCESS) continue;
        if (flags & CUBLASLT_NUMERICAL_IMPL_FLAGS_TENSOR_OP_MASK) continue;  /* skip DMMA/tensor */
        cublasLtMatmulHeuristicResult_t heur; memset(&heur,0,sizeof heur);
        if (cublasLtMatmulAlgoCheck(x->lt, x->desc, x->la, x->lb, x->lc, x->lc, &algo, &heur)
                != CUBLAS_STATUS_SUCCESS) continue;
        if (heur.state != CUBLAS_STATUS_SUCCESS) continue;
        if (heur.workspaceSize > x->ws_size) continue;
        x->algo = algo; x->have_algo = 1; break;
    }
    return x->have_algo ? 0 : -2;   /* -2: no non-tensor FP64 algo validated */
}

static cublasStatus_t do_gemm_lt_fp64(lt_fp64_t *x, cudaStream_t st,
                                      const void *dA, const void *dB, void *dC) {
    static const double one=1.0, zero=0.0;
    return cublasLtMatmul(x->lt, x->desc, &one, dA, x->la, dB, x->lb, &zero,
                          dC, x->lc, dC, x->lc, &x->algo, x->ws, x->ws_size, st);
}

static void lt_fp64_free(lt_fp64_t *x) {
    if (x->la)   cublasLtMatrixLayoutDestroy(x->la);
    if (x->lb)   cublasLtMatrixLayoutDestroy(x->lb);
    if (x->lc)   cublasLtMatrixLayoutDestroy(x->lc);
    if (x->desc) cublasLtMatmulDescDestroy(x->desc);
    if (x->ws)   cudaFree(x->ws);
    if (x->lt)   cublasLtDestroy(x->lt);
    memset(x, 0, sizeof *x);
}

/* sampled correctness check against an fp64 host reference.
 * hC is the already-downloaded host copy of the C slice (S x ncols, col-major). */
static double validate_slice(job_t *j, const void *hC) {
    const long S = j->S, ncols = j->ncols;
    double worst = 0.0;
    uint64_t rng = 0xC0FFEEull ^ (uint64_t)j->dev;
    for (int s = 0; s < 128; s++) {
        rng ^= rng<<13; rng ^= rng>>7; rng ^= rng<<17;
        long i = (long)(rng % (uint64_t)S);
        rng ^= rng<<13; rng ^= rng>>7; rng ^= rng<<17;
        long jc = (long)(rng % (uint64_t)ncols);     /* local column */
        long gj = j->col0 + jc;                       /* global column */
        double ref = 0.0;
        for (long k = 0; k < S; k++) {
            double a = (double)j->hA[i + k*S], b = (double)j->hB[k + gj*S];
            if (j->prec == PREC_BF16) {   /* match the bf16-rounded inputs cuBLAS saw */
                a = (double)bf16_to_f32(f32_to_bf16(j->hA[i + k*S]));
                b = (double)bf16_to_f32(f32_to_bf16(j->hB[k + gj*S]));
            }
            ref += a * b;
        }
        double got;
        if (j->prec == PREC_FP64) got = ((double*)hC)[i + jc*S];
        else                      got = (double)((float*)hC)[i + jc*S];
        double denom = fabs(ref) > 1e-12 ? fabs(ref) : 1e-12;
        double rel = fabs(got - ref) / denom;
        if (rel > worst) worst = rel;
    }
    return worst;
}

static void *gpu_worker(void *arg) {
    job_t *j = (job_t *)arg;
    j->rc = 0; j->max_rel_err = -1.0; j->batch_ms = 0.0;
    j->h2d_ms = 0.0; j->d2h_ms = 0.0;
    const long S = j->S, ncols = j->ncols;
    const int  M = (int)S, N = (int)ncols, K = (int)S;

    /* All resources declared + NULL-initialized BEFORE any goto, so the
       barrier_out cleanup never reads an indeterminate handle/pointer no
       matter which failure point jumped there. */
    cublasHandle_t h = NULL; cudaStream_t st = NULL;
    void *dA=NULL, *dB=NULL, *dC=NULL;
    cudaEvent_t e0=NULL, e1=NULL;
    pthread_t smp_th; int smp_started=0;
    lt_fp64_t lt = {0}; int use_lt=0;   /* cuBLASLt non-tensor FP64 path */

    if (cudaSetDevice(j->dev) != cudaSuccess)    { j->rc=-1; goto barrier_out; }
    if (cudaStreamCreate(&st)!=cudaSuccess)      { j->rc=-1; goto barrier_out; }
    if (cublasCreate(&h)!=CUBLAS_STATUS_SUCCESS) { j->rc=-1; goto barrier_out; }
    cublasSetStream(h, st);
    /* Math mode: PEDANTIC (no tensor cores) for fp32 and fp64-on-CUDA-cores;
       DEFAULT (tensor ops allowed) for bf16 and fp64-DMMA — otherwise a
       PEDANTIC handle would silently suppress the --fp64-tensor DMMA path. */
    {
        cublasMath_t mm = CUBLAS_PEDANTIC_MATH;
        if (j->prec==PREC_BF16 || (j->prec==PREC_FP64 && j->fp64_tensor))
            mm = CUBLAS_DEFAULT_MATH;
        cublasSetMathMode(h, mm);
    }

    /* --fp64-cudacore: build a cuBLASLt context bound to a non-tensor FP64 algo.
       If no such algo validates, report the config as an error (forcing failed). */
    if (j->fp64_cudacore && j->prec==PREC_FP64) {
        int lr = lt_fp64_setup(&lt, M, N, K);
        if (lr != 0) {
            fprintf(stderr, "fp64-cudacore: no non-tensor FP64 algo (rc=%d) on dev %d\n", lr, j->dev);
            j->rc=-1; goto barrier_out;
        }
        use_lt = 1;
    }

    /* allocate + upload: A full (S*S), B slice (S*ncols), C slice (S*ncols).
       h2d_ms accumulates the PCIe time of both operand uploads. */
    if (alloc_upload(j->prec, j->hA, (size_t)S*S, &dA, &j->h2d_ms) != 0)          { j->rc=-1; goto barrier_out; }
    if (alloc_upload(j->prec, j->hB + (size_t)j->col0*S, (size_t)S*ncols, &dB, &j->h2d_ms)!=0){ j->rc=-1; goto barrier_out; }
    if (cudaMalloc(&dC, (size_t)S*ncols*out_bytes(j->prec))!=cudaSuccess){ j->rc=-1; goto barrier_out; }
    cudaMemset(dC, 0, (size_t)S*ncols*out_bytes(j->prec));
    if (cudaEventCreate(&e0)!=cudaSuccess || cudaEventCreate(&e1)!=cudaSuccess){ j->rc=-1; goto barrier_out; }

    /* warmup */
    for (int w=0; w<j->warmup; w++) {
        cublasStatus_t gs = use_lt ? do_gemm_lt_fp64(&lt, st, dA,dB,dC)
                                   : do_gemm(h, j->prec, j->fp64_tensor, M,N,K, dA,dB,dC);
        if (gs!=CUBLAS_STATUS_SUCCESS) j->rc=-1;
    }
    cudaStreamSynchronize(st);
    if (j->rc) goto barrier_out;

    /* all GPUs cross the start line together so the run truly overlaps */
    pthread_barrier_wait(j->bar);

    /* start NVML sampler AFTER the barrier so it only covers the timed region
       (not the setup/sync window) — keeps the clock/power averages honest */
    if (j->nvml_on && j->smp.valid) {
        atomic_store(&j->smp.stop, 0);
        if (pthread_create(&smp_th, NULL, sampler_run, &j->smp)==0) smp_started=1;
    }

    /* timed batch (no per-iter host sync -> steady-state) */
    int loop_err=0;
    cudaEventRecord(e0, st);
    for (int it=0; it<j->iters; it++) {
        cublasStatus_t gs = use_lt ? do_gemm_lt_fp64(&lt, st, dA,dB,dC)
                                   : do_gemm(h, j->prec, j->fp64_tensor, M,N,K, dA,dB,dC);
        if (gs!=CUBLAS_STATUS_SUCCESS) loop_err=1;
    }
    cudaEventRecord(e1, st);
    cudaEventSynchronize(e1);

    if (smp_started) { atomic_store(&j->smp.stop, 1); pthread_join(smp_th, NULL); }
    if (loop_err) { j->rc=-1; goto done; }   /* report as error, not a fake number */
    { float ms=0; cudaEventElapsedTime(&ms, e0, e1); j->batch_ms = (double)ms; }

    /* timed device->host download of the full C slice (the result a real caller
       would fetch back), then validate from that same host copy if requested. */
    int want_c = j->measure_xfer || (j->validate && S <= j->valid_max);
    if (want_c) {
        size_t cbytes = (size_t)S*ncols*out_bytes(j->prec);
        void *hC = my_malloc(cbytes);
        if (hC) {
            double t0 = now_ms();
            cudaMemcpy(hC, dC, cbytes, cudaMemcpyDeviceToHost);
            j->d2h_ms = now_ms() - t0;
            if (j->validate && S <= j->valid_max)
                j->max_rel_err = validate_slice(j, hC);
            my_free(hC);
        }
    }

done:   /* success/timed-error path: barrier already passed, so do NOT re-wait */
    if (e0) cudaEventDestroy(e0);
    if (e1) cudaEventDestroy(e1);
    if (dA) cudaFree(dA);
    if (dB) cudaFree(dB);
    if (dC) cudaFree(dC);
    lt_fp64_free(&lt);   /* no-op when not set up ({0}) */
    if (h)  cublasDestroy(h);
    if (st) cudaStreamDestroy(st);
    return NULL;

barrier_out:
    /* ensure the barrier is satisfied even on the error path so peers proceed */
    if (e0) cudaEventDestroy(e0);
    if (e1) cudaEventDestroy(e1);
    if (dA) cudaFree(dA);
    if (dB) cudaFree(dB);
    if (dC) cudaFree(dC);
    lt_fp64_free(&lt);   /* no-op when not set up ({0}) */
    if (h)  cublasDestroy(h);
    if (st) cudaStreamDestroy(st);
    pthread_barrier_wait(j->bar);
    return NULL;
}

/* sampled correctness check for the sparse path: C(i,gj)=sum_k A.val[k]*B(A.col[k],gj) */
static double validate_spmm(job_t *j, const void *hC) {
    const long S = j->S, ncols = j->ncols;
    double worst = 0.0;
    uint64_t rng = 0x5EED5EEDull ^ (uint64_t)j->dev;
    for (int s = 0; s < 128; s++) {
        rng ^= rng<<13; rng ^= rng>>7; rng ^= rng<<17;
        long i = (long)(rng % (uint64_t)S);
        rng ^= rng<<13; rng ^= rng>>7; rng ^= rng<<17;
        long jc = (long)(rng % (uint64_t)ncols);
        long gj = j->col0 + jc;
        double ref = 0.0;
        for (long k = (long)j->hRowOff[i]; k < (long)j->hRowOff[i+1]; k++) {
            long col = j->hColInd[k];
            double a = (double)j->hVals[k], b = (double)j->hB[col + gj*S];
            if (j->prec == PREC_BF16) {
                a = (double)bf16_to_f32(f32_to_bf16(j->hVals[k]));
                b = (double)bf16_to_f32(f32_to_bf16(j->hB[col + gj*S]));
            }
            ref += a * b;
        }
        double got = (j->prec==PREC_FP64) ? ((double*)hC)[i + jc*S]
                                          : (double)((float*)hC)[i + jc*S];
        double denom = fabs(ref) > 1e-12 ? fabs(ref) : 1e-12;
        double rel = fabs(got - ref) / denom;
        if (rel > worst) worst = rel;
    }
    return worst;
}

/* sparse engine worker: cuSPARSE SpMM, C(dense SxNcols) = A(sparse SxS) * B(dense SxNcols).
 * Mirrors gpu_worker's scaffolding (per-device, NVML, barrier, timed transfers).
 * Single cleanup label + pre_barrier flag => the barrier is hit exactly once. */
static void *spmm_worker(void *arg) {
    job_t *j = (job_t *)arg;
    j->rc = 0; j->max_rel_err = -1.0; j->batch_ms = 0.0;
    j->h2d_ms = 0.0; j->d2h_ms = 0.0;
    const long S = j->S, ncols = j->ncols, nnz = j->nnz;
    int pre_barrier = 0, smp_started = 0;

    cusparseHandle_t sph = NULL; cudaStream_t st = NULL;
    void *dVals=NULL, *dB=NULL, *dC=NULL, *dBuf=NULL;
    int  *dRow=NULL, *dCol=NULL;
    cusparseSpMatDescr_t matA = NULL;
    cusparseDnMatDescr_t matB = NULL, matC = NULL;
    cudaEvent_t e0=NULL, e1=NULL;
    pthread_t smp_th;

    if (cudaSetDevice(j->dev) != cudaSuccess)         { j->rc=-1; goto sout; }
    if (cudaStreamCreate(&st)!=cudaSuccess)           { j->rc=-1; goto sout; }
    if (cusparseCreate(&sph)!=CUSPARSE_STATUS_SUCCESS){ j->rc=-1; goto sout; }
    cusparseSetStream(sph, st);

    /* upload CSR (rowOff/colInd ints, no conversion; vals converted) + B slice */
    if (cudaMalloc((void**)&dRow,(size_t)(S+1)*sizeof(int))!=cudaSuccess){ j->rc=-1; goto sout; }
    if (cudaMalloc((void**)&dCol,(size_t)nnz*sizeof(int))!=cudaSuccess)  { j->rc=-1; goto sout; }
    {
        double t0 = now_ms();
        cudaError_t er = cudaMemcpy(dRow, j->hRowOff, (size_t)(S+1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaError_t ec = cudaMemcpy(dCol, j->hColInd, (size_t)nnz*sizeof(int),   cudaMemcpyHostToDevice);
        j->h2d_ms += now_ms() - t0;
        if (er!=cudaSuccess || ec!=cudaSuccess){ j->rc=-1; goto sout; }
    }
    if (alloc_upload(j->prec, j->hVals, (size_t)nnz, &dVals, &j->h2d_ms)!=0){ j->rc=-1; goto sout; }
    if (alloc_upload(j->prec, j->hB + (size_t)j->col0*S, (size_t)S*ncols, &dB, &j->h2d_ms)!=0){ j->rc=-1; goto sout; }
    if (cudaMalloc(&dC,(size_t)S*ncols*out_bytes(j->prec))!=cudaSuccess){ j->rc=-1; goto sout; }
    cudaMemset(dC, 0, (size_t)S*ncols*out_bytes(j->prec));

    if (cusparseCreateCsr(&matA, S, S, nnz, dRow, dCol, dVals,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
            cuda_type(j->prec))!=CUSPARSE_STATUS_SUCCESS) { j->rc=-1; goto sout; }
    if (cusparseCreateDnMat(&matB, S, ncols, S, dB, cuda_type(j->prec),
            CUSPARSE_ORDER_COL)!=CUSPARSE_STATUS_SUCCESS) { j->rc=-1; goto sout; }
    if (cusparseCreateDnMat(&matC, S, ncols, S, dC, out_type(j->prec),
            CUSPARSE_ORDER_COL)!=CUSPARSE_STATUS_SUCCESS) { j->rc=-1; goto sout; }

    cudaDataType compute = (j->prec==PREC_FP64) ? CUDA_R_64F : CUDA_R_32F;
    static const float  f1=1.0f, f0=0.0f; static const double d1=1.0, d0=0.0;
    const void *alpha = (j->prec==PREC_FP64) ? (const void*)&d1 : (const void*)&f1;
    const void *beta  = (j->prec==PREC_FP64) ? (const void*)&d0 : (const void*)&f0;
    const cusparseOperation_t NT = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseSpMMAlg_t   alg = CUSPARSE_SPMM_ALG_DEFAULT;

    size_t bufSize = 0;
    if (cusparseSpMM_bufferSize(sph, NT, NT, alpha, matA, matB, beta, matC,
            compute, alg, &bufSize)!=CUSPARSE_STATUS_SUCCESS) { j->rc=-1; goto sout; }
    if (bufSize && cudaMalloc(&dBuf, bufSize)!=cudaSuccess) { j->rc=-1; goto sout; }

    if (cudaEventCreate(&e0)!=cudaSuccess || cudaEventCreate(&e1)!=cudaSuccess){ j->rc=-1; goto sout; }

    for (int w=0; w<j->warmup; w++)
        if (cusparseSpMM(sph, NT, NT, alpha, matA, matB, beta, matC,
                compute, alg, dBuf)!=CUSPARSE_STATUS_SUCCESS) { j->rc=-1; }
    cudaStreamSynchronize(st);
    if (j->rc) goto sout;

    pthread_barrier_wait(j->bar); pre_barrier = 1;

    if (j->nvml_on && j->smp.valid) {
        atomic_store(&j->smp.stop, 0);
        if (pthread_create(&smp_th, NULL, sampler_run, &j->smp)==0) smp_started=1;
    }

    int loop_err=0;
    cudaEventRecord(e0, st);
    for (int it=0; it<j->iters; it++)
        if (cusparseSpMM(sph, NT, NT, alpha, matA, matB, beta, matC, compute, alg, dBuf)
                != CUSPARSE_STATUS_SUCCESS) loop_err=1;
    cudaEventRecord(e1, st);
    cudaEventSynchronize(e1);
    if (smp_started) { atomic_store(&j->smp.stop, 1); pthread_join(smp_th, NULL); }
    if (loop_err) { j->rc=-1; goto sout; }   /* sampler already joined; pre_barrier set */
    { float ms=0; cudaEventElapsedTime(&ms, e0, e1); j->batch_ms = (double)ms; }

    {
        int want_c = j->measure_xfer || (j->validate && S <= j->valid_max);
        if (want_c) {
            size_t cb = (size_t)S*ncols*out_bytes(j->prec);
            void *hC = my_malloc(cb);
            if (hC) {
                double t0 = now_ms();
                cudaMemcpy(hC, dC, cb, cudaMemcpyDeviceToHost);
                j->d2h_ms = now_ms() - t0;
                if (j->validate && S <= j->valid_max)
                    j->max_rel_err = validate_spmm(j, hC);
                my_free(hC);
            }
        }
    }

sout:
    if (matA) cusparseDestroySpMat(matA);
    if (matB) cusparseDestroyDnMat(matB);
    if (matC) cusparseDestroyDnMat(matC);
    if (e0) cudaEventDestroy(e0);
    if (e1) cudaEventDestroy(e1);
    if (dVals) cudaFree(dVals);
    if (dB) cudaFree(dB);
    if (dC) cudaFree(dC);
    if (dBuf) cudaFree(dBuf);
    if (dRow) cudaFree(dRow);
    if (dCol) cudaFree(dCol);
    if (sph) cusparseDestroy(sph);
    if (st) cudaStreamDestroy(st);
    if (!pre_barrier) pthread_barrier_wait(j->bar);  /* errored before timed region */
    return NULL;
}

/* ── config ──────────────────────────────────────────────────────────────── */
typedef struct {
    long   sizes[32];   int n_sizes;
    prec_t precs[3];    int n_precs;
    int    gpus[8];     int n_gpus;
    bench_mode_t mode;
    engine_t engines[2]; int n_engines;   /* dense and/or sparse                 */
    double densities[8]; int n_dens;       /* sparse-engine A density sweep        */
    int    fill_sparse; double fill_density; /* dense-engine data: --fill sparse:d */
    int    iters, warmup;
    int    fp64_tensor, fp64_cudacore;
    int    nvml_on, validate, valid_max, measure_xfer;
    double mem_frac;
    const char *csv;
    const char *tag;
} cfg_t;

static long mem_footprint(cfg_t *c, prec_t p, bench_mode_t m, long S, int G, long *max_ncols) {
    long base = S/G, rem = S%G;
    long ncols = base + (rem?1:0);                 /* largest slot */
    *max_ncols = (m==MODE_REPLICAS) ? S : ncols;
    size_t ib=in_bytes(p), ob=out_bytes(p);
    if (m==MODE_REPLICAS)
        return (long)((size_t)ib*S*S + (size_t)ib*S*S + (size_t)ob*S*S);
    return (long)((size_t)ib*S*S + (size_t)ib*S*(*max_ncols) + (size_t)ob*S*(*max_ncols));
    (void)c;
}

/* sparse (cuSPARSE SpMM) device footprint: full CSR(A) replicated + B,C dense
 * slices. The SpMM workspace is small for ALG_DEFAULT and folded into the cap
 * margin. */
static long sparse_footprint(prec_t p, bench_mode_t m, long S, long nnz, int G, long *max_ncols) {
    long base = S/G, rem = S%G;
    *max_ncols = (m==MODE_REPLICAS) ? S : (base + (rem?1:0));
    size_t ib=in_bytes(p), ob=out_bytes(p);
    size_t csr = (size_t)(S+1)*sizeof(int) + (size_t)nnz*sizeof(int) + (size_t)nnz*ib;
    return (long)(csr + ib*(size_t)S*(size_t)(*max_ncols) + ob*(size_t)S*(size_t)(*max_ncols));
}

static void usage(const char *p) {
    printf(
"Usage: %s [options]\n"
"  --sizes a,b,..     matrix sizes (M=N=K)         [512..65536 pow2]\n"
"  --precisions p,..  fp32,fp64,bf16               [all]\n"
"  --gpus n,..        gpu counts to test           [1,2,3,4 capped to avail]\n"
"  --mode m           split | replicas             [split]\n"
"  --engine e,..      dense (cuBLAS) , sparse (cuSPARSE SpMM) [dense]\n"
"  --density d,..     sparse-engine A nonzero frac  [0.01,0.05,0.1,0.25]\n"
"  --fill f           dense-engine data: dense | sparse[:d]   [dense]\n"
"  --iters n          timed iterations             [10]\n"
"  --warmup n         warmup iterations            [3]\n"
"  --fp64-tensor      FP64 via DMMA tensor cores    [default path on A100]\n"
"  --fp64-cudacore    force non-tensor FP64 (cuBLASLt) -> ~9.7 TFLOPS / half\n"
"  --no-nvml          disable NVML telemetry       [enabled]\n"
"  --validate         sampled correctness check    [off]\n"
"  --validate-max n   max size to validate         [2048]\n"
"  --no-transfers     skip timed H2D/D2H measure   [measured]\n"
"  --mem-frac f       usable fraction of GPU mem   [0.90]\n"
"  --csv path         append CSV results           [stdout only]\n"
"  --tag s            free-form label in CSV        [\"\"]\n"
"  --help\n", p);
}

static int parse_longs(const char *s, long *out, int max) {
    int n=0; char buf[512]; snprintf(buf,sizeof buf,"%s",s);
    for (char *t=strtok(buf,","); t && n<max; t=strtok(NULL,",")) out[n++]=atol(t);
    return n;
}

/* Run one (engine, precision, gpus, density) point: footprint-check, spawn G
 * workers, aggregate, emit a stdout line + CSV row. hRow/hCol/hVals are the
 * host CSR of A for the sparse engine (NULL for dense). */
static void run_config(const cfg_t *c, FILE *csv, const char *host, int cudart,
                       long cap, int nvml_ready, long S, prec_t p, int G,
                       engine_t engine, double density, long nnz,
                       const float *hA, const float *hB,
                       const int *hRow, const int *hCol, const float *hVals)
{
    long ts=(long)time(NULL);
    const char *mode_s = c->mode==MODE_SPLIT?"split":"replicas";
    const char *eng_s  = eng_name(engine);
    const char *pn = (p==PREC_FP64 && c->fp64_cudacore) ? "fp64c" : prec_name(p);
    long max_ncols=0, foot;
    if (engine==ENG_SPARSE) foot = sparse_footprint(p,c->mode,S,nnz,G,&max_ncols);
    else                    foot = mem_footprint((cfg_t*)c,p,c->mode,S,G,&max_ncols);
    double mem_gb=(double)foot/1e9;

    if (foot > cap){
        printf("%-6s %-5s %-3d %6ld %6.3f %9s %9s %6s %8s %8s %6s %7s %s\n",
            eng_s,pn,G,S,density,"-","-","-","-","-","-","-","oom-skip");
        if(csv) fprintf(csv,"%s,%ld,%s,%d,%s,%s,%d,%ld,%ld,%ld,%d,%d,"
            "0,0,0,0,0,0,0,0,0,0,none,%.3f,%d,%d,%.3g,"
            "0,0,0,0,0,0,0,0,%s,%.4g,%ld,oom-skip\n",
            c->tag,ts,host,cudart,mode_s,pn,G,S,S,S,c->iters,c->warmup,
            mem_gb,c->fp64_tensor,0,-1.0,eng_s,density,nnz);
        return;
    }

    job_t job[8]; pthread_barrier_t bar; pthread_barrier_init(&bar,NULL,(unsigned)G);
    long base=S/G, rem=S%G, col=0;
    for(int g=0;g<G;g++){
        long ncols=(c->mode==MODE_REPLICAS)?S:(base+(g<rem?1:0));
        memset(&job[g],0,sizeof job[g]);
        job[g].dev=g; job[g].S=S;
        job[g].col0=(c->mode==MODE_REPLICAS)?0:col;
        job[g].ncols=ncols;
        job[g].prec=p; job[g].engine=engine; job[g].density=density; job[g].nnz=nnz;
        job[g].hRowOff=hRow; job[g].hColInd=hCol; job[g].hVals=hVals;
        job[g].fp64_tensor=c->fp64_tensor;
        job[g].fp64_cudacore=c->fp64_cudacore;
        job[g].warmup=c->warmup; job[g].iters=c->iters;
        job[g].hA=hA; job[g].hB=hB;
        job[g].nvml_on=nvml_ready; job[g].validate=c->validate; job[g].valid_max=c->valid_max;
        job[g].measure_xfer=c->measure_xfer;
        job[g].bar=&bar; job[g].mem_gb=mem_gb;
        if (c->mode!=MODE_REPLICAS) col+=ncols;
        job[g].smp.valid=0;
        if (nvml_ready){
            char bus[32]; if (cudaDeviceGetPCIBusId(bus,sizeof bus,g)==cudaSuccess){
                if (nvmlDeviceGetHandleByPciBusId_v2(bus,&job[g].smp.dev)==NVML_SUCCESS)
                    job[g].smp.valid=1;
            }
            atomic_init(&job[g].smp.stop,0);
        }
    }

    pthread_t th[8];
    for(int g=0;g<G;g++)
        pthread_create(&th[g],NULL, engine==ENG_SPARSE?spmm_worker:gpu_worker, &job[g]);
    for(int g=0;g<G;g++) pthread_join(th[g],NULL);
    pthread_barrier_destroy(&bar);

    int err=0; double wall=0, sum_flops=0, rel=-1.0;
    unsigned long long sm_sum=0,mem_sum=0,pow_sum=0; long nsamp=0;
    unsigned sm_max=0,mem_max=0,pow_max=0,temp_max=0; unsigned long long tr=0;
    for(int g=0;g<G;g++){
        if(job[g].rc){err=1;continue;}
        double tg=job[g].batch_ms/1000.0; if(tg>wall) wall=tg;
        sum_flops += (engine==ENG_SPARSE)
            ? (double)c->iters*2.0*(double)nnz*(double)job[g].ncols
            : (double)c->iters*2.0*(double)S*(double)job[g].ncols*(double)S;
        if(job[g].max_rel_err>rel) rel=job[g].max_rel_err;
        sampler_t*s=&job[g].smp;
        if(s->n>0){ sm_sum+=s->sm_sum; mem_sum+=s->mem_sum; pow_sum+=s->pow_sum; nsamp+=s->n;
            if(s->sm_max>sm_max)   sm_max=s->sm_max;
            if(s->mem_max>mem_max) mem_max=s->mem_max;
            if(s->pow_max>pow_max) pow_max=s->pow_max;
            if(s->temp_max>temp_max) temp_max=s->temp_max;
            tr|=s->throttle_or; }
    }
    if(err){
        printf("%-6s %-5s %-3d %6ld %6.3f %9s %9s %6s %8s %8s %6s %7s %s\n",
            eng_s,pn,G,S,density,"-","-","-","-","-","-","-","error");
        if(csv) fprintf(csv,"%s,%ld,%s,%d,%s,%s,%d,%ld,%ld,%ld,%d,%d,"
            "0,0,0,0,0,0,0,0,0,0,none,%.3f,%d,%d,%.3g,"
            "0,0,0,0,0,0,0,0,%s,%.4g,%ld,error\n",
            c->tag,ts,host,cudart,mode_s,pn,G,S,S,S,c->iters,c->warmup,
            mem_gb,c->fp64_tensor,c->validate&&S<=c->valid_max,rel,eng_s,density,nnz);
        return;
    }

    double agg_tflops = sum_flops/wall/1e12;
    double per_gpu = agg_tflops/(double)G;
    double pct = (engine==ENG_SPARSE) ? 0.0
               : 100.0*agg_tflops/((double)G*peak_tflops(p,c->fp64_cudacore));

    /* bottleneck GPU (largest upload + 1 compute + download path) */
    int bn=0; double max_e2e=0;
    for(int g=0;g<G;g++){
        double comp=job[g].batch_ms/(double)c->iters;
        double e2e=job[g].h2d_ms+comp+job[g].d2h_ms;
        if(e2e>max_e2e){max_e2e=e2e; bn=g;}
    }
    double b_h2d=job[bn].h2d_ms, b_d2h=job[bn].d2h_ms;
    double b_comp=job[bn].batch_ms/(double)c->iters;
    double e2e_ms=max_e2e;
    double single = (engine==ENG_SPARSE) ? 2.0*(double)nnz*(double)S
                                         : 2.0*(double)S*(double)S*(double)S;
    double prob_flops=(c->mode==MODE_REPLICAS?(double)G:1.0)*single;
    double eff_tflops = e2e_ms>0 ? prob_flops/(e2e_ms/1000.0)/1e12 : 0;
    double xfer_pct   = e2e_ms>0 ? 100.0*(b_h2d+b_d2h)/e2e_ms : 0;
    size_t hb, db;
    if (engine==ENG_SPARSE){
        hb=(size_t)(S+1)*sizeof(int) + (size_t)nnz*sizeof(int) + (size_t)nnz*in_bytes(p)
           + in_bytes(p)*(size_t)S*(size_t)job[bn].ncols;
        db=out_bytes(p)*(size_t)S*(size_t)job[bn].ncols;
    } else {
        hb=in_bytes(p)*((size_t)S*S+(size_t)S*(size_t)job[bn].ncols);
        db=out_bytes(p)*((size_t)S*(size_t)job[bn].ncols);
    }
    double h2d_gbps = b_h2d>0 ? (double)hb/1e9/(b_h2d/1000.0) : 0;
    double d2h_gbps = b_d2h>0 ? (double)db/1e9/(b_d2h/1000.0) : 0;
    double sm_avg = nsamp? (double)sm_sum/(double)nsamp : 0;
    double mem_avg= nsamp? (double)mem_sum/(double)nsamp: 0;
    double pow_avg= nsamp? (double)pow_sum/(double)nsamp/1000.0:0;
    char trbuf[128]; throttle_str(tr,trbuf,sizeof trbuf);

    double tol = (p==PREC_BF16)?5e-2:(p==PREC_FP32)?1e-3:1e-9;
    const char *vstr = rel>=0 ? (rel<tol?"ok/val":"ok/HIERR") : "ok";
    printf("%-6s %-5s %-3d %6ld %6.3f %9.1f %9.1f %5.1f%% %8.2f %8.2f %5.1f%% %7.0f %s\n",
        eng_s,pn,G,S,density,agg_tflops,eff_tflops,pct,b_comp,e2e_ms,xfer_pct,sm_avg,vstr);
    if(csv){
        fprintf(csv,"%s,%ld,%s,%d,%s,%s,%d,%ld,%ld,%ld,%d,%d,"
            "%.2f,%.2f,%.2f,%.3f,"
            "%.0f,%u,%.0f,%.1f,%.1f,%u,%s,"
            "%.3f,%d,%d,%.3g,"
            "%.3f,%.3f,%.3f,%.3f,%.1f,%.1f,%.2f,%.1f,"
            "%s,%.4g,%ld,ok\n",
            c->tag,ts,host,cudart,mode_s,pn,G,S,S,S,c->iters,c->warmup,
            agg_tflops,per_gpu,pct,wall*1000.0,
            sm_avg,sm_max,mem_avg,pow_avg,(double)pow_max/1000.0,temp_max,trbuf,
            mem_gb,c->fp64_tensor,c->validate&&S<=c->valid_max,rel,
            b_h2d,b_d2h,b_comp,e2e_ms,h2d_gbps,d2h_gbps,eff_tflops,xfer_pct,
            eng_s,density,nnz);
        fflush(csv);
    }
}

int main(int argc, char **argv) {
    cfg_t c = {0};
    c.mode=MODE_SPLIT; c.iters=10; c.warmup=3; c.nvml_on=1;
    c.valid_max=2048; c.mem_frac=0.90; c.tag=""; c.measure_xfer=1;
    /* defaults filled after we know device count */
    int set_sizes=0, set_precs=0, set_gpus=0;

    for (int i=1;i<argc;i++){
        const char *a=argv[i];
        if (!strcmp(a,"--sizes")&&i+1<argc){ c.n_sizes=parse_longs(argv[++i],c.sizes,32); set_sizes=1; }
        else if (!strcmp(a,"--precisions")&&i+1<argc){
            char b[128]; snprintf(b,sizeof b,"%s",argv[++i]); c.n_precs=0;
            for(char*t=strtok(b,",");t;t=strtok(NULL,",")){
                if(!strcmp(t,"fp32"))c.precs[c.n_precs++]=PREC_FP32;
                else if(!strcmp(t,"fp64"))c.precs[c.n_precs++]=PREC_FP64;
                else if(!strcmp(t,"bf16"))c.precs[c.n_precs++]=PREC_BF16;
            } set_precs=1;
        }
        else if (!strcmp(a,"--gpus")&&i+1<argc){ long g[8]; int n=parse_longs(argv[++i],g,8);
            for(int k=0;k<n;k++) c.gpus[k]=(int)g[k];
            c.n_gpus=n; set_gpus=1; }
        else if (!strcmp(a,"--mode")&&i+1<argc){ c.mode=strcmp(argv[++i],"replicas")?MODE_SPLIT:MODE_REPLICAS; }
        else if (!strcmp(a,"--engine")&&i+1<argc){
            char b[64]; snprintf(b,sizeof b,"%s",argv[++i]); c.n_engines=0;
            for(char*t=strtok(b,",");t&&c.n_engines<2;t=strtok(NULL,",")){
                if(!strcmp(t,"dense"))  c.engines[c.n_engines++]=ENG_DENSE;
                else if(!strcmp(t,"sparse")) c.engines[c.n_engines++]=ENG_SPARSE;
            }
        }
        else if (!strcmp(a,"--density")&&i+1<argc){
            char b[128]; snprintf(b,sizeof b,"%s",argv[++i]); c.n_dens=0;
            for(char*t=strtok(b,",");t&&c.n_dens<8;t=strtok(NULL,",")) c.densities[c.n_dens++]=atof(t);
        }
        else if (!strcmp(a,"--fill")&&i+1<argc){
            const char *f=argv[++i];
            if(!strncmp(f,"sparse",6)){ c.fill_sparse=1;
                const char*cl=strchr(f,':'); c.fill_density=cl?atof(cl+1):0.1; }
            else c.fill_sparse=0;
        }
        else if (!strcmp(a,"--iters")&&i+1<argc) c.iters=atoi(argv[++i]);
        else if (!strcmp(a,"--warmup")&&i+1<argc) c.warmup=atoi(argv[++i]);
        else if (!strcmp(a,"--fp64-tensor")) c.fp64_tensor=1;
        else if (!strcmp(a,"--fp64-cudacore")) c.fp64_cudacore=1;
        else if (!strcmp(a,"--no-nvml")) c.nvml_on=0;
        else if (!strcmp(a,"--validate")) c.validate=1;
        else if (!strcmp(a,"--validate-max")&&i+1<argc) c.valid_max=atoi(argv[++i]);
        else if (!strcmp(a,"--no-transfers")) c.measure_xfer=0;
        else if (!strcmp(a,"--mem-frac")&&i+1<argc) c.mem_frac=atof(argv[++i]);
        else if (!strcmp(a,"--csv")&&i+1<argc) c.csv=argv[++i];
        else if (!strcmp(a,"--tag")&&i+1<argc) c.tag=argv[++i];
        else if (!strcmp(a,"--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr,"unknown arg: %s\n",a); usage(argv[0]); return 2; }
    }

    int ndev=0;
    if (cudaGetDeviceCount(&ndev)!=cudaSuccess || ndev<1) {
        fprintf(stderr,"no CUDA devices (is the driver loaded? run nvidia-smi)\n"); return 1;
    }
    if (!set_sizes){ long d[]={512,1024,2048,4096,8192,16384,32768,65536};
        c.n_sizes=8; memcpy(c.sizes,d,sizeof d); }
    if (!set_precs){ c.precs[0]=PREC_FP32;c.precs[1]=PREC_FP64;c.precs[2]=PREC_BF16;c.n_precs=3; }
    if (!set_gpus){ c.n_gpus=0; for(int g=1;g<=ndev&&g<=4;g++) c.gpus[c.n_gpus++]=g; }
    if (c.n_engines==0){ c.engines[0]=ENG_DENSE; c.n_engines=1; }
    if (c.n_dens==0){ double dd[]={0.01,0.05,0.1,0.25}; c.n_dens=4; memcpy(c.densities,dd,sizeof dd); }
    if (c.fill_sparse && c.fill_density<=0) c.fill_density=0.1;

    /* total memory of device 0 (assume homogeneous A100s) */
    CUDA_CHECK(cudaSetDevice(0));
    size_t mfree=0,mtotal=0; cudaMemGetInfo(&mfree,&mtotal);
    long cap = (long)((double)mtotal * c.mem_frac);

    int nvml_ready=0;
    if (c.nvml_on && nvmlInit_v2()==NVML_SUCCESS) nvml_ready=1;
    else if (c.nvml_on) fprintf(stderr,"warning: nvmlInit failed; telemetry off\n");

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
    int cudart=0; cudaRuntimeGetVersion(&cudart);

    FILE *csv=NULL; int new_file=1;
    if (c.csv){ FILE *t=fopen(c.csv,"r"); if(t){fseek(t,0,SEEK_END); if(ftell(t)>0)new_file=0; fclose(t);}
        csv=fopen(c.csv,"a"); if(!csv){ perror("csv"); } }
    const char *HDR="tag,ts,host,cudart,mode,precision,gpus,M,N,K,iters,warmup,"
        "agg_tflops,per_gpu_tflops,pct_peak,wall_ms,"
        "sm_mhz_avg,sm_mhz_max,mem_mhz_avg,power_w_avg,power_w_max,temp_c_max,throttle,"
        "mem_gb_per_gpu,fp64_tensor,validated,max_rel_err,"
        "h2d_ms,d2h_ms,compute_ms,e2e_ms,h2d_gbps,d2h_gbps,eff_tflops,transfer_pct,"
        "engine,density,nnz,status\n";
    if (csv && new_file) fputs(HDR,csv);
    printf("# GPU GEMM bench  host=%s devices=%d mode=%s cudart=%d cap=%.1fGB\n",
        host, ndev, c.mode==MODE_SPLIT?"split":"replicas", cudart, (double)cap/1e9);
    printf("%-6s %-5s %-3s %6s %6s %9s %9s %6s %8s %8s %6s %7s %s\n",
        "eng","prec","G","N","dens","aggTF","effTF","pk%","comp_ms","e2e_ms","xfer%","SMclk","status");

    for (int si=0; si<c.n_sizes; si++){
        long S=c.sizes[si];
        /* host operands: B is dense fp32 for both engines; A (dense) is needed
           only for a dense-engine run; --fill sparse zeroes the dense data. */
        int has_dense=0;
        for(int ei=0;ei<c.n_engines;ei++) if(c.engines[ei]==ENG_DENSE) has_dense=1;
        float *hA=NULL, *hB=(float*)my_malloc((size_t)S*S*sizeof(float));
        if(!hB){ fprintf(stderr,"host OOM (B) at S=%ld\n",S); continue; }
        fill_rand(hB,(size_t)S*S,2);
        if(has_dense){
            hA=(float*)my_malloc((size_t)S*S*sizeof(float));
            if(!hA){ fprintf(stderr,"host OOM (A) at S=%ld\n",S); my_free(hB); continue; }
            fill_rand(hA,(size_t)S*S,1);
            if(c.fill_sparse){
                sparsify_inplace(hA,(size_t)S*S,c.fill_density,11);
                sparsify_inplace(hB,(size_t)S*S,c.fill_density,22);
            }
        }

        for(int ei=0; ei<c.n_engines; ei++){
            engine_t eng=c.engines[ei];
            if(eng==ENG_DENSE){
                double dens = c.fill_sparse ? c.fill_density : 1.0;
                long approx_nnz = (long)llround(dens*(double)S*(double)S);
                for(int pi=0;pi<c.n_precs;pi++)
                    for(int gi=0;gi<c.n_gpus;gi++){
                        int G=c.gpus[gi]; if(G>ndev) continue;
                        run_config(&c,csv,host,cudart,cap,nvml_ready,S,c.precs[pi],G,
                                   ENG_DENSE,dens,approx_nnz, hA,hB,NULL,NULL,NULL);
                    }
            } else {
                for(int di=0; di<c.n_dens; di++){
                    double d=c.densities[di];
                    int *hRow=NULL,*hCol=NULL; float *hVals=NULL; long nnz=0;
                    int gr=gen_csr(S,d,&hRow,&hCol,&hVals,&nnz);
                    if(gr!=0){
                        printf("sparse %-5s %-3s %6ld %6.3f  CSR-gen-skip (%s)\n",
                               "","",S,d, gr==-2?"nnz>2^31":"host-OOM");
                        continue;
                    }
                    for(int pi=0;pi<c.n_precs;pi++)
                        for(int gi=0;gi<c.n_gpus;gi++){
                            int G=c.gpus[gi]; if(G>ndev) continue;
                            run_config(&c,csv,host,cudart,cap,nvml_ready,S,c.precs[pi],G,
                                       ENG_SPARSE,d,nnz, NULL,hB,hRow,hCol,hVals);
                        }
                    my_free(hRow); my_free(hCol); my_free(hVals);
                }
            }
        }
        my_free(hA); my_free(hB);
    }

    if (csv) fclose(csv);
    if (nvml_ready) nvmlShutdown();
    return 0;
}
