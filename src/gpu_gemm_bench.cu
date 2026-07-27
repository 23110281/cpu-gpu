/* gpu_gemm_bench.cu -- multi-GPU cuBLAS GEMM benchmark (nvcc, CUDA C++).
 *
 * Companion to gpu_gemm_bench.c (gcc). Key improvements vs the C version:
 *
 *  1. PINNED host memory (cudaMallocHost/cudaFreeHost) for A and B operands.
 *     H2D/D2H transfers achieve near-peak PCIe Gen4 bandwidth (~25 GB/s)
 *     instead of the ~2-6 GB/s seen with pageable malloc.
 *
 *  2. CUDA Events for ALL three timing phases (GPU-side clock, jitter-free):
 *       e_h2d_start/end  -- cudaMemcpyAsync H2D (A + B slice)
 *       e0/e1            -- compute kernel batch
 *       e_d2h_start/end  -- cudaMemcpyAsync D2H (C slice)
 *
 *  3. Host-side BF16/FP64 conversion loops (OpenMP parallelized).
 *     Conversion runs on host CPU before the timed H2D copy, so it does NOT pollute
 *     the PCIe bandwidth measurement and ensures we only ship native-sized data.
 *
 *  4. Fixes the void* / return-1 type mismatch in worker threads.  Workers set
 *     j->rc and return NULL; CUDA errors jump to a cleanup label via
 *     CUDA_CHECK_J / CUBLAS_CHECK_J / SPARSE_CHECK_J macros.
 *
 *  5. GPU utilisation (%) now printed in stdout and CSV (was collected by NVML
 *     in the C version but the util_sum field was never used in any output).
 *
 *  6. Default sweep starts at size=256 (was 512 in C version) per user request.
 *
 *  7. cudaStreamSynchronize after cudaMemset(dC) before cuBLAS stream starts --
 *     eliminates the latent race between the default-stream zero-fill and the
 *     compute stream.
 *
 * All benchmark logic (multi-GPU split, NVML, CSV, validation, sparse SpMM)
 * is preserved from gpu_gemm_bench.c; only the implementation details change.
 *
 * Compile:  make bench_cuda
 * Run:      ./bin/gpu_gemm_bench_cu --help
 */

/* -- includes -------------------------------------------------------------- */
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cusparse.h>
#include <cusparseLt.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>          /* __nv_bfloat16, __float2bfloat16_rn        */
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
#include <sys/mman.h>
#include "bf16_cvt.h"

/* -- error-checking macros ------------------------------------------------- */
/* J-variants set j->rc and jump to a cleanup label -- safe inside void*.     */
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

/* For use in main / helper context where exit() is acceptable.               */
#define CUDA_CHECK(expr) do {                                                 \
    cudaError_t _e = (expr);                                                  \
    if (_e != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA %s:%d %s -> %s\n",                             \
                __FILE__, __LINE__, #expr, cudaGetErrorString(_e));          \
        exit(1);                                                              \
    }                                                                         \
} while (0)

/* -- precision / mode / engine types -------------------------------------- */
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
static cudaDataType out_type(prec_t p) {
    return p == PREC_FP64 ? CUDA_R_64F : CUDA_R_32F;
}

/* A100-SXM4-40GB dense peak TFLOPS per GPU.
 * FP32 : 19.5  CUDA cores (TF32 disabled by CUBLAS_PEDANTIC_MATH).
 * FP64 : 9.7   CUDA cores  (default in this build -- cuBLASLt non-tensor algo).
 *        19.5  DMMA tensor cores  (opt-in via --fp64-tensor).
 * BF16 : 312.0 Tensor cores.                                                  *
 *                                                                              *
 * WHY NOT CUBLAS_COMPUTE_64F_PEDANTIC?                                         *
 * On cuBLAS 12.x / A100 the PEDANTIC flag is documented to prevent DMMA but   *
 * in practice still dispatches the FP64 DMMA pipe (README #precision-routing,  *
 * observed 135-145% of 9.7 TFLOPS CUDA-core peak, i.e. ~13-14 TFLOPS).        *
 * The only reliable mechanism is cuBLASLt algorithm enumeration: every algo    *
 * is queried for CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS and rejected if the    *
 * CUBLASLT_NUMERICAL_IMPL_FLAGS_TENSOR_OP_MASK bit is set (lt_fp64_setup).    *
 * Verify with ncu:                                                             *
 *   ncu --metrics sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active,\
 *                sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active   *
 *       ./bin/gpu_gemm_bench_cu --sizes 8192 --precisions fp64 --gpus 1 --no-nvml
 * DMMA% ~= 0, FMA% > 0  =>  CUDA cores confirmed.                             */
static double peak_tflops(prec_t p, int fp64_tensor) {
    switch (p) {
    case PREC_FP32: return 19.5;
    case PREC_FP64: return fp64_tensor ? 19.5 : 9.7;
    case PREC_BF16: return 312.0;
    }
    return 1.0;
}

/* -- GPU conversion kernels (nvcc only) ----------------------------------- */
/* Legacy GPU-side conversion kernels (no longer used since conversion is pre-formatted on the CPU before the transfer)
__global__ void kern_f32_to_bf16(const float * __restrict__ src,
                                  __nv_bfloat16 * __restrict__ dst, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16_rn(src[i]);
}
__global__ void kern_f32_to_fp64(const float * __restrict__ src,
                                  double * __restrict__ dst, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (double)src[i];
}
static void launch_f32_to_bf16(const float *src, void *dst, size_t n, cudaStream_t st) {
    int tpb = 256, blk = (int)((n + tpb - 1) / tpb);
    kern_f32_to_bf16<<<blk, tpb, 0, st>>>(src, (__nv_bfloat16 *)dst, n);
}
static void launch_f32_to_fp64(const float *src, void *dst, size_t n, cudaStream_t st) {
    int tpb = 256, blk = (int)((n + tpb - 1) / tpb);
    kern_f32_to_fp64<<<blk, tpb, 0, st>>>(src, (double *)dst, n);
}
*/

/* -- deterministic host RNG ----------------------------------------------- */
static inline float hashf(uint64_t i, uint64_t seed) {
    uint64_t x = i * 0x9E3779B97F4A7C15ULL + seed;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL; x ^= x >> 31;
    return (float)((double)(uint32_t)(x >> 32) / (double)UINT32_MAX) * 2.0f - 1.0f;
}
static void fill_rand(float *p, size_t n, uint64_t seed) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) p[i] = hashf(i, seed);
}
static inline double hashu(uint64_t i, uint64_t seed) {
    uint64_t x = (i * 0x9E3779B97F4A7C15ULL) ^ seed;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL; x ^= x >> 31;
    return (double)(uint32_t)(x >> 32) / ((double)UINT32_MAX + 1.0);
}
static void sparsify_inplace(float *p, size_t n, double density, uint64_t seed) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++)
        if (hashu(i, seed) >= density) p[i] = 0.0f;
}

/* Build sparse SxS CSR with ~density nonzeros/row, sorted distinct columns.  */
static int gen_csr(long S, double density,
                   int **rowOff_o, int **colInd_o, float **vals_o, long *nnz_o) {
    int npr = (int)llround(density * (double)S);
    if (npr < 1) npr = 1; if (npr > S) npr = (int)S;
    long nnz = (long)npr * S;
    if (nnz >= (long)2147483647) return -2;
    int   *rowOff = (int *)  malloc((size_t)(S+1) * sizeof(int));
    int   *colInd = (int *)  malloc((size_t)nnz   * sizeof(int));
    float *vals   = (float *)malloc((size_t)nnz   * sizeof(float));
    if (!rowOff || !colInd || !vals) { free(rowOff); free(colInd); free(vals); return -1; }
    long stride = S / npr; if (stride < 1) stride = 1;
#pragma omp parallel for schedule(static)
    for (long r = 0; r < S; r++) {
        rowOff[r] = (int)(r * (long)npr);
        for (int jj = 0; jj < npr; jj++) {
            uint64_t x = ((uint64_t)r * 1469598103934665603ULL)
                       ^ ((uint64_t)jj * 1099511628211ULL);
            x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33;
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

/* -- NVML live sampler ----------------------------------------------------- */
typedef struct {
    nvmlDevice_t dev;
    int          valid;
    std::atomic<int> stop;
    unsigned long long sm_sum, mem_sum, pow_sum, util_sum; /* MHz, MHz, mW, % */
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

/* -- host-side BF16 helpers (validation reference only) ------------------- */
static inline uint16_t f32_to_bf16_host(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    const uint32_t exp8 = (u >> 23) & 0xFFu;
    if (exp8 == 0xFFu) {
        const uint32_t mant = u & 0x007FFFFFu;
        return mant ? (uint16_t)((u >> 16) | 0x0040u) : (uint16_t)(u >> 16);
    }
    const uint32_t lsb = (u >> 16) & 1u, bias = 0x7FFFu + lsb;
    u += bias; return (uint16_t)(u >> 16);
}
static inline float bf16_to_f32_host(uint16_t b) {
    uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f;
}

/* -- per-GPU job descriptor ------------------------------------------------ */
typedef struct {
    /* inputs */
    int      dev;
    long     S, col0, ncols;
    prec_t   prec;
    engine_t engine;
    double   density; long nnz;
    const int   *hRowOff, *hColInd;
    const float *hVals;
    int      fp64_tensor, fp64_cudacore;
    int      warmup, iters;
    /* PINNED host operands (full A and B in fp32) */
    const float *hA_pin;
    const float *hB_pin;
    void  *hA_nat;
    void  *hB_nat;
    void  *hC_pin;
    const void  *hVals_nat;
    int      nvml_on, validate, valid_max, measure_xfer;
    pthread_barrier_t *bar;
    /* outputs */
    int      rc;
    double   batch_ms, h2d_ms, d2h_ms, max_rel_err, mem_gb;
    sampler_t smp;
} job_t;

/* -- cuBLASLt non-tensor FP64 (--fp64-cudacore) --------------------------- */
typedef struct {
    cublasLtHandle_t       lt;
    cublasLtMatmulDesc_t   desc;
    cublasLtMatrixLayout_t la, lb, lc;
    cublasLtMatmulAlgo_t   algo;
    int                    have_algo;
    void                  *ws; size_t ws_size;
} lt_fp64_t;

static int lt_fp64_setup(lt_fp64_t *x, int M, int N, int K) {
    memset(x, 0, sizeof *x);
    if (cublasLtCreate(&x->lt) != CUBLAS_STATUS_SUCCESS) return -1;
    if (cublasLtMatmulDescCreate(&x->desc, CUBLAS_COMPUTE_64F,
                                 CUDA_R_64F) != CUBLAS_STATUS_SUCCESS) return -1;
    cublasOperation_t opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(x->desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof opN);
    cublasLtMatmulDescSetAttribute(x->desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof opN);
    if (cublasLtMatrixLayoutCreate(&x->la, CUDA_R_64F, (uint64_t)M, (uint64_t)K, M)
            != CUBLAS_STATUS_SUCCESS) return -1;
    if (cublasLtMatrixLayoutCreate(&x->lb, CUDA_R_64F, (uint64_t)K, (uint64_t)N, K)
            != CUBLAS_STATUS_SUCCESS) return -1;
    if (cublasLtMatrixLayoutCreate(&x->lc, CUDA_R_64F, (uint64_t)M, (uint64_t)N, M)
            != CUBLAS_STATUS_SUCCESS) return -1;
    x->ws_size = (size_t)64*1024*1024;
    if (cudaMalloc(&x->ws, x->ws_size) != cudaSuccess) { x->ws = NULL; x->ws_size = 0; }
    int ids[256]; int nids = 0;
    if (cublasLtMatmulAlgoGetIds(x->lt, CUBLAS_COMPUTE_64F, CUDA_R_64F,
            CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F,
            256, ids, &nids) != CUBLAS_STATUS_SUCCESS) return -1;
    for (int i = 0; i < nids; i++) {
        cublasLtMatmulAlgo_t algo;
        if (cublasLtMatmulAlgoInit(x->lt, CUBLAS_COMPUTE_64F, CUDA_R_64F,
                CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F,
                ids[i], &algo) != CUBLAS_STATUS_SUCCESS) continue;
        uint64_t flags = 0; size_t got = 0;
        if (cublasLtMatmulAlgoCapGetAttribute(&algo,
                CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
                &flags, sizeof flags, &got) != CUBLAS_STATUS_SUCCESS) continue;
        if (flags & CUBLASLT_NUMERICAL_IMPL_FLAGS_TENSOR_OP_MASK) continue;
        cublasLtMatmulHeuristicResult_t heur; memset(&heur, 0, sizeof heur);
        if (cublasLtMatmulAlgoCheck(x->lt, x->desc, x->la, x->lb, x->lc, x->lc,
                &algo, &heur) != CUBLAS_STATUS_SUCCESS) continue;
        if (heur.state != CUBLAS_STATUS_SUCCESS) continue;
        if (heur.workspaceSize > x->ws_size) continue;
        x->algo = algo; x->have_algo = 1; break;
    }
    return x->have_algo ? 0 : -2;
    (void)M; (void)N; (void)K;
}
static cublasStatus_t do_gemm_lt_fp64(lt_fp64_t *x, cudaStream_t st,
                                      const void *dA, const void *dB, void *dC) {
    static const double one = 1.0, zero = 0.0;
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

/* -- cuBLAS GEMM dispatch -------------------------------------------------- */
/*
 * FP32: CUBLAS_COMPUTE_32F_PEDANTIC disables TF32 and forces IEEE-754 FP32
 *       through CUDA FFMA cores.  Verified: achieved TFLOPS <= 19.5 (CUDA-core
 *       peak) on A100.  CUBLAS_PEDANTIC_MATH on the handle is set as a second
 *       guard (see gpu_worker setup).
 *
 * FP64 (fp64_tensor=true, --fp64-tensor): CUBLAS_COMPUTE_64F +
 *       GEMM_DEFAULT_TENSOR_OP explicitly selects the FP64 DMMA pipe (19.5
 *       TFLOPS A100 peak). This is the *opt-in* path.
 *       NOTE: do NOT pass fp64_tensor=false here for FP64 -- the PEDANTIC
 *       variant does not reliably block DMMA on cuBLAS 12.x.  Non-tensor FP64
 *       goes through do_gemm_lt_fp64() / cuBLASLt instead.
 *
 * BF16: CUBLAS_COMPUTE_32F + GEMM_DEFAULT_TENSOR_OP routes BF16 inputs through
 *       the BF16 Tensor Core HMMA pipe (312 TFLOPS A100 peak) with FP32
 *       accumulation.  This is the standard / intended path for BF16.
 */
static cublasStatus_t do_gemm(cublasHandle_t h, prec_t p, int fp64_tensor,
                               int M, int N, int K,
                               const void *dA, const void *dB, void *dC) {
    static const float  f1 = 1.0f, f0 = 0.0f;
    static const double d1 = 1.0,  d0 = 0.0;
    switch (p) {
    case PREC_FP32:
        /* CUBLAS_PEDANTIC_MATH (set on handle) + COMPUTE_32F_PEDANTIC:
         * two independent gates that both disable TF32.  Double-guarded
         * because cublasSetMathMode() and the compute-type flag operate on
         * different dispatch layers inside cuBLAS.                           */
        return cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
            &f1, dA, CUDA_R_32F, M, dB, CUDA_R_32F, K, &f0, dC, CUDA_R_32F, M,
            CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT);
    case PREC_FP64:
        /* ONLY called when fp64_tensor=true (DMMA opt-in path).             *
         * Non-tensor FP64 is handled by do_gemm_lt_fp64() / cuBLASLt.       *
         * Assertion guard: if this branch is reached without fp64_tensor,    *
         * DMMA may still run -- that is a caller bug, not a cuBLAS guarantee. */
        assert(fp64_tensor && "non-tensor FP64 must go via do_gemm_lt_fp64");
        return cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
            &d1, dA, CUDA_R_64F, M, dB, CUDA_R_64F, K, &d0, dC, CUDA_R_64F, M,
            CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    case PREC_BF16:
        return cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
            &f1, dA, CUDA_R_16BF, M, dB, CUDA_R_16BF, K, &f0, dC, CUDA_R_32F, M,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

/* -- CUDA-event elapsed time ----------------------------------------------- */
static float event_ms(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f; cudaEventElapsedTime(&ms, a, b); return ms;
}

/* -- sampled correctness checks ------------------------------------------- */
static double validate_slice(job_t *j, const void *hC) {
    const long S = j->S, ncols = j->ncols; double worst = 0.0;
    uint64_t rng = 0xC0FFEEull ^ (uint64_t)j->dev;
    for (int s = 0; s < 128; s++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        long i  = (long)(rng % (uint64_t)S);
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        long jc = (long)(rng % (uint64_t)ncols), gj = j->col0 + jc;
        double ref = 0.0;
        for (long k = 0; k < S; k++) {
            double a = (double)j->hA_pin[i + k*S], b = (double)j->hB_pin[k + gj*S];
            if (j->prec == PREC_BF16) {
                a = (double)bf16_to_f32_host(f32_to_bf16_host(j->hA_pin[i + k*S]));
                b = (double)bf16_to_f32_host(f32_to_bf16_host(j->hB_pin[k + gj*S]));
            }
            ref += a * b;
        }
        double got = (j->prec == PREC_FP64) ? ((double*)hC)[i + jc*S]
                                             : (double)((float*)hC)[i + jc*S];
        double denom = fabs(ref) > 1e-12 ? fabs(ref) : 1e-12;
        double rel   = fabs(got - ref) / denom;
        if (rel > worst) worst = rel;
    }
    return worst;
}

static inline float f32_to_tf32_host(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    // Truncate the 13 LSBs of mantissa to emulate TF32 (19-bit precision)
    u &= 0xFFFFE000u;
    float f; memcpy(&f, &u, 4); return f;
}

static double validate_structured_spmm(job_t *j, const void *hA_pruned, const void *hC) {
    const long S = j->S, ncols = j->ncols; double worst = 0.0;
    uint64_t rng = 0x5EED5EEDull ^ (uint64_t)j->dev;
    for (int s = 0; s < 128; s++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        long i  = (long)(rng % (uint64_t)S);
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        long jc = (long)(rng % (uint64_t)ncols), gj = j->col0 + jc;
        double ref = 0.0;
        for (long k = 0; k < S; k++) {
            double a, b;
            if (j->prec == PREC_BF16) {
                a = (double)bf16_to_f32_host(((const uint16_t*)hA_pruned)[i * S + k]);
                b = (double)bf16_to_f32_host(f32_to_bf16_host(j->hB_pin[k + gj*S]));
            } else {
                // FP32 mode uses TF32 on Sparse Tensor Cores, so emulate TF32 truncation
                a = (double)f32_to_tf32_host(((const float*)hA_pruned)[i * S + k]);
                b = (double)f32_to_tf32_host(j->hB_pin[k + gj*S]);
            }
            ref += a * b;
        }
        /* For sparse BF16: output is BF16 (CUDA_R_16BF), stored as uint16_t.
         * For sparse FP32: output is FP32.  FP64 sparse is rejected earlier. */
        double got;
        if (j->prec == PREC_BF16)
            got = (double)bf16_to_f32_host(((const uint16_t*)hC)[i + jc*S]);
        else
            got = (j->prec == PREC_FP64) ? ((double*)hC)[i + jc*S]
                                         : (double)((const float*)hC)[i + jc*S];
        double denom = fabs(ref) > 1e-12 ? fabs(ref) : 1e-12;
        double rel   = fabs(got - ref) / denom;
        if (rel > worst) worst = rel;
    }
    return worst;
}

/* -- dense GPU worker ------------------------------------------------------ */
/*
 * Transfer flow:
 *  FP32: cudaMemcpyAsync pinned->dA/dB (timed), no staging.
 *  BF16/FP64: cudaMemcpyAsync pinned->dA_f32/dB_f32 staging (timed),
 *              then kern_f32_to_bf16 / kern_f32_to_fp64 on same stream (not timed).
 *  D2H: cudaMemcpyAsync dC->pinned hC_pin (timed).
 */
static void *gpu_worker(void *arg) {
    job_t *j = (job_t *)arg;
    j->rc = 0; j->max_rel_err = -1.0;
    j->batch_ms = j->h2d_ms = j->d2h_ms = 0.0;
    const long S = j->S, ncols = j->ncols;
    const int  M = (int)S, N = (int)ncols, K = (int)S;

    cublasHandle_t h = NULL; cudaStream_t st = NULL;
    void  *dA = NULL, *dB = NULL, *dC = NULL;
    float *dA_f32 = NULL, *dB_f32 = NULL;
    cudaEvent_t e_h2d0 = NULL, e_h2d1 = NULL;
    cudaEvent_t e0 = NULL, e1 = NULL;
    cudaEvent_t e_d2h0 = NULL, e_d2h1 = NULL;
    lt_fp64_t lt = {}; int use_lt = 0;
    pthread_t smp_th; int smp_started = 0;
    void *hC_pin = NULL;
    void *hA_nat = (void*)j->hA_nat;
    void *hB_nat = (void*)j->hB_nat;
    size_t eb = in_bytes(j->prec);
    if (cudaSetDevice(j->dev) != cudaSuccess) { j->rc = -1; goto barrier_out; }
    CUDA_CHECK_J(cudaStreamCreate(&st), barrier_out);
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { j->rc = -1; goto barrier_out; }
    cublasSetStream(h, st);
    {
        /* CUBLAS_PEDANTIC_MATH: disables TF32 on the cuBLAS handle level.
         * Combined with CUBLAS_COMPUTE_32F_PEDANTIC in do_gemm() this gives
         * two independent guards for FP32 (TF32 disabled at both the handle
         * dispatch layer and the kernel-selection layer).
         * For BF16 and FP64-DMMA we need DEFAULT_MATH to allow tensor ops.   */
        cublasMath_t mm = CUBLAS_PEDANTIC_MATH;          /* FP32: both gates  */
        if (j->prec == PREC_BF16)                         /* BF16: tensor OK   */
            mm = CUBLAS_DEFAULT_MATH;
        if (j->prec == PREC_FP64 && j->fp64_tensor)       /* FP64 DMMA opt-in  */
            mm = CUBLAS_DEFAULT_MATH;
        cublasSetMathMode(h, mm);
    }
    /*
     * FP64 CUDA-core path (DEFAULT for FP64 in this build):
     *
     * cuBLASLt enumerates every available algorithm and rejects any that have
     * CUBLASLT_NUMERICAL_IMPL_FLAGS_TENSOR_OP_MASK set in its capability flags.
     * The first surviving algo is validated against the problem size before use
     * (cublasLtMatmulAlgoCheck).  This provides a hardware-level guarantee that
     * no DMMA instruction is issued -- unlike CUBLAS_COMPUTE_64F_PEDANTIC which
     * is a scheduling hint that cuBLAS 12.x ignores for sm_80.
     *
     * If no non-tensor algo exists on this GPU/CUDA version, we abort loudly
     * rather than silently fall back to DMMA (that would corrupt the numbers).
     */
    if (j->prec == PREC_FP64 && !j->fp64_tensor) {
        int lr = lt_fp64_setup(&lt, M, N, K);
        if (lr != 0) {
            fprintf(stderr,
                "[dev %d] FP64 CUDA-core: lt_fp64_setup failed (rc=%d).\n"
                "  No non-tensor FP64 algorithm found for M=%d N=%d K=%d.\n"
                "  Add --fp64-tensor to benchmark DMMA instead.\n",
                j->dev, lr, M, N, K);
            j->rc = -1; goto barrier_out;
        }
        use_lt = 1;
    }

    CUDA_CHECK_J(cudaMalloc(&dA, (size_t)S*S    *in_bytes(j->prec)), barrier_out);
    CUDA_CHECK_J(cudaMalloc(&dB, (size_t)S*ncols*in_bytes(j->prec)), barrier_out);
    CUDA_CHECK_J(cudaMalloc(&dC, (size_t)S*ncols*out_bytes(j->prec)), barrier_out);
    CUDA_CHECK_J(cudaMemset(dC, 0, (size_t)S*ncols*out_bytes(j->prec)), barrier_out);
    /* Sync default-stream zero-fill before cuBLAS stream. */
    CUDA_CHECK_J(cudaStreamSynchronize(st), barrier_out);

    CUDA_CHECK_J(cudaEventCreate(&e_h2d0), barrier_out);
    CUDA_CHECK_J(cudaEventCreate(&e_h2d1), barrier_out);
    CUDA_CHECK_J(cudaEventCreate(&e0),     barrier_out);
    CUDA_CHECK_J(cudaEventCreate(&e1),     barrier_out);
    CUDA_CHECK_J(cudaEventCreate(&e_d2h0), barrier_out);
    CUDA_CHECK_J(cudaEventCreate(&e_d2h1), barrier_out);

    cudaEventRecord(e_h2d0, st);
    CUDA_CHECK_J(cudaMemcpyAsync(dA, hA_nat, (size_t)S*S*eb, cudaMemcpyHostToDevice, st), barrier_out);
    CUDA_CHECK_J(cudaMemcpyAsync(dB, hB_nat, (size_t)S*ncols*eb, cudaMemcpyHostToDevice, st), barrier_out);
    cudaEventRecord(e_h2d1, st);
    CUDA_CHECK_J(cudaStreamSynchronize(st), barrier_out);
    j->h2d_ms = (double)event_ms(e_h2d0, e_h2d1);

    /* warmup */
    for (int w = 0; w < j->warmup; w++) {
        cublasStatus_t gs = use_lt
            ? do_gemm_lt_fp64(&lt, st, dA, dB, dC)
            : do_gemm(h, j->prec, j->fp64_tensor, M, N, K, dA, dB, dC);
        if (gs != CUBLAS_STATUS_SUCCESS) j->rc = -1;
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
            cublasStatus_t gs = use_lt
                ? do_gemm_lt_fp64(&lt, st, dA, dB, dC)
                : do_gemm(h, j->prec, j->fp64_tensor, M, N, K, dA, dB, dC);
            if (gs != CUBLAS_STATUS_SUCCESS) loop_err = 1;
        }
        cudaEventRecord(e1, st);
        cudaEventSynchronize(e1);
        if (smp_started) { j->smp.stop.store(1, std::memory_order_relaxed); pthread_join(smp_th, NULL); }
        if (loop_err) { j->rc = -1; goto done; }
        j->batch_ms = (double)event_ms(e0, e1);
    }

    if (j->measure_xfer || (j->validate && S <= j->valid_max)) {
        size_t cbytes = (size_t)S * ncols * out_bytes(j->prec);
        hC_pin = j->hC_pin;
        if (hC_pin) {
            cudaEventRecord(e_d2h0, st);
            cudaMemcpyAsync(hC_pin, dC, cbytes, cudaMemcpyDeviceToHost, st);
            cudaEventRecord(e_d2h1, st);
            cudaStreamSynchronize(st);
            j->d2h_ms = (double)event_ms(e_d2h0, e_d2h1);
            if (j->validate && S <= j->valid_max)
                j->max_rel_err = validate_slice(j, hC_pin);
        }
    }

done:
    if (e_h2d0) cudaEventDestroy(e_h2d0);
    if (e_h2d1) cudaEventDestroy(e_h2d1);
    if (e0)     cudaEventDestroy(e0);
    if (e1)     cudaEventDestroy(e1);
    if (e_d2h0) cudaEventDestroy(e_d2h0);
    if (e_d2h1) cudaEventDestroy(e_d2h1);
    if (dA_f32) cudaFree(dA_f32);
    if (dB_f32) cudaFree(dB_f32);
    if (dA) cudaFree(dA); if (dB) cudaFree(dB); if (dC) cudaFree(dC);
    lt_fp64_free(&lt);
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
    if (dA_f32) cudaFree(dA_f32);
    if (dB_f32) cudaFree(dB_f32);
    if (dA) cudaFree(dA); if (dB) cudaFree(dB); if (dC) cudaFree(dC);
    lt_fp64_free(&lt);
    if (h)  cublasDestroy(h);
    if (st) cudaStreamDestroy(st);
    pthread_barrier_wait(j->bar);
    return NULL;
}

/* cuSPARSELt on sm_80 (A100): BF16 inputs must use BF16 output with
 * CUSPARSE_ORDER_COL for C/D.  FP32 output with COL-major C returns
 * CUSPARSE_STATUS_NOT_SUPPORTED (10) on this driver/cuSPARSELt version.
 *
 * A device that raises an async (context-level) CUDA fault during the
 * timed matmul loop is NOT reset here.  cudaDeviceReset() mid-function,
 * followed by the teardown calls below that reference handles/pointers
 * from the just-destroyed context, is undefined behaviour -- that pattern
 * previously caused unrecoverable corruption of the whole node requiring a
 * physical power cycle.  A context in the sticky-error state left behind by
 * an async fault safely rejects further API calls on its own; the device
 * is instead marked poisoned (below) and skipped for the rest of this
 * process's sparse jobs.                                                  */
static inline cudaDataType sparse_out_type(prec_t p) {
    return p == PREC_BF16 ? CUDA_R_16BF : out_type(p);
}
static inline size_t sparse_out_bytes(prec_t p) {
    return p == PREC_BF16 ? 2 : out_bytes(p);
}

/* Sparse-engine device poisoning: an async (context-level) CUDA fault during
 * one sparse job leaves that physical GPU's driver-visible state unconfirmed
 * safe for reuse.  Rather than blindly re-entering it on the next
 * back-to-back sparse job (the pattern observed to corrupt the node), skip
 * that device for the remainder of this process. Indexed by physical device
 * id, matches cfg_t.gpus[8] / job_t.dev range. */
static std::atomic<int> g_sparse_dev_poisoned[8] = {};

/* cuSPARSELt (unlike core cuBLAS/cuSPARSE) is young enough that its
 * process-wide first-touch behaviour on cusparseLtInit()/descriptor/plan
 * creation is not something this codebase can assume is safe to race across
 * threads. spmm_worker() runs one thread per GPU concurrently; this mutex
 * serializes only the one-time per-job setup (init through compress), never
 * the timed matmul loop, so it does not affect measured throughput.        */
static std::mutex g_cusparselt_setup_mutex;

/* -- sparse SpMM worker ---------------------------------------------------- */
static void *spmm_worker(void *arg) {
    job_t *j = (job_t *)arg;
    j->rc = 0; j->max_rel_err = -1.0;
    j->batch_ms = j->h2d_ms = j->d2h_ms = 0.0;
    const long S = j->S, ncols = j->ncols;
    int pre_barrier = 0, smp_started = 0;

    cusparseLtHandle_t handle;
    int handle_inited = 0;
    cusparseLtMatDescriptor_t matA, matB, matC;
    int descA_inited = 0, descB_inited = 0, descC_inited = 0;
    cusparseLtMatmulDescriptor_t matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t plan;
    int plan_inited = 0;

    cudaStream_t st = NULL;
    void *dA = NULL, *dB = NULL, *dC = NULL;
    void *dA_compressed = NULL, *dA_compress_buf = NULL;
    void *d_workspace = NULL;
    void *hC_pin = NULL;
    void *hA_pruned_host = NULL;
    void *hB_nat = (void*)j->hB_nat;
    size_t ev = in_bytes(j->prec);
    pthread_t smp_th;

    if (j->dev >= 0 && j->dev < 8 &&
        g_sparse_dev_poisoned[j->dev].load(std::memory_order_relaxed)) {
        fprintf(stderr,
            "[dev %d] skipping sparse job: device was marked unsafe after an "
            "async CUDA fault earlier this run. Restart the process (and check "
            "driver/GPU health) before running sparse on this device again.\n",
            j->dev);
        j->rc = -1; goto sout;
    }

    if (j->prec == PREC_FP64) {
        fprintf(stderr, "[dev %d] FP64 is not supported for structured sparsity on Tensor Cores (cuSPARSELt).\n", j->dev);
        j->rc = -1; goto sout;
    }

    if (cudaSetDevice(j->dev) != cudaSuccess) { j->rc = -1; goto sout; }
    CUDA_CHECK_J(cudaStreamCreate(&st), sout);

    CUDA_CHECK_J(cudaMalloc(&dA, (size_t)S*S    *ev), sout);
    CUDA_CHECK_J(cudaMalloc(&dB, (size_t)S*ncols*ev), sout);
    CUDA_CHECK_J(cudaMalloc(&dC, (size_t)S*ncols*sparse_out_bytes(j->prec)), sout);
    CUDA_CHECK_J(cudaMemset(dC, 0, (size_t)S*ncols*sparse_out_bytes(j->prec)), sout);
    CUDA_CHECK_J(cudaStreamSynchronize(st), sout);

    {
        cudaEvent_t e_h2d0, e_h2d1;
        CUDA_CHECK_J(cudaEventCreate(&e_h2d0), sout);
        CUDA_CHECK_J(cudaEventCreate(&e_h2d1), sout);
        cudaEventRecord(e_h2d0, st);
        CUDA_CHECK_J(cudaMemcpyAsync(dA, j->hA_nat, (size_t)S*S*ev, cudaMemcpyHostToDevice, st), sout);
        CUDA_CHECK_J(cudaMemcpyAsync(dB, hB_nat, (size_t)S*ncols*ev, cudaMemcpyHostToDevice, st), sout);
        cudaEventRecord(e_h2d1, st);
        CUDA_CHECK_J(cudaStreamSynchronize(st), sout);
        j->h2d_ms = (double)event_ms(e_h2d0, e_h2d1);
        cudaEventDestroy(e_h2d0); cudaEventDestroy(e_h2d1);
    }

    {
        /* Serialize cusparseLt's one-time per-job setup (init through
         * compress) across the per-GPU worker threads. Released
         * automatically (RAII) on every exit path, including the
         * SPARSE_CHECK_J/CUDA_CHECK_J `goto sout` error jumps below. Scope
         * ends before the warmup loop, so the timed region is never
         * serialized. */
        std::lock_guard<std::mutex> setup_lock(g_cusparselt_setup_mutex);

        SPARSE_CHECK_J(cusparseLtInit(&handle), sout);
        handle_inited = 1;

        SPARSE_CHECK_J(cusparseLtStructuredDescriptorInit(&handle, &matA, S, S, S, 16,
            cuda_type(j->prec), CUSPARSE_ORDER_ROW, CUSPARSELT_SPARSITY_50_PERCENT), sout);
        descA_inited = 1;

        SPARSE_CHECK_J(cusparseLtDenseDescriptorInit(&handle, &matB, S, ncols, S, 16,
            cuda_type(j->prec), CUSPARSE_ORDER_COL), sout);
        descB_inited = 1;

        SPARSE_CHECK_J(cusparseLtDenseDescriptorInit(&handle, &matC, S, ncols, S, 16,
            sparse_out_type(j->prec), CUSPARSE_ORDER_COL), sout);
        descC_inited = 1;

        SPARSE_CHECK_J(cusparseLtMatmulDescriptorInit(&handle, &matmul,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &matA, &matB, &matC, &matC, CUSPARSE_COMPUTE_32F), sout);

        SPARSE_CHECK_J(cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel, &matmul,
            CUSPARSELT_MATMUL_ALG_DEFAULT), sout);

        SPARSE_CHECK_J(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel), sout);
        plan_inited = 1;

        SPARSE_CHECK_J(cusparseLtSpMMAPrune(&handle, &matmul, dA, dA,
            CUSPARSELT_PRUNE_SPMMA_TILE, st), sout);

        size_t compressed_size, compressed_buf_size;
        SPARSE_CHECK_J(cusparseLtSpMMACompressedSize(&handle, &plan,
            &compressed_size, &compressed_buf_size), sout);
        CUDA_CHECK_J(cudaMalloc(&dA_compressed, compressed_size), sout);
        if (compressed_buf_size) {
            CUDA_CHECK_J(cudaMalloc(&dA_compress_buf, compressed_buf_size), sout);
        }

        SPARSE_CHECK_J(cusparseLtSpMMACompress(&handle, &plan, dA,
            dA_compressed, dA_compress_buf, st), sout);

        size_t workspace_size;
        SPARSE_CHECK_J(cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size), sout);
        if (workspace_size) {
            CUDA_CHECK_J(cudaMalloc(&d_workspace, workspace_size), sout);
        }
    }

    {
        float alpha = 1.0f, beta = 0.0f;
        for (int w = 0; w < j->warmup; w++) {
            if (cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB, &beta, dC, dC,
                    d_workspace, &st, 1) != CUSPARSE_STATUS_SUCCESS) {
                j->rc = -1;
            }
        }
    }
    CUDA_CHECK_J(cudaStreamSynchronize(st), sout);
    if (j->rc) goto sout;

    pthread_barrier_wait(j->bar); pre_barrier = 1;

    if (j->nvml_on && j->smp.valid) {
        j->smp.stop.store(0, std::memory_order_relaxed);
        if (pthread_create(&smp_th, NULL, sampler_run, &j->smp) == 0)
            smp_started = 1;
    }

    {
        int loop_err = 0;
        cudaEvent_t e0, e1;
        CUDA_CHECK_J(cudaEventCreate(&e0), sout);
        CUDA_CHECK_J(cudaEventCreate(&e1), sout);
        float alpha = 1.0f, beta = 0.0f;

        cudaEventRecord(e0, st);
        for (int it = 0; it < j->iters; it++) {
            if (cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB, &beta, dC, dC,
                    d_workspace, &st, 1) != CUSPARSE_STATUS_SUCCESS) {
                loop_err = 1;
            }
        }
        cudaEventRecord(e1, st);
        cudaEventSynchronize(e1);

        /* Stop the NVML sampler before inspecting/handling any fault below --
         * it must not still be touching this device while we decide the
         * device's fate. */
        if (smp_started) {
            j->smp.stop.store(1, std::memory_order_relaxed);
            pthread_join(smp_th, NULL);
            smp_started = 0;
        }

        /* Check for asynchronous CUDA errors that cusparseLt may not surface
         * as a return-code (e.g. context-level device errors). Do NOT call
         * cudaDeviceReset() here: resetting mid-function and then continuing
         * to call cudaFree/cudaStreamDestroy/cusparseLt*Destroy below against
         * handles from the just-destroyed context is undefined behaviour, and
         * was the actual cause of unrecoverable node corruption previously
         * observed with this benchmark. A sticky-error context safely rejects
         * further API calls on its own; just mark the device poisoned so the
         * next back-to-back sparse job on it is skipped instead of blindly
         * re-entering unconfirmed-safe state.                              */
        {
            cudaError_t aerr = cudaGetLastError();
            if (aerr != cudaSuccess) {
                fprintf(stderr, "[dev %d] async CUDA error after cusparseLt loop: %s\n",
                        j->dev, cudaGetErrorString(aerr));
                loop_err = 1;
                if (j->dev >= 0 && j->dev < 8)
                    g_sparse_dev_poisoned[j->dev].store(1, std::memory_order_relaxed);
            }
        }
        j->batch_ms = (double)event_ms(e0, e1);
        cudaEventDestroy(e0); cudaEventDestroy(e1);

        if (loop_err) { j->rc = -1; goto sout; }
    }

    if (j->measure_xfer || (j->validate && S <= j->valid_max)) {
        size_t cb = (size_t)S * ncols * sparse_out_bytes(j->prec);
        if (cudaMallocHost(&hC_pin, cb) == cudaSuccess) {
            cudaEvent_t e_d2h0, e_d2h1;
            CUDA_CHECK_J(cudaEventCreate(&e_d2h0), sout);
            CUDA_CHECK_J(cudaEventCreate(&e_d2h1), sout);
            cudaEventRecord(e_d2h0, st);
            cudaMemcpyAsync(hC_pin, dC, cb, cudaMemcpyDeviceToHost, st);
            cudaEventRecord(e_d2h1, st);
            cudaStreamSynchronize(st);
            j->d2h_ms = (double)event_ms(e_d2h0, e_d2h1);
            cudaEventDestroy(e_d2h0); cudaEventDestroy(e_d2h1);

            if (j->validate && S <= j->valid_max) {
                size_t ab = (size_t)S * S * ev;
                if (cudaMallocHost(&hA_pruned_host, ab) == cudaSuccess) {
                    cudaMemcpy(hA_pruned_host, dA, ab, cudaMemcpyDeviceToHost);
                    j->max_rel_err = validate_structured_spmm(j, hA_pruned_host, hC_pin);
                    cudaFreeHost(hA_pruned_host); hA_pruned_host = NULL;
                }
            }
            cudaFreeHost(hC_pin); hC_pin = NULL;
        }
    }

sout:
    if (smp_started) {
        j->smp.stop.store(1, std::memory_order_relaxed);
        pthread_join(smp_th, NULL);
    }
    if (plan_inited) cusparseLtMatmulPlanDestroy(&plan);
    if (descC_inited) cusparseLtMatDescriptorDestroy(&matC);
    if (descB_inited) cusparseLtMatDescriptorDestroy(&matB);
    if (descA_inited) cusparseLtMatDescriptorDestroy(&matA);
    if (handle_inited) cusparseLtDestroy(&handle);

    if (dA) cudaFree(dA);
    if (dB) cudaFree(dB);
    if (dC) cudaFree(dC);
    if (dA_compressed) cudaFree(dA_compressed);
    if (dA_compress_buf) cudaFree(dA_compress_buf);
    if (d_workspace) cudaFree(d_workspace);

    if (st) cudaStreamDestroy(st);
    if (!pre_barrier) pthread_barrier_wait(j->bar);
    return NULL;
}

/* -- config ---------------------------------------------------------------- */
typedef struct {
    long   sizes[32]; int n_sizes;
    prec_t precs[3];  int n_precs;
    int    gpus[8];   int n_gpus;
    bench_mode_t mode;
    engine_t engines[2]; int n_engines;
    double densities[8]; int n_dens;
    int    fill_sparse; double fill_density;
    int    iters, warmup, fp64_tensor, fp64_cudacore;
    int    nvml_on, validate, valid_max, measure_xfer;
    double mem_frac;
    const char *csv, *tag;
} cfg_t;

static long mem_footprint(const cfg_t *c, prec_t p, bench_mode_t m,
                           long S, int G, long *max_ncols) {
    long base = S/G, rem = S%G, ncols = base + (rem?1:0);
    *max_ncols = (m == MODE_REPLICAS) ? S : ncols;
    size_t ib = in_bytes(p), ob = out_bytes(p);
    if (m == MODE_REPLICAS)
        return (long)((size_t)ib*S*S + (size_t)ib*S*S + (size_t)ob*S*S);
    return (long)((size_t)ib*S*S + (size_t)ib*S*(*max_ncols) + (size_t)ob*S*(*max_ncols));
    (void)c;
}


static void usage(const char *p) {
    printf(
"Usage: %s [options]\n"
"  --sizes a,b,..     matrix sizes (M=N=K)         [256..65536 pow2]\n"
"  --precisions p,..  fp32,fp64,bf16               [all]\n"
"  --gpus n,..        gpu counts to test           [1..4]\n"
"  --mode m           split | replicas             [split]\n"
"  --engine e,..      dense | sparse               [dense]\n"
"  --density d,..     sparse A nonzero fraction    [0.01,0.05,0.1,0.25]\n"
"  --fill f           dense data: dense|sparse[:d] [dense]\n"
"  --iters n          timed iterations             [10]\n"
"  --warmup n         warmup iterations            [3]\n"
"  --fp64-tensor      FP64 via DMMA tensor cores (19.5 TFLOPS peak)\n"
"                     [default: FP64 uses CUDA cores, 9.7 TFLOPS peak]\n"
"  --no-nvml          disable NVML telemetry\n"
"  --validate         sampled correctness check\n"
"  --validate-max n   max size to validate         [2048]\n"
"  --no-transfers     skip timed H2D/D2H\n"
"  --mem-frac f       usable GPU mem fraction      [0.90]\n"
"  --csv path         append CSV results\n"
"  --tag s            CSV label\n"
"  --help\n"
"\n"
"Precision -> execution-unit routing (this nvcc build):\n"
"  fp32   CUDA cores  CUBLAS_COMPUTE_32F_PEDANTIC + CUBLAS_PEDANTIC_MATH\n"
"                     (TF32 disabled at both the handle and kernel-select layers)\n"
"  fp64   CUDA cores  cuBLASLt non-tensor algo (TENSOR_OP_MASK filtered)\n"
"                     CUBLAS_COMPUTE_64F_PEDANTIC is NOT used: on cuBLAS 12.x /\n"
"                     A100 it still dispatches DMMA (measured 135-145%% of 9.7T).\n"
"                     Use --fp64-tensor to benchmark the DMMA pipe instead.\n"
"  fp64t  DMMA cores  CUBLAS_COMPUTE_64F + GEMM_DEFAULT_TENSOR_OP  (opt-in)\n"
"  bf16   Tensor crse CUBLAS_COMPUTE_32F + GEMM_DEFAULT_TENSOR_OP, FP32 accum\n"
"\n"
"nvcc build differences vs gcc build:\n"
"  * Pinned host memory -> near PCIe Gen4 peak H2D/D2H (~25 GB/s)\n"
"  * CUDA Events for all timings (H2D, compute, D2H)  -- no host-clock jitter\n"
"  * BF16/FP64 conversion on host (CPU loop), not GPU kernel\n"
"  * GPU utilisation %% column in stdout and CSV\n"
"  * Default sweep starts at size=256\n"
"  * FP64 defaults to CUDA cores, not DMMA  (inverted vs gcc build)\n"
"\n"
"Verify execution units with Nsight Compute:\n"
"  ncu --metrics sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active,\\"
"               sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active\\"
"      ./bin/gpu_gemm_bench_cu --sizes 8192 --precisions fp64 --gpus 1 --no-nvml\n"
"  Expected: dmma%%=0, fma%%>0  (CUDA cores confirmed)\n", p);
}

static int parse_longs(const char *s, long *out, int max) {
    int n = 0; char buf[512]; snprintf(buf, sizeof buf, "%s", s);
    for (char *t = strtok(buf, ","); t && n < max; t = strtok(NULL, ","))
        out[n++] = atol(t);
    return n;
}

/* -- aggregate and report one benchmark point ------------------------------ */
static void run_config(const cfg_t *c, FILE *csv, const char *host, int cudart,
                       long cap, int nvml_ready, long S, prec_t p, int G,
                       engine_t engine, double density, long nnz,
                       const float *hA_pin, const float *hB_pin,
                       const int *hRow, const int *hCol, const float *hVals,
                       void *pre_hA_nat, void *pre_hB_nat, void *pre_hC_pin_pool)
{
    long ts = (long)time(NULL);
    const char *mode_s = (c->mode == MODE_SPLIT) ? "split" : "replicas";
    const char *eng_s  = eng_name(engine);
    /* precision label in output:
     *  fp32   -- CUDA cores (CUBLAS_COMPUTE_32F_PEDANTIC + PEDANTIC_MATH)
     *  fp64   -- CUDA cores (cuBLASLt non-tensor algo, default)
     *  fp64t  -- DMMA tensor cores (--fp64-tensor opt-in)
     *  bf16   -- Tensor cores (intended)
     */
    const char *pn = (p == PREC_FP64 && c->fp64_tensor) ? "fp64t" : prec_name(p);
    long max_ncols = 0, foot = mem_footprint(c, p, c->mode, S, G, &max_ncols);
    double mem_gb = (double)foot / 1e9;

    double display_density = (engine == ENG_SPARSE) ? 0.50 : density;
    long display_nnz = (engine == ENG_SPARSE) ? (S * S / 2) : nnz;

    if (foot > cap) {
        printf("%-6s %-5s %-3d %6ld %6.3f %9s %9s %6s %8s %8s %6s %5s %7s %s\n",
               eng_s, pn, G, S, display_density, "-","-","-","-","-","-","-","-","oom-skip");
        if (csv) fprintf(csv,
            "%s,%ld,%s,%d,%s,%s,%d,%ld,%ld,%ld,%d,%d,"
            "0,0,0,0,0,0,0,0,0,0,none,0,%.3f,%d,%d,%.3g,"
            "0,0,0,0,0,0,0,0,%s,%.4g,%ld,oom-skip\n",
            c->tag,ts,host,cudart,mode_s,pn,G,S,S,S,c->iters,c->warmup,
            mem_gb,c->fp64_tensor,0,-1.0,eng_s,display_density,display_nnz);
        return;
    }

    void *hA_nat = NULL;
    void *hB_nat = NULL;
    void *hVals_nat = NULL;
    size_t eb = in_bytes(p);

    if (p == PREC_FP32) {
        hA_nat = (void*)hA_pin;
        hB_nat = (void*)hB_pin;
        hVals_nat = (void*)hVals;
    } else {
        if (hA_pin && pre_hA_nat) {
            hA_nat = pre_hA_nat;
            if (p == PREC_FP64) {
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < (size_t)S*S; i++) ((double*)hA_nat)[i] = (double)hA_pin[i];
            } else {
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < (size_t)S*S; i++) ((uint16_t*)hA_nat)[i] = f32_to_bf16(hA_pin[i]);
            }
        }
        if (hB_pin && pre_hB_nat) {
            hB_nat = pre_hB_nat;
            if (p == PREC_FP64) {
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < (size_t)S*S; i++) ((double*)hB_nat)[i] = (double)hB_pin[i];
            } else {
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < (size_t)S*S; i++) ((uint16_t*)hB_nat)[i] = f32_to_bf16(hB_pin[i]);
            }
        }
        if (hVals) {
            // Unchanged for now, sparse uses hVals
            if (cudaMallocHost(&hVals_nat, (size_t)nnz * eb) != cudaSuccess) {
                fprintf(stderr, "Host OOM for pre-converted hVals_nat\n");
                return;
            }
            if (p == PREC_FP64) {
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < (size_t)nnz; i++) ((double*)hVals_nat)[i] = (double)hVals[i];
            } else {
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < (size_t)nnz; i++) ((uint16_t*)hVals_nat)[i] = f32_to_bf16(hVals[i]);
            }
        }
    }

    job_t job[8]; pthread_barrier_t bar;
    pthread_barrier_init(&bar, NULL, (unsigned)G);
    long base = S/G, rem = S%G, col = 0;
    for (int g = 0; g < G; g++) {
        long ncols = (c->mode == MODE_REPLICAS) ? S : (base + (g < rem ? 1 : 0));
        job[g].rc = 0;
        job[g].batch_ms = 0.0; job[g].h2d_ms = 0.0; job[g].d2h_ms = 0.0;
        job[g].max_rel_err = 0.0; job[g].mem_gb = 0.0;
        job[g].smp.valid = 0;
        job[g].smp.sm_sum = 0; job[g].smp.mem_sum = 0; job[g].smp.pow_sum = 0; job[g].smp.util_sum = 0;
        job[g].smp.sm_max = 0; job[g].smp.mem_max = 0; job[g].smp.pow_max = 0; job[g].smp.temp_max = 0;
        job[g].smp.throttle_or = 0; job[g].smp.n = 0;
        job[g].dev     = g; job[g].S = S;
        job[g].col0    = (c->mode == MODE_REPLICAS) ? 0 : col;
        job[g].ncols   = ncols; job[g].prec = p; job[g].engine = engine;
        job[g].density = density; job[g].nnz = nnz;
        job[g].hRowOff = hRow; job[g].hColInd = hCol; job[g].hVals = hVals;
        job[g].fp64_tensor   = c->fp64_tensor;
        job[g].fp64_cudacore = c->fp64_cudacore;
        job[g].warmup  = c->warmup; job[g].iters = c->iters;
        job[g].hA_pin  = hA_pin;   job[g].hB_pin = hB_pin;
        job[g].hA_nat  = hA_nat;
        job[g].hB_nat  = hB_nat ? (void*)((char*)hB_nat + job[g].col0 * S * eb) : NULL;
        job[g].hC_pin  = pre_hC_pin_pool ? (void*)((char*)pre_hC_pin_pool + job[g].col0 * S * 8) : NULL;
        job[g].hVals_nat = hVals_nat;
        job[g].nvml_on = nvml_ready;
        job[g].validate    = c->validate;
        job[g].valid_max   = c->valid_max;
        job[g].measure_xfer = c->measure_xfer;
        job[g].bar     = &bar; job[g].mem_gb = mem_gb;
        if (c->mode != MODE_REPLICAS) col += ncols;
        job[g].smp.valid = 0;
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
        pthread_create(&th[g], NULL,
                       engine == ENG_SPARSE ? spmm_worker : gpu_worker, &job[g]);
    for (int g = 0; g < G; g++) pthread_join(th[g], NULL);
    pthread_barrier_destroy(&bar);

    if (p != PREC_FP32) {
        if (hVals_nat)  cudaFreeHost(hVals_nat);
    }

    int err = 0; double wall = 0, sum_flops = 0, rel = -1.0;
    unsigned long long sm_sum = 0, mem_sum = 0, pow_sum = 0, util_sum = 0;
    long nsamp = 0;
    unsigned sm_max = 0, mem_max = 0, pow_max = 0, temp_max = 0;
    unsigned long long tr = 0;
    for (int g = 0; g < G; g++) {
        if (job[g].rc) { err = 1; continue; }
        double tg = job[g].batch_ms / 1000.0; if (tg > wall) wall = tg;
        sum_flops += (engine == ENG_SPARSE)
            ? (double)c->iters * 2.0 * (double)display_nnz * (double)job[g].ncols
            : (double)c->iters * 2.0 * (double)S * (double)job[g].ncols * (double)S;
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

    if (err) {
        printf("%-6s %-5s %-3d %6ld %6.3f %9s %9s %6s %8s %8s %6s %5s %7s %s\n",
               eng_s, pn, G, S, display_density, "-","-","-","-","-","-","-","-","error");
        if (csv) fprintf(csv,
            "%s,%ld,%s,%d,%s,%s,%d,%ld,%ld,%ld,%d,%d,"
            "0,0,0,0,0,0,0,0,0,0,none,0,%.3f,%d,%d,%.3g,"
            "0,0,0,0,0,0,0,0,%s,%.4g,%ld,error\n",
            c->tag,ts,host,cudart,mode_s,pn,G,S,S,S,c->iters,c->warmup,
            mem_gb,c->fp64_tensor,c->validate&&S<=c->valid_max,rel,eng_s,display_density,display_nnz);
        return;
    }

    /* Guard: if all workers returned rc=0 but wall==0 (async error not caught
     * by return-code, or event timer malfunction), report as time-zero error
     * rather than printing inf TFLOPS and corrupting subsequent runs.         */
    if (wall <= 0.0) {
        printf("%-6s %-5s %-3d %6ld %6.3f %9s %9s %6s %8s %8s %6s %5s %7s %s\n",
               eng_s, pn, G, S, display_density, "-","-","-","-","-","-","-","-","time-zero");
        if (csv) fprintf(csv,
            "%s,%ld,%s,%d,%s,%s,%d,%ld,%ld,%ld,%d,%d,"
            "0,0,0,0,0,0,0,0,0,0,none,0,%.3f,%d,%d,%.3g,"
            "0,0,0,0,0,0,0,0,%s,%.4g,%ld,time-zero\n",
            c->tag,ts,host,cudart,mode_s,pn,G,S,S,S,c->iters,c->warmup,
            mem_gb,c->fp64_tensor,c->validate&&S<=c->valid_max,rel,
            eng_s,display_density,display_nnz);
        return;
    }

    double agg_tflops = sum_flops / wall / 1e12;
    double per_gpu    = agg_tflops / (double)G;
    double peak = peak_tflops(p, c->fp64_tensor);
    if (engine == ENG_SPARSE) peak *= 2.0; // Sparse Tensor Cores double dense peak
    double pct = 100.0 * agg_tflops / ((double)G * peak);

    int bn = 0; double max_e2e = 0;
    for (int g = 0; g < G; g++) {
        double comp = job[g].batch_ms / (double)c->iters;
        double e2e  = job[g].h2d_ms + comp + job[g].d2h_ms;
        if (e2e > max_e2e) { max_e2e = e2e; bn = g; }
    }
    double b_h2d  = job[bn].h2d_ms;
    double b_d2h  = job[bn].d2h_ms;
    double b_comp = job[bn].batch_ms / (double)c->iters;
    double e2e_ms = max_e2e;

    double single = (engine == ENG_SPARSE)
                  ? 2.0*(double)display_nnz*(double)S
                  : 2.0*(double)S*(double)S*(double)S;
    double prob_flops = (c->mode == MODE_REPLICAS ? (double)G : 1.0) * single;
    double eff_tflops = e2e_ms > 0 ? prob_flops/(e2e_ms/1000.0)/1e12 : 0;
    double xfer_pct   = e2e_ms > 0 ? 100.0*(b_h2d+b_d2h)/e2e_ms : 0;

    size_t hb = in_bytes(p)*((size_t)S*S+(size_t)S*(size_t)job[bn].ncols);
    size_t db = out_bytes(p)*((size_t)S*(size_t)job[bn].ncols);

    double h2d_gbps = b_h2d > 0 ? (double)hb/1e9/(b_h2d/1000.0) : 0;
    double d2h_gbps = b_d2h > 0 ? (double)db/1e9/(b_d2h/1000.0) : 0;

    double sm_avg   = nsamp ? (double)sm_sum  /(double)nsamp : 0;
    double mem_avg  = nsamp ? (double)mem_sum /(double)nsamp : 0;
    double pow_avg  = nsamp ? (double)pow_sum /(double)nsamp/1000.0 : 0;
    double util_avg = nsamp ? (double)util_sum/(double)nsamp : 0; /* NEW */
    char trbuf[128]; throttle_str(tr, trbuf, sizeof trbuf);

    double tol  = (p == PREC_BF16) ? 5e-2 : (p == PREC_FP32 && engine == ENG_SPARSE) ? 5e-2 : (p == PREC_FP32) ? 1e-3 : 1e-9;
    const char *vstr = rel >= 0 ? (rel < tol ? "ok/val" : "ok/HIERR") : "ok";

    printf("%-6s %-5s %-3d %6ld %6.3f %9.1f %9.1f %5.1f%% %8.2f %8.2f %5.1f%% %5.1f%% %7.0f %s\n",
           eng_s, pn, G, S, display_density,
           agg_tflops, eff_tflops, pct,
           b_comp, e2e_ms, xfer_pct, util_avg, sm_avg, vstr);

    if (csv) {
        fprintf(csv,
            "%s,%ld,%s,%d,%s,%s,%d,%ld,%ld,%ld,%d,%d,"
            "%.2f,%.2f,%.2f,%.3f,"
            "%.0f,%u,%.0f,%.1f,%.1f,%u,%s,"
            "%.1f,"
            "%.3f,%d,%d,%.3g,"
            "%.3f,%.3f,%.3f,%.3f,%.1f,%.1f,%.3f,%.1f,"
            "%s,%.4g,%ld,ok\n",
            c->tag,ts,host,cudart,mode_s,pn,G,S,S,S,c->iters,c->warmup,
            agg_tflops,per_gpu,pct,wall*1000.0,
            sm_avg,sm_max,mem_avg,pow_avg,(double)pow_max/1000.0,temp_max,trbuf,
            util_avg,
            mem_gb,c->fp64_tensor,c->validate&&S<=c->valid_max,rel,
            b_h2d,b_d2h,b_comp,e2e_ms,h2d_gbps,d2h_gbps,eff_tflops,xfer_pct,
            eng_s,display_density,display_nnz);
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
        if (!strcmp(a,"--sizes")&&i+1<argc) {
            c.n_sizes = parse_longs(argv[++i], c.sizes, 32); set_sizes = 1;
        } else if (!strcmp(a,"--precisions")&&i+1<argc) {
            char b[128]; snprintf(b,sizeof b,"%s",argv[++i]); c.n_precs = 0;
            for (char *t = strtok(b,","); t; t = strtok(NULL,",")) {
                if      (!strcmp(t,"fp32")) c.precs[c.n_precs++] = PREC_FP32;
                else if (!strcmp(t,"fp64")) c.precs[c.n_precs++] = PREC_FP64;
                else if (!strcmp(t,"bf16")) c.precs[c.n_precs++] = PREC_BF16;
            } set_precs = 1;
        } else if (!strcmp(a,"--gpus")&&i+1<argc) {
            long g[8]; int n = parse_longs(argv[++i], g, 8);
            for (int k = 0; k < n; k++) c.gpus[k] = (int)g[k];
            c.n_gpus = n; set_gpus = 1;
        } else if (!strcmp(a,"--mode")&&i+1<argc) {
            c.mode = strcmp(argv[++i],"replicas") ? MODE_SPLIT : MODE_REPLICAS;
        } else if (!strcmp(a,"--engine")&&i+1<argc) {
            char b[64]; snprintf(b,sizeof b,"%s",argv[++i]); c.n_engines = 0;
            for (char *t = strtok(b,","); t&&c.n_engines<2; t = strtok(NULL,",")) {
                if      (!strcmp(t,"dense"))  c.engines[c.n_engines++] = ENG_DENSE;
                else if (!strcmp(t,"sparse")) c.engines[c.n_engines++] = ENG_SPARSE;
            }
        } else if (!strcmp(a,"--density")&&i+1<argc) {
            char b[128]; snprintf(b,sizeof b,"%s",argv[++i]); c.n_dens = 0;
            for (char *t = strtok(b,","); t&&c.n_dens<8; t = strtok(NULL,","))
                c.densities[c.n_dens++] = atof(t);
        } else if (!strcmp(a,"--fill")&&i+1<argc) {
            const char *f = argv[++i];
            if (!strncmp(f,"sparse",6)) {
                c.fill_sparse = 1;
                const char *cl = strchr(f,':'); c.fill_density = cl ? atof(cl+1) : 0.1;
            } else c.fill_sparse = 0;
        } else if (!strcmp(a,"--iters")   &&i+1<argc) c.iters   = atoi(argv[++i]);
        else if   (!strcmp(a,"--warmup")  &&i+1<argc) c.warmup  = atoi(argv[++i]);
        else if   (!strcmp(a,"--fp64-tensor"))         c.fp64_tensor = 1;
        else if   (!strcmp(a,"--fp64-cudacore"))  {
            /* --fp64-cudacore is retired: CUDA-core path is now the DEFAULT
             * for FP64.  Accept the flag for backwards-compat; it is a no-op. */
            fprintf(stderr, "note: --fp64-cudacore is now the default; flag ignored.\n");
        }
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
    /* Default sweep includes 256 per user request. */
    if (!set_sizes) {
        long d[] = {256,512,1024,2048,4096,8192,16384,32768,65536};
        c.n_sizes = 9; memcpy(c.sizes, d, sizeof d);
    }
    if (!set_precs) {
        c.precs[0]=PREC_FP32; c.precs[1]=PREC_FP64; c.precs[2]=PREC_BF16; c.n_precs=3;
    }
    if (!set_gpus) {
        c.n_gpus = 0;
        for (int g = 1; g <= ndev && g <= 4; g++) c.gpus[c.n_gpus++] = g;
    }
    if (c.n_engines == 0) { c.engines[0] = ENG_DENSE; c.n_engines = 1; }
    if (c.n_dens    == 0) {
        double dd[] = {0.01,0.05,0.1,0.25}; c.n_dens = 4;
        memcpy(c.densities, dd, sizeof dd);
    }
    if (c.fill_sparse && c.fill_density <= 0) c.fill_density = 0.1;

    CUDA_CHECK(cudaSetDevice(0));
    size_t mfree = 0, mtotal = 0; cudaMemGetInfo(&mfree, &mtotal);
    long cap = (long)((double)mtotal * c.mem_frac);

    int nvml_ready = 0;
    if (c.nvml_on && nvmlInit_v2() == NVML_SUCCESS) nvml_ready = 1;
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
    int cudart = 0; cudaRuntimeGetVersion(&cudart);

    FILE *csv = NULL; int new_file = 1;
    if (c.csv) {
        FILE *t = fopen(c.csv, "r");
        if (t) { fseek(t,0,SEEK_END); if (ftell(t)>0) new_file=0; fclose(t); }
        csv = fopen(c.csv, "a"); if (!csv) perror("csv");
    }
    /* CSV header -- util_pct_avg is the new column vs the gcc build. */
    const char *HDR =
        "tag,ts,host,cudart,mode,precision,gpus,M,N,K,iters,warmup,"
        "agg_tflops,per_gpu_tflops,pct_peak,wall_ms,"
        "sm_mhz_avg,sm_mhz_max,mem_mhz_avg,power_w_avg,power_w_max,temp_c_max,throttle,"
        "util_pct_avg,"
        "mem_gb_per_gpu,fp64_tensor,validated,max_rel_err,"
        "h2d_ms,d2h_ms,compute_ms,e2e_ms,h2d_gbps,d2h_gbps,eff_tflops,transfer_pct,"
        "engine,density,nnz,status\n";
    if (csv && new_file) fputs(HDR, csv);

    printf("# GPU GEMM bench (nvcc/CUDA C++)  host=%s devices=%d mode=%s cudart=%d cap=%.1fGB\n",
           host, ndev, c.mode==MODE_SPLIT?"split":"replicas", cudart, (double)cap/1e9);
    printf("# Pinned memory | CUDA-event timers | Host-side BF16/FP64 conversion\n");
    printf("# Precision -> execution-unit routing:\n");
    printf("#   fp32  -> CUDA cores  (COMPUTE_32F_PEDANTIC + PEDANTIC_MATH, TF32 off)\n");
    if (c.fp64_tensor)
        printf("#   fp64t -> DMMA cores  (COMPUTE_64F + TENSOR_OP, --fp64-tensor)\n");
    else
        printf("#   fp64  -> CUDA cores  (cuBLASLt non-tensor algo, TENSOR_OP_MASK=0)\n");
    printf("#   bf16  -> Tensor cores (COMPUTE_32F + TENSOR_OP, FP32 accum)\n");
    printf("# Peak refs (per GPU): fp32=19.5T fp64=%s bf16=312T\n",
           c.fp64_tensor ? "19.5T(DMMA)" : "9.7T(CUDA)");
    printf("%-6s %-5s %-3s %6s %6s %9s %9s %6s %8s %8s %6s %5s %7s %s\n",
           "eng","prec","G","N","dens",
           "aggTF","effTF","pk%","comp_ms","e2e_ms","xfer%","util%","SMclk","status");

    for (int si = 0; si < c.n_sizes; si++) {
        long S = c.sizes[si];
        float *hB_pin = NULL;
        if (cudaMallocHost((void**)&hB_pin, (size_t)S*S*sizeof(float)) != cudaSuccess) {
            fprintf(stderr, "pinned host OOM (B) at S=%ld\n", S); continue;
        }
        madvise(hB_pin, (size_t)S*S*sizeof(float), MADV_NOHUGEPAGE);
        fill_rand(hB_pin, (size_t)S*S, 2);

        float *hA_pin = NULL;
        if (cudaMallocHost((void**)&hA_pin, (size_t)S*S*sizeof(float)) != cudaSuccess) {
            fprintf(stderr, "pinned host OOM (A) at S=%ld\n", S);
            cudaFreeHost(hB_pin); continue;
        }
        madvise(hA_pin, (size_t)S*S*sizeof(float), MADV_NOHUGEPAGE);
        fill_rand(hA_pin, (size_t)S*S, 1);
        if (c.fill_sparse) {
            sparsify_inplace(hA_pin, (size_t)S*S, c.fill_density, 11);
            sparsify_inplace(hB_pin, (size_t)S*S, c.fill_density, 22);
        }

        void *pre_hA_nat = NULL, *pre_hB_nat = NULL, *pre_hC_pin_pool = NULL;
        if (cudaMallocHost(&pre_hA_nat, (size_t)S*S*8) == cudaSuccess && pre_hA_nat)
            madvise(pre_hA_nat, (size_t)S*S*8, MADV_NOHUGEPAGE);
        if (cudaMallocHost(&pre_hB_nat, (size_t)S*S*8) == cudaSuccess && pre_hB_nat)
            madvise(pre_hB_nat, (size_t)S*S*8, MADV_NOHUGEPAGE);
        if (cudaMallocHost(&pre_hC_pin_pool, (size_t)S*S*8) == cudaSuccess && pre_hC_pin_pool)
            madvise(pre_hC_pin_pool, (size_t)S*S*8, MADV_NOHUGEPAGE);


        for (int ei = 0; ei < c.n_engines; ei++) {
            engine_t eng = c.engines[ei];
            if (eng == ENG_DENSE) {
                double dens = c.fill_sparse ? c.fill_density : 1.0;
                long approx_nnz = (long)llround(dens * (double)S * (double)S);
                for (int pi = 0; pi < c.n_precs; pi++)
                    for (int gi = 0; gi < c.n_gpus; gi++) {
                        int G = c.gpus[gi]; if (G > ndev) continue;
                        run_config(&c, csv, host, cudart, cap, nvml_ready,
                                   S, c.precs[pi], G,
                                   ENG_DENSE, dens, approx_nnz,
                                   hA_pin, hB_pin, NULL, NULL, NULL,
                                   pre_hA_nat, pre_hB_nat, pre_hC_pin_pool);
                    }
            } else {
                /* Structured 2:4 sparsity (cuSPARSELt): density is always
                 * fixed at 50% by the hardware pruning step -- the --density
                 * flag and CSR generation are irrelevant. Run once per
                 * (prec, gpu-count) combination, passing NULL for CSR args.
                 * display_density / display_nnz are overridden to 0.5/S²/2
                 * inside run_config regardless.
                 *
                 * NOTE: cuSPARSELt prunes every A to 2:4 (50%) via PRUNE_SPMMA_TILE.
                 * The --density flag sets the INPUT initialization density only.
                 * All hardware computations run at 50% density regardless of this value.
                 * See paper footnote: "GEMM sparse density column records input density." */
                for (int pi = 0; pi < c.n_precs; pi++)
                    for (int gi = 0; gi < c.n_gpus; gi++) {
                        int G = c.gpus[gi]; if (G > ndev) continue;
                        run_config(&c, csv, host, cudart, cap, nvml_ready,
                                   S, c.precs[pi], G,
                                   ENG_SPARSE, 0.5, (long)S*(long)S/2,
                                   hA_pin, hB_pin, NULL, NULL, NULL,
                                   pre_hA_nat, pre_hB_nat, pre_hC_pin_pool);
                    }
            }
        }
        cudaFreeHost(hA_pin);  /* NULL-safe */
        cudaFreeHost(hB_pin);
        cudaFreeHost(pre_hA_nat);
        cudaFreeHost(pre_hB_nat);
        cudaFreeHost(pre_hC_pin_pool);
    }

    if (csv) fclose(csv);
    if (nvml_ready) nvmlShutdown();
    return 0;
}
