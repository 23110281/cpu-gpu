/* gpu_saxpy_bench.cu -- multi-GPU cuBLAS SAXPY benchmark (nvcc, CUDA C++).
 *
 * y[i] = alpha * x[i] + y[i]  (BLAS Level-1 AXPY)
 * Arithmetic intensity: 0.167 FLOPs/byte (fp32), 0.083 FLOPs/byte (fp64).
 * Bandwidth-bound at all practical vector sizes on A100.
 * Headline metric: achieved GB/s and % of 1555 GB/s HBM peak.
 *
 * IMPORTANT: Sizes below N=32M (33554432) elements are L2-cache-bound on A100
 * (L2=40MB; 2-vector fp32 working set at N=8M is 64MB, borderline).
 * The benchmark still runs at small N but prints "l2-bound" in the status
 * column when agg_gbps > 1555.0 (exceeds HBM peak => L2 bandwidth measured).
 *
 * Validation uses post-hoc single-call approach: after timed loop, reset dy
 * from hy_orig, run one fresh cublasSaxpy, compare against y_ref = alpha*x + y_orig.
 * hy_orig is only allocated when (validate && N <= validate_max).
 *
 * Structure mirrors gpu_gemv_bench.cu exactly. Read that file as the reference.
 * Compile: make saxpy
 * Run:     ./bin/gpu_saxpy_bench --help
 */

/* -- includes -------------------------------------------------------------- */
#include <cublas_v2.h>
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
#include <unistd.h>
#include "bf16_cvt.h"

/* -- error-checking macros ------------------------------------------------- */
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

static const char *prec_name(prec_t p) {
    return p == PREC_FP32 ? "fp32" : p == PREC_FP64 ? "fp64" : "bf16";
}
static size_t in_bytes(prec_t p) { return p == PREC_FP32 ? 4 : p == PREC_FP64 ? 8 : 2; }

/* A100-SXM4-40GB HBM2e peak. Copy from gemv_host.h L71. */
static inline double peak_gbps(void) { return 1555.0; }

static inline float hashf(uint64_t i, uint64_t seed) {
    uint64_t x = i * 0x9E3779B97F4A7C15ULL + seed;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL; x ^= x >> 31;
    return (float)((double)(uint32_t)(x >> 32) / (double)UINT32_MAX) * 2.0f - 1.0f;
}

static inline void fill_rand(float *p, size_t n, uint64_t seed) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) p[i] = hashf(i, seed);
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
    /* inputs */
    int      dev;
    long     N;          /* full problem size (vector length) */
    long     n_local;    /* this GPU's element slice = ceil(N/G) for split, N for replicas */
    long     offset;     /* starting element index in global vectors (0 for replicas) */
    prec_t   prec;
    int      warmup, iters;
    float    alpha;      /* SAXPY scalar, always fp32; cuBLAS internally converts for fp64 */
    /* pinned host vectors, full fp32 (always float*, even for fp64 -- converted before H2D) */
    const float *hx_pin;   /* full vector x, fp32, pinned */
    const float *hy_pin;   /* full vector y, fp32, pinned */
    const float *hy_orig;  /* copy of y before any SAXPY, for post-hoc validation */
                           /* NULL if !validate || N > validate_max */
    /* native-precision slices actually shipped to device */
    const void  *hx_nat;   /* this GPU's x slice, native precision, pinned */
    const void  *hy_nat;   /* this GPU's y slice, native precision, pinned */
    int      nvml_on, validate, valid_max, measure_xfer;
    pthread_barrier_t *bar;
    /* outputs (written by worker, read by run_config) */
    int      rc;
    double   batch_ms, h2d_ms, d2h_ms, max_rel_err, mem_gb;
    sampler_t smp;
} job_t;

/* Dispatch cublasSaxpy / cublasDaxpy based on precision.
 * bf16 is not supported by any cuBLAS AXPY variant -- caller must skip bf16. */
static cublasStatus_t do_saxpy(cublasHandle_t h, prec_t p, int n_local,
                                float alpha, const void *dx, void *dy) {
    double dalpha = (double)alpha;
    switch (p) {
    case PREC_FP32:
        return cublasSaxpy(h, n_local, &alpha,
                           (const float *)dx, 1, (float *)dy, 1);
    case PREC_FP64:
        return cublasDaxpy(h, n_local, &dalpha,
                           (const double *)dx, 1, (double *)dy, 1);
    case PREC_BF16:
        return CUBLAS_STATUS_NOT_SUPPORTED;  /* caller skips this case */
    }
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

/* Post-hoc validation: compare device result of ONE fresh saxpy call against
 * y_ref[i] = alpha * hx[i] + hy_orig[i] for 128 sampled indices.
 * Called AFTER the timed loop: reset dy from hy_orig, run one call, sample.
 * Returns max relative error. Returns -1.0 if skipped. */
static double validate_saxpy(cublasHandle_t h, cudaStream_t st,
                              job_t *j, void *dy, const void *dx) {
    /* Step 1: reset dy by copying hy_orig (native precision) back to device */
    size_t eb = in_bytes(j->prec);
    if (cudaMemcpyAsync(dy, j->hy_orig, (size_t)j->n_local * eb,
                        cudaMemcpyHostToDevice, st) != cudaSuccess) return -1.0;
    cudaStreamSynchronize(st);
    /* Step 2: one fresh saxpy call */
    if (do_saxpy(h, j->prec, (int)j->n_local, j->alpha, dx, dy) != CUBLAS_STATUS_SUCCESS)
        return -1.0;
    cudaStreamSynchronize(st);
    /* Step 3: copy result back */
    void *hy_val = NULL;
    if (cudaMallocHost(&hy_val, (size_t)j->n_local * eb) != cudaSuccess) return -1.0;
    cudaMemcpy(hy_val, dy, (size_t)j->n_local * eb, cudaMemcpyDeviceToHost);
    /* Step 4: sample 128 elements */
    double worst = 0.0;
    uint64_t rng = 0xC0FFEEull ^ (uint64_t)j->dev;
    int nsamp = (j->n_local < 128) ? (int)j->n_local : 128;
    for (int s = 0; s < nsamp; s++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        long i = (long)(rng % (uint64_t)j->n_local);
        double xi   = (j->prec == PREC_FP64) ? (double)j->hx_pin[j->offset + i]
                                              : (double)j->hx_pin[j->offset + i];
        double yi0  = (double)j->hy_orig[i];  /* hy_orig is always fp32 */
        double ref  = (double)j->alpha * xi + yi0;
        double got  = (j->prec == PREC_FP64) ? ((const double *)hy_val)[i]
                                              : (double)((const float *)hy_val)[i];
        double denom = fabs(ref) > 1e-12 ? fabs(ref) : 1e-12;
        double rel  = fabs(got - ref) / denom;
        if (rel > worst) worst = rel;
    }
    cudaFreeHost(hy_val);
    return worst;
}

static void *saxpy_worker(void *arg) {
    job_t *j = (job_t *)arg;
    j->rc = 0; j->max_rel_err = -1.0;
    j->batch_ms = j->h2d_ms = j->d2h_ms = 0.0;
    int pre_barrier = 0, smp_started = 0;       /* ← MUST have pre_barrier */
    pthread_t smp_th;

    cublasHandle_t h = NULL; cudaStream_t st = NULL;
    void *dx = NULL, *dy = NULL;
    cudaEvent_t e_h2d0=NULL,e_h2d1=NULL,e0=NULL,e1=NULL,e_d2h0=NULL,e_d2h1=NULL;
    void *hy_pin = NULL;

    /* ── Setup ─────────────────────────────────────────────────────── */
    if (cudaSetDevice(j->dev) != cudaSuccess) { j->rc = -1; goto barrier_out; }
    CUDA_CHECK_J(cudaStreamCreate(&st), barrier_out);
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { j->rc=-1; goto barrier_out; }
    cublasSetStream(h, st);

    CUDA_CHECK_J(cudaMalloc(&dx, (size_t)j->n_local * in_bytes(j->prec)), barrier_out);
    CUDA_CHECK_J(cudaMalloc(&dy, (size_t)j->n_local * in_bytes(j->prec)), barrier_out);

    /* ── H2D transfer (timed) ───────────────────────────────────────── */
    CUDA_CHECK_J(cudaEventCreate(&e_h2d0), barrier_out);
    CUDA_CHECK_J(cudaEventCreate(&e_h2d1), barrier_out);
    CUDA_CHECK_J(cudaEventCreate(&e0),     barrier_out);
    CUDA_CHECK_J(cudaEventCreate(&e1),     barrier_out);
    CUDA_CHECK_J(cudaEventCreate(&e_d2h0), barrier_out);
    CUDA_CHECK_J(cudaEventCreate(&e_d2h1), barrier_out);

    cudaEventRecord(e_h2d0, st);
    CUDA_CHECK_J(cudaMemcpyAsync(dx, j->hx_nat,
        (size_t)j->n_local * in_bytes(j->prec), cudaMemcpyHostToDevice, st), barrier_out);
    CUDA_CHECK_J(cudaMemcpyAsync(dy, j->hy_nat,
        (size_t)j->n_local * in_bytes(j->prec), cudaMemcpyHostToDevice, st), barrier_out);
    cudaEventRecord(e_h2d1, st);
    CUDA_CHECK_J(cudaStreamSynchronize(st), barrier_out);
    j->h2d_ms = (double)event_ms(e_h2d0, e_h2d1);

    /* ── Warmup ─────────────────────────────────────────────────────── */
    for (int w = 0; w < j->warmup; w++) {
        if (do_saxpy(h, j->prec, (int)j->n_local, j->alpha, dx, dy)
                != CUBLAS_STATUS_SUCCESS) j->rc = -1;
    }
    CUDA_CHECK_J(cudaStreamSynchronize(st), barrier_out);
    if (j->rc) goto barrier_out;

    /* ── BARRIER: placed immediately before cudaEventRecord(e0) ──────
     * This ensures all GPU threads start their timed loop simultaneously.
     * CUDA events record on the GPU stream, so CPU jitter after the barrier
     * does not affect timing accuracy -- but simultaneous start is still
     * good practice for clean aggregate bandwidth in replicas mode.      */
    pthread_barrier_wait(j->bar);
    pre_barrier = 1;    /* ← set AFTER barrier, before timed loop */

    /* ── NVML sampler start ─────────────────────────────────────────── */
    if (j->nvml_on && j->smp.valid) {
        j->smp.stop.store(0, std::memory_order_relaxed);
        if (pthread_create(&smp_th, NULL, sampler_run, &j->smp) == 0)
            smp_started = 1;
    }

    /* ── Timed loop ──────────────────────────────────────────────────── */
    {
        int loop_err = 0;
        cudaEventRecord(e0, st);
        for (int it = 0; it < j->iters; it++) {
            if (do_saxpy(h, j->prec, (int)j->n_local, j->alpha, dx, dy)
                    != CUBLAS_STATUS_SUCCESS) loop_err = 1;
        }
        cudaEventRecord(e1, st);
        cudaEventSynchronize(e1);
        /* Stop NVML BEFORE checking/handling any fault */
        if (smp_started) {
            j->smp.stop.store(1, std::memory_order_relaxed);
            pthread_join(smp_th, NULL);
            smp_started = 0;
        }
        /* Check for async CUDA errors (do NOT call cudaDeviceReset) */
        if (cudaGetLastError() != cudaSuccess) loop_err = 1;
        if (loop_err) { j->rc = -1; goto done; }
        j->batch_ms = (double)event_ms(e0, e1);
    }

    /* ── Validation (post-hoc single-call, only if requested) ──────── */
    if (j->validate && j->N <= j->valid_max && j->hy_orig != NULL) {
        j->max_rel_err = validate_saxpy(h, st, j, dy, dx);
    }

    /* ── D2H transfer (optional timing) ─────────────────────────────── */
    if (j->measure_xfer) {
        size_t yb = (size_t)j->n_local * in_bytes(j->prec);
        if (cudaMallocHost(&hy_pin, yb) == cudaSuccess) {
            cudaEventRecord(e_d2h0, st);
            cudaMemcpyAsync(hy_pin, dy, yb, cudaMemcpyDeviceToHost, st);
            cudaEventRecord(e_d2h1, st);
            cudaStreamSynchronize(st);
            j->d2h_ms = (double)event_ms(e_d2h0, e_d2h1);
            cudaFreeHost(hy_pin); hy_pin = NULL;
        }
    }

done:
    if (smp_started) { j->smp.stop.store(1, std::memory_order_relaxed); pthread_join(smp_th, NULL); }
    if (e_h2d0) cudaEventDestroy(e_h2d0); if (e_h2d1) cudaEventDestroy(e_h2d1);
    if (e0)     cudaEventDestroy(e0);     if (e1)     cudaEventDestroy(e1);
    if (e_d2h0) cudaEventDestroy(e_d2h0); if (e_d2h1) cudaEventDestroy(e_d2h1);
    if (dx) cudaFree(dx); if (dy) cudaFree(dy);
    if (hy_pin) cudaFreeHost(hy_pin);
    if (h) cublasDestroy(h);
    if (st) cudaStreamDestroy(st);
    /* pre_barrier=1 means barrier was already hit; don't hit again */
    if (!pre_barrier) pthread_barrier_wait(j->bar);
    return NULL;

barrier_out:
    /* Error before barrier: still MUST hit barrier so other threads don't hang */
    if (smp_started) { j->smp.stop.store(1, std::memory_order_relaxed); pthread_join(smp_th, NULL); }
    if (e_h2d0) cudaEventDestroy(e_h2d0); if (e_h2d1) cudaEventDestroy(e_h2d1);
    if (e0)     cudaEventDestroy(e0);     if (e1)     cudaEventDestroy(e1);
    if (e_d2h0) cudaEventDestroy(e_d2h0); if (e_d2h1) cudaEventDestroy(e_d2h1);
    if (dx) cudaFree(dx); if (dy) cudaFree(dy);
    if (h) cublasDestroy(h);
    if (st) cudaStreamDestroy(st);
    pthread_barrier_wait(j->bar);   /* ← ALWAYS hit, never skip */
    return NULL;
}

/* -- main ------------------------------------------------------------------ */
typedef struct {
    long   sizes[32]; int n_sizes;
    prec_t precs[4];  int n_precs;   /* fp32 and fp64 only; bf16 is skipped */
    int    gpus[8];   int n_gpus;
    bench_mode_t mode;
    int    iters, warmup;
    float  alpha;
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

static void usage(const char *p) {
    printf(
"Usage: %s [options]\n"
"  --sizes a,b,..     vector sizes                 [33554432,67108864,...]\n"
"  --precisions p,..  fp32,fp64                    [all]\n"
"  --gpus n,..        gpu counts to test           [1..4]\n"
"  --mode m           split | replicas             [split]\n"
"  --iters n          timed iterations             [20]\n"
"  --warmup n         warmup iterations            [5]\n"
"  --alpha f          scalar alpha                 [2.0]\n"
"  --no-nvml          disable NVML telemetry\n"
"  --validate         sampled correctness check\n"
"  --validate-max n   max size to validate         [33554432]\n"
"  --no-transfers     skip timed H2D/D2H\n"
"  --mem-frac f       usable GPU mem fraction      [0.90]\n"
"  --csv path         append CSV results\n"
"  --tag s            CSV label\n"
"  --help\n", p);
}

static void run_config(const cfg_t *c, FILE *csv, const char *host, int cudart,
                       long cap, int nvml_ready, long N, prec_t p, int G,
                       const float *hx_pin, const float *hy_pin, const float *hy_orig_f32)
{
    long ts = (long)time(NULL);
    const char *mode_s = (c->mode == MODE_SPLIT) ? "split" : "replicas";
    const char *pn = prec_name(p);

    if (p == PREC_BF16) {
        printf("bf16 not supported by cuBLAS AXPY\n");
        return;
    }

    long n_max_local = c->mode == MODE_REPLICAS ? N : N / G + 1;
    long foot = 3L * n_max_local * (long)in_bytes(p);
    double mem_gb = (double)foot / 1e9;

    if (foot > cap) {
        printf("%-5s %-3d %12ld %9s %9s %6s %8s %8s %6s %6s %7s %s\n",
               pn, G, N, "-","-","-","-","-","-","-","-","oom-skip");
        if (csv) fprintf(csv,
            "%s,%ld,%s,%d,%s,%s,%d,%ld,%d,%d,"
            "0,0,0,0,"
            "0,0,0,0,0,0,none,"
            "0,"
            "%.3f,%d,-1.0,"
            "0,0,0,0,0,"
            "oom-skip\n",
            c->tag,ts,host,cudart,mode_s,pn,G,N,c->iters,c->warmup,
            mem_gb,c->validate&&N<=c->valid_max);
        return;
    }

    void *hx_nat = NULL;
    void *hy_nat = NULL;
    size_t eb = in_bytes(p);

    if (p == PREC_FP32) {
        hx_nat = (void*)hx_pin;
        hy_nat = (void*)hy_pin;
    } else if (p == PREC_FP64) {
        if (cudaMallocHost(&hx_nat, (size_t)N * eb) != cudaSuccess) {
            fprintf(stderr, "Host OOM for pre-converted hx_nat\n");
            return;
        }
        if (cudaMallocHost(&hy_nat, (size_t)N * eb) != cudaSuccess) {
            fprintf(stderr, "Host OOM for pre-converted hy_nat\n");
            if (hx_nat) cudaFreeHost(hx_nat);
            return;
        }
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < (size_t)N; i++) ((double*)hx_nat)[i] = (double)hx_pin[i];
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < (size_t)N; i++) ((double*)hy_nat)[i] = (double)hy_pin[i];
    }

    job_t job[8]; pthread_barrier_t bar;
    pthread_barrier_init(&bar, NULL, (unsigned)G);
    long cumulative_offset = 0;
    for (int g = 0; g < G; g++) {
        long n_local = (c->mode == MODE_REPLICAS) ? N : (N / G + (g < N % G ? 1 : 0));
        job[g].rc = 0;
        job[g].batch_ms = 0.0; job[g].h2d_ms = 0.0; job[g].d2h_ms = 0.0;
        job[g].max_rel_err = 0.0; job[g].mem_gb = mem_gb;
        job[g].smp.valid = 0;
        job[g].smp.sm_sum = 0; job[g].smp.mem_sum = 0; job[g].smp.pow_sum = 0; job[g].smp.util_sum = 0;
        job[g].smp.sm_max = 0; job[g].smp.mem_max = 0; job[g].smp.pow_max = 0; job[g].smp.temp_max = 0;
        job[g].smp.throttle_or = 0; job[g].smp.n = 0;
        job[g].dev = g; job[g].N = N;
        job[g].n_local = n_local; job[g].offset = (c->mode == MODE_REPLICAS) ? 0 : cumulative_offset;
        job[g].prec = p;
        job[g].warmup = c->warmup; job[g].iters = c->iters;
        job[g].alpha = c->alpha;
        job[g].hx_pin = hx_pin; job[g].hy_pin = hy_pin;
        job[g].hy_orig = (c->validate && N <= c->valid_max) ? hy_orig_f32 + job[g].offset : NULL;
        job[g].hx_nat = (const char*)hx_nat + job[g].offset * eb;
        job[g].hy_nat = (const char*)hy_nat + job[g].offset * eb;
        job[g].nvml_on = nvml_ready;
        job[g].validate = c->validate;
        job[g].valid_max = c->valid_max;
        job[g].measure_xfer = c->measure_xfer;
        job[g].bar = &bar;
        if (c->mode != MODE_REPLICAS) cumulative_offset += n_local;
        
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
        pthread_create(&th[g], NULL, saxpy_worker, &job[g]);
    for (int g = 0; g < G; g++) pthread_join(th[g], NULL);
    pthread_barrier_destroy(&bar);

    if (p != PREC_FP32) {
        if (hx_nat) cudaFreeHost(hx_nat);
        if (hy_nat) cudaFreeHost(hy_nat);
    }

    int err = 0; double wall = 0, rel = -1.0;
    unsigned long long sm_sum = 0, mem_sum = 0, pow_sum = 0, util_sum = 0;
    long nsamp = 0;
    unsigned sm_max = 0, mem_max = 0, pow_max = 0, temp_max = 0;
    unsigned long long tr = 0;
    for (int g = 0; g < G; g++) {
        if (job[g].rc) { err = 1; continue; }
        double tg = job[g].batch_ms / 1000.0; if (tg > wall) wall = tg;
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

    if (err || wall <= 0.0) {
        const char* estat = err ? "error" : "time-zero";
        printf("%-5s %-3d %12ld %9s %9s %6s %8s %8s %6s %6s %7s %s\n",
               pn, G, N, "-","-","-","-","-","-","-","-",estat);
        if (csv) fprintf(csv,
            "%s,%ld,%s,%d,%s,%s,%d,%ld,%d,%d,"
            "0,0,0,0,"
            "0,0,0,0,0,0,none,"
            "0,"
            "%.3f,%d,%.3g,"
            "0,0,0,0,0,"
            "%s\n",
            c->tag,ts,host,cudart,mode_s,pn,G,N,c->iters,c->warmup,
            mem_gb,c->validate&&N<=c->valid_max,rel,estat);
        return;
    }

    double sum_bytes = 0;
    for (int g = 0; g < G; g++)
        if (!job[g].rc)
            sum_bytes += (double)c->iters * (double)job[g].n_local * 3.0 * in_bytes(p);
    double agg_gbps = (wall > 0) ? sum_bytes / wall / 1e9 : 0;
    double per_gpu_gbps = agg_gbps / (double)G;
    double pct = 100.0 * agg_gbps / (peak_gbps() * (double)G);

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
    double xfer_pct = e2e_ms > 0 ? 100.0*(b_h2d+b_d2h)/e2e_ms : 0;

    double sm_avg   = nsamp ? (double)sm_sum  /(double)nsamp : 0;
    double mem_avg  = nsamp ? (double)mem_sum /(double)nsamp : 0;
    double pow_avg  = nsamp ? (double)pow_sum /(double)nsamp/1000.0 : 0;
    double util_avg = nsamp ? (double)util_sum/(double)nsamp : 0;
    char trbuf[128]; throttle_str(tr, trbuf, sizeof trbuf);

    double tol = (p == PREC_FP64) ? 1e-12 : 1e-5;
    const char *vstr = (rel >= 0) ? (rel < tol ? "ok/val" : "ok/HIERR") : "ok";
    
    const char *status_str = "ok";
    if (agg_gbps > peak_gbps() * (double)G) status_str = "l2-bound";

    printf("%-5s %-3d %12ld %9.1f %9.1f %5.1f%% %8.2f %8.2f %5.1f%% %5.1f%% %7.0f %s\n",
           pn, G, N, agg_gbps, per_gpu_gbps, pct,
           b_comp, e2e_ms, xfer_pct, util_avg, sm_avg, vstr);

    if (csv) {
        fprintf(csv,
            "%s,%ld,%s,%d,%s,%s,%d,%ld,%d,%d,"
            "%.2f,%.2f,%.3f,%.3f,"
            "%.0f,%u,%.0f,%.1f,%.1f,%u,%s,"
            "%.1f,"
            "%.3f,%d,%.3g,"
            "%.3f,%.3f,%.3f,%.3f,%.1f,"
            "%s\n",
            c->tag,ts,host,cudart,mode_s,pn,G,N,c->iters,c->warmup,
            agg_gbps,per_gpu_gbps,pct,wall*1000.0,
            sm_avg,sm_max,mem_avg,pow_avg,(double)pow_max/1000.0,temp_max,trbuf,
            util_avg,
            mem_gb,c->validate&&N<=c->valid_max,rel,
            b_h2d,b_d2h,b_comp,e2e_ms,xfer_pct,
            status_str);
        fflush(csv);
    }
}

int main(int argc, char **argv) {
    cfg_t c = {};
    c.mode = MODE_SPLIT; c.iters = 20; c.warmup = 5; c.nvml_on = 1;
    c.valid_max = 33554432; /* 32M — below this N is L2-bound on A100, validate is still useful */
    c.mem_frac = 0.90; c.alpha = 2.0f; c.tag = ""; c.measure_xfer = 1;
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
        } else if (!strcmp(a,"--iters")   &&i+1<argc) c.iters   = atoi(argv[++i]);
        else if   (!strcmp(a,"--warmup")  &&i+1<argc) c.warmup  = atoi(argv[++i]);
        else if   (!strcmp(a,"--alpha")   &&i+1<argc) c.alpha   = atof(argv[++i]);
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
        long d[] = {33554432,67108864,134217728,268435456,536870912,1073741824};
        c.n_sizes = 6; memcpy(c.sizes, d, sizeof d);
    }
    if (!set_precs) {
        c.precs[0]=PREC_FP32; c.precs[1]=PREC_FP64; c.n_precs=2;
    }
    if (!set_gpus) {
        c.n_gpus = 0;
        for (int g = 1; g <= ndev && g <= 4; g++) c.gpus[c.n_gpus++] = g;
    }

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
        "tag,ts,host,cudart,mode,precision,gpus,N,iters,warmup,"
        "agg_gbps,per_gpu_gbps,pct_peak_bw,wall_ms,"
        "sm_mhz_avg,sm_mhz_max,mem_mhz_avg,power_w_avg,power_w_max,temp_c_max,throttle,"
        "util_pct_avg,"
        "mem_gb_per_gpu,validated,max_rel_err,"
        "h2d_ms,d2h_ms,compute_ms,e2e_ms,transfer_pct,"
        "status\n";
    if (csv && new_file) fputs(HDR, csv);

    printf("# GPU SAXPY bench (nvcc/CUDA C++)  host=%s devices=%d mode=%s cudart=%d cap=%.1fGB\n",
           host, ndev, c.mode==MODE_SPLIT?"split":"replicas", cudart, (double)cap/1e9);
    printf("# Peak refs (per GPU): %.1f GB/s\n", peak_gbps());
    printf("%-5s %-3s %12s %9s %9s %6s %8s %8s %6s %6s %7s %s\n",
           "prec","G","N",
           "aggGB/s","pGPUGB/s","pk%","comp_ms","e2e_ms","xfer%","util%","SMclk","status");

    for (int si = 0; si < c.n_sizes; si++) {
        long N = c.sizes[si];
        float *hx_pin = NULL, *hy_pin = NULL, *hy_orig_f32 = NULL;
        if (cudaMallocHost((void**)&hx_pin, (size_t)N*sizeof(float)) != cudaSuccess) {
            fprintf(stderr, "pinned host OOM (x) at N=%ld\n", N); continue;
        }
        if (cudaMallocHost((void**)&hy_pin, (size_t)N*sizeof(float)) != cudaSuccess) {
            fprintf(stderr, "pinned host OOM (y) at N=%ld\n", N);
            cudaFreeHost(hx_pin); continue;
        }
        fill_rand(hx_pin, (size_t)N, 1);
        fill_rand(hy_pin, (size_t)N, 2);
        
        if (c.validate && N <= c.valid_max) {
            if (cudaMallocHost((void**)&hy_orig_f32, (size_t)N*sizeof(float)) != cudaSuccess) {
                fprintf(stderr, "pinned host OOM (y_orig) at N=%ld\n", N);
                cudaFreeHost(hx_pin); cudaFreeHost(hy_pin); continue;
            }
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < (size_t)N; i++) hy_orig_f32[i] = hy_pin[i];
        }

        for (int pi = 0; pi < c.n_precs; pi++)
            for (int gi = 0; gi < c.n_gpus; gi++) {
                int G = c.gpus[gi]; if (G > ndev) continue;
                long n_max_local = c.mode == MODE_REPLICAS ? N : N / G + 1;
                long foot = 3L * n_max_local * (long)in_bytes(c.precs[pi]);
                if ((double)foot > (double)mtotal * c.mem_frac) {
                    printf("%-5s %-3d %12ld %9s %9s %6s %8s %8s %6s %6s %7s %s\n",
                           prec_name(c.precs[pi]), G, N, "-","-","-","-","-","-","-","-","oom-skip");
                    if (csv) {
                        long ts = (long)time(NULL);
                        fprintf(csv,
                            "%s,%ld,%s,%d,%s,%s,%d,%ld,%d,%d,"
                            "0,0,0,0,"
                            "0,0,0,0,0,0,none,"
                            "0,"
                            "%.3f,%d,-1.0,"
                            "0,0,0,0,0,"
                            "oom-skip\n",
                            c.tag,ts,host,cudart,c.mode==MODE_SPLIT?"split":"replicas",
                            prec_name(c.precs[pi]),G,N,c.iters,c.warmup,
                            (double)foot/1e9,c.validate&&N<=c.valid_max);
                        fflush(csv);
                    }
                    continue;
                }
                
                run_config(&c, csv, host, cudart, cap, nvml_ready,
                           N, c.precs[pi], G,
                           hx_pin, hy_pin, hy_orig_f32);
            }
        cudaFreeHost(hx_pin); cudaFreeHost(hy_pin);
        if (hy_orig_f32) cudaFreeHost(hy_orig_f32);
    }

    if (csv) fclose(csv);
    if (nvml_ready) nvmlShutdown();
    return 0;
}
