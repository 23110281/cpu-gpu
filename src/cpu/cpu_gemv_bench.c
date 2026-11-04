/* src/cpu/cpu_gemv_bench.c
 *
 * Multi-threaded Intel oneMKL GEMV / SpMV benchmark (y = alpha * A * x + beta * y).
 *
 * Workloads:
 *   1. Dense GEMV:
 *      - fp32: cblas_sgemv
 *      - fp64: cblas_dgemv
 *   2. Sparse SpMV (CSR format):
 *      - fp32: mkl_sparse_s_mv
 *      - fp64: mkl_sparse_d_mv
 *
 * Arithmetic Intensity: ~2 FLOPs / element. Strictly memory-bandwidth bound!
 * Headline metrics: Achieved Memory Bandwidth (GB/s) and Compute (TFLOPS).
 *
 * Compile: make cpu_gemv (or gcc with MKL flags)
 * Run:     ./bin/cpu_gemv_bench --help
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>

#if defined(USE_OPENBLAS)
  #include <cblas.h>
  #define HAVE_MKL 0
  #define HAVE_CBLAS 1
#elif defined(USE_MKL)
  #if defined(__has_include) && __has_include(<mkl.h>)
    #include <mkl.h>
  #elif defined(__has_include) && __has_include(<mkl_cblas.h>)
    #include <mkl_cblas.h>
    #include <mkl_spblas.h>
    #include <mkl_service.h>
  #else
    #include <mkl.h>
  #endif
  #define HAVE_MKL 1
  #define HAVE_CBLAS 1
#elif defined(__has_include)
  #if __has_include(<mkl.h>)
    #include <mkl.h>
    #define HAVE_MKL 1
    #define HAVE_CBLAS 1
  #elif __has_include(<mkl_cblas.h>)
    #include <mkl_cblas.h>
    #include <mkl_spblas.h>
    #include <mkl_service.h>
    #define HAVE_MKL 1
    #define HAVE_CBLAS 1
  #elif __has_include(<cblas.h>)
    #include <cblas.h>
    #define HAVE_MKL 0
    #define HAVE_CBLAS 1
  #else
    #define HAVE_MKL 0
    #define HAVE_CBLAS 0
  #endif
#else
  #define HAVE_MKL 0
  #define HAVE_CBLAS 0
#endif

#include "gemv_host.h"
#include "bf16_cvt.h"
#include "cpu_telemetry.h"

/* ── Fallback stubs if compiling without CBLAS ───────────────────────────── */
#if !HAVE_CBLAS
typedef enum { CblasRowMajor=101, CblasColMajor=102 } CBLAS_LAYOUT;
typedef enum { CblasNoTrans=111, CblasTrans=112 } CBLAS_TRANSPOSE;

static inline void cblas_sgemv(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                               const int M, const int N, const float alpha,
                               const float *A, const int lda,
                               const float *X, const int incX,
                               const float beta, float *Y, const int incY) {
    (void)Layout; (void)TransA; (void)lda; (void)incX; (void)incY;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) sum += A[i * N + j] * X[j];
        Y[i] = alpha * sum + beta * Y[i];
    }
}

static inline void cblas_dgemv(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                               const int M, const int N, const double alpha,
                               const double *A, const int lda,
                               const double *X, const int incX,
                               const double beta, double *Y, const int incY) {
    (void)Layout; (void)TransA; (void)lda; (void)incX; (void)incY;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) sum += A[i * N + j] * X[j];
        Y[i] = alpha * sum + beta * Y[i];
    }
}
#endif

/* ── Fallback stubs for non-MKL environments (OpenBLAS / generic) ────────── */
#if !HAVE_MKL
static inline void *mkl_malloc(size_t size, int align) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, align, size) != 0) return NULL;
    return ptr;
}

static inline void mkl_free(void *ptr) { free(ptr); }

__attribute__((weak)) void openblas_set_num_threads(int num_threads);

static inline void mkl_set_num_threads(int t) {
    if (openblas_set_num_threads) openblas_set_num_threads(t);
    omp_set_num_threads(t);
}

static inline void mkl_get_version_string(char *buf, int len) {
#if HAVE_CBLAS
    snprintf(buf, len, "OpenBLAS_Engine");
#else
    snprintf(buf, len, "Fallback_OpenMP");
#endif
}
#endif

typedef enum { PREC_FP32, PREC_FP64 } prec_t;
typedef enum { ENG_DENSE, ENG_SPARSE } engine_t;

static const char *prec_name(prec_t p) { return p == PREC_FP32 ? "fp32" : "fp64"; }
static const char *eng_name(engine_t e) { return e == ENG_DENSE ? "dense" : "sparse"; }
static size_t elem_bytes(prec_t p) { return p == PREC_FP32 ? 4 : 8; }

typedef struct {
    long sizes[32];
    int n_sizes;
    prec_t precs[4];
    int n_precs;
    int threads[32];
    int n_threads;
    engine_t engines[2];
    int n_engines;
    double densities[8];
    int n_densities;
    int iters;
    int warmup;
    int validate;
    int valid_max;
    const char *csv;
    const char *tag;
} cfg_t;

static void usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --sizes S,S,...     Matrix dimensions (SxS)      [default: 1024..16384]\n");
    printf("  --precisions P,P    Precisions: fp32,fp64        [default: fp32,fp64]\n");
    printf("  --threads T,T,...   Active thread counts         [default: 1..max_cores]\n");
    printf("  --engine E,E        dense,sparse                 [default: dense,sparse]\n");
    printf("  --density D,D       CSR matrix densities         [default: 0.01,0.05,0.1]\n");
    printf("  --iters N           Timed loop iterations        [default: 15]\n");
    printf("  --warmup N          Untimed warmup runs          [default: 3]\n");
    printf("  --validate          Check numerical accuracy     [default: off]\n");
    printf("  --validate-max N    Max size to validate         [default: 4096]\n");
    printf("  --csv PATH          Append results to CSV file\n");
    printf("  --tag STRING        Label identifier in CSV      [default: cpu_gemv]\n");
    printf("  --help              Display this help message\n");
}

static int parse_ints(const char *s, int *out, int max) {
    int n = 0; char buf[512]; snprintf(buf, sizeof(buf), "%s", s);
    for (char *t = strtok(buf, ","); t && n < max; t = strtok(NULL, ",")) out[n++] = atoi(t);
    return n;
}

static int parse_longs(const char *s, long *out, int max) {
    int n = 0; char buf[512]; snprintf(buf, sizeof(buf), "%s", s);
    for (char *t = strtok(buf, ","); t && n < max; t = strtok(NULL, ",")) out[n++] = atol(t);
    return n;
}

static int parse_doubles(const char *s, double *out, int max) {
    int n = 0; char buf[512]; snprintf(buf, sizeof(buf), "%s", s);
    for (char *t = strtok(buf, ","); t && n < max; t = strtok(NULL, ",")) out[n++] = atof(t);
    return n;
}

/* ── Validation for Dense GEMV ───────────────────────────────────────────── */
static double validate_dense(long S, prec_t p, const void *A, const void *X, const void *Y) {
    int check_samples = 32;
    double max_rel = 0.0;
    for (int sample = 0; sample < check_samples; sample++) {
        long r = (sample * 1337 + 7) % S;
        double ref = 0.0;
        for (long c = 0; c < S; c++) {
            double a_val = (p == PREC_FP32) ? ((const float*)A)[r * S + c] : ((const double*)A)[r * S + c];
            double x_val = (p == PREC_FP32) ? ((const float*)X)[c] : ((const double*)X)[c];
            ref += a_val * x_val;
        }
        double got = (p == PREC_FP32) ? ((const float*)Y)[r] : ((const double*)Y)[r];
        double denom = fabs(ref) > 1e-12 ? fabs(ref) : 1e-12;
        double rel = fabs(got - ref) / denom;
        if (rel > max_rel) max_rel = rel;
    }
    return max_rel;
}

/* ── Run Single Dense GEMV ────────────────────────────────────────────────── */
static void run_dense_gemv(const cfg_t *c, FILE *csv, const char *host, const char *mkl_info,
                           long S, prec_t p, int T, cpu_telemetry_t *telem) {
    long ts = (long)time(NULL);
    const char *pn = prec_name(p);
    size_t eb = elem_bytes(p);
    size_t bytes_A = (size_t)S * S * eb;
    size_t bytes_vec = (size_t)S * eb;
    double mem_gb = (double)(bytes_A + 2 * bytes_vec) / 1e9;

    void *A = mkl_malloc(bytes_A, 64);
    void *X = mkl_malloc(bytes_vec, 64);
    void *Y = mkl_malloc(bytes_vec, 64);

    if (!A || !X || !Y) {
        printf("%-6s %-5s %-3d %6ld %6.2f %9s %8s %s\n", "dense", pn, T, S, 1.0, "-", "-", "oom-skip");
        if (csv) {
            fprintf(csv, "%s,%ld,%s,%s,dense,split,%s,%d,%ld,1.0,%ld,%d,%d,"
                         "0.0,0.0,0.0,0.0,0.0,0.0,0.0,%.3f,0,-1.0,oom-skip\n",
                    c->tag, ts, host, mkl_info, pn, T, S, S * S, c->iters, c->warmup, mem_gb);
        }
        if (A) mkl_free(A); if (X) mkl_free(X); if (Y) mkl_free(Y);
        return;
    }

    if (p == PREC_FP32) {
        gemv_fill_rand((float*)A, (size_t)S * S, 1);
        gemv_fill_rand((float*)X, (size_t)S, 2);
    } else {
        float *tmpA = (float*)malloc(bytes_A / 2);
        float *tmpX = (float*)malloc(bytes_vec / 2);
        gemv_fill_rand(tmpA, (size_t)S * S, 1);
        gemv_fill_rand(tmpX, (size_t)S, 2);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < (size_t)S * S; i++) ((double*)A)[i] = (double)tmpA[i];
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < (size_t)S; i++) ((double*)X)[i] = (double)tmpX[i];
        free(tmpA); free(tmpX);
    }
    memset(Y, 0, bytes_vec);

    mkl_set_num_threads(T);

    /* Warmup */
    for (int w = 0; w < c->warmup; w++) {
        if (p == PREC_FP32) {
            cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)S, (int)S, 1.0f,
                        (const float*)A, (int)S, (const float*)X, 1, 0.0f, (float*)Y, 1);
        } else {
            cblas_dgemv(CblasRowMajor, CblasNoTrans, (int)S, (int)S, 1.0,
                        (const double*)A, (int)S, (const double*)X, 1, 0.0, (double*)Y, 1);
        }
    }

    /* Timed Loop */
    cpu_telemetry_start(telem);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

    for (int it = 0; it < c->iters; it++) {
        if (p == PREC_FP32) {
            cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)S, (int)S, 1.0f,
                        (const float*)A, (int)S, (const float*)X, 1, 0.0f, (float*)Y, 1);
        } else {
            cblas_dgemv(CblasRowMajor, CblasNoTrans, (int)S, (int)S, 1.0,
                        (const double*)A, (int)S, (const double*)X, 1, 0.0, (double*)Y, 1);
        }
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
    cpu_telemetry_stop(telem);

    double wall_sec = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double comp_ms = (wall_sec * 1000.0) / (double)c->iters;

    /* Memory Bandwidth (GB/s): Matrix A is read + Vector X is read + Vector Y is written */
    double bytes_transferred = ((double)S * (double)S + 2.0 * (double)S) * (double)eb * (double)c->iters;
    double agg_gbps = (wall_sec > 0.0) ? (bytes_transferred / wall_sec / 1e9) : 0.0;
    double flops = 2.0 * (double)S * (double)S * (double)c->iters;
    double agg_tflops = (wall_sec > 0.0) ? (flops / wall_sec / 1e12) : 0.0;

    double max_rel = -1.0;
    int validated = 0;
    if (c->validate && S <= c->valid_max) {
        max_rel = validate_dense(S, p, A, X, Y);
        validated = 1;
    }

    printf("%-6s %-5s %-3d %6ld %6.2f %8.2f %7.3f %6.1fW %7.1f %s\n",
           "dense", pn, T, S, 1.0, agg_gbps, agg_tflops,
           telem->avg_power_watts, telem->avg_freq_mhz,
           (validated && max_rel > 1e-2) ? "HIERR" : "ok");
    fflush(stdout);

    if (csv) {
        fprintf(csv,
            "%s,%ld,%s,%s,dense,split,%s,%d,%ld,1.0,%ld,%d,%d,"
            "%.2f,%.4f,%.3f,%.3f,"
            "%.1f,%.1f,%.1f,"
            "%.3f,%d,%.3e,ok\n",
            c->tag, ts, host, mkl_info, pn, T, S, S * S, c->iters, c->warmup,
            agg_gbps, agg_tflops, comp_ms, wall_sec * 1000.0,
            telem->avg_freq_mhz, telem->avg_power_watts, telem->max_temp_c,
            mem_gb, validated, max_rel);
        fflush(csv);
    }

    mkl_free(A); mkl_free(X); mkl_free(Y);
}

/* ── Run Single Sparse SpMV ───────────────────────────────────────────────── */
static void run_sparse_spmv(const cfg_t *c, FILE *csv, const char *host, const char *mkl_info,
                            long S, prec_t p, int T, double density, cpu_telemetry_t *telem) {
    long ts = (long)time(NULL);
    const char *pn = prec_name(p);
    size_t eb = elem_bytes(p);

    int *rowOff = NULL, *colInd = NULL;
    float *vals_f32 = NULL;
    long nnz = 0;

    if (gemv_gen_csr(S, density, &rowOff, &colInd, &vals_f32, &nnz, 42) != 0) {
        fprintf(stderr, "gemv_gen_csr failed for S=%ld dens=%.2f\n", S, density);
        return;
    }

    void *vals = vals_f32;
    if (p == PREC_FP64) {
        double *vals_f64 = (double*)malloc((size_t)nnz * sizeof(double));
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < nnz; i++) vals_f64[i] = (double)vals_f32[i];
        free(vals_f32);
        vals = vals_f64;
    }

    void *X = mkl_malloc((size_t)S * eb, 64);
    void *Y = mkl_malloc((size_t)S * eb, 64);
    if (p == PREC_FP32) gemv_fill_rand((float*)X, (size_t)S, 2);
    else {
        float *tmpX = (float*)malloc((size_t)S * sizeof(float));
        gemv_fill_rand(tmpX, (size_t)S, 2);
        for (long i = 0; i < S; i++) ((double*)X)[i] = (double)tmpX[i];
        free(tmpX);
    }
    memset(Y, 0, (size_t)S * eb);

    mkl_set_num_threads(T);

#if HAVE_MKL
    sparse_matrix_t csrA;
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    if (p == PREC_FP32) {
        mkl_sparse_s_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO, (int)S, (int)S,
                                rowOff, rowOff + 1, colInd, (float*)vals);
    } else {
        mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO, (int)S, (int)S,
                                rowOff, rowOff + 1, colInd, (double*)vals);
    }
    mkl_sparse_optimize(csrA);

    /* Warmup */
    for (int w = 0; w < c->warmup; w++) {
        if (p == PREC_FP32) {
            mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0f, csrA, descr, (const float*)X, 0.0f, (float*)Y);
        } else {
            mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrA, descr, (const double*)X, 0.0, (double*)Y);
        }
    }

    /* Timed Loop */
    cpu_telemetry_start(telem);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

    for (int it = 0; it < c->iters; it++) {
        if (p == PREC_FP32) {
            mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0f, csrA, descr, (const float*)X, 0.0f, (float*)Y);
        } else {
            mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrA, descr, (const double*)X, 0.0, (double*)Y);
        }
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
    cpu_telemetry_stop(telem);
    mkl_sparse_destroy(csrA);
#else
    /* Fallback SpMV loop */
    cpu_telemetry_start(telem);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

    for (int it = 0; it < c->iters; it++) {
        #pragma omp parallel for schedule(static)
        for (long r = 0; r < S; r++) {
            double sum = 0.0;
            int start = rowOff[r], end = rowOff[r + 1];
            for (int idx = start; idx < end; idx++) {
                int col = colInd[idx];
                double v = (p == PREC_FP32) ? ((float*)vals)[idx] : ((double*)vals)[idx];
                double x = (p == PREC_FP32) ? ((float*)X)[col] : ((double*)X)[col];
                sum += v * x;
            }
            if (p == PREC_FP32) ((float*)Y)[r] = (float)sum;
            else ((double*)Y)[r] = sum;
        }
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
    cpu_telemetry_stop(telem);
#endif

    double wall_sec = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double comp_ms = (wall_sec * 1000.0) / (double)c->iters;

    /* CSR memory bandwidth: vals read + colInd read + rowOff read + X read + Y write */
    double bytes_transferred = ((double)nnz * (double)eb + (double)nnz * 4.0 + (double)S * 4.0 + 2.0 * (double)S * (double)eb) * (double)c->iters;
    double agg_gbps = (wall_sec > 0.0) ? (bytes_transferred / wall_sec / 1e9) : 0.0;
    double flops = 2.0 * (double)nnz * (double)c->iters;
    double agg_tflops = (wall_sec > 0.0) ? (flops / wall_sec / 1e12) : 0.0;
    double mem_gb = (double)(nnz * eb + nnz * 4 + (S + 1) * 4 + 2 * S * eb) / 1e9;

    printf("%-6s %-5s %-3d %6ld %6.2f %8.2f %7.3f %6.1fW %7.1f ok\n",
           "sparse", pn, T, S, density, agg_gbps, agg_tflops,
           telem->avg_power_watts, telem->avg_freq_mhz);
    fflush(stdout);

    if (csv) {
        fprintf(csv,
            "%s,%ld,%s,%s,sparse,split,%s,%d,%ld,%.4f,%ld,%d,%d,"
            "%.2f,%.4f,%.3f,%.3f,"
            "%.1f,%.1f,%.1f,"
            "%.3f,0,-1.0,ok\n",
            c->tag, ts, host, mkl_info, pn, T, S, density, nnz, c->iters, c->warmup,
            agg_gbps, agg_tflops, comp_ms, wall_sec * 1000.0,
            telem->avg_freq_mhz, telem->avg_power_watts, telem->max_temp_c,
            mem_gb);
        fflush(csv);
    }

    free(rowOff); free(colInd); free(vals);
    mkl_free(X); mkl_free(Y);
}

/* ── Main Entry Point ─────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    cfg_t c;
    memset(&c, 0, sizeof(c));

    c.iters = 15;
    c.warmup = 3;
    c.validate = 0;
    c.valid_max = 4096;
    c.tag = "cpu_gemv";

    int set_sizes = 0, set_precs = 0, set_threads = 0, set_engs = 0, set_dens = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--sizes") && i + 1 < argc) {
            c.n_sizes = parse_longs(argv[++i], c.sizes, 32); set_sizes = 1;
        } else if (!strcmp(a, "--precisions") && i + 1 < argc) {
            char b[128]; snprintf(b, sizeof(b), "%s", argv[++i]); c.n_precs = 0;
            for (char *t = strtok(b, ","); t; t = strtok(NULL, ",")) {
                if (!strcmp(t, "fp32")) c.precs[c.n_precs++] = PREC_FP32;
                else if (!strcmp(t, "fp64")) c.precs[c.n_precs++] = PREC_FP64;
            }
            set_precs = 1;
        } else if ((!strcmp(a, "--threads") || !strcmp(a, "--gpus")) && i + 1 < argc) {
            c.n_threads = parse_ints(argv[++i], c.threads, 32); set_threads = 1;
        } else if (!strcmp(a, "--engine") && i + 1 < argc) {
            char b[64]; snprintf(b, sizeof(b), "%s", argv[++i]); c.n_engines = 0;
            for (char *t = strtok(b, ","); t && c.n_engines < 2; t = strtok(NULL, ",")) {
                if (!strcmp(t, "dense")) c.engines[c.n_engines++] = ENG_DENSE;
                else if (!strcmp(t, "sparse")) c.engines[c.n_engines++] = ENG_SPARSE;
            }
            set_engs = 1;
        } else if (!strcmp(a, "--density") && i + 1 < argc) {
            c.n_densities = parse_doubles(argv[++i], c.densities, 8); set_dens = 1;
        } else if (!strcmp(a, "--iters") && i + 1 < argc) {
            c.iters = atoi(argv[++i]);
        } else if (!strcmp(a, "--warmup") && i + 1 < argc) {
            c.warmup = atoi(argv[++i]);
        } else if (!strcmp(a, "--validate")) {
            c.validate = 1;
        } else if (!strcmp(a, "--validate-max") && i + 1 < argc) {
            c.valid_max = atoi(argv[++i]);
        } else if (!strcmp(a, "--csv") && i + 1 < argc) {
            c.csv = argv[++i];
        } else if (!strcmp(a, "--tag") && i + 1 < argc) {
            c.tag = argv[++i];
        } else if (!strcmp(a, "--help")) {
            usage(argv[0]); return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", a);
            usage(argv[0]); return 2;
        }
    }

    if (!set_sizes) {
        long d[] = {1024, 2048, 4096, 8192, 16384};
        c.n_sizes = 5; memcpy(c.sizes, d, sizeof(d));
    }
    if (!set_precs) {
        c.precs[0] = PREC_FP32; c.precs[1] = PREC_FP64; c.n_precs = 2;
    }
    if (!set_threads) {
        int max_cores = omp_get_num_procs();
        if (max_cores <= 0) max_cores = 4;
        c.n_threads = 0;
        for (int t = 1; t <= max_cores; t *= 2) c.threads[c.n_threads++] = t;
        if (c.threads[c.n_threads - 1] != max_cores) c.threads[c.n_threads++] = max_cores;
    }
    if (!set_engs) {
        c.engines[0] = ENG_DENSE; c.engines[1] = ENG_SPARSE; c.n_engines = 2;
    }
    if (!set_dens) {
        double dd[] = {0.01, 0.05, 0.10};
        c.n_densities = 3; memcpy(c.densities, dd, sizeof(dd));
    }

    char host[128] = "Unknown_CPU";
    gethostname(host, sizeof(host));

    char mkl_info[128] = "oneMKL";
#if HAVE_MKL
    mkl_get_version_string(mkl_info, sizeof(mkl_info));
    for (int i = 0; mkl_info[i]; i++) if (mkl_info[i] == ',' || mkl_info[i] == ' ') mkl_info[i] = '_';
#endif

    cpu_telemetry_t telem;
    cpu_telemetry_init(&telem);

    FILE *csv = NULL;
    int new_file = 1;
    if (c.csv) {
        FILE *chk = fopen(c.csv, "r");
        if (chk) {
            fseek(chk, 0, SEEK_END);
            if (ftell(chk) > 0) new_file = 0;
            fclose(chk);
        }
        csv = fopen(c.csv, "a");
    }

    const char *HDR =
        "tag,ts,host,mkl_version,engine,mode,precision,threads,S,density,nnz,iters,warmup,"
        "agg_gbps,agg_tflops,comp_ms,wall_ms,"
        "cpu_mhz_avg,power_w_avg,temp_c_max,"
        "mem_gb,validated,max_rel_err,status\n";

    if (csv && new_file) {
        fputs(HDR, csv);
        fflush(csv);
    }

    printf("# CPU GEMV/SpMV Benchmark (Intel oneMKL / OpenMP) host=%s threads_max=%d\n",
           host, omp_get_num_procs());
    printf("%-6s %-5s %-3s %6s %6s %8s %7s %7s %7s %s\n",
           "eng", "prec", "T", "S", "dens", "GB/s", "TFLOPS", "power", "MHz", "status");

    for (int si = 0; si < c.n_sizes; si++) {
        long S = c.sizes[si];
        for (int ei = 0; ei < c.n_engines; ei++) {
            engine_t eng = c.engines[ei];
            for (int pi = 0; pi < c.n_precs; pi++) {
                prec_t p = c.precs[pi];
                for (int ti = 0; ti < c.n_threads; ti++) {
                    int T = c.threads[ti];
                    if (eng == ENG_DENSE) {
                        run_dense_gemv(&c, csv, host, mkl_info, S, p, T, &telem);
                    } else {
                        for (int di = 0; di < c.n_densities; di++) {
                            double dens = c.densities[di];
                            run_sparse_spmv(&c, csv, host, mkl_info, S, p, T, dens, &telem);
                        }
                    }
                }
            }
        }
    }

    if (csv) fclose(csv);
    return 0;
}
