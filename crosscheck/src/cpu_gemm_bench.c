/* src/cpu/cpu_gemm_bench.c
 *
 * Supported precisions:
 *   fp32: cblas_sgemm
 *   fp64: cblas_dgemm
 *   bf16: cblas_gemm_bf16bf16f32
 *
 * Scaling:
 *   Sweeps problem sizes (S) x precisions x thread counts (T) x modes.
 *   Emits identical CSV schema for direct ingestion by results/plotting/gpu/generate_plots.py.
 *
 * Compile: make cpu_gemm
 * Run:     ./bin/cpu_gemm_bench --help
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

/* ── Fallback stubs if compiling on a machine without standard CBLAS headers ─ */
#if !HAVE_CBLAS
typedef enum { CblasRowMajor=101, CblasColMajor=102 } CBLAS_LAYOUT;
typedef enum { CblasNoTrans=111, CblasTrans=112 } CBLAS_TRANSPOSE;

static inline void cblas_sgemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                 const float alpha, const float *A, const int lda,
                 const float *B, const int ldb, const float beta,
                 float *C, const int ldc) {
    (void)Layout; (void)TransA; (void)TransB; (void)lda; (void)ldb; (void)ldc;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

static inline void cblas_dgemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                 const double alpha, const double *A, const int lda,
                 const double *B, const int ldb, const double beta,
                 double *C, const int ldc) {
    (void)Layout; (void)TransA; (void)TransB; (void)lda; (void)ldb; (void)ldc;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}
#endif

/* ── Fallback stubs for non-MKL environments (OpenBLAS / generic) ────────── */
#if !HAVE_MKL
typedef unsigned short MKL_BF16;

static inline void cblas_gemm_bf16bf16f32(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                            const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                            const float alpha, const MKL_BF16 *A, const int lda,
                            const MKL_BF16 *B, const int ldb, const float beta,
                            float *C, const int ldc) {
#if HAVE_CBLAS
    size_t szA = (size_t)M * (size_t)K;
    size_t szB = (size_t)K * (size_t)N;
    float *fA = (float *)malloc(szA * sizeof(float));
    float *fB = (float *)malloc(szB * sizeof(float));
    if (fA && fB) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < szA; i++) fA[i] = bf16_to_f32(A[i]);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < szB; i++) fB[i] = bf16_to_f32(B[i]);

        cblas_sgemm(Layout, TransA, TransB, M, N, K, alpha, fA, lda, fB, ldb, beta, C, ldc);

        free(fA);
        free(fB);
        return;
    }
    if (fA) free(fA);
    if (fB) free(fB);
#endif
    (void)Layout; (void)TransA; (void)TransB; (void)lda; (void)ldb; (void)ldc;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += bf16_to_f32(A[i * K + k]) * bf16_to_f32(B[k * N + j]);
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

static inline void *mkl_malloc(size_t size, int align) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, align, size) != 0) return NULL;
    return ptr;
}

static inline void mkl_free(void *ptr) {
    free(ptr);
}

__attribute__((weak)) void openblas_set_num_threads(int num_threads);

static inline void mkl_set_num_threads(int t) {
    if (openblas_set_num_threads) openblas_set_num_threads(t);
    omp_set_num_threads(t);
}

static inline void mkl_get_version_string(char *buf, int len) {
#if HAVE_CBLAS
    snprintf(buf, len, "OpenBLAS_Engine");
#else
    snprintf(buf, len, "Fallback_OpenMP_Engine");
#endif
}
#endif

/* ── Precision & Benchmark Configurations ─────────────────────────────────── */
typedef enum { PREC_FP32, PREC_FP64, PREC_BF16 } prec_t;
typedef enum { MODE_SPLIT, MODE_REPLICAS } bench_mode_t;

static const char *prec_name(prec_t p) {
    return p == PREC_FP32 ? "fp32" : (p == PREC_FP64 ? "fp64" : "bf16");
}

static size_t elem_bytes(prec_t p) {
    return p == PREC_FP32 ? 4 : (p == PREC_FP64 ? 8 : 2);
}

typedef struct {
    long sizes[32];
    int n_sizes;
    prec_t precs[4];
    int n_precs;
    int threads[32];
    int n_threads;
    bench_mode_t mode;
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
    printf("  --sizes S,S,...     Matrix dimensions (SxS)      [default: 512..16384]\n");
    printf("  --precisions P,P    Precisions: fp32,fp64,bf16   [default: fp32,fp64,bf16]\n");
    printf("  --threads T,T,...   Active thread counts         [default: 1..max_cores]\n");
    printf("  --gpus T,T,...      Alias for --threads (for script compatibility)\n");
    printf("  --mode split|replicas Execution mode             [default: split]\n");
    printf("  --iters N           Timed loop iterations        [default: 15]\n");
    printf("  --warmup N          Untimed warmup runs          [default: 3]\n");
    printf("  --validate          Check numerical accuracy     [default: off]\n");
    printf("  --validate-max N    Max size to validate         [default: 2048]\n");
    printf("  --csv PATH          Append results to CSV file\n");
    printf("  --tag STRING        Label identifier in CSV      [default: cpu_gemm]\n");
    printf("  --help              Display this help message\n");
}

static int parse_ints(const char *s, int *out, int max) {
    int n = 0;
    char buf[512];
    snprintf(buf, sizeof(buf), "%s", s);
    for (char *t = strtok(buf, ","); t && n < max; t = strtok(NULL, ","))
        out[n++] = atoi(t);
    return n;
}

static int parse_longs(const char *s, long *out, int max) {
    int n = 0;
    char buf[512];
    snprintf(buf, sizeof(buf), "%s", s);
    for (char *t = strtok(buf, ","); t && n < max; t = strtok(NULL, ","))
        out[n++] = atol(t);
    return n;
}

/* ── Validation Helper ────────────────────────────────────────────────────── */
static double validate_gemm(long S, prec_t p, const void *hA, const void *hB, const void *hC) {
    int check_samples = 32;
    double max_rel = 0.0;

    for (int sample = 0; sample < check_samples; sample++) {
        long r = (sample * 1337 + 7) % S;
        long c = (sample * 31337 + 13) % S;

        double ref = 0.0;
        for (long k = 0; k < S; k++) {
            double a_val, b_val;
            if (p == PREC_FP32) {
                a_val = ((const float*)hA)[r * S + k];
                b_val = ((const float*)hB)[k * S + c];
            } else if (p == PREC_FP64) {
                a_val = ((const double*)hA)[r * S + k];
                b_val = ((const double*)hB)[k * S + c];
            } else {
                a_val = bf16_to_f32(((const uint16_t*)hA)[r * S + k]);
                b_val = bf16_to_f32(((const uint16_t*)hB)[k * S + c]);
            }
            ref += a_val * b_val;
        }

        double got = 0.0;
        if (p == PREC_FP32) got = ((const float*)hC)[r * S + c];
        else if (p == PREC_FP64) got = ((const double*)hC)[r * S + c];
        else got = ((const float*)hC)[r * S + c]; /* bf16 output accumulates in float */

        double denom = fabs(ref) > 1e-12 ? fabs(ref) : 1e-12;
        double rel = fabs(got - ref) / denom;
        if (rel > max_rel) max_rel = rel;
    }
    return max_rel;
}

/* ── Benchmark Runner for a Single Configuration ─────────────────────────── */
static void run_cpu_gemm(const cfg_t *c, FILE *csv, const char *host, const char *mkl_info,
                         long S, prec_t p, int T, cpu_telemetry_t *telem) {
    long ts = (long)time(NULL);
    const char *mode_s = (c->mode == MODE_SPLIT) ? "split" : "replicas";
    const char *pn = prec_name(p);
    size_t bytes_in = (size_t)S * S * elem_bytes(p);
    size_t bytes_out = (size_t)S * S * (p == PREC_FP64 ? 8 : 4);
    double mem_gb = (double)(2 * bytes_in + bytes_out) / 1e9;

    /* 64-byte aligned memory allocations for vector execution */
    void *A = mkl_malloc(bytes_in, 64);
    void *B = mkl_malloc(bytes_in, 64);
    void *C = mkl_malloc(bytes_out, 64);

    if (!A || !B || !C) {
        printf("%-6s %-5s %-3d %6ld %9s %8s %s\n", "dense", pn, T, S, "-", "-", "oom-skip");
        if (csv) {
            fprintf(csv, "%s,%ld,%s,%s,%s,%s,%d,%ld,%ld,%ld,%d,%d,"
                         "0.0,0.0,0.0,"
                         "0.0,0.0,0.0,"
                         "%.3f,0,-1.0,oom-skip\n",
                    c->tag, ts, host, mkl_info, mode_s, pn, T, S, S, S, c->iters, c->warmup,
                    mem_gb);
        }
        if (A) mkl_free(A);
        if (B) mkl_free(B);
        if (C) mkl_free(C);
        return;
    }

    /* Initialize inputs using deterministic host PRNG from gemv_host.h */
    if (p == PREC_FP32) {
        gemv_fill_rand((float*)A, (size_t)S * S, 1);
        gemv_fill_rand((float*)B, (size_t)S * S, 2);
    } else if (p == PREC_FP64) {
        float *tmp = (float*)malloc((size_t)S * S * sizeof(float));
        gemv_fill_rand(tmp, (size_t)S * S, 1);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < (size_t)S * S; i++) ((double*)A)[i] = (double)tmp[i];
        gemv_fill_rand(tmp, (size_t)S * S, 2);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < (size_t)S * S; i++) ((double*)B)[i] = (double)tmp[i];
        free(tmp);
    } else {
        float *tmp = (float*)malloc((size_t)S * S * sizeof(float));
        gemv_fill_rand(tmp, (size_t)S * S, 1);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < (size_t)S * S; i++) ((uint16_t*)A)[i] = f32_to_bf16(tmp[i]);
        gemv_fill_rand(tmp, (size_t)S * S, 2);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < (size_t)S * S; i++) ((uint16_t*)B)[i] = f32_to_bf16(tmp[i]);
        free(tmp);
    }
    memset(C, 0, bytes_out);

    /* Set OpenMP thread concurrency */
    mkl_set_num_threads(T);

    /* Warmup executions */
    for (int w = 0; w < c->warmup; w++) {
        if (p == PREC_FP32) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        (int)S, (int)S, (int)S, 1.0f, (const float*)A, (int)S,
                        (const float*)B, (int)S, 0.0f, (float*)C, (int)S);
        } else if (p == PREC_FP64) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        (int)S, (int)S, (int)S, 1.0, (const double*)A, (int)S,
                        (const double*)B, (int)S, 0.0, (double*)C, (int)S);
        } else {
            cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                  (int)S, (int)S, (int)S, 1.0f,
                                  (const MKL_BF16*)A, (int)S,
                                  (const MKL_BF16*)B, (int)S, 0.0f,
                                  (float*)C, (int)S);
        }
    }

    /* Timed Region with Telemetry */
    cpu_telemetry_start(telem);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

    for (int it = 0; it < c->iters; it++) {
        if (p == PREC_FP32) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        (int)S, (int)S, (int)S, 1.0f, (const float*)A, (int)S,
                        (const float*)B, (int)S, 0.0f, (float*)C, (int)S);
        } else if (p == PREC_FP64) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        (int)S, (int)S, (int)S, 1.0, (const double*)A, (int)S,
                        (const double*)B, (int)S, 0.0, (double*)C, (int)S);
        } else {
            cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                  (int)S, (int)S, (int)S, 1.0f,
                                  (const MKL_BF16*)A, (int)S,
                                  (const MKL_BF16*)B, (int)S, 0.0f,
                                  (float*)C, (int)S);
        }
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
    cpu_telemetry_stop(telem);

    double wall_sec = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double comp_ms = (wall_sec * 1000.0) / (double)c->iters;
    double total_flops = 2.0 * (double)S * (double)S * (double)S * (double)c->iters;
    double agg_tflops = (wall_sec > 0.0) ? (total_flops / wall_sec / 1e12) : 0.0;

    /* Numerical accuracy validation */
    double max_rel = -1.0;
    int validated = 0;
    if (c->validate && S <= c->valid_max) {
        max_rel = validate_gemm(S, p, A, B, C);
        validated = 1;
    }

    /* Print console log */
    printf("%-6s %-5s %-3d %6ld %9.2f %8.2f %6.1fW %7.1f %s\n",
           "dense", pn, T, S, agg_tflops, comp_ms,
           telem->avg_power_watts, telem->avg_freq_mhz,
           (validated && max_rel > 1e-2) ? "HIERR" : "ok");
    fflush(stdout);

    /* Append to CPU-specific CSV */
    if (csv) {
        fprintf(csv,
            "%s,%ld,%s,%s,%s,%s,%d,%ld,%ld,%ld,%d,%d,"
            "%.2f,%.3f,%.3f,"
            "%.1f,%.1f,%.1f,"
            "%.3f,%d,%.3e,%s\n",
            c->tag, ts, host, mkl_info, mode_s, pn, T, S, S, S, c->iters, c->warmup,
            agg_tflops, comp_ms, wall_sec * 1000.0,
            telem->avg_freq_mhz, telem->avg_power_watts, telem->max_temp_c,
            mem_gb, validated, max_rel,
            (validated && max_rel > 1e-2) ? "validation_failed" : "ok");
        fflush(csv);
    }

    mkl_free(A);
    mkl_free(B);
    mkl_free(C);
}

/* ── Main Entry Point ─────────────────────────────────────────────────────── */
__attribute__((weak)) char *openblas_get_config(void);

int main(int argc, char **argv) {
    cfg_t c;
    memset(&c, 0, sizeof(c));

    c.mode = MODE_SPLIT;
    c.iters = 15;
    c.warmup = 3;
    c.validate = 0;
    c.valid_max = 2048;
    c.tag = "cpu_gemm";

    int set_sizes = 0, set_precs = 0, set_threads = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--sizes") && i + 1 < argc) {
            c.n_sizes = parse_longs(argv[++i], c.sizes, 32); set_sizes = 1;
        } else if (!strcmp(a, "--precisions") && i + 1 < argc) {
            char b[128]; snprintf(b, sizeof(b), "%s", argv[++i]); c.n_precs = 0;
            for (char *t = strtok(b, ","); t; t = strtok(NULL, ",")) {
                if (!strcmp(t, "fp32")) c.precs[c.n_precs++] = PREC_FP32;
                else if (!strcmp(t, "fp64")) c.precs[c.n_precs++] = PREC_FP64;
                else if (!strcmp(t, "bf16")) c.precs[c.n_precs++] = PREC_BF16;
            }
            set_precs = 1;
        } else if ((!strcmp(a, "--threads") || !strcmp(a, "--gpus")) && i + 1 < argc) {
            c.n_threads = parse_ints(argv[++i], c.threads, 32); set_threads = 1;
        } else if (!strcmp(a, "--mode") && i + 1 < argc) {
            c.mode = strcmp(argv[++i], "replicas") ? MODE_SPLIT : MODE_REPLICAS;
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

    /* Set default parameters if not provided */
    if (!set_sizes) {
        long d[] = {512, 1024, 2048, 4096, 8192, 16384};
        c.n_sizes = 6;
        memcpy(c.sizes, d, sizeof(d));
    }
    if (!set_precs) {
        c.precs[0] = PREC_FP32; c.precs[1] = PREC_FP64; c.precs[2] = PREC_BF16;
        c.n_precs = 3;
    }
    if (!set_threads) {
        int max_cores = omp_get_num_procs();
        if (max_cores <= 0) max_cores = 4;
        c.n_threads = 0;
        for (int t = 1; t <= max_cores; t *= 2) c.threads[c.n_threads++] = t;
        if (c.threads[c.n_threads - 1] != max_cores) c.threads[c.n_threads++] = max_cores;
    }

    /* Host info */
    char host[128] = "Unknown_CPU";
    gethostname(host, sizeof(host));

    char mkl_info[128] = "unknown_BLAS";
#if HAVE_MKL
    mkl_get_version_string(mkl_info, sizeof(mkl_info));
#else
    /* label the library actually linked (was hard-coded "oneMKL" under OpenBLAS) */
    snprintf(mkl_info, sizeof(mkl_info), "%s", openblas_get_config ? openblas_get_config() : "OpenBLAS");
#endif
    for (int i = 0; mkl_info[i]; i++) if (mkl_info[i] == ',' || mkl_info[i] == ' ') mkl_info[i] = '_';

    /* Setup Telemetry */
    cpu_telemetry_t telem;
    cpu_telemetry_init(&telem);

    /* Open CSV file */
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
        "tag,ts,host,mkl_version,mode,precision,threads,M,N,K,iters,warmup,"
        "agg_tflops,comp_ms,wall_ms,"
        "cpu_mhz_avg,power_w_avg,temp_c_max,"
        "mem_gb,validated,max_rel_err,status\n";

    if (csv && new_file) {
        fputs(HDR, csv);
        fflush(csv);
    }

    printf("# CPU GEMM Benchmark (Intel oneMKL / OpenMP) host=%s threads_max=%d\n",
           host, omp_get_num_procs());
    printf("# Precisions: fp32, fp64, bf16 (Intel AMX / AVX-512)\n");
    printf("%-6s %-5s %-3s %6s %9s %8s %7s %7s %s\n",
           "eng", "prec", "T", "N", "aggTF", "comp_ms", "power", "MHz", "status");

    for (int si = 0; si < c.n_sizes; si++) {
        long S = c.sizes[si];
        for (int pi = 0; pi < c.n_precs; pi++) {
            prec_t p = c.precs[pi];
            for (int ti = 0; ti < c.n_threads; ti++) {
                int T = c.threads[ti];
                run_cpu_gemm(&c, csv, host, mkl_info, S, p, T, &telem);
            }
        }
    }

    if (csv) fclose(csv);
    return 0;
}
