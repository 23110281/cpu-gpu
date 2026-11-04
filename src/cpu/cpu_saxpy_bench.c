/* src/cpu/cpu_saxpy_bench.c
 *
 * Multi-threaded Intel oneMKL SAXPY / DAXPY benchmark (y = alpha * x + y).
 *
 * Precisions:
 *   fp32: cblas_saxpy
 *   fp64: cblas_daxpy
 *
 * Arithmetic Intensity:
 *   - fp32: 2 FLOPs / (3 * 4 bytes) = 0.167 FLOPs/byte
 *   - fp64: 2 FLOPs / (3 * 8 bytes) = 0.083 FLOPs/byte
 * Strictly memory-bandwidth bound at large vector sizes (N).
 * Headline metric: achieved memory bandwidth (GB/s).
 *
 * Compile: make cpu_saxpy (or gcc with MKL flags)
 * Run:     ./bin/cpu_saxpy_bench --help
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
#include "cpu_telemetry.h"

/* ── Fallback stubs if compiling without CBLAS ───────────────────────────── */
#if !HAVE_CBLAS
static inline void cblas_saxpy(const int N, const float alpha,
                               const float *X, const int incX,
                               float *Y, const int incY) {
    (void)incX; (void)incY;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) Y[i] = alpha * X[i] + Y[i];
}

static inline void cblas_daxpy(const int N, const double alpha,
                               const float *X, const int incX,
                               double *Y, const int incY) {
    (void)incX; (void)incY;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) Y[i] = alpha * X[i] + Y[i];
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

static const char *prec_name(prec_t p) { return p == PREC_FP32 ? "fp32" : "fp64"; }
static size_t elem_bytes(prec_t p) { return p == PREC_FP32 ? 4 : 8; }

typedef struct {
    long sizes[32];
    int n_sizes;
    prec_t precs[4];
    int n_precs;
    int threads[32];
    int n_threads;
    int iters;
    int warmup;
    int validate;
    long valid_max;
    const char *csv;
    const char *tag;
} cfg_t;

static void usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --sizes N,N,...     Vector elements (N)          [default: 1M..128M]\n");
    printf("  --precisions P,P    Precisions: fp32,fp64        [default: fp32,fp64]\n");
    printf("  --threads T,T,...   Active thread counts         [default: 1..max_cores]\n");
    printf("  --iters N           Timed loop iterations        [default: 20]\n");
    printf("  --warmup N          Untimed warmup runs          [default: 3]\n");
    printf("  --validate          Check numerical accuracy     [default: off]\n");
    printf("  --validate-max N    Max vector size to validate  [default: 33554432]\n");
    printf("  --csv PATH          Append results to CSV file\n");
    printf("  --tag STRING        Label identifier in CSV      [default: cpu_saxpy]\n");
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

/* ── Validation for Vector Addition ───────────────────────────────────────── */
static double validate_axpy(long N, prec_t p, double alpha, const void *X, const void *Y_orig, const void *Y) {
    int check_samples = 64;
    double max_rel = 0.0;
    for (int s = 0; s < check_samples; s++) {
        long idx = (s * 104729 + 17) % N;
        double x_val = (p == PREC_FP32) ? ((const float*)X)[idx] : ((const double*)X)[idx];
        double y_orig_val = (p == PREC_FP32) ? ((const float*)Y_orig)[idx] : ((const double*)Y_orig)[idx];
        double ref = alpha * x_val + y_orig_val;
        double got = (p == PREC_FP32) ? ((const float*)Y)[idx] : ((const double*)Y)[idx];

        double denom = fabs(ref) > 1e-12 ? fabs(ref) : 1e-12;
        double rel = fabs(got - ref) / denom;
        if (rel > max_rel) max_rel = rel;
    }
    return max_rel;
}

/* ── Benchmark Runner for Single Vector Size ──────────────────────────────── */
static void run_cpu_saxpy(const cfg_t *c, FILE *csv, const char *host, const char *mkl_info,
                          long N, prec_t p, int T, cpu_telemetry_t *telem) {
    long ts = (long)time(NULL);
    const char *pn = prec_name(p);
    size_t eb = elem_bytes(p);
    size_t vec_bytes = (size_t)N * eb;
    double mem_gb = (double)(2 * vec_bytes) / 1e9;

    void *X = mkl_malloc(vec_bytes, 64);
    void *Y = mkl_malloc(vec_bytes, 64);
    void *Y_orig = NULL;
    if (c->validate && N <= c->valid_max) {
        Y_orig = malloc(vec_bytes);
    }

    if (!X || !Y || (c->validate && N <= c->valid_max && !Y_orig)) {
        printf("%-6s %-5s %-3d %10ld %9s %8s %s\n", "axpy", pn, T, N, "-", "-", "oom-skip");
        if (csv) {
            fprintf(csv, "%s,%ld,%s,%s,split,%s,%d,%ld,%d,%d,"
                         "0.0,0.0,0.0,0.0,0.0,0.0,0.0,%.3f,0,-1.0,oom-skip\n",
                    c->tag, ts, host, mkl_info, pn, T, N, c->iters, c->warmup, mem_gb);
        }
        if (X) mkl_free(X); if (Y) mkl_free(Y); if (Y_orig) free(Y_orig);
        return;
    }

    /* Initialize vectors */
    if (p == PREC_FP32) {
        gemv_fill_rand((float*)X, (size_t)N, 1);
        gemv_fill_rand((float*)Y, (size_t)N, 2);
        if (Y_orig) memcpy(Y_orig, Y, vec_bytes);
    } else {
        float *tmp = (float*)malloc(vec_bytes / 2);
        gemv_fill_rand(tmp, (size_t)N, 1);
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < N; i++) ((double*)X)[i] = (double)tmp[i];
        gemv_fill_rand(tmp, (size_t)N, 2);
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < N; i++) ((double*)Y)[i] = (double)tmp[i];
        free(tmp);
        if (Y_orig) memcpy(Y_orig, Y, vec_bytes);
    }

    mkl_set_num_threads(T);
    const float alpha_f32 = 2.5f;
    const double alpha_f64 = 2.5;

    /* Warmup */
    for (int w = 0; w < c->warmup; w++) {
        if (p == PREC_FP32) cblas_saxpy((int)N, alpha_f32, (const float*)X, 1, (float*)Y, 1);
        else cblas_daxpy((int)N, alpha_f64, (const double*)X, 1, (double*)Y, 1);
    }

    /* Timed Loop */
    cpu_telemetry_start(telem);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

    for (int it = 0; it < c->iters; it++) {
        if (p == PREC_FP32) cblas_saxpy((int)N, alpha_f32, (const float*)X, 1, (float*)Y, 1);
        else cblas_daxpy((int)N, alpha_f64, (const double*)X, 1, (double*)Y, 1);
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
    cpu_telemetry_stop(telem);

    double wall_sec = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double comp_ms = (wall_sec * 1000.0) / (double)c->iters;

    /* Bandwidth: 2 reads (X, Y) + 1 write (Y) = 3 * N * elem_bytes per iteration */
    double bytes_transferred = 3.0 * (double)N * (double)eb * (double)c->iters;
    double agg_gbps = (wall_sec > 0.0) ? (bytes_transferred / wall_sec / 1e9) : 0.0;
    double flops = 2.0 * (double)N * (double)c->iters;
    double agg_tflops = (wall_sec > 0.0) ? (flops / wall_sec / 1e12) : 0.0;

    double max_rel = -1.0;
    int validated = 0;
    if (c->validate && N <= c->valid_max && Y_orig) {
        /* Run one single fresh reference step from Y_orig to test accuracy */
        if (p == PREC_FP32) {
            float *test_y = (float*)malloc(vec_bytes);
            memcpy(test_y, Y_orig, vec_bytes);
            cblas_saxpy((int)N, alpha_f32, (const float*)X, 1, test_y, 1);
            max_rel = validate_axpy(N, p, (double)alpha_f32, X, Y_orig, test_y);
            free(test_y);
        } else {
            double *test_y = (double*)malloc(vec_bytes);
            memcpy(test_y, Y_orig, vec_bytes);
            cblas_daxpy((int)N, alpha_f64, (const double*)X, 1, test_y, 1);
            max_rel = validate_axpy(N, p, alpha_f64, X, Y_orig, test_y);
            free(test_y);
        }
        validated = 1;
    }

    printf("%-6s %-5s %-3d %10ld %8.2f %7.4f %7.2f %6.1fW %7.1f %s\n",
           "axpy", pn, T, N, agg_gbps, agg_tflops, comp_ms,
           telem->avg_power_watts, telem->avg_freq_mhz,
           (validated && max_rel > 1e-2) ? "HIERR" : "ok");
    fflush(stdout);

    if (csv) {
        fprintf(csv,
            "%s,%ld,%s,%s,split,%s,%d,%ld,%d,%d,"
            "%.2f,%.5f,%.3f,%.3f,"
            "%.1f,%.1f,%.1f,"
            "%.3f,%d,%.3e,ok\n",
            c->tag, ts, host, mkl_info, pn, T, N, c->iters, c->warmup,
            agg_gbps, agg_tflops, comp_ms, wall_sec * 1000.0,
            telem->avg_freq_mhz, telem->avg_power_watts, telem->max_temp_c,
            mem_gb, validated, max_rel);
        fflush(csv);
    }

    mkl_free(X); mkl_free(Y);
    if (Y_orig) free(Y_orig);
}

/* ── Main Entry Point ─────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    cfg_t c;
    memset(&c, 0, sizeof(c));

    c.iters = 20;
    c.warmup = 3;
    c.validate = 0;
    c.valid_max = 33554432;
    c.tag = "cpu_saxpy";

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
            }
            set_precs = 1;
        } else if ((!strcmp(a, "--threads") || !strcmp(a, "--gpus")) && i + 1 < argc) {
            c.n_threads = parse_ints(argv[++i], c.threads, 32); set_threads = 1;
        } else if (!strcmp(a, "--iters") && i + 1 < argc) {
            c.iters = atoi(argv[++i]);
        } else if (!strcmp(a, "--warmup") && i + 1 < argc) {
            c.warmup = atoi(argv[++i]);
        } else if (!strcmp(a, "--validate")) {
            c.validate = 1;
        } else if (!strcmp(a, "--validate-max") && i + 1 < argc) {
            c.valid_max = atol(argv[++i]);
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
        long d[] = {1048576, 4194304, 16777216, 33554432, 67108864, 134217728};
        c.n_sizes = 6; memcpy(c.sizes, d, sizeof(d));
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
        "tag,ts,host,mkl_version,mode,precision,threads,N,iters,warmup,"
        "agg_gbps,agg_tflops,comp_ms,wall_ms,"
        "cpu_mhz_avg,power_w_avg,temp_c_max,"
        "mem_gb,validated,max_rel_err,status\n";

    if (csv && new_file) {
        fputs(HDR, csv);
        fflush(csv);
    }

    printf("# CPU SAXPY/DAXPY Benchmark (Intel oneMKL / OpenMP) host=%s threads_max=%d\n",
           host, omp_get_num_procs());
    printf("%-6s %-5s %-3s %10s %8s %7s %7s %7s %7s %s\n",
           "work", "prec", "T", "N", "GB/s", "TFLOPS", "comp_ms", "power", "MHz", "status");

    for (int si = 0; si < c.n_sizes; si++) {
        long N = c.sizes[si];
        for (int pi = 0; pi < c.n_precs; pi++) {
            prec_t p = c.precs[pi];
            for (int ti = 0; ti < c.n_threads; ti++) {
                int T = c.threads[ti];
                run_cpu_saxpy(&c, csv, host, mkl_info, N, p, T, &telem);
            }
        }
    }

    if (csv) fclose(csv);
    return 0;
}
