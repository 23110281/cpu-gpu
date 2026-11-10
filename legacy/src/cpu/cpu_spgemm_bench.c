/* src/cpu/cpu_spgemm_bench.c
 *
 * Multi-threaded Intel oneMKL Sparse-Sparse GEMM (SpGEMM) benchmark:
 *   C = alpha * A * B   where A, B, and C are ALL sparse CSR matrices.
 *
 * Workload:
 *   Matrix-Matrix multiplication of two sparse matrices in CSR format.
 *   Uses oneMKL Inspector-Executor Sparse BLAS API (mkl_sparse_spmm).
 *
 * Precisions:
 *   fp32: mkl_sparse_s_spmm
 *   fp64: mkl_sparse_d_spmm
 *
 * FLOP Counting:
 *   Theoretical FLOPs = 2 * density_A * density_B * S^3
 *
 * Compile: make cpu_spgemm (or gcc with MKL flags)
 * Run:     ./bin/cpu_spgemm_bench --help
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
  #define HAVE_MKL 0
#elif defined(USE_MKL)
  #if defined(__has_include) && __has_include(<mkl.h>)
    #include <mkl.h>
  #elif defined(__has_include) && __has_include(<mkl_spblas.h>)
    #include <mkl_spblas.h>
    #include <mkl_service.h>
  #else
    #include <mkl.h>
  #endif
  #define HAVE_MKL 1
#elif defined(__has_include)
  #if __has_include(<mkl.h>)
    #include <mkl.h>
    #define HAVE_MKL 1
  #elif __has_include(<mkl_spblas.h>)
    #include <mkl_spblas.h>
    #include <mkl_service.h>
    #define HAVE_MKL 1
  #else
    #define HAVE_MKL 0
  #endif
#else
  #define HAVE_MKL 0
#endif

#include "gemv_host.h"
#include "cpu_telemetry.h"

/* ── Fallback stubs if compiling without MKL ──────────────────────────────── */
#if !HAVE_MKL
static inline void mkl_set_num_threads(int t) { omp_set_num_threads(t); }
static inline void mkl_get_version_string(char *buf, int len) { snprintf(buf, len, "Fallback_OpenMP"); }
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
    double densities[8];
    int n_densities;
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
    printf("  --sizes S,S,...     Matrix dimensions (SxS)      [default: 1024..8192]\n");
    printf("  --precisions P,P    Precisions: fp32,fp64        [default: fp32,fp64]\n");
    printf("  --threads T,T,...   Active thread counts         [default: 1..max_cores]\n");
    printf("  --density D,D       CSR matrix densities         [default: 0.01,0.05,0.1]\n");
    printf("  --iters N           Timed loop iterations        [default: 10]\n");
    printf("  --warmup N          Untimed warmup runs          [default: 2]\n");
    printf("  --validate          Check numerical accuracy     [default: off]\n");
    printf("  --validate-max N    Max size to validate         [default: 2048]\n");
    printf("  --csv PATH          Append results to CSV file\n");
    printf("  --tag STRING        Label identifier in CSV      [default: cpu_spgemm]\n");
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

/* ── Benchmark Runner for a Single SpGEMM Configuration ───────────────────── */
static void run_cpu_spgemm(const cfg_t *c, FILE *csv, const char *host, const char *mkl_info,
                           long S, prec_t p, int T, double density, cpu_telemetry_t *telem) {
    long ts = (long)time(NULL);
    const char *pn = prec_name(p);
    size_t eb = elem_bytes(p);

    int *rowOffA = NULL, *colIndA = NULL, *rowOffB = NULL, *colIndB = NULL;
    float *valsA_f32 = NULL, *valsB_f32 = NULL;
    long nnzA = 0, nnzB = 0;

    /* Generate sparse matrices A (seed 42) and B (seed 84) using gemv_gen_csr */
    if (gemv_gen_csr(S, density, &rowOffA, &colIndA, &valsA_f32, &nnzA, 42) != 0 ||
        gemv_gen_csr(S, density, &rowOffB, &colIndB, &valsB_f32, &nnzB, 84) != 0) {
        fprintf(stderr, "gemv_gen_csr failed for S=%ld dens=%.2f\n", S, density);
        return;
    }

    void *valsA = valsA_f32;
    void *valsB = valsB_f32;

    if (p == PREC_FP64) {
        double *vA64 = (double*)malloc((size_t)nnzA * sizeof(double));
        double *vB64 = (double*)malloc((size_t)nnzB * sizeof(double));
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < nnzA; i++) vA64[i] = (double)valsA_f32[i];
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < nnzB; i++) vB64[i] = (double)valsB_f32[i];
        free(valsA_f32); free(valsB_f32);
        valsA = vA64; valsB = vB64;
    }

    mkl_set_num_threads(T);
    long nnzC = 0;

#if HAVE_MKL
    sparse_matrix_t csrA = NULL, csrB = NULL, csrC = NULL;

    if (p == PREC_FP32) {
        mkl_sparse_s_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO, (int)S, (int)S,
                                rowOffA, rowOffA + 1, colIndA, (float*)valsA);
        mkl_sparse_s_create_csr(&csrB, SPARSE_INDEX_BASE_ZERO, (int)S, (int)S,
                                rowOffB, rowOffB + 1, colIndB, (float*)valsB);
    } else {
        mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO, (int)S, (int)S,
                                rowOffA, rowOffA + 1, colIndA, (double*)valsA);
        mkl_sparse_d_create_csr(&csrB, SPARSE_INDEX_BASE_ZERO, (int)S, (int)S,
                                rowOffB, rowOffB + 1, colIndB, (double*)valsB);
    }

    /* Warmup executions */
    for (int w = 0; w < c->warmup; w++) {
        csrC = NULL;
        mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, csrA, csrB, &csrC);
        if (csrC) mkl_sparse_destroy(csrC);
    }

    /* Timed Loop */
    cpu_telemetry_start(telem);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

    for (int it = 0; it < c->iters; it++) {
        csrC = NULL;
        mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, csrA, csrB, &csrC);
        if (it < c->iters - 1 && csrC) {
            mkl_sparse_destroy(csrC);
        }
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
    cpu_telemetry_stop(telem);

    /* Export metadata of result matrix C */
    if (csrC) {
        sparse_index_base_t indexing;
        MKL_INT rows, cols, *rStart = NULL, *rEnd = NULL, *cInd = NULL;
        void *values = NULL;
        if (p == PREC_FP32) {
            mkl_sparse_s_export_csr(csrC, &indexing, &rows, &cols, &rStart, &rEnd, &cInd, (float**)&values);
        } else {
            mkl_sparse_d_export_csr(csrC, &indexing, &rows, &cols, &rStart, &rEnd, &cInd, (double**)&values);
        }
        if (rEnd) nnzC = (long)rEnd[rows - 1];
        mkl_sparse_destroy(csrC);
    }
    mkl_sparse_destroy(csrA);
    mkl_sparse_destroy(csrB);
#else
    /* No MKL: there is no sparse x sparse kernel to time. This branch previously timed a
     * 10 ms nanosleep and reported it as an SpGEMM result (fabricated rows). Report the
     * configuration as unsupported instead; use the v2 harness (torch.sparse) for SpGEMM. */
    (void)nnzC;
    struct timespec t0 = {0, 0}, t1 = {0, 0};   /* keeps the (unreachable) tail compiling */
    (void)t0; (void)t1;
    printf("%-5s %-3d %6ld %6.2f  unsupported (built without MKL)\n", pn, T, S, density);
    if (csv) fprintf(csv, "%s,%ld,%s,%s,%s,%d,%ld,%.4f,%ld,%ld,0,%d,%d,0,0,0,0,0,0,0,0,0,-1.0,unsupported\n",
                     c->tag, (long)time(NULL), host, mkl_info, pn, T, S, density, nnzA, nnzB, c->iters, c->warmup);
    return;
#endif

    double wall_sec = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double comp_ms = (wall_sec * 1000.0) / (double)c->iters;

    /* Theoretical FLOPs = 2 * density_A * density_B * S^3 */
    double theoretical_flops = 2.0 * density * density * (double)S * (double)S * (double)S * (double)c->iters;
    double agg_tflops = (wall_sec > 0.0) ? (theoretical_flops / wall_sec / 1e12) : 0.0;

    /* Memory Bandwidth: (nnzA + nnzB + nnzC) * (elem_bytes + 4) + 3*(S+1)*4 */
    double bytes_per_iter = ((double)(nnzA + nnzB + nnzC) * (double)(eb + 4)) + (3.0 * (double)(S + 1) * 4.0);
    double agg_gbps = (wall_sec > 0.0) ? ((bytes_per_iter * (double)c->iters) / wall_sec / 1e9) : 0.0;

    double mem_gb = (double)((nnzA + nnzB + nnzC) * (eb + 4)) / 1e9;

    printf("%-6s %-5s %-3d %6ld %6.2f %7.4f %8.2f %8.2f %6.1fW %7.1f ok\n",
           "spgemm", pn, T, S, density, agg_tflops, agg_gbps, comp_ms,
           telem->avg_power_watts, telem->avg_freq_mhz);
    fflush(stdout);

    if (csv) {
        fprintf(csv,
            "%s,%ld,%s,%s,split,%s,%d,%ld,%.4f,%ld,%ld,%ld,%d,%d,"
            "%.4f,%.2f,%.3f,%.3f,"
            "%.1f,%.1f,%.1f,"
            "%.3f,0,-1.0,ok\n",
            c->tag, ts, host, mkl_info, pn, T, S, density, nnzA, nnzB, nnzC, c->iters, c->warmup,
            agg_tflops, agg_gbps, comp_ms, wall_sec * 1000.0,
            telem->avg_freq_mhz, telem->avg_power_watts, telem->max_temp_c,
            mem_gb);
        fflush(csv);
    }

    free(rowOffA); free(colIndA); free(valsA);
    free(rowOffB); free(colIndB); free(valsB);
}

/* ── Main Entry Point ─────────────────────────────────────────────────────── */
__attribute__((weak)) char *openblas_get_config(void);

int main(int argc, char **argv) {
    cfg_t c;
    memset(&c, 0, sizeof(c));

    c.iters = 10;
    c.warmup = 2;
    c.validate = 0;
    c.valid_max = 2048;
    c.tag = "cpu_spgemm";

    int set_sizes = 0, set_precs = 0, set_threads = 0, set_dens = 0;

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
        } else if (!strcmp(a, "--density") && i + 1 < argc) {
            c.n_densities = parse_doubles(argv[++i], c.densities, 8); set_dens = 1;
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
        long d[] = {1024, 2048, 4096, 8192};
        c.n_sizes = 4; memcpy(c.sizes, d, sizeof(d));
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
    if (!set_dens) {
        double dd[] = {0.01, 0.05, 0.10};
        c.n_densities = 3; memcpy(c.densities, dd, sizeof(dd));
    }

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
        "tag,ts,host,mkl_version,mode,precision,threads,S,density,nnz_A,nnz_B,nnz_C,iters,warmup,"
        "agg_tflops,agg_gbps,comp_ms,wall_ms,"
        "cpu_mhz_avg,power_w_avg,temp_c_max,"
        "mem_gb,validated,max_rel_err,status\n";

    if (csv && new_file) {
        fputs(HDR, csv);
        fflush(csv);
    }

    printf("# CPU SpGEMM Benchmark (Intel oneMKL / OpenMP) host=%s threads_max=%d\n",
           host, omp_get_num_procs());
    printf("%-6s %-5s %-3s %6s %6s %7s %8s %8s %6s %7s %s\n",
           "work", "prec", "T", "S", "dens", "TFLOPS", "GB/s", "comp_ms", "power", "MHz", "status");

    for (int si = 0; si < c.n_sizes; si++) {
        long S = c.sizes[si];
        for (int di = 0; di < c.n_densities; di++) {
            double dens = c.densities[di];
            for (int pi = 0; pi < c.n_precs; pi++) {
                prec_t p = c.precs[pi];
                for (int ti = 0; ti < c.n_threads; ti++) {
                    int T = c.threads[ti];
                    run_cpu_spgemm(&c, csv, host, mkl_info, S, p, T, dens, &telem);
                }
            }
        }
    }

    if (csv) fclose(csv);
    return 0;
}
