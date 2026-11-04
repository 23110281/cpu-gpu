/* src/cpu/cpu_pca_bench.c
 *
 * Multi-threaded Principal Component Analysis (PCA) Benchmark.
 *
 * Algorithm Pipeline:
 *   1. Mean Centering: Compute column-wise feature means mu, center X -> X_tilde.
 *   2. Sample Covariance Matrix: Sigma = (1 / (N - 1)) * X_tilde^T * X_tilde  (D x D).
 *   3. Top-k Eigensolver: Multithreaded Power Iteration with Gram-Schmidt orthogonalization.
 *   4. Feature Projection: Y = X_tilde * V_k  (N x k).
 *
 * Stresses memory streaming (mean/centering/projection) and compute-dense outer-products
 * (sample covariance accumulation).
 *
 * Compile: gcc -std=c11 -O3 -march=native -fopenmp -I src/common -I src/cpu src/cpu/cpu_pca_bench.c -o bin/cpu_pca_bench -lm
 * Run:     ./bin/cpu_pca_bench --help
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

#include "cpu_telemetry.h"

#ifndef CLOCK_MONOTONIC_RAW
#define CLOCK_MONOTONIC_RAW 4
#endif

typedef enum { PREC_FP32, PREC_FP64 } prec_t;

static const char *prec_name(prec_t p) {
    return p == PREC_FP32 ? "fp32" : "fp64";
}

typedef struct {
    long sizes[32];
    int n_sizes;
    int features;
    int components;
    prec_t precs[4];
    int n_precs;
    int threads[32];
    int n_threads;
    int iters;
    int warmup;
    int validate;
    const char *csv;
    const char *tag;
} cfg_t;

/* ── FP64 PCA Pipeline ────────────────────────────────────────────────────── */
static double run_pca_f64(long N, int D, int K, int threads,
                          const double *X, double *X_centered, double *cov,
                          double *eigvecs, double *eigvals, double *Y) {
    /* 1. Mean Centering */
    double *means = (double *)calloc(D, sizeof(double));
    if (!means) return -1.0;

    #pragma omp parallel num_threads(threads)
    {
        double *local_means = (double *)calloc(D, sizeof(double));
        #pragma omp for schedule(static)
        for (long i = 0; i < N; i++) {
            for (int j = 0; j < D; j++) {
                local_means[j] += X[i * D + j];
            }
        }
        #pragma omp critical
        {
            for (int j = 0; j < D; j++) {
                means[j] += local_means[j];
            }
        }
        free(local_means);
    }

    for (int j = 0; j < D; j++) {
        means[j] /= (double)N;
    }

    #pragma omp parallel for num_threads(threads) schedule(static) collapse(2)
    for (long i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            X_centered[i * D + j] = X[i * D + j] - means[j];
        }
    }
    free(means);

    /* 2. Covariance Matrix Accumulation: cov = X_centered^T * X_centered / (N - 1) */
    memset(cov, 0, (size_t)D * (size_t)D * sizeof(double));

    #pragma omp parallel num_threads(threads)
    {
        double *local_cov = (double *)calloc((size_t)D * (size_t)D, sizeof(double));
        #pragma omp for schedule(static)
        for (long i = 0; i < N; i++) {
            const double *row = &X_centered[i * D];
            for (int j = 0; j < D; j++) {
                double rj = row[j];
                for (int l = j; l < D; l++) {
                    local_cov[j * D + l] += rj * row[l];
                }
            }
        }
        #pragma omp critical
        {
            for (int j = 0; j < D; j++) {
                for (int l = j; l < D; l++) {
                    cov[j * D + l] += local_cov[j * D + l];
                }
            }
        }
        free(local_cov);
    }

    double inv_n1 = 1.0 / (double)(N > 1 ? N - 1 : 1);
    for (int j = 0; j < D; j++) {
        for (int l = j; l < D; l++) {
            cov[j * D + l] *= inv_n1;
            cov[l * D + j] = cov[j * D + l]; /* Symmetric mirror */
        }
    }

    /* 3. Top-k Eigensolver via Power Iteration with Gram-Schmidt */
    double *v = (double *)malloc(D * sizeof(double));
    double *w = (double *)malloc(D * sizeof(double));

    for (int k = 0; k < K; k++) {
        /* Initialize deterministic probe vector */
        for (int j = 0; j < D; j++) v[j] = 1.0 / sqrt((double)D);

        double lambda = 0.0;
        int max_power_iters = 25;
        for (int iter = 0; iter < max_power_iters; iter++) {
            /* Gram-Schmidt deflation against previous eigenvectors */
            for (int prev = 0; prev < k; prev++) {
                double dot = 0.0;
                for (int j = 0; j < D; j++) dot += v[j] * eigvecs[prev * D + j];
                for (int j = 0; j < D; j++) v[j] -= dot * eigvecs[prev * D + j];
            }

            /* Normalize */
            double norm = 0.0;
            for (int j = 0; j < D; j++) norm += v[j] * v[j];
            norm = sqrt(norm);
            if (norm > 1e-12) {
                for (int j = 0; j < D; j++) v[j] /= norm;
            }

            /* Matrix-Vector multiplication: w = cov * v */
            #pragma omp parallel for num_threads(threads) schedule(static)
            for (int j = 0; j < D; j++) {
                double sum = 0.0;
                for (int l = 0; l < D; l++) sum += cov[j * D + l] * v[l];
                w[j] = sum;
            }

            /* Rayleigh quotient for eigenvalue */
            lambda = 0.0;
            for (int j = 0; j < D; j++) lambda += v[j] * w[j];

            /* Update probe vector */
            memcpy(v, w, D * sizeof(double));
        }

        /* Store final normalized eigenvector and eigenvalue */
        double norm = 0.0;
        for (int j = 0; j < D; j++) norm += v[j] * v[j];
        norm = sqrt(norm);
        if (norm > 1e-12) {
            for (int j = 0; j < D; j++) v[j] /= norm;
        }

        memcpy(&eigvecs[k * D], v, D * sizeof(double));
        eigvals[k] = lambda;
    }

    free(v);
    free(w);

    /* 4. Projection: Y = X_centered * V_k^T  (N x K) */
    #pragma omp parallel for num_threads(threads) schedule(static) collapse(2)
    for (long i = 0; i < N; i++) {
        for (int k = 0; k < K; k++) {
            double dot = 0.0;
            for (int j = 0; j < D; j++) {
                dot += X_centered[i * D + j] * eigvecs[k * D + j];
            }
            Y[i * K + k] = dot;
        }
    }

    /* Compute max orthogonality error for validation: max |v_a . v_b - delta_ab| */
    double max_ortho_err = 0.0;
    for (int a = 0; a < K; a++) {
        for (int b = 0; b < K; b++) {
            double dot = 0.0;
            for (int j = 0; j < D; j++) dot += eigvecs[a * D + j] * eigvecs[b * D + j];
            double expected = (a == b) ? 1.0 : 0.0;
            double err = fabs(dot - expected);
            if (err > max_ortho_err) max_ortho_err = err;
        }
    }

    return max_ortho_err;
}

/* ── FP32 PCA Pipeline ────────────────────────────────────────────────────── */
static float run_pca_f32(long N, int D, int K, int threads,
                         const float *X, float *X_centered, float *cov,
                         float *eigvecs, float *eigvals, float *Y) {
    float *means = (float *)calloc(D, sizeof(float));
    if (!means) return -1.0f;

    #pragma omp parallel num_threads(threads)
    {
        float *local_means = (float *)calloc(D, sizeof(float));
        #pragma omp for schedule(static)
        for (long i = 0; i < N; i++) {
            for (int j = 0; j < D; j++) local_means[j] += X[i * D + j];
        }
        #pragma omp critical
        {
            for (int j = 0; j < D; j++) means[j] += local_means[j];
        }
        free(local_means);
    }

    for (int j = 0; j < D; j++) means[j] /= (float)N;

    #pragma omp parallel for num_threads(threads) schedule(static) collapse(2)
    for (long i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) X_centered[i * D + j] = X[i * D + j] - means[j];
    }
    free(means);

    memset(cov, 0, (size_t)D * (size_t)D * sizeof(float));
    #pragma omp parallel num_threads(threads)
    {
        float *local_cov = (float *)calloc((size_t)D * (size_t)D, sizeof(float));
        #pragma omp for schedule(static)
        for (long i = 0; i < N; i++) {
            const float *row = &X_centered[i * D];
            for (int j = 0; j < D; j++) {
                float rj = row[j];
                for (int l = j; l < D; l++) local_cov[j * D + l] += rj * row[l];
            }
        }
        #pragma omp critical
        {
            for (int j = 0; j < D; j++) {
                for (int l = j; l < D; l++) cov[j * D + l] += local_cov[j * D + l];
            }
        }
        free(local_cov);
    }

    float inv_n1 = 1.0f / (float)(N > 1 ? N - 1 : 1);
    for (int j = 0; j < D; j++) {
        for (int l = j; l < D; l++) {
            cov[j * D + l] *= inv_n1;
            cov[l * D + j] = cov[j * D + l];
        }
    }

    float *v = (float *)malloc(D * sizeof(float));
    float *w = (float *)malloc(D * sizeof(float));

    for (int k = 0; k < K; k++) {
        for (int j = 0; j < D; j++) v[j] = 1.0f / sqrtf((float)D);
        float lambda = 0.0f;
        int max_power_iters = 25;

        for (int iter = 0; iter < max_power_iters; iter++) {
            for (int prev = 0; prev < k; prev++) {
                float dot = 0.0f;
                for (int j = 0; j < D; j++) dot += v[j] * eigvecs[prev * D + j];
                for (int j = 0; j < D; j++) v[j] -= dot * eigvecs[prev * D + j];
            }

            float norm = 0.0f;
            for (int j = 0; j < D; j++) norm += v[j] * v[j];
            norm = sqrtf(norm);
            if (norm > 1e-6f) {
                for (int j = 0; j < D; j++) v[j] /= norm;
            }

            #pragma omp parallel for num_threads(threads) schedule(static)
            for (int j = 0; j < D; j++) {
                float sum = 0.0f;
                for (int l = 0; l < D; l++) sum += cov[j * D + l] * v[l];
                w[j] = sum;
            }

            lambda = 0.0f;
            for (int j = 0; j < D; j++) lambda += v[j] * w[j];
            memcpy(v, w, D * sizeof(float));
        }

        float norm = 0.0f;
        for (int j = 0; j < D; j++) norm += v[j] * v[j];
        norm = sqrtf(norm);
        if (norm > 1e-6f) {
            for (int j = 0; j < D; j++) v[j] /= norm;
        }

        memcpy(&eigvecs[k * D], v, D * sizeof(float));
        eigvals[k] = lambda;
    }

    free(v);
    free(w);

    #pragma omp parallel for num_threads(threads) schedule(static) collapse(2)
    for (long i = 0; i < N; i++) {
        for (int k = 0; k < K; k++) {
            float dot = 0.0f;
            for (int j = 0; j < D; j++) dot += X_centered[i * D + j] * eigvecs[k * D + j];
            Y[i * K + k] = dot;
        }
    }

    float max_ortho_err = 0.0f;
    for (int a = 0; a < K; a++) {
        for (int b = 0; b < K; b++) {
            float dot = 0.0f;
            for (int j = 0; j < D; j++) dot += eigvecs[a * D + j] * eigvecs[b * D + j];
            float expected = (a == b) ? 1.0f : 0.0f;
            float err = fabsf(dot - expected);
            if (err > max_ortho_err) max_ortho_err = err;
        }
    }

    return max_ortho_err;
}

/* ── Benchmark Driver Function ────────────────────────────────────────────── */
static void run_pca_benchmark(const cfg_t *c, FILE *csv, const char *host,
                              long N, int D, int K, prec_t p, int T, cpu_telemetry_t *telem) {
    const char *pn = prec_name(p);
    size_t elem_size = (p == PREC_FP32) ? sizeof(float) : sizeof(double);
    double mem_gb = (double)(N * D + N * D + D * D + K * D + K + N * K) * (double)elem_size / 1e9;

    void *X = NULL, *X_centered = NULL, *cov = NULL, *eigvecs = NULL, *eigvals = NULL, *Y = NULL;
    posix_memalign(&X, 64, N * D * elem_size);
    posix_memalign(&X_centered, 64, N * D * elem_size);
    posix_memalign(&cov, 64, D * D * elem_size);
    posix_memalign(&eigvecs, 64, K * D * elem_size);
    posix_memalign(&eigvals, 64, K * elem_size);
    posix_memalign(&Y, 64, N * K * elem_size);

    if (!X || !X_centered || !cov || !eigvecs || !eigvals || !Y) {
        fprintf(stderr, "Allocation failed for PCA (N=%ld, D=%d)\n", N, D);
        return;
    }

    /* Initialize X with deterministic pseudo-random values */
    if (p == PREC_FP32) {
        float *fX = (float *)X;
        #pragma omp parallel for num_threads(T) schedule(static)
        for (long i = 0; i < N * D; i++) fX[i] = (float)((i * 1337 + 17) % 1000) / 100.0f;
    } else {
        double *dX = (double *)X;
        #pragma omp parallel for num_threads(T) schedule(static)
        for (long i = 0; i < N * D; i++) dX[i] = (double)((i * 1337 + 17) % 1000) / 100.0;
    }

    /* Untimed Warmup */
    for (int it = 0; it < c->warmup; it++) {
        if (p == PREC_FP32) {
            run_pca_f32(N > 50000 ? 50000 : N, D, K, T, (float *)X, (float *)X_centered, (float *)cov,
                        (float *)eigvecs, (float *)eigvals, (float *)Y);
        } else {
            run_pca_f64(N > 50000 ? 50000 : N, D, K, T, (double *)X, (double *)X_centered, (double *)cov,
                        (double *)eigvecs, (double *)eigvals, (double *)Y);
        }
    }

    /* Timed Region with Telemetry */
    cpu_telemetry_start(telem);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

    double max_ortho_err = 0.0;
    for (int it = 0; it < c->iters; it++) {
        if (p == PREC_FP32) {
            max_ortho_err = (double)run_pca_f32(N, D, K, T, (float *)X, (float *)X_centered, (float *)cov,
                                                (float *)eigvecs, (float *)eigvals, (float *)Y);
        } else {
            max_ortho_err = run_pca_f64(N, D, K, T, (double *)X, (double *)X_centered, (double *)cov,
                                        (double *)eigvecs, (double *)eigvals, (double *)Y);
        }
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
    cpu_telemetry_stop(telem);

    double wall_sec = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double comp_ms = (wall_sec * 1000.0) / (double)c->iters;
    double total_samples = (double)N * (double)c->iters;
    double msamples_sec = (wall_sec > 0.0) ? (total_samples / wall_sec / 1e6) : 0.0;

    /* Total FLOPs approximation: Covariance N*D*D + Projection 2*N*D*K */
    double flops_per_run = (double)N * (double)D * (double)D + 2.0 * (double)N * (double)D * (double)K;
    double agg_tflops = (wall_sec > 0.0) ? ((flops_per_run * (double)c->iters) / wall_sec / 1e12) : 0.0;

    int validated = c->validate ? 1 : 0;
    const char *status = (validated && max_ortho_err > 0.01) ? "HIERR" : "ok";

    printf("%-6s %-5s %-3d %9ld %4d %3d %8.2f %7.3f %8.2f %6.1fW %7.1f %s\n",
           "pca", pn, T, N, D, K, msamples_sec, agg_tflops, comp_ms,
           telem->avg_power_watts, telem->avg_freq_mhz, status);
    fflush(stdout);

    if (csv) {
        time_t ts = time(NULL);
        fprintf(csv,
            "%s,%ld,%s,OpenMP,pca_eigensolver,%s,%d,%ld,%d,%d,%d,%d,"
            "%.2f,%.4f,%.3f,%.3f,"
            "%.1f,%.1f,%.1f,%.3f,%d,%.3e,%s\n",
            c->tag, (long)ts, host, pn, T, N, D, K, c->iters, c->warmup,
            msamples_sec, agg_tflops, comp_ms, wall_sec * 1000.0,
            telem->avg_freq_mhz, telem->avg_power_watts, telem->max_temp_c,
            mem_gb, validated, max_ortho_err, status);
        fflush(csv);
    }

    free(X); free(X_centered); free(cov); free(eigvecs); free(eigvals); free(Y);
}

static void usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --sizes N,N,...     Number of samples N       [default: 100K,500K,2M]\n");
    printf("  --features D        Feature dimensions D      [default: 64]\n");
    printf("  --components K      Top-k principal axes K    [default: 8]\n");
    printf("  --precisions P,P    Precisions: fp32,fp64     [default: fp32,fp64]\n");
    printf("  --threads T,T,...   Active thread counts      [default: 1..max_cores]\n");
    printf("  --iters N           Timed loop iterations     [default: 5]\n");
    printf("  --warmup N          Untimed warmup runs       [default: 1]\n");
    printf("  --validate          Verify eigenvector orthonormality\n");
    printf("  --csv PATH          Append results to CSV file\n");
    printf("  --tag STRING        Label identifier in CSV   [default: cpu_pca]\n");
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

int main(int argc, char **argv) {
    cfg_t c;
    memset(&c, 0, sizeof(c));

    c.features = 64;
    c.components = 8;
    c.iters = 5;
    c.warmup = 1;
    c.validate = 0;
    c.tag = "cpu_pca";

    int set_sizes = 0, set_precs = 0, set_threads = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--sizes") && i + 1 < argc) {
            c.n_sizes = parse_longs(argv[++i], c.sizes, 32); set_sizes = 1;
        } else if (!strcmp(a, "--features") && i + 1 < argc) {
            c.features = atoi(argv[++i]);
        } else if (!strcmp(a, "--components") && i + 1 < argc) {
            c.components = atoi(argv[++i]);
        } else if (!strcmp(a, "--precisions") && i + 1 < argc) {
            char b[128]; snprintf(b, sizeof(b), "%s", argv[++i]); c.n_precs = 0;
            for (char *t = strtok(b, ","); t; t = strtok(NULL, ",")) {
                if (!strcmp(t, "fp32")) c.precs[c.n_precs++] = PREC_FP32;
                else if (!strcmp(t, "fp64")) c.precs[c.n_precs++] = PREC_FP64;
            }
            set_precs = 1;
        } else if (!strcmp(a, "--threads") && i + 1 < argc) {
            c.n_threads = parse_ints(argv[++i], c.threads, 32); set_threads = 1;
        } else if (!strcmp(a, "--iters") && i + 1 < argc) {
            c.iters = atoi(argv[++i]);
        } else if (!strcmp(a, "--warmup") && i + 1 < argc) {
            c.warmup = atoi(argv[++i]);
        } else if (!strcmp(a, "--validate")) {
            c.validate = 1;
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
        long d[] = {100000, 500000, 2000000};
        c.n_sizes = 3; memcpy(c.sizes, d, sizeof(d));
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
        "tag,ts,host,engine,mode,precision,threads,samples,features,components,iters,warmup,"
        "msamples_sec,agg_tflops,comp_ms,wall_ms,"
        "cpu_mhz_avg,power_w_avg,temp_c_max,mem_gb,validated,max_ortho_err,status\n";

    if (csv && new_file) {
        fputs(HDR, csv);
        fflush(csv);
    }

    printf("# Principal Component Analysis (PCA) Benchmark host=%s threads_max=%d\n",
           host, omp_get_num_procs());
    printf("%-6s %-5s %-3s %9s %4s %3s %8s %7s %8s %7s %7s %s\n",
           "work", "prec", "T", "samples", "feat", "dim", "MSamp/s", "TFLOPS", "comp_ms", "power", "MHz", "status");

    for (int si = 0; si < c.n_sizes; si++) {
        long N = c.sizes[si];
        for (int pi = 0; pi < c.n_precs; pi++) {
            prec_t p = c.precs[pi];
            for (int ti = 0; ti < c.n_threads; ti++) {
                int T = c.threads[ti];
                run_pca_benchmark(&c, csv, host, N, c.features, c.components, p, T, &telem);
            }
        }
    }

    if (csv) fclose(csv);
    return 0;
}
