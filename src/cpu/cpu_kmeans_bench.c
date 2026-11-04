/* src/cpu/cpu_kmeans_bench.c
 *
 * Multi-threaded K-Means Clustering Benchmark.
 *
 * Lloyd's Algorithm Pipeline:
 *   1. Assignment Step: Find argmin_k || x_i - mu_k ||^2 for all N points.
 *      Uses thread-local accumulators to eliminate atomic contention.
 *   2. Update Step: mu_k = (1 / |S_k|) * sum_{i in S_k} x_i.
 *
 * Stresses memory bandwidth (points streaming from RAM/L3) interleaved with
 * Fused Multiply-Add (FMA) Euclidean distance reductions.
 *
 * Compile: gcc -std=c11 -O3 -march=native -fopenmp -I src/common -I src/cpu src/cpu/cpu_kmeans_bench.c -o bin/cpu_kmeans_bench -lm
 * Run:     ./bin/cpu_kmeans_bench --help
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
    int clusters;
    int features;
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

/* ── FP64 K-Means Engine ──────────────────────────────────────────────────── */
static double run_kmeans_f64(long N, int K, int D, int iters, int threads,
                             const double *points, double *centroids, int *assignments,
                             double *wcss_history) {
    /* Allocate thread-local accumulators to eliminate atomic write bottlenecks */
    double **local_centroids = (double **)malloc((size_t)threads * sizeof(double *));
    long **local_counts = (long **)malloc((size_t)threads * sizeof(long *));
    for (int t = 0; t < threads; t++) {
        local_centroids[t] = (double *)calloc((size_t)K * (size_t)D, sizeof(double));
        local_counts[t] = (long *)calloc((size_t)K, sizeof(long));
    }

    double final_wcss = 0.0;

    for (int it = 0; it < iters; it++) {
        for (int t = 0; t < threads; t++) {
            memset(local_centroids[t], 0, (size_t)K * (size_t)D * sizeof(double));
            memset(local_counts[t], 0, (size_t)K * sizeof(long));
        }

        double iter_wcss = 0.0;

        /* Phase 1: Assignment Step */
        #pragma omp parallel num_threads(threads) reduction(+:iter_wcss)
        {
            int tid = omp_get_thread_num();
            double *t_cent = local_centroids[tid];
            long *t_cnt = local_counts[tid];

            #pragma omp for schedule(static)
            for (long i = 0; i < N; i++) {
                const double *pt = &points[i * D];
                double min_dist = 1e30;
                int best_k = 0;

                for (int k = 0; k < K; k++) {
                    const double *c = &centroids[k * D];
                    double dist = 0.0;
                    for (int d = 0; d < D; d++) {
                        double diff = pt[d] - c[d];
                        dist += diff * diff;
                    }
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_k = k;
                    }
                }

                assignments[i] = best_k;
                iter_wcss += min_dist;

                t_cnt[best_k]++;
                double *dst = &t_cent[best_k * D];
                for (int d = 0; d < D; d++) {
                    dst[d] += pt[d];
                }
            }
        }

        wcss_history[it] = iter_wcss;
        final_wcss = iter_wcss;

        /* Phase 2: Update Step (Reduction across thread-local buffers) */
        for (int k = 0; k < K; k++) {
            long total_count = 0;
            for (int t = 0; t < threads; t++) {
                total_count += local_counts[t][k];
            }

            if (total_count > 0) {
                double inv_c = 1.0 / (double)total_count;
                for (int d = 0; d < D; d++) {
                    double sum_d = 0.0;
                    for (int t = 0; t < threads; t++) {
                        sum_d += local_centroids[t][k * D + d];
                    }
                    centroids[k * D + d] = sum_d * inv_c;
                }
            }
        }
    }

    for (int t = 0; t < threads; t++) {
        free(local_centroids[t]);
        free(local_counts[t]);
    }
    free(local_centroids);
    free(local_counts);

    return final_wcss;
}

/* ── FP32 K-Means Engine ──────────────────────────────────────────────────── */
static float run_kmeans_f32(long N, int K, int D, int iters, int threads,
                            const float *points, float *centroids, int *assignments,
                            float *wcss_history) {
    float **local_centroids = (float **)malloc((size_t)threads * sizeof(float *));
    long **local_counts = (long **)malloc((size_t)threads * sizeof(long *));
    for (int t = 0; t < threads; t++) {
        local_centroids[t] = (float *)calloc((size_t)K * (size_t)D, sizeof(float));
        local_counts[t] = (long *)calloc((size_t)K, sizeof(long));
    }

    float final_wcss = 0.0f;

    for (int it = 0; it < iters; it++) {
        for (int t = 0; t < threads; t++) {
            memset(local_centroids[t], 0, (size_t)K * (size_t)D * sizeof(float));
            memset(local_counts[t], 0, (size_t)K * sizeof(long));
        }

        double iter_wcss = 0.0;

        #pragma omp parallel num_threads(threads) reduction(+:iter_wcss)
        {
            int tid = omp_get_thread_num();
            float *t_cent = local_centroids[tid];
            long *t_cnt = local_counts[tid];

            #pragma omp for schedule(static)
            for (long i = 0; i < N; i++) {
                const float *pt = &points[i * D];
                float min_dist = 1e30f;
                int best_k = 0;

                for (int k = 0; k < K; k++) {
                    const float *c = &centroids[k * D];
                    float dist = 0.0f;
                    for (int d = 0; d < D; d++) {
                        float diff = pt[d] - c[d];
                        dist += diff * diff;
                    }
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_k = k;
                    }
                }

                assignments[i] = best_k;
                iter_wcss += (double)min_dist;

                t_cnt[best_k]++;
                float *dst = &t_cent[best_k * D];
                for (int d = 0; d < D; d++) {
                    dst[d] += pt[d];
                }
            }
        }

        wcss_history[it] = (float)iter_wcss;
        final_wcss = (float)iter_wcss;

        for (int k = 0; k < K; k++) {
            long total_count = 0;
            for (int t = 0; t < threads; t++) {
                total_count += local_counts[t][k];
            }

            if (total_count > 0) {
                float inv_c = 1.0f / (float)total_count;
                for (int d = 0; d < D; d++) {
                    float sum_d = 0.0f;
                    for (int t = 0; t < threads; t++) {
                        sum_d += local_centroids[t][k * D + d];
                    }
                    centroids[k * D + d] = sum_d * inv_c;
                }
            }
        }
    }

    for (int t = 0; t < threads; t++) {
        free(local_centroids[t]);
        free(local_counts[t]);
    }
    free(local_centroids);
    free(local_counts);

    return final_wcss;
}

/* ── Benchmark Driver Function ────────────────────────────────────────────── */
static void run_kmeans_benchmark(const cfg_t *c, FILE *csv, const char *host,
                                 long N, int K, int D, prec_t p, int T, cpu_telemetry_t *telem) {
    const char *pn = prec_name(p);
    size_t elem_size = (p == PREC_FP32) ? sizeof(float) : sizeof(double);
    double mem_gb = (double)(N * D + K * D) * (double)elem_size / 1e9;

    void *points = NULL, *centroids = NULL, *wcss_hist = NULL;
    int *assignments = (int *)malloc(N * sizeof(int));
    posix_memalign(&points, 64, N * D * elem_size);
    posix_memalign(&centroids, 64, K * D * elem_size);
    posix_memalign(&wcss_hist, 64, c->iters * elem_size);

    if (!points || !centroids || !assignments || !wcss_hist) {
        fprintf(stderr, "Allocation failed for K-Means (N=%ld, K=%d, D=%d)\n", N, K, D);
        return;
    }

    /* Initialize points and initial centroids deterministically */
    if (p == PREC_FP32) {
        float *fpts = (float *)points;
        float *fcents = (float *)centroids;
        #pragma omp parallel for num_threads(T) schedule(static)
        for (long i = 0; i < N * D; i++) fpts[i] = (float)((i * 1337 + 13) % 1000) / 100.0f;
        for (int k = 0; k < K * D; k++) fcents[k] = fpts[k * (N / (K * D > 0 ? K : 1))];
    } else {
        double *dpts = (double *)points;
        double *dcents = (double *)centroids;
        #pragma omp parallel for num_threads(T) schedule(static)
        for (long i = 0; i < N * D; i++) dpts[i] = (double)((i * 1337 + 13) % 1000) / 100.0;
        for (int k = 0; k < K * D; k++) dcents[k] = dpts[k * (N / (K * D > 0 ? K : 1))];
    }

    /* Untimed Warmup */
    for (int it = 0; it < c->warmup; it++) {
        if (p == PREC_FP32) {
            run_kmeans_f32(N > 50000 ? 50000 : N, K, D, 2, T, (float *)points, (float *)centroids, assignments, (float *)wcss_hist);
        } else {
            run_kmeans_f64(N > 50000 ? 50000 : N, K, D, 2, T, (double *)points, (double *)centroids, assignments, (double *)wcss_hist);
        }
    }

    /* Timed Region with Telemetry */
    cpu_telemetry_start(telem);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

    double final_wcss = 0.0;
    if (p == PREC_FP32) {
        final_wcss = (double)run_kmeans_f32(N, K, D, c->iters, T, (float *)points, (float *)centroids, assignments, (float *)wcss_hist);
    } else {
        final_wcss = run_kmeans_f64(N, K, D, c->iters, T, (double *)points, (double *)centroids, assignments, (double *)wcss_hist);
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
    cpu_telemetry_stop(telem);

    double wall_sec = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double comp_ms = (wall_sec * 1000.0) / (double)c->iters;
    double total_point_iters = (double)N * (double)c->iters;
    double mpoints_sec = (wall_sec > 0.0) ? (total_point_iters / wall_sec / 1e6) : 0.0;

    /* FLOPs per point-centroid comparison: 2 * D (subtraction + multiply-accumulate) */
    double total_flops = 2.0 * (double)N * (double)K * (double)D * (double)c->iters;
    double agg_tflops = (wall_sec > 0.0) ? (total_flops / wall_sec / 1e12) : 0.0;

    /* Validate monotonic convergence of WCSS */
    int validated = 0;
    int valid_ok = 1;
    if (c->validate) {
        if (p == PREC_FP32) {
            float *hist = (float *)wcss_hist;
            for (int it = 1; it < c->iters; it++) {
                if (hist[it] > hist[it - 1] * 1.001f) valid_ok = 0; /* Should not increase */
            }
        } else {
            double *hist = (double *)wcss_hist;
            for (int it = 1; it < c->iters; it++) {
                if (hist[it] > hist[it - 1] * 1.001) valid_ok = 0;
            }
        }
        validated = 1;
    }

    const char *status = (validated && !valid_ok) ? "HIERR" : "ok";

    printf("%-6s %-5s %-3d %9ld %4d %3d %8.2f %7.3f %8.2f %6.1fW %7.1f %s\n",
           "kmeans", pn, T, N, K, D, mpoints_sec, agg_tflops, comp_ms,
           telem->avg_power_watts, telem->avg_freq_mhz, status);
    fflush(stdout);

    if (csv) {
        time_t ts = time(NULL);
        fprintf(csv,
            "%s,%ld,%s,OpenMP,lloyd_kmeans,%s,%d,%ld,%d,%d,%d,%d,"
            "%.2f,%.4f,%.3f,%.3f,"
            "%.1f,%.1f,%.1f,%.3f,%.2e,%d,%s\n",
            c->tag, (long)ts, host, pn, T, N, K, D, c->iters, c->warmup,
            mpoints_sec, agg_tflops, comp_ms, wall_sec * 1000.0,
            telem->avg_freq_mhz, telem->avg_power_watts, telem->max_temp_c,
            mem_gb, final_wcss, validated, status);
        fflush(csv);
    }

    free(points); free(centroids); free(assignments); free(wcss_hist);
}

static void usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --sizes N,N,...     Number of points N        [default: 200K,1M,5M]\n");
    printf("  --clusters K        Number of clusters K      [default: 32]\n");
    printf("  --features D        Feature dimensions D      [default: 32]\n");
    printf("  --precisions P,P    Precisions: fp32,fp64     [default: fp32,fp64]\n");
    printf("  --threads T,T,...   Active thread counts      [default: 1..max_cores]\n");
    printf("  --iters N           Clustering iterations     [default: 15]\n");
    printf("  --warmup N          Untimed warmup runs       [default: 2]\n");
    printf("  --validate          Verify monotonic WCSS convergence\n");
    printf("  --csv PATH          Append results to CSV file\n");
    printf("  --tag STRING        Label identifier in CSV   [default: cpu_kmeans]\n");
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

    c.clusters = 32;
    c.features = 32;
    c.iters = 15;
    c.warmup = 2;
    c.validate = 0;
    c.tag = "cpu_kmeans";

    int set_sizes = 0, set_precs = 0, set_threads = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--sizes") && i + 1 < argc) {
            c.n_sizes = parse_longs(argv[++i], c.sizes, 32); set_sizes = 1;
        } else if (!strcmp(a, "--clusters") && i + 1 < argc) {
            c.clusters = atoi(argv[++i]);
        } else if (!strcmp(a, "--features") && i + 1 < argc) {
            c.features = atoi(argv[++i]);
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
        long d[] = {200000, 1000000, 5000000};
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
        "tag,ts,host,engine,mode,precision,threads,points,clusters,features,iters,warmup,"
        "mpoints_sec,agg_tflops,comp_ms,wall_ms,"
        "cpu_mhz_avg,power_w_avg,temp_c_max,mem_gb,final_wcss,validated,status\n";

    if (csv && new_file) {
        fputs(HDR, csv);
        fflush(csv);
    }

    printf("# K-Means Clustering Benchmark host=%s threads_max=%d\n",
           host, omp_get_num_procs());
    printf("%-6s %-5s %-3s %9s %4s %3s %8s %7s %8s %7s %7s %s\n",
           "work", "prec", "T", "points", "K", "D", "Mpts/s", "TFLOPS", "comp_ms", "power", "MHz", "status");

    for (int si = 0; si < c.n_sizes; si++) {
        long N = c.sizes[si];
        for (int pi = 0; pi < c.n_precs; pi++) {
            prec_t p = c.precs[pi];
            for (int ti = 0; ti < c.n_threads; ti++) {
                int T = c.threads[ti];
                run_kmeans_benchmark(&c, csv, host, N, c.clusters, c.features, p, T, &telem);
            }
        }
    }

    if (csv) fclose(csv);
    return 0;
}
