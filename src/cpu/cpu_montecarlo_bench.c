/* src/cpu/cpu_montecarlo_bench.c
 *
 * Multi-threaded Monte Carlo European Option Pricing Benchmark (Black-Scholes model).
 *
 * Simulates asset paths under Geometric Brownian Motion:
 *   S_T = S_0 * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z),  Z ~ N(0, 1)
 *
 * Call Payoff:
 *   C = max(S_T - K, 0) * exp(-r * T)
 *
 * Stresses CPU floating-point registers and transcendental SIMD instructions (exp, sqrt, log)
 * without memory bandwidth bottlenecks.
 *
 * Compile: gcc -std=c11 -O3 -march=native -fopenmp -I src/common -I src/cpu src/cpu/cpu_montecarlo_bench.c -o bin/cpu_montecarlo_bench -lm
 * Run:     ./bin/cpu_montecarlo_bench --help
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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef enum { PREC_FP32, PREC_FP64 } prec_t;

static const char *prec_name(prec_t p) {
    return p == PREC_FP32 ? "fp32" : "fp64";
}

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
    const char *csv;
    const char *tag;
} cfg_t;

/* ── Fast Thread-Local XORShift128+ PRNG ──────────────────────────────────── */
typedef struct {
    uint64_t s[2];
} xorshift128p_state_t;

static inline uint64_t xorshift128p_next(xorshift128p_state_t *st) {
    uint64_t s1 = st->s[0];
    const uint64_t s0 = st->s[1];
    st->s[0] = s0;
    s1 ^= s1 << 23;
    st->s[1] = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26);
    return st->s[1] + s0;
}

static inline void xorshift128p_seed(xorshift128p_state_t *st, uint64_t seed) {
    uint64_t z = seed + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    st->s[0] = z ^ (z >> 31);
    z = (st->s[0] + seed) + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    st->s[1] = z ^ (z >> 31);
    if (st->s[0] == 0 && st->s[1] == 0) {
        st->s[0] = 1337;
        st->s[1] = 42;
    }
}

/* Uniform random number in (0, 1] */
static inline double rand_uniform_f64(xorshift128p_state_t *st) {
    return ((double)(xorshift128p_next(st) >> 11) + 1.0) / 9007199254740993.0;
}

static inline float rand_uniform_f32(xorshift128p_state_t *st) {
    return ((float)(xorshift128p_next(st) >> 40) + 1.0f) / 16777217.0f;
}

/* Standard normal variable via Box-Muller transform */
static inline void box_muller_f64(xorshift128p_state_t *st, double *z0, double *z1) {
    double u1 = rand_uniform_f64(st);
    double u2 = rand_uniform_f64(st);
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;
    *z0 = r * cos(theta);
    *z1 = r * sin(theta);
}

static inline void box_muller_f32(xorshift128p_state_t *st, float *z0, float *z1) {
    float u1 = rand_uniform_f32(st);
    float u2 = rand_uniform_f32(st);
    float r = sqrtf(-2.0f * logf(u1));
    float theta = (float)(2.0 * M_PI) * u2;
    *z0 = r * cosf(theta);
    *z1 = r * sinf(theta);
}

/* ── Closed-form Black-Scholes Analytical Call Option Price ──────────────── */
static double black_scholes_call_analytical(double S0, double K, double T, double r, double sigma) {
    double sig_sqrt_t = sigma * sqrt(T);
    double d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / sig_sqrt_t;
    double d2 = d1 - sig_sqrt_t;
    double cdf_d1 = 0.5 * (1.0 + erf(d1 / sqrt(2.0)));
    double cdf_d2 = 0.5 * (1.0 + erf(d2 / sqrt(2.0)));
    return S0 * cdf_d1 - K * exp(-r * T) * cdf_d2;
}

/* ── Monte Carlo Kernel (FP64) ────────────────────────────────────────────── */
static double run_montecarlo_f64(long total_paths, int threads,
                                 double S0, double K, double T, double r, double sigma) {
    double drift = (r - 0.5 * sigma * sigma) * T;
    double vol = sigma * sqrt(T);
    double discount = exp(-r * T);

    double global_payoff_sum = 0.0;
    long pairs = total_paths / 2;

    #pragma omp parallel num_threads(threads) reduction(+:global_payoff_sum)
    {
        int tid = omp_get_thread_num();
        xorshift128p_state_t rng;
        xorshift128p_seed(&rng, 42ULL + (uint64_t)tid * 1000003ULL);

        #pragma omp for schedule(static)
        for (long i = 0; i < pairs; i++) {
            double z0, z1;
            box_muller_f64(&rng, &z0, &z1);

            double sT0 = S0 * exp(drift + vol * z0);
            double sT1 = S0 * exp(drift + vol * z1);

            double p0 = sT0 > K ? (sT0 - K) : 0.0;
            double p1 = sT1 > K ? (sT1 - K) : 0.0;

            global_payoff_sum += (p0 + p1);
        }
    }

    return discount * (global_payoff_sum / (double)total_paths);
}

/* ── Monte Carlo Kernel (FP32) ────────────────────────────────────────────── */
static float run_montecarlo_f32(long total_paths, int threads,
                                float S0, float K, float T, float r, float sigma) {
    float drift = (r - 0.5f * sigma * sigma) * T;
    float vol = sigma * sqrtf(T);
    float discount = expf(-r * T);

    double global_payoff_sum = 0.0;
    long pairs = total_paths / 2;

    #pragma omp parallel num_threads(threads) reduction(+:global_payoff_sum)
    {
        int tid = omp_get_thread_num();
        xorshift128p_state_t rng;
        xorshift128p_seed(&rng, 42ULL + (uint64_t)tid * 1000003ULL);

        #pragma omp for schedule(static)
        for (long i = 0; i < pairs; i++) {
            float z0, z1;
            box_muller_f32(&rng, &z0, &z1);

            float sT0 = S0 * expf(drift + vol * z0);
            float sT1 = S0 * expf(drift + vol * z1);

            float p0 = sT0 > K ? (sT0 - K) : 0.0f;
            float p1 = sT1 > K ? (sT1 - K) : 0.0f;

            global_payoff_sum += (double)(p0 + p1);
        }
    }

    return (float)(discount * (global_payoff_sum / (double)total_paths));
}

static void usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --sizes N,N,...     Number of simulated paths [default: 1M..100M]\n");
    printf("  --precisions P,P    Precisions: fp32,fp64     [default: fp32,fp64]\n");
    printf("  --threads T,T,...   Active thread counts      [default: 1..max_cores]\n");
    printf("  --iters N           Timed loop iterations     [default: 10]\n");
    printf("  --warmup N          Untimed warmup runs       [default: 2]\n");
    printf("  --validate          Compare with Black-Scholes exact formula\n");
    printf("  --csv PATH          Append results to CSV file\n");
    printf("  --tag STRING        Label identifier in CSV   [default: cpu_montecarlo]\n");
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

static void run_benchmark(const cfg_t *c, FILE *csv, const char *host,
                          long paths, prec_t p, int T, cpu_telemetry_t *telem) {
    const char *pn = prec_name(p);
    double S0 = 100.0, K = 100.0, maturity = 1.0, r = 0.05, sigma = 0.20;

    double ref_price = black_scholes_call_analytical(S0, K, maturity, r, sigma);
    double sim_price = 0.0;

    /* Untimed Warmup */
    for (int it = 0; it < c->warmup; it++) {
        if (p == PREC_FP32) {
            sim_price = (double)run_montecarlo_f32(100000, T, (float)S0, (float)K, (float)maturity, (float)r, (float)sigma);
        } else {
            sim_price = run_montecarlo_f64(100000, T, S0, K, maturity, r, sigma);
        }
    }

    /* Timed Region with Telemetry */
    cpu_telemetry_start(telem);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

    for (int it = 0; it < c->iters; it++) {
        if (p == PREC_FP32) {
            sim_price = (double)run_montecarlo_f32(paths, T, (float)S0, (float)K, (float)maturity, (float)r, (float)sigma);
        } else {
            sim_price = run_montecarlo_f64(paths, T, S0, K, maturity, r, sigma);
        }
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
    cpu_telemetry_stop(telem);

    double wall_sec = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double comp_ms = (wall_sec * 1000.0) / (double)c->iters;
    double total_paths = (double)paths * (double)c->iters;
    double mpaths_sec = (wall_sec > 0.0) ? (total_paths / wall_sec / 1e6) : 0.0;
    /* Box-Muller (~14 FLOPs) + Brownian Step (~10 FLOPs) = ~24 FLOPs/path */
    double agg_tflops = (wall_sec > 0.0) ? ((total_paths * 24.0) / wall_sec / 1e12) : 0.0;

    double rel_err = fabs(sim_price - ref_price) / ref_price;
    int validated = c->validate ? 1 : 0;
    const char *status = (validated && rel_err > 0.02) ? "HIERR" : "ok";

    printf("%-6s %-5s %-3d %10ld %10.2f %9.3f %8.2f %6.1fW %7.1f %s\n",
           "mc_bs", pn, T, paths, mpaths_sec, agg_tflops, comp_ms,
           telem->avg_power_watts, telem->avg_freq_mhz, status);
    fflush(stdout);

    if (csv) {
        time_t ts = time(NULL);
        fprintf(csv,
            "%s,%ld,%s,OpenMP,mc_black_scholes,%s,%d,%ld,%d,%d,"
            "%.2f,%.4f,%.3f,%.3f,"
            "%.1f,%.1f,%.1f,"
            "%.4f,%.4f,%d,%.3e,%s\n",
            c->tag, (long)ts, host, pn, T, paths, c->iters, c->warmup,
            mpaths_sec, agg_tflops, comp_ms, wall_sec * 1000.0,
            telem->avg_freq_mhz, telem->avg_power_watts, telem->max_temp_c,
            sim_price, ref_price, validated, rel_err, status);
        fflush(csv);
    }
}

int main(int argc, char **argv) {
    cfg_t c;
    memset(&c, 0, sizeof(c));

    c.iters = 10;
    c.warmup = 2;
    c.validate = 0;
    c.tag = "cpu_montecarlo";

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
        long d[] = {1000000, 5000000, 20000000, 50000000, 100000000};
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
        "tag,ts,host,engine,mode,precision,threads,paths,iters,warmup,"
        "mpaths_sec,agg_tflops,comp_ms,wall_ms,"
        "cpu_mhz_avg,power_w_avg,temp_c_max,"
        "price_sim,price_ref,validated,max_rel_err,status\n";

    if (csv && new_file) {
        fputs(HDR, csv);
        fflush(csv);
    }

    printf("# Monte Carlo Option Pricing Benchmark (Black-Scholes) host=%s threads_max=%d\n",
           host, omp_get_num_procs());
    printf("%-6s %-5s %-3s %10s %10s %9s %8s %7s %7s %s\n",
           "work", "prec", "T", "paths", "MPaths/s", "TFLOPS", "comp_ms", "power", "MHz", "status");

    for (int si = 0; si < c.n_sizes; si++) {
        long paths = c.sizes[si];
        for (int pi = 0; pi < c.n_precs; pi++) {
            prec_t p = c.precs[pi];
            for (int ti = 0; ti < c.n_threads; ti++) {
                int T = c.threads[ti];
                run_benchmark(&c, csv, host, paths, p, T, &telem);
            }
        }
    }

    if (csv) fclose(csv);
    return 0;
}
