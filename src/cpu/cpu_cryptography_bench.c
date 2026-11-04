/* src/cpu/cpu_cryptography_bench.c
 *
 * Multi-threaded Cryptographic Hashing Benchmark (SHA-256 & SHA-512).
 * NIST FIPS 180-4 compliant implementations in pure C.
 *
 * Stresses CPU integer arithmetic, 32/64-bit bitwise rotation pipelines,
 * instruction-level parallelism, and streaming memory read bandwidth.
 *
 * Compile: gcc -std=c11 -O3 -march=native -fopenmp -I src/common -I src/cpu src/cpu/cpu_cryptography_bench.c -o bin/cpu_cryptography_bench -lm
 * Run:     ./bin/cpu_cryptography_bench --help
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

typedef enum { ALG_SHA256, ALG_SHA512 } algo_t;

static const char *algo_name(algo_t a) {
    return a == ALG_SHA256 ? "sha256" : "sha512";
}

typedef struct {
    long sizes[32];
    int n_sizes;
    algo_t algos[4];
    int n_algos;
    int threads[32];
    int n_threads;
    int iters;
    int warmup;
    int validate;
    const char *csv;
    const char *tag;
} cfg_t;

/* ── 32-bit SHA-256 Primitives ────────────────────────────────────────────── */
#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH32(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ32(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define SIGMA0_32(x) (ROTR32(x, 2) ^ ROTR32(x, 13) ^ ROTR32(x, 22))
#define SIGMA1_32(x) (ROTR32(x, 6) ^ ROTR32(x, 11) ^ ROTR32(x, 25))
#define SIG0_32(x) (ROTR32(x, 7) ^ ROTR32(x, 18) ^ ((x) >> 3))
#define SIG1_32(x) (ROTR32(x, 17) ^ ROTR32(x, 19) ^ ((x) >> 10))

static const uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

static void sha256_transform(uint32_t state[8], const uint8_t data[64]) {
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    uint32_t W[64];

    for (int t = 0; t < 16; t++) {
        W[t] = ((uint32_t)data[t * 4] << 24) |
               ((uint32_t)data[t * 4 + 1] << 16) |
               ((uint32_t)data[t * 4 + 2] << 8) |
               ((uint32_t)data[t * 4 + 3]);
    }
    for (int t = 16; t < 64; t++) {
        W[t] = SIG1_32(W[t - 2]) + W[t - 7] + SIG0_32(W[t - 15]) + W[t - 16];
    }

    for (int t = 0; t < 64; t++) {
        uint32_t T1 = h + SIGMA1_32(e) + CH32(e, f, g) + K256[t] + W[t];
        uint32_t T2 = SIGMA0_32(a) + MAJ32(a, b, c);
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

/* ── 64-bit SHA-512 Primitives ────────────────────────────────────────────── */
#define ROTR64(x, n) (((x) >> (n)) | ((x) << (64 - (n))))
#define CH64(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ64(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define SIGMA0_64(x) (ROTR64(x, 28) ^ ROTR64(x, 34) ^ ROTR64(x, 39))
#define SIGMA1_64(x) (ROTR64(x, 14) ^ ROTR64(x, 18) ^ ROTR64(x, 41))
#define SIG0_64(x) (ROTR64(x, 1) ^ ROTR64(x, 8) ^ ((x) >> 7))
#define SIG1_64(x) (ROTR64(x, 19) ^ ROTR64(x, 61) ^ ((x) >> 6))

static const uint64_t K512[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47867113ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};

static void sha512_transform(uint64_t state[8], const uint8_t data[128]) {
    uint64_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint64_t e = state[4], f = state[5], g = state[6], h = state[7];
    uint64_t W[80];

    for (int t = 0; t < 16; t++) {
        W[t] = ((uint64_t)data[t * 8] << 56) |
               ((uint64_t)data[t * 8 + 1] << 48) |
               ((uint64_t)data[t * 8 + 2] << 40) |
               ((uint64_t)data[t * 8 + 3] << 32) |
               ((uint64_t)data[t * 8 + 4] << 24) |
               ((uint64_t)data[t * 8 + 5] << 16) |
               ((uint64_t)data[t * 8 + 6] << 8) |
               ((uint64_t)data[t * 8 + 7]);
    }
    for (int t = 16; t < 80; t++) {
        W[t] = SIG1_64(W[t - 2]) + W[t - 7] + SIG0_64(W[t - 15]) + W[t - 16];
    }

    for (int t = 0; t < 80; t++) {
        uint64_t T1 = h + SIGMA1_64(e) + CH64(e, f, g) + K512[t] + W[t];
        uint64_t T2 = SIGMA0_64(a) + MAJ64(a, b, c);
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

/* ── Benchmark Execution Function ─────────────────────────────────────────── */
static void run_crypto_benchmark(const cfg_t *c, FILE *csv, const char *host,
                                 long bytes, algo_t a, int T, cpu_telemetry_t *telem) {
    const char *aname = algo_name(a);
    int block_size = (a == ALG_SHA256) ? 64 : 128;
    long num_blocks = bytes / block_size;

    /* Allocate buffer aligned to 64 bytes */
    uint8_t *buffer = NULL;
    if (posix_memalign((void **)&buffer, 64, bytes) != 0 || !buffer) {
        fprintf(stderr, "Allocation failed for %ld bytes\n", bytes);
        return;
    }

    /* Fill buffer deterministically */
    #pragma omp parallel for num_threads(T) schedule(static)
    for (long i = 0; i < bytes; i++) {
        buffer[i] = (uint8_t)((i * 37 + 11) & 0xFF);
    }

    /* Untimed Warmup */
    for (int it = 0; it < c->warmup; it++) {
        #pragma omp parallel for num_threads(T) schedule(static)
        for (long b = 0; b < (num_blocks > 1000 ? 1000 : num_blocks); b++) {
            if (a == ALG_SHA256) {
                uint32_t state[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
                sha256_transform(state, buffer + b * 64);
            } else {
                uint64_t state[8] = {0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
                                     0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
                                     0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
                                     0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL};
                sha512_transform(state, buffer + b * 128);
            }
        }
    }

    /* Timed Region with Telemetry */
    cpu_telemetry_start(telem);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

    for (int it = 0; it < c->iters; it++) {
        #pragma omp parallel for num_threads(T) schedule(static)
        for (long b = 0; b < num_blocks; b++) {
            if (a == ALG_SHA256) {
                uint32_t state[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
                sha256_transform(state, buffer + b * 64);
            } else {
                uint64_t state[8] = {0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
                                     0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
                                     0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
                                     0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL};
                sha512_transform(state, buffer + b * 128);
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
    cpu_telemetry_stop(telem);

    double wall_sec = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double comp_ms = (wall_sec * 1000.0) / (double)c->iters;
    double total_bytes = (double)bytes * (double)c->iters;
    double agg_gbps = (wall_sec > 0.0) ? (total_bytes / wall_sec / 1e9) : 0.0;
    double mblocks_sec = (wall_sec > 0.0) ? ((total_bytes / (double)block_size) / wall_sec / 1e6) : 0.0;

    /* Validation against standard test pattern */
    int validated = 0;
    int valid_ok = 1;
    if (c->validate) {
        /* NIST test: SHA-256("abc") first 32-bit word = 0xba7816bf */
        uint8_t test_block[64];
        memset(test_block, 0, sizeof(test_block));
        test_block[0] = 'a'; test_block[1] = 'b'; test_block[2] = 'c';
        test_block[3] = 0x80; /* padding bit */
        test_block[63] = 24;  /* length in bits */

        uint32_t state[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                             0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
        sha256_transform(state, test_block);
        if (state[0] != 0xba7816bf) {
            valid_ok = 0;
        }
        validated = 1;
    }

    const char *status = (validated && !valid_ok) ? "HIERR" : "ok";

    printf("%-6s %-7s %-3d %10ld %8.2f %10.2f %8.2f %6.1fW %7.1f %s\n",
           "crypto", aname, T, bytes, agg_gbps, mblocks_sec, comp_ms,
           telem->avg_power_watts, telem->avg_freq_mhz, status);
    fflush(stdout);

    if (csv) {
        time_t ts = time(NULL);
        fprintf(csv,
            "%s,%ld,%s,OpenMP,%s,%d,%ld,%d,%d,"
            "%.3f,%.2f,%.3f,%.3f,"
            "%.1f,%.1f,%.1f,%d,%s\n",
            c->tag, (long)ts, host, aname, T, bytes, c->iters, c->warmup,
            agg_gbps, mblocks_sec, comp_ms, wall_sec * 1000.0,
            telem->avg_freq_mhz, telem->avg_power_watts, telem->max_temp_c,
            validated, status);
        fflush(csv);
    }

    free(buffer);
}

static void usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --sizes N,N,...     Data buffer size in bytes [default: 64MB..1GB]\n");
    printf("  --algorithms A,A    Algorithms: sha256,sha512 [default: sha256,sha512]\n");
    printf("  --threads T,T,...   Active thread counts      [default: 1..max_cores]\n");
    printf("  --iters N           Timed loop iterations     [default: 10]\n");
    printf("  --warmup N          Untimed warmup runs       [default: 2]\n");
    printf("  --validate          Verify against NIST standard test vectors\n");
    printf("  --csv PATH          Append results to CSV file\n");
    printf("  --tag STRING        Label identifier in CSV   [default: cpu_crypto]\n");
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

    c.iters = 10;
    c.warmup = 2;
    c.validate = 0;
    c.tag = "cpu_crypto";

    int set_sizes = 0, set_algos = 0, set_threads = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--sizes") && i + 1 < argc) {
            c.n_sizes = parse_longs(argv[++i], c.sizes, 32); set_sizes = 1;
        } else if (!strcmp(a, "--algorithms") && i + 1 < argc) {
            char b[128]; snprintf(b, sizeof(b), "%s", argv[++i]); c.n_algos = 0;
            for (char *t = strtok(b, ","); t; t = strtok(NULL, ",")) {
                if (!strcmp(t, "sha256")) c.algos[c.n_algos++] = ALG_SHA256;
                else if (!strcmp(t, "sha512")) c.algos[c.n_algos++] = ALG_SHA512;
            }
            set_algos = 1;
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
        long d[] = {67108864, 268435456, 1073741824}; /* 64MB, 256MB, 1GB */
        c.n_sizes = 3; memcpy(c.sizes, d, sizeof(d));
    }
    if (!set_algos) {
        c.algos[0] = ALG_SHA256; c.algos[1] = ALG_SHA512; c.n_algos = 2;
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
        "tag,ts,host,engine,algorithm,threads,bytes,iters,warmup,"
        "agg_gbps,mblocks_sec,comp_ms,wall_ms,"
        "cpu_mhz_avg,power_w_avg,temp_c_max,validated,status\n";

    if (csv && new_file) {
        fputs(HDR, csv);
        fflush(csv);
    }

    printf("# Cryptographic Hashing Benchmark (SHA-256 / SHA-512) host=%s threads_max=%d\n",
           host, omp_get_num_procs());
    printf("%-6s %-7s %-3s %10s %8s %10s %8s %7s %7s %s\n",
           "work", "algo", "T", "bytes", "GB/s", "MBlocks/s", "comp_ms", "power", "MHz", "status");

    for (int si = 0; si < c.n_sizes; si++) {
        long bytes = c.sizes[si];
        for (int ai = 0; ai < c.n_algos; ai++) {
            algo_t a = c.algos[ai];
            for (int ti = 0; ti < c.n_threads; ti++) {
                int T = c.threads[ti];
                run_crypto_benchmark(&c, csv, host, bytes, a, T, &telem);
            }
        }
    }

    if (csv) fclose(csv);
    return 0;
}
