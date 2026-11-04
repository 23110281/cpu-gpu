/*
 * gpu_c2c_streaming_bench.cu
 *
 * Cycle 1: Regime 2 (Unit-Stride Streaming Bandwidth: SAXPY & GEMV)
 * across 4 Coherent NUMA Residency Modes on NVIDIA Grace Hopper (GH200):
 *   1. CPU_LOCAL         (72 Grace Neoverse-V2 OpenMP cores, LPDDR5X)
 *   2. GPU_EXPLICIT_COPY (Pinned LPDDR5X -> HBM3 -> cuBLAS -> LPDDR5X)
 *   3. GPU_ZERO_COPY     (Mapped LPDDR5X streamed directly over NVLink-C2C)
 *   4. GPU_UM_MIGRATE    (cudaMallocManaged with cold ATS faulting vs warm)
 *
 * Strictly follows vendor/standard libraries:
 *   - GPU: cuBLAS (cublasSaxpy, cublasDaxpy, cublasSgemv, cublasDgemv)
 *   - CPU: OpenMP vector loops on Grace Neoverse-V2
 */

#include "bench_harness.h"
#include "gpu_device_specs.h"
#include <cublas_v2.h>
#include <cblas.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_cpu_threads = 72;

#define CUBLAS_CHECK(expr) do { \
    cublasStatus_t st = (expr); \
    if (st != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %d at line %d\n", (int)st, __LINE__); \
        return -1; \
    } \
} while (0)

#define CUDA_CHECK(expr) do { \
    cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        return -1; \
    } \
} while (0)

/* ── Benchmark SAXPY ───────────────────────────────────────────────────────── */
int run_saxpy_bench(FILE *csv_out, residency_mode_t residency, const char *prec, long N, int iters, int warmup) {
    bench_record_t rec;
    memset(&rec, 0, sizeof(rec));
    snprintf(rec.workload, sizeof(rec.workload), "saxpy");
    rec.regime = REGIME_STREAMING_BANDWIDTH;
    rec.residency = residency;
    snprintf(rec.precision, sizeof(rec.precision), "%s", prec);
    snprintf(rec.problem_size, sizeof(rec.problem_size), "N=%ld", N);
    snprintf(rec.status, sizeof(rec.status), "ok");

    size_t elem_size = (strcmp(prec, "fp64") == 0) ? sizeof(double) : sizeof(float);
    size_t vec_bytes = (size_t)N * elem_size;
    rec.working_set_bytes = 3 * vec_bytes; /* 2 reads (x, y) + 1 write (y) */

    gpu_spec_t spec = get_gpu_spec_for_device(0);

    double t0_alloc = bench_get_time_ms();
    bench_buf_t bx = bench_alloc_buf(vec_bytes, residency);
    bench_buf_t by = bench_alloc_buf(vec_bytes, residency);
    bench_buf_t by_orig = bench_alloc_buf(vec_bytes, RESIDENCY_CPU_LOCAL);
    rec.t_alloc_ms = bench_get_time_ms() - t0_alloc;

    if (!bx.host_ptr || !by.host_ptr || !by_orig.host_ptr ||
        (residency == RESIDENCY_GPU_EXPLICIT_COPY && (!bx.dev_ptr || !by.dev_ptr))) {
        snprintf(rec.status, sizeof(rec.status), "oom-skip");
        log_bench_record(csv_out, &rec);
        bench_free_buf(&bx);
        bench_free_buf(&by);
        bench_free_buf(&by_orig);
        return 0;
    }

    /* Initialize inputs on CPU */
    if (strcmp(prec, "fp64") == 0) {
        double *x = (double *)bx.host_ptr;
        double *y = (double *)by.host_ptr;
        double *yo = (double *)by_orig.host_ptr;
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < N; i++) {
            x[i] = 1.0;
            y[i] = 2.0;
            yo[i] = 2.0;
        }
    } else {
        float *x = (float *)bx.host_ptr;
        float *y = (float *)by.host_ptr;
        float *yo = (float *)by_orig.host_ptr;
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < N; i++) {
            x[i] = 1.0f;
            y[i] = 2.0f;
            yo[i] = 2.0f;
        }
    }

    if (residency == RESIDENCY_CPU_LOCAL) {
        rec.threads_or_sms = g_cpu_threads;
        openblas_set_num_threads(g_cpu_threads);
        omp_set_num_threads(g_cpu_threads);

        long chunk_n = 536870912L;
        if (chunk_n > N) chunk_n = N;

        /* CPU OpenBLAS execution */
        if (strcmp(prec, "fp64") == 0) {
            double *x = (double *)bx.host_ptr;
            double *y = (double *)by.host_ptr;
            double alpha = 2.5;

            /* Cold launch (1 iteration) */
            double t0 = bench_get_time_ms();
            for (long off = 0; off < N; off += chunk_n) {
                int cur_n = (off + chunk_n <= N) ? (int)chunk_n : (int)(N - off);
                cblas_daxpy(cur_n, alpha, x + off, 1, y + off, 1);
            }
            rec.t_kernel_cold_ms = bench_get_time_ms() - t0;

            /* Warm iterations */
            for (int w = 0; w < warmup; w++) {
                for (long off = 0; off < N; off += chunk_n) {
                    int cur_n = (off + chunk_n <= N) ? (int)chunk_n : (int)(N - off);
                    cblas_daxpy(cur_n, alpha, x + off, 1, y + off, 1);
                }
            }
            t0 = bench_get_time_ms();
            for (int it = 0; it < iters; it++) {
                for (long off = 0; off < N; off += chunk_n) {
                    int cur_n = (off + chunk_n <= N) ? (int)chunk_n : (int)(N - off);
                    cblas_daxpy(cur_n, alpha, x + off, 1, y + off, 1);
                }
            }
            rec.t_kernel_warm_ms = (bench_get_time_ms() - t0) / iters;
        } else {
            float *x = (float *)bx.host_ptr;
            float *y = (float *)by.host_ptr;
            float alpha = 2.5f;

            double t0 = bench_get_time_ms();
            for (long off = 0; off < N; off += chunk_n) {
                int cur_n = (off + chunk_n <= N) ? (int)chunk_n : (int)(N - off);
                cblas_saxpy(cur_n, alpha, x + off, 1, y + off, 1);
            }
            rec.t_kernel_cold_ms = bench_get_time_ms() - t0;

            for (int w = 0; w < warmup; w++) {
                for (long off = 0; off < N; off += chunk_n) {
                    int cur_n = (off + chunk_n <= N) ? (int)chunk_n : (int)(N - off);
                    cblas_saxpy(cur_n, alpha, x + off, 1, y + off, 1);
                }
            }
            t0 = bench_get_time_ms();
            for (int it = 0; it < iters; it++) {
                for (long off = 0; off < N; off += chunk_n) {
                    int cur_n = (off + chunk_n <= N) ? (int)chunk_n : (int)(N - off);
                    cblas_saxpy(cur_n, alpha, x + off, 1, y + off, 1);
                }
            }
            rec.t_kernel_warm_ms = (bench_get_time_ms() - t0) / iters;
        }

        rec.t_h2d_ms = 0.0;
        rec.t_d2h_ms = 0.0;
        rec.t_e2e_ms = rec.t_kernel_warm_ms;
        rec.eff_bw_gbps = (double)rec.working_set_bytes / (rec.t_e2e_ms * 1e6);
        rec.c2c_bw_gbps = 0.0; /* Fits locally in Grace LPDDR5X */
        rec.gflops = ((double)N * 2.0) / (rec.t_kernel_warm_ms * 1e6);
        rec.pct_peak = (rec.eff_bw_gbps / 500.0) * 100.0; /* Grace LPDDR5X ~500 GB/s peak */
    } else {
        /* GPU execution via cuBLAS */
        rec.threads_or_sms = 132;
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetStream(handle, stream));

        cudaEvent_t ev_h2d0, ev_h2d1, ev_k0, ev_k1, ev_d2h0, ev_d2h1;
        CUDA_CHECK(cudaEventCreate(&ev_h2d0));
        CUDA_CHECK(cudaEventCreate(&ev_h2d1));
        CUDA_CHECK(cudaEventCreate(&ev_k0));
        CUDA_CHECK(cudaEventCreate(&ev_k1));
        CUDA_CHECK(cudaEventCreate(&ev_d2h0));
        CUDA_CHECK(cudaEventCreate(&ev_d2h1));

        /* Explicit H2D */
        if (residency == RESIDENCY_GPU_EXPLICIT_COPY) {
            cudaEventRecord(ev_h2d0, stream);
            CUDA_CHECK(cudaMemcpyAsync(bx.dev_ptr, bx.host_ptr, vec_bytes, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(by.dev_ptr, by.host_ptr, vec_bytes, cudaMemcpyHostToDevice, stream));
            cudaEventRecord(ev_h2d1, stream);
            cudaEventSynchronize(ev_h2d1);
            float h2d_ms = 0;
            cudaEventElapsedTime(&h2d_ms, ev_h2d0, ev_h2d1);
            rec.t_h2d_ms = (double)h2d_ms;
        } else {
            rec.t_h2d_ms = 0.0;
        }

        void *dx = (residency == RESIDENCY_GPU_EXPLICIT_COPY) ? bx.dev_ptr : bx.host_ptr;
        void *dy = (residency == RESIDENCY_GPU_EXPLICIT_COPY) ? by.dev_ptr : by.host_ptr;

        /* Chunked cuBLAS execution to prevent 32-bit int overflow for N >= 2^31 */
        long chunk_n = 536870912L; /* 512M elements per chunk */
        if (chunk_n > N) chunk_n = N;

        /* Measure Cold Launch */
        cudaEventRecord(ev_k0, stream);
        if (strcmp(prec, "fp64") == 0) {
            double alpha = 2.5;
            for (long off = 0; off < N; off += chunk_n) {
                int cur_n = (off + chunk_n <= N) ? (int)chunk_n : (int)(N - off);
                cublasDaxpy(handle, cur_n, &alpha, (const double *)dx + off, 1, (double *)dy + off, 1);
            }
        } else {
            float alpha = 2.5f;
            for (long off = 0; off < N; off += chunk_n) {
                int cur_n = (off + chunk_n <= N) ? (int)chunk_n : (int)(N - off);
                cublasSaxpy(handle, cur_n, &alpha, (const float *)dx + off, 1, (float *)dy + off, 1);
            }
        }
        cudaEventRecord(ev_k1, stream);
        cudaEventSynchronize(ev_k1);
        float cold_ms = 0;
        cudaEventElapsedTime(&cold_ms, ev_k0, ev_k1);
        rec.t_kernel_cold_ms = (double)cold_ms;

        /* Warmup */
        for (int w = 0; w < warmup; w++) {
            if (strcmp(prec, "fp64") == 0) {
                double alpha = 2.5;
                for (long off = 0; off < N; off += chunk_n) {
                    int cur_n = (off + chunk_n <= N) ? (int)chunk_n : (int)(N - off);
                    cublasDaxpy(handle, cur_n, &alpha, (const double *)dx + off, 1, (double *)dy + off, 1);
                }
            } else {
                float alpha = 2.5f;
                for (long off = 0; off < N; off += chunk_n) {
                    int cur_n = (off + chunk_n <= N) ? (int)chunk_n : (int)(N - off);
                    cublasSaxpy(handle, cur_n, &alpha, (const float *)dx + off, 1, (float *)dy + off, 1);
                }
            }
        }
        cudaStreamSynchronize(stream);

        /* Timed Warm Loop */
        cudaEventRecord(ev_k0, stream);
        for (int it = 0; it < iters; it++) {
            if (strcmp(prec, "fp64") == 0) {
                double alpha = 2.5;
                for (long off = 0; off < N; off += chunk_n) {
                    int cur_n = (off + chunk_n <= N) ? (int)chunk_n : (int)(N - off);
                    cublasDaxpy(handle, cur_n, &alpha, (const double *)dx + off, 1, (double *)dy + off, 1);
                }
            } else {
                float alpha = 2.5f;
                for (long off = 0; off < N; off += chunk_n) {
                    int cur_n = (off + chunk_n <= N) ? (int)chunk_n : (int)(N - off);
                    cublasSaxpy(handle, cur_n, &alpha, (const float *)dx + off, 1, (float *)dy + off, 1);
                }
            }
        }
        cudaEventRecord(ev_k1, stream);
        cudaEventSynchronize(ev_k1);
        float warm_ms = 0;
        cudaEventElapsedTime(&warm_ms, ev_k0, ev_k1);
        rec.t_kernel_warm_ms = (double)warm_ms / iters;

        /* Explicit D2H */
        if (residency == RESIDENCY_GPU_EXPLICIT_COPY) {
            cudaEventRecord(ev_d2h0, stream);
            CUDA_CHECK(cudaMemcpyAsync(by.host_ptr, by.dev_ptr, vec_bytes, cudaMemcpyDeviceToHost, stream));
            cudaEventRecord(ev_d2h1, stream);
            cudaEventSynchronize(ev_d2h1);
            float d2h_ms = 0;
            cudaEventElapsedTime(&d2h_ms, ev_d2h0, ev_d2h1);
            rec.t_d2h_ms = (double)d2h_ms;
        } else {
            rec.t_d2h_ms = 0.0;
        }

        if (residency == RESIDENCY_GPU_UM_MIGRATE) {
            rec.t_e2e_ms = rec.t_kernel_cold_ms + rec.t_kernel_warm_ms;
            rec.c2c_bw_gbps = (rec.t_kernel_cold_ms > 0) ? ((double)rec.working_set_bytes / (rec.t_kernel_cold_ms * 1e6)) : 0.0;
        } else if (residency == RESIDENCY_GPU_ZERO_COPY) {
            rec.t_e2e_ms = rec.t_kernel_warm_ms;
            rec.c2c_bw_gbps = (rec.t_kernel_warm_ms > 0) ? ((double)rec.working_set_bytes / (rec.t_kernel_warm_ms * 1e6)) : 0.0;
        } else {
            rec.t_e2e_ms = rec.t_h2d_ms + rec.t_kernel_warm_ms + rec.t_d2h_ms;
            rec.c2c_bw_gbps = (((double)vec_bytes * 2.0) / (rec.t_h2d_ms * 1e6 + 1e-9));
        }
        rec.eff_bw_gbps = (double)rec.working_set_bytes / (rec.t_e2e_ms * 1e6);
        rec.gflops = ((double)N * 2.0) / (rec.t_kernel_warm_ms * 1e6);

        double peak_ref = (residency == RESIDENCY_GPU_ZERO_COPY) ? spec.pcie_or_nvlink_gbps : spec.peak_hbm_gbps;
        rec.pct_peak = (rec.eff_bw_gbps / peak_ref) * 100.0;

        cublasDestroy(handle);
        cudaStreamDestroy(stream);
        cudaEventDestroy(ev_h2d0);
        cudaEventDestroy(ev_h2d1);
        cudaEventDestroy(ev_k0);
        cudaEventDestroy(ev_k1);
        cudaEventDestroy(ev_d2h0);
        cudaEventDestroy(ev_d2h1);
    }

    log_bench_record(csv_out, &rec);
    printf("SAXPY [%-14s] %-4s N=%-10ld: t_e2e=%8.2fms (h2d=%6.2fms, kern=%6.2fms, d2h=%6.2fms) | EffBW=%7.1f GB/s (%5.1f%% pk)\n",
           residency_to_str(residency), prec, N, rec.t_e2e_ms, rec.t_h2d_ms, rec.t_kernel_warm_ms, rec.t_d2h_ms, rec.eff_bw_gbps, rec.pct_peak);

    bench_free_buf(&bx);
    bench_free_buf(&by);
    bench_free_buf(&by_orig);
    return 0;
}

/* ── Benchmark GEMV ────────────────────────────────────────────────────────── */
int run_gemv_bench(FILE *csv_out, residency_mode_t residency, const char *prec, int S, int iters, int warmup) {
    bench_record_t rec;
    memset(&rec, 0, sizeof(rec));
    snprintf(rec.workload, sizeof(rec.workload), "gemv");
    rec.regime = REGIME_STREAMING_BANDWIDTH;
    rec.residency = residency;
    snprintf(rec.precision, sizeof(rec.precision), "%s", prec);
    snprintf(rec.problem_size, sizeof(rec.problem_size), "S=%d", S);
    snprintf(rec.status, sizeof(rec.status), "ok");

    size_t elem_size = (strcmp(prec, "fp64") == 0) ? sizeof(double) : sizeof(float);
    size_t mat_bytes = (size_t)S * (size_t)S * elem_size;
    size_t vec_bytes = (size_t)S * elem_size;
    rec.working_set_bytes = mat_bytes + 2 * vec_bytes;

    gpu_spec_t spec = get_gpu_spec_for_device(0);

    double t0_alloc = bench_get_time_ms();
    bench_buf_t bA = bench_alloc_buf(mat_bytes, residency);
    bench_buf_t bx = bench_alloc_buf(vec_bytes, residency);
    bench_buf_t by = bench_alloc_buf(vec_bytes, residency);
    rec.t_alloc_ms = bench_get_time_ms() - t0_alloc;

    if (!bA.host_ptr || !bx.host_ptr || !by.host_ptr ||
        (residency == RESIDENCY_GPU_EXPLICIT_COPY && (!bA.dev_ptr || !bx.dev_ptr || !by.dev_ptr))) {
        snprintf(rec.status, sizeof(rec.status), "oom-skip");
        log_bench_record(csv_out, &rec);
        bench_free_buf(&bA);
        bench_free_buf(&bx);
        bench_free_buf(&by);
        return 0;
    }

    /* Initialize matrix and vectors */
    if (strcmp(prec, "fp64") == 0) {
        double *A = (double *)bA.host_ptr;
        double *x = (double *)bx.host_ptr;
        double *y = (double *)by.host_ptr;
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < (long)S * S; i++) A[i] = 1.0;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < S; i++) { x[i] = 1.0; y[i] = 0.0; }
    } else {
        float *A = (float *)bA.host_ptr;
        float *x = (float *)bx.host_ptr;
        float *y = (float *)by.host_ptr;
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < (long)S * S; i++) A[i] = 1.0f;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < S; i++) { x[i] = 1.0f; y[i] = 0.0f; }
    }

    if (residency == RESIDENCY_CPU_LOCAL) {
        rec.threads_or_sms = g_cpu_threads;
        openblas_set_num_threads(g_cpu_threads);
        omp_set_num_threads(g_cpu_threads);
        double alpha = 1.0, beta = 0.0;

        double t0 = bench_get_time_ms();
        if (strcmp(prec, "fp64") == 0) {
            double *A = (double *)bA.host_ptr;
            double *x = (double *)bx.host_ptr;
            double *y = (double *)by.host_ptr;
            cblas_dgemv(CblasRowMajor, CblasNoTrans, S, S, alpha, A, S, x, 1, beta, y, 1);
            rec.t_kernel_cold_ms = bench_get_time_ms() - t0;

            for (int w = 0; w < warmup; w++) {
                cblas_dgemv(CblasRowMajor, CblasNoTrans, S, S, alpha, A, S, x, 1, beta, y, 1);
            }
            t0 = bench_get_time_ms();
            for (int it = 0; it < iters; it++) {
                cblas_dgemv(CblasRowMajor, CblasNoTrans, S, S, alpha, A, S, x, 1, beta, y, 1);
            }
            rec.t_kernel_warm_ms = (bench_get_time_ms() - t0) / iters;
        } else {
            float *A = (float *)bA.host_ptr;
            float *x = (float *)bx.host_ptr;
            float *y = (float *)by.host_ptr;
            cblas_sgemv(CblasRowMajor, CblasNoTrans, S, S, (float)alpha, A, S, x, 1, (float)beta, y, 1);
            rec.t_kernel_cold_ms = bench_get_time_ms() - t0;

            for (int w = 0; w < warmup; w++) {
                cblas_sgemv(CblasRowMajor, CblasNoTrans, S, S, (float)alpha, A, S, x, 1, (float)beta, y, 1);
            }
            t0 = bench_get_time_ms();
            for (int it = 0; it < iters; it++) {
                cblas_sgemv(CblasRowMajor, CblasNoTrans, S, S, (float)alpha, A, S, x, 1, (float)beta, y, 1);
            }
            rec.t_kernel_warm_ms = (bench_get_time_ms() - t0) / iters;
        }
        rec.t_h2d_ms = 0.0;
        rec.t_d2h_ms = 0.0;
        rec.t_e2e_ms = rec.t_kernel_warm_ms;
        rec.eff_bw_gbps = (double)rec.working_set_bytes / (rec.t_e2e_ms * 1e6);
        rec.c2c_bw_gbps = 0.0;
        rec.gflops = ((double)S * (double)S * 2.0) / (rec.t_kernel_warm_ms * 1e6);
        rec.pct_peak = (rec.eff_bw_gbps / 500.0) * 100.0;
    } else {
        rec.threads_or_sms = 132;
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetStream(handle, stream));

        cudaEvent_t ev_h2d0, ev_h2d1, ev_k0, ev_k1, ev_d2h0, ev_d2h1;
        CUDA_CHECK(cudaEventCreate(&ev_h2d0));
        CUDA_CHECK(cudaEventCreate(&ev_h2d1));
        CUDA_CHECK(cudaEventCreate(&ev_k0));
        CUDA_CHECK(cudaEventCreate(&ev_k1));
        CUDA_CHECK(cudaEventCreate(&ev_d2h0));
        CUDA_CHECK(cudaEventCreate(&ev_d2h1));

        /* Explicit H2D */
        if (residency == RESIDENCY_GPU_EXPLICIT_COPY) {
            cudaEventRecord(ev_h2d0, stream);
            CUDA_CHECK(cudaMemcpyAsync(bA.dev_ptr, bA.host_ptr, mat_bytes, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(bx.dev_ptr, bx.host_ptr, vec_bytes, cudaMemcpyHostToDevice, stream));
            cudaEventRecord(ev_h2d1, stream);
            cudaEventSynchronize(ev_h2d1);
            float h2d_ms = 0;
            cudaEventElapsedTime(&h2d_ms, ev_h2d0, ev_h2d1);
            rec.t_h2d_ms = (double)h2d_ms;
        } else {
            rec.t_h2d_ms = 0.0;
        }

        void *dA = (residency == RESIDENCY_GPU_EXPLICIT_COPY) ? bA.dev_ptr : bA.host_ptr;
        void *dx = (residency == RESIDENCY_GPU_EXPLICIT_COPY) ? bx.dev_ptr : bx.host_ptr;
        void *dy = (residency == RESIDENCY_GPU_EXPLICIT_COPY) ? by.dev_ptr : by.host_ptr;

        /* Chunked cuBLAS execution to prevent 32-bit offset overflow */
        int chunk_rows = (strcmp(prec, "fp64") == 0) ? 8192 : 16384;
        if (chunk_rows > S) chunk_rows = S;

        /* Measure Cold Launch */
        cudaEventRecord(ev_k0, stream);
        if (strcmp(prec, "fp64") == 0) {
            double alpha = 1.0, beta = 0.0;
            for (int r = 0; r < S; r += chunk_rows) {
                int cur_rows = (r + chunk_rows <= S) ? chunk_rows : (S - r);
                double *subA = (double *)dA + (long)r * S;
                double *subY = (double *)dy + r;
                cublasDgemv(handle, CUBLAS_OP_T, S, cur_rows, &alpha, subA, S, (const double *)dx, 1, &beta, subY, 1);
            }
        } else {
            float alpha = 1.0f, beta = 0.0f;
            for (int r = 0; r < S; r += chunk_rows) {
                int cur_rows = (r + chunk_rows <= S) ? chunk_rows : (S - r);
                float *subA = (float *)dA + (long)r * S;
                float *subY = (float *)dy + r;
                cublasSgemv(handle, CUBLAS_OP_T, S, cur_rows, &alpha, subA, S, (const float *)dx, 1, &beta, subY, 1);
            }
        }
        cudaEventRecord(ev_k1, stream);
        cudaEventSynchronize(ev_k1);
        float cold_ms = 0;
        cudaEventElapsedTime(&cold_ms, ev_k0, ev_k1);
        rec.t_kernel_cold_ms = (double)cold_ms;

        /* Warmup */
        for (int w = 0; w < warmup; w++) {
            if (strcmp(prec, "fp64") == 0) {
                double alpha = 1.0, beta = 0.0;
                for (int r = 0; r < S; r += chunk_rows) {
                    int cur_rows = (r + chunk_rows <= S) ? chunk_rows : (S - r);
                    cublasDgemv(handle, CUBLAS_OP_T, S, cur_rows, &alpha, (double *)dA + (long)r*S, S, (const double *)dx, 1, &beta, (double *)dy + r, 1);
                }
            } else {
                float alpha = 1.0f, beta = 0.0f;
                for (int r = 0; r < S; r += chunk_rows) {
                    int cur_rows = (r + chunk_rows <= S) ? chunk_rows : (S - r);
                    cublasSgemv(handle, CUBLAS_OP_T, S, cur_rows, &alpha, (float *)dA + (long)r*S, S, (const float *)dx, 1, &beta, (float *)dy + r, 1);
                }
            }
        }
        cudaStreamSynchronize(stream);

        /* Timed Warm Loop */
        cudaEventRecord(ev_k0, stream);
        for (int it = 0; it < iters; it++) {
            if (strcmp(prec, "fp64") == 0) {
                double alpha = 1.0, beta = 0.0;
                for (int r = 0; r < S; r += chunk_rows) {
                    int cur_rows = (r + chunk_rows <= S) ? chunk_rows : (S - r);
                    cublasDgemv(handle, CUBLAS_OP_T, S, cur_rows, &alpha, (double *)dA + (long)r*S, S, (const double *)dx, 1, &beta, (double *)dy + r, 1);
                }
            } else {
                float alpha = 1.0f, beta = 0.0f;
                for (int r = 0; r < S; r += chunk_rows) {
                    int cur_rows = (r + chunk_rows <= S) ? chunk_rows : (S - r);
                    cublasSgemv(handle, CUBLAS_OP_T, S, cur_rows, &alpha, (float *)dA + (long)r*S, S, (const float *)dx, 1, &beta, (float *)dy + r, 1);
                }
            }
        }
        cudaEventRecord(ev_k1, stream);
        cudaEventSynchronize(ev_k1);
        float warm_ms = 0;
        cudaEventElapsedTime(&warm_ms, ev_k0, ev_k1);
        rec.t_kernel_warm_ms = (double)warm_ms / iters;

        /* Explicit D2H */
        if (residency == RESIDENCY_GPU_EXPLICIT_COPY) {
            cudaEventRecord(ev_d2h0, stream);
            CUDA_CHECK(cudaMemcpyAsync(by.host_ptr, by.dev_ptr, vec_bytes, cudaMemcpyDeviceToHost, stream));
            cudaEventRecord(ev_d2h1, stream);
            cudaEventSynchronize(ev_d2h1);
            float d2h_ms = 0;
            cudaEventElapsedTime(&d2h_ms, ev_d2h0, ev_d2h1);
            rec.t_d2h_ms = (double)d2h_ms;
        } else {
            rec.t_d2h_ms = 0.0;
        }

        if (residency == RESIDENCY_GPU_UM_MIGRATE) {
            rec.t_e2e_ms = rec.t_kernel_cold_ms + rec.t_kernel_warm_ms;
            rec.c2c_bw_gbps = (rec.t_kernel_cold_ms > 0) ? ((double)rec.working_set_bytes / (rec.t_kernel_cold_ms * 1e6)) : 0.0;
        } else if (residency == RESIDENCY_GPU_ZERO_COPY) {
            rec.t_e2e_ms = rec.t_kernel_warm_ms;
            rec.c2c_bw_gbps = (rec.t_kernel_warm_ms > 0) ? ((double)rec.working_set_bytes / (rec.t_kernel_warm_ms * 1e6)) : 0.0;
        } else {
            rec.t_e2e_ms = rec.t_h2d_ms + rec.t_kernel_warm_ms + rec.t_d2h_ms;
            rec.c2c_bw_gbps = ((double)(mat_bytes + vec_bytes) / (rec.t_h2d_ms * 1e6 + 1e-9));
        }
        rec.eff_bw_gbps = (double)rec.working_set_bytes / (rec.t_e2e_ms * 1e6);
        rec.gflops = ((double)S * (double)S * 2.0) / (rec.t_kernel_warm_ms * 1e6);

        double peak_ref = (residency == RESIDENCY_GPU_ZERO_COPY) ? spec.pcie_or_nvlink_gbps : spec.peak_hbm_gbps;
        rec.pct_peak = (rec.eff_bw_gbps / peak_ref) * 100.0;

        cublasDestroy(handle);
        cudaStreamDestroy(stream);
        cudaEventDestroy(ev_h2d0);
        cudaEventDestroy(ev_h2d1);
        cudaEventDestroy(ev_k0);
        cudaEventDestroy(ev_k1);
        cudaEventDestroy(ev_d2h0);
        cudaEventDestroy(ev_d2h1);
    }

    log_bench_record(csv_out, &rec);
    printf("GEMV  [%-14s] %-4s S=%-6d: t_e2e=%8.2fms (h2d=%6.2fms, kern=%6.2fms, d2h=%6.2fms) | EffBW=%7.1f GB/s (%5.1f%% pk)\n",
           residency_to_str(residency), prec, S, rec.t_e2e_ms, rec.t_h2d_ms, rec.t_kernel_warm_ms, rec.t_d2h_ms, rec.eff_bw_gbps, rec.pct_peak);

    bench_free_buf(&bA);
    bench_free_buf(&bx);
    bench_free_buf(&by);
    return 0;
}

/* ── Benchmark RMSNorm ─────────────────────────────────────────────────────── */
__global__ void rmsnorm_kernel(float *out, const float *in, const float *weight, int B, int D, float eps) {
    int row = blockIdx.x;
    if (row >= B) return;
    const float *x = in + (long)row * D;
    float *y = out + (long)row * D;

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = x[i];
        thread_sum += val * val;
    }

    /* Warp reduction */
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    __shared__ float warp_sums[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    if (lane == 0) warp_sums[wid] = thread_sum;
    __syncthreads();

    float block_sum = 0.0f;
    if (wid == 0) {
        block_sum = (lane < (blockDim.x + 31) / 32) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) warp_sums[0] = block_sum;
    }
    __syncthreads();

    float mean_sq = warp_sums[0] / (float)D;
    float inv_rms = rsqrtf(mean_sq + eps);

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float w = weight ? weight[i] : 1.0f;
        y[i] = x[i] * inv_rms * w;
    }
}

int run_rmsnorm_bench(FILE *csv_out, residency_mode_t residency, int B, int D, int iters, int warmup) {
    bench_record_t rec;
    memset(&rec, 0, sizeof(rec));
    snprintf(rec.workload, sizeof(rec.workload), "rmsnorm");
    rec.regime = REGIME_STREAMING_BANDWIDTH;
    rec.residency = residency;
    snprintf(rec.precision, sizeof(rec.precision), "fp32");
    snprintf(rec.problem_size, sizeof(rec.problem_size), "B=%d,D=%d", B, D);
    snprintf(rec.status, sizeof(rec.status), "ok");

    size_t mat_bytes = (size_t)B * (size_t)D * sizeof(float);
    size_t weight_bytes = (size_t)D * sizeof(float);
    rec.working_set_bytes = 2 * mat_bytes + weight_bytes;

    gpu_spec_t spec = get_gpu_spec_for_device(0);

    double t0_alloc = bench_get_time_ms();
    bench_buf_t bX = bench_alloc_buf(mat_bytes, residency);
    bench_buf_t bY = bench_alloc_buf(mat_bytes, residency);
    bench_buf_t bW = bench_alloc_buf(weight_bytes, residency);
    rec.t_alloc_ms = bench_get_time_ms() - t0_alloc;

    if (!bX.host_ptr || !bY.host_ptr || !bW.host_ptr ||
        (residency == RESIDENCY_GPU_EXPLICIT_COPY && (!bX.dev_ptr || !bY.dev_ptr || !bW.dev_ptr))) {
        snprintf(rec.status, sizeof(rec.status), "oom-skip");
        log_bench_record(csv_out, &rec);
        bench_free_buf(&bX); bench_free_buf(&bY); bench_free_buf(&bW);
        return 0;
    }

    float *hx = (float *)bX.host_ptr;
    float *hy = (float *)bY.host_ptr;
    float *hw = (float *)bW.host_ptr;
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)B * D; i++) { hx[i] = 1.0f; hy[i] = 0.0f; }
    for (int i = 0; i < D; i++) hw[i] = 1.0f;

    if (residency == RESIDENCY_CPU_LOCAL) {
        rec.threads_or_sms = g_cpu_threads;
        omp_set_num_threads(g_cpu_threads);

        double t0 = bench_get_time_ms();
        #pragma omp parallel for schedule(static)
        for (int b = 0; b < B; b++) {
            float sum_sq = 0.0f;
            const float *x = hx + (long)b * D;
            float *y = hy + (long)b * D;
            for (int i = 0; i < D; i++) sum_sq += x[i] * x[i];
            float inv_rms = 1.0f / sqrtf(sum_sq / (float)D + 1e-5f);
            for (int i = 0; i < D; i++) y[i] = x[i] * inv_rms * hw[i];
        }
        rec.t_kernel_cold_ms = bench_get_time_ms() - t0;

        for (int w = 0; w < warmup; w++) {
            #pragma omp parallel for schedule(static)
            for (int b = 0; b < B; b++) {
                float sum_sq = 0.0f;
                const float *x = hx + (long)b * D;
                float *y = hy + (long)b * D;
                for (int i = 0; i < D; i++) sum_sq += x[i] * x[i];
                float inv_rms = 1.0f / sqrtf(sum_sq / (float)D + 1e-5f);
                for (int i = 0; i < D; i++) y[i] = x[i] * inv_rms * hw[i];
            }
        }

        t0 = bench_get_time_ms();
        for (int it = 0; it < iters; it++) {
            #pragma omp parallel for schedule(static)
            for (int b = 0; b < B; b++) {
                float sum_sq = 0.0f;
                const float *x = hx + (long)b * D;
                float *y = hy + (long)b * D;
                for (int i = 0; i < D; i++) sum_sq += x[i] * x[i];
                float inv_rms = 1.0f / sqrtf(sum_sq / (float)D + 1e-5f);
                for (int i = 0; i < D; i++) y[i] = x[i] * inv_rms * hw[i];
            }
        }
        rec.t_kernel_warm_ms = (bench_get_time_ms() - t0) / iters;
        rec.t_h2d_ms = 0.0;
        rec.t_d2h_ms = 0.0;
        rec.t_e2e_ms = rec.t_kernel_warm_ms;
        rec.eff_bw_gbps = (double)rec.working_set_bytes / (rec.t_e2e_ms * 1e6);
        rec.c2c_bw_gbps = 0.0;
        rec.gflops = ((double)B * (double)D * 4.0) / (rec.t_kernel_warm_ms * 1e6);
        rec.pct_peak = (rec.eff_bw_gbps / 500.0) * 100.0;
    } else {
        rec.threads_or_sms = 132;
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0); cudaEventCreate(&ev1);

        if (residency == RESIDENCY_GPU_EXPLICIT_COPY) {
            double th0 = bench_get_time_ms();
            cudaMemcpyAsync(bX.dev_ptr, bX.host_ptr, mat_bytes, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(bW.dev_ptr, bW.host_ptr, weight_bytes, cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
            rec.t_h2d_ms = bench_get_time_ms() - th0;
        }

        float *dx = (float *)((residency == RESIDENCY_GPU_EXPLICIT_COPY) ? bX.dev_ptr : bX.host_ptr);
        float *dy = (float *)((residency == RESIDENCY_GPU_EXPLICIT_COPY) ? bY.dev_ptr : bY.host_ptr);
        float *dw = (float *)((residency == RESIDENCY_GPU_EXPLICIT_COPY) ? bW.dev_ptr : bW.host_ptr);

        int block_size = 256;
        if (D < block_size) block_size = D;

        cudaEventRecord(ev0, stream);
        rmsnorm_kernel<<<B, block_size, 0, stream>>>(dy, dx, dw, B, D, 1e-5f);
        cudaEventRecord(ev1, stream);
        cudaStreamSynchronize(stream);
        float cold_ms = 0;
        cudaEventElapsedTime(&cold_ms, ev0, ev1);
        rec.t_kernel_cold_ms = (double)cold_ms;

        for (int w = 0; w < warmup; w++) {
            rmsnorm_kernel<<<B, block_size, 0, stream>>>(dy, dx, dw, B, D, 1e-5f);
        }
        cudaStreamSynchronize(stream);

        cudaEventRecord(ev0, stream);
        for (int it = 0; it < iters; it++) {
            rmsnorm_kernel<<<B, block_size, 0, stream>>>(dy, dx, dw, B, D, 1e-5f);
        }
        cudaEventRecord(ev1, stream);
        cudaStreamSynchronize(stream);
        float warm_ms = 0;
        cudaEventElapsedTime(&warm_ms, ev0, ev1);
        rec.t_kernel_warm_ms = (double)warm_ms / iters;

        if (residency == RESIDENCY_GPU_EXPLICIT_COPY) {
            double td0 = bench_get_time_ms();
            cudaMemcpyAsync(bY.host_ptr, bY.dev_ptr, mat_bytes, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            rec.t_d2h_ms = bench_get_time_ms() - td0;
        }

        if (residency == RESIDENCY_GPU_UM_MIGRATE) {
            rec.t_e2e_ms = rec.t_kernel_cold_ms + rec.t_kernel_warm_ms;
            rec.c2c_bw_gbps = (rec.t_kernel_cold_ms > 0) ? ((double)rec.working_set_bytes / (rec.t_kernel_cold_ms * 1e6)) : 0.0;
        } else if (residency == RESIDENCY_GPU_ZERO_COPY) {
            rec.t_e2e_ms = rec.t_kernel_warm_ms;
            rec.c2c_bw_gbps = (rec.t_kernel_warm_ms > 0) ? ((double)rec.working_set_bytes / (rec.t_kernel_warm_ms * 1e6)) : 0.0;
        } else {
            rec.t_e2e_ms = rec.t_h2d_ms + rec.t_kernel_warm_ms + rec.t_d2h_ms;
            rec.c2c_bw_gbps = ((double)mat_bytes * 2.0 / (rec.t_h2d_ms * 1e6 + 1e-9));
        }

        rec.eff_bw_gbps = (double)rec.working_set_bytes / (rec.t_e2e_ms * 1e6);
        rec.gflops = ((double)B * (double)D * 4.0) / (rec.t_kernel_warm_ms * 1e6);
        double peak_ref = (residency == RESIDENCY_GPU_ZERO_COPY) ? spec.pcie_or_nvlink_gbps : spec.peak_hbm_gbps;
        rec.pct_peak = (rec.eff_bw_gbps / peak_ref) * 100.0;

        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        cudaStreamDestroy(stream);
    }

    log_bench_record(csv_out, &rec);
    printf("RMSNorm [%-14s] B=%-6d D=%-5d : e2e=%7.2fms (kern=%7.2fms) | EffBW=%7.1f GB/s (%5.1f%% pk)\n",
           residency_to_str(residency), B, D, rec.t_e2e_ms, rec.t_kernel_warm_ms, rec.eff_bw_gbps, rec.pct_peak);

    bench_free_buf(&bX); bench_free_buf(&bY); bench_free_buf(&bW);
    return 0;
}

int main(int argc, char **argv) {
    const char *csv_file = "residency_streaming_sweep.csv";
    int iters = 10;
    int warmup = 3;
    int cpu_threads = 72;
    const char *workload = "all";
    const char *modes_arg = "all";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv_file = argv[++i];
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) cpu_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--workload") && i + 1 < argc) workload = argv[++i];
        else if (!strcmp(argv[i], "--modes") && i + 1 < argc) modes_arg = argv[++i];
    }

    g_cpu_threads = cpu_threads;
    openblas_set_num_threads(cpu_threads);
    omp_set_num_threads(cpu_threads);

    FILE *fout = fopen(csv_file, "a+");
    if (!fout) { perror("fopen csv"); return 1; }
    fseek(fout, 0, SEEK_END);
    if (ftell(fout) == 0) print_bench_csv_header(fout);

    printf("========================================================================================\n");
    printf(" NVIDIA Grace Hopper (GH200): Regime 2 Unit-Stride Streaming Across 4 Coherent NUMA Modes\n");
    printf(" Target CSV: %s | Iters=%d | Warmup=%d | CPU Threads=%d\n", csv_file, iters, warmup, cpu_threads);
    printf("========================================================================================\n\n");

    residency_mode_t modes[4];
    int n_modes = 0;
    if (!strcmp(modes_arg, "all")) {
        modes[0] = RESIDENCY_CPU_LOCAL;
        modes[1] = RESIDENCY_GPU_EXPLICIT_COPY;
        modes[2] = RESIDENCY_GPU_ZERO_COPY;
        modes[3] = RESIDENCY_GPU_UM_MIGRATE;
        n_modes = 4;
    } else {
        char buf[256]; snprintf(buf, sizeof(buf), "%s", modes_arg);
        for (char *m = strtok(buf, ","); m; m = strtok(NULL, ",")) {
            if (!strcmp(m, "cpu_local") || !strcmp(m, "cpu")) modes[n_modes++] = RESIDENCY_CPU_LOCAL;
            else if (!strcmp(m, "gpu_explicit") || !strcmp(m, "explicit")) modes[n_modes++] = RESIDENCY_GPU_EXPLICIT_COPY;
            else if (!strcmp(m, "gpu_zero_copy") || !strcmp(m, "zerocopy")) modes[n_modes++] = RESIDENCY_GPU_ZERO_COPY;
            else if (!strcmp(m, "gpu_um_migrate") || !strcmp(m, "um")) modes[n_modes++] = RESIDENCY_GPU_UM_MIGRATE;
        }
    }

    /* 1. SAXPY Sweeps: 16 working set sizes spanning L2 (4MB), SLC (64-512MB), HBM3 (1-88GB), Oversubscribed (104-256GB) */
    if (!strcmp(workload, "all") || !strcmp(workload, "saxpy")) {
        long saxpy_sizes[] = {
            524288L,       /* 4 MB (Grace L2 / Hopper L2) */
            2097152L,      /* 16 MB */
            4194304L,      /* 32 MB */
            6291456L,      /* 48 MB */
            8388608L,      /* 64 MB (Grace 114MB SLC) */
            12582912L,     /* 96 MB */
            16777216L,     /* 128 MB */
            33554432L,     /* 256 MB */
            67108864L,     /* 512 MB */
            134217728L,    /* 1 GB (HBM3) */
            536870912L,    /* 4 GB */
            2147483648L,   /* 16 GB */
            4294967296L,   /* 32 GB */
            8589934592L,   /* 64 GB */
            11811160064L,  /* 88 GB */
            13958643712L,  /* 104 GB (Oversubscribed > 96GB!) */
            17179869184L,  /* 128 GB */
            21474836480L,  /* 160 GB */
            27649568768L,  /* 206 GB */
            34359738368L   /* 256 GB */
        };
        int n_saxpy = sizeof(saxpy_sizes) / sizeof(saxpy_sizes[0]);

        printf("── [Part 1] SAXPY (1-Pass Streaming Across L2 -> SLC -> HBM3 -> Oversubscribed) ──────\n");
        for (int s = 0; s < n_saxpy; s++) {
            long N = saxpy_sizes[s];
            for (int m = 0; m < n_modes; m++) {
                run_saxpy_bench(fout, modes[m], "fp32", N, iters, warmup);
                run_saxpy_bench(fout, modes[m], "fp64", N, iters, warmup);
            }
            printf("----------------------------------------------------------------------------------------\n");
        }
    }

    /* 2. GEMV Sweeps: 1024 up to 65536 */
    if (!strcmp(workload, "all") || !strcmp(workload, "gemv")) {
        int gemv_sizes[] = {1024, 2048, 4096, 8192, 16384, 32768, 65536};
        int n_gemv = sizeof(gemv_sizes) / sizeof(gemv_sizes[0]);

        printf("\n── [Part 2] GEMV (1-Pass Matrix Streaming, AI=0.25) ──────────────────────────────────\n");
        for (int s = 0; s < n_gemv; s++) {
            int S = gemv_sizes[s];
            for (int m = 0; m < n_modes; m++) {
                run_gemv_bench(fout, modes[m], "fp32", S, iters, warmup);
                run_gemv_bench(fout, modes[m], "fp64", S, iters, warmup);
            }
            printf("----------------------------------------------------------------------------------------\n");
        }
    }

    /* 3. RMSNorm Sweeps: Token Batches with D=4096 / D=8192 */
    if (!strcmp(workload, "all") || !strcmp(workload, "rmsnorm")) {
        int rmsnorm_batches[] = {256, 1024, 4096, 16384, 65536, 262144, 1048576};
        int n_rms = sizeof(rmsnorm_batches) / sizeof(rmsnorm_batches[0]);

        printf("\n── [Part 3] RMSNorm (Transformer Normalization Streaming) ───────────────────────────\n");
        for (int s = 0; s < n_rms; s++) {
            int B = rmsnorm_batches[s];
            for (int m = 0; m < n_modes; m++) {
                run_rmsnorm_bench(fout, modes[m], B, 4096, iters, warmup);
            }
            printf("----------------------------------------------------------------------------------------\n");
        }
    }

    fclose(fout);
    printf("\nBenchmark complete! Results recorded to %s\n", csv_file);
    return 0;
}
