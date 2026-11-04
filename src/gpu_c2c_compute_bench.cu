/*
 * gpu_c2c_compute_bench.cu
 *
 * Cycle 2: Regime 1 (Arithmetic Compute-Bound: GEMM & Monte Carlo)
 * across 4 Coherent NUMA Residency Modes on NVIDIA Grace Hopper (GH200):
 *   1. CPU_LOCAL         (72 Grace Neoverse-V2 OpenMP cores, LPDDR5X)
 *   2. GPU_EXPLICIT_COPY (Pinned LPDDR5X -> HBM3 -> cuBLAS -> LPDDR5X)
 *   3. GPU_ZERO_COPY     (Mapped LPDDR5X: Demonstrates the multi-pass L2 eviction collapse!)
 *   4. GPU_UM_MIGRATE    (cudaMallocManaged with cold ATS faulting vs warm)
 *
 * Strictly follows vendor/standard libraries:
 *   - GPU: cuBLAS (cublasGemmEx for BF16/FP32/FP64)
 *   - CPU: OpenMP parallel tiled compute on Grace Neoverse-V2
 */

#include "bench_harness.h"
#include "gpu_device_specs.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cblas.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

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

/* ── Benchmark GEMM ────────────────────────────────────────────────────────── */
int run_gemm_bench(FILE *csv_out, residency_mode_t residency, const char *prec, int M, int N, int K, int iters, int warmup) {
    bench_record_t rec;
    memset(&rec, 0, sizeof(rec));
    snprintf(rec.workload, sizeof(rec.workload), "gemm");
    rec.regime = REGIME_ARITHMETIC_COMPUTE;
    rec.residency = residency;
    snprintf(rec.precision, sizeof(rec.precision), "%s", prec);
    snprintf(rec.problem_size, sizeof(rec.problem_size), "M=%d,N=%d,K=%d", M, N, K);
    snprintf(rec.status, sizeof(rec.status), "ok");

    size_t in_elem_bytes = sizeof(float);
    size_t out_elem_bytes = sizeof(float);
    cudaDataType_t cuda_in_type = CUDA_R_32F;
    cudaDataType_t cuda_out_type = CUDA_R_32F;
    cublasComputeType_t cuda_comp_type = CUBLAS_COMPUTE_32F;

    if (strcmp(prec, "fp64") == 0) {
        in_elem_bytes = sizeof(double);
        out_elem_bytes = sizeof(double);
        cuda_in_type = CUDA_R_64F;
        cuda_out_type = CUDA_R_64F;
        cuda_comp_type = CUBLAS_COMPUTE_64F;
    } else if (strcmp(prec, "bf16") == 0) {
        in_elem_bytes = sizeof(__nv_bfloat16);
        out_elem_bytes = sizeof(__nv_bfloat16);
        cuda_in_type = CUDA_R_16BF;
        cuda_out_type = CUDA_R_16BF;
        cuda_comp_type = CUBLAS_COMPUTE_32F_FAST_16BF;
    } else if (strcmp(prec, "fp16") == 0) {
        in_elem_bytes = sizeof(__half);
        out_elem_bytes = sizeof(__half);
        cuda_in_type = CUDA_R_16F;
        cuda_out_type = CUDA_R_16F;
        cuda_comp_type = CUBLAS_COMPUTE_32F_FAST_16F;
    } else if (strcmp(prec, "tf32") == 0) {
        in_elem_bytes = sizeof(float);
        out_elem_bytes = sizeof(float);
        cuda_in_type = CUDA_R_32F;
        cuda_out_type = CUDA_R_32F;
        cuda_comp_type = CUBLAS_COMPUTE_32F_FAST_TF32;
    }

    size_t bytes_A = (size_t)M * K * in_elem_bytes;
    size_t bytes_B = (size_t)K * N * in_elem_bytes;
    size_t bytes_C = (size_t)M * N * out_elem_bytes;
    rec.working_set_bytes = bytes_A + bytes_B + bytes_C;

    double num_flops = 2.0 * (double)M * (double)N * (double)K;
    gpu_spec_t spec = get_gpu_spec_for_device(0);

    double t0_alloc = bench_get_time_ms();
    bench_buf_t bA = bench_alloc_buf(bytes_A, residency);
    bench_buf_t bB = bench_alloc_buf(bytes_B, residency);
    bench_buf_t bC = bench_alloc_buf(bytes_C, residency);
    rec.t_alloc_ms = bench_get_time_ms() - t0_alloc;

    if (!bA.host_ptr || !bB.host_ptr || !bC.host_ptr ||
        (residency == RESIDENCY_GPU_EXPLICIT_COPY && (!bA.dev_ptr || !bB.dev_ptr || !bC.dev_ptr))) {
        snprintf(rec.status, sizeof(rec.status), "oom-skip");
        log_bench_record(csv_out, &rec);
        bench_free_buf(&bA);
        bench_free_buf(&bB);
        bench_free_buf(&bC);
        return 0;
    }

    /* Initialize matrices */
    if (strcmp(prec, "fp64") == 0) {
        double *A = (double *)bA.host_ptr;
        double *B = (double *)bB.host_ptr;
        double *C = (double *)bC.host_ptr;
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < (long)M * K; i++) A[i] = 1.0;
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < (long)K * N; i++) B[i] = 1.0;
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < (long)M * N; i++) C[i] = 0.0;
    } else if (strcmp(prec, "bf16") == 0) {
        __nv_bfloat16 *A = (__nv_bfloat16 *)bA.host_ptr;
        __nv_bfloat16 *B = (__nv_bfloat16 *)bB.host_ptr;
        __nv_bfloat16 *C = (__nv_bfloat16 *)bC.host_ptr;
        __nv_bfloat16 v1 = __float2bfloat16(1.0f);
        __nv_bfloat16 v0 = __float2bfloat16(0.0f);
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < (long)M * K; i++) A[i] = v1;
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < (long)K * N; i++) B[i] = v1;
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < (long)M * N; i++) C[i] = v0;
    } else if (strcmp(prec, "fp16") == 0) {
        __half *A = (__half *)bA.host_ptr;
        __half *B = (__half *)bB.host_ptr;
        __half *C = (__half *)bC.host_ptr;
        __half v1 = __float2half(1.0f);
        __half v0 = __float2half(0.0f);
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < (long)M * K; i++) A[i] = v1;
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < (long)K * N; i++) B[i] = v1;
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < (long)M * N; i++) C[i] = v0;
    } else {
        float *A = (float *)bA.host_ptr;
        float *B = (float *)bB.host_ptr;
        float *C = (float *)bC.host_ptr;
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < (long)M * K; i++) A[i] = 1.0f;
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < (long)K * N; i++) B[i] = 1.0f;
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < (long)M * N; i++) C[i] = 0.0f;
    }

    if (residency == RESIDENCY_CPU_LOCAL) {
        rec.threads_or_sms = 72;
        openblas_set_num_threads(72);

        auto cpu_gemm_step = [&]() {
            if (strcmp(prec, "fp64") == 0) {
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            M, N, K, 1.0,
                            (const double *)bA.host_ptr, K,
                            (const double *)bB.host_ptr, N,
                            0.0, (double *)bC.host_ptr, N);
            } else if (strcmp(prec, "fp32") == 0 || strcmp(prec, "tf32") == 0) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            M, N, K, 1.0f,
                            (const float *)bA.host_ptr, K,
                            (const float *)bB.host_ptr, N,
                            0.0f, (float *)bC.host_ptr, N);
            } else if (strcmp(prec, "fp16") == 0) {
                std::vector<float> fA((size_t)M * K), fB((size_t)K * N), fC((size_t)M * N);
                const __half *hA = (const __half *)bA.host_ptr;
                const __half *hB = (const __half *)bB.host_ptr;
                #pragma omp parallel for schedule(static)
                for (long i = 0; i < (long)M * K; i++) fA[i] = __half2float(hA[i]);
                #pragma omp parallel for schedule(static)
                for (long i = 0; i < (long)K * N; i++) fB[i] = __half2float(hB[i]);
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            M, N, K, 1.0f, fA.data(), K, fB.data(), N, 0.0f, fC.data(), N);
                __half *hC = (__half *)bC.host_ptr;
                #pragma omp parallel for schedule(static)
                for (long i = 0; i < (long)M * N; i++) hC[i] = __float2half(fC[i]);
            } else {
                /* BF16 on CPU: convert to FP32, run cblas_sgemm, convert back */
                std::vector<float> fA((size_t)M * K), fB((size_t)K * N), fC((size_t)M * N);
                const __nv_bfloat16 *b16A = (const __nv_bfloat16 *)bA.host_ptr;
                const __nv_bfloat16 *b16B = (const __nv_bfloat16 *)bB.host_ptr;
                #pragma omp parallel for schedule(static)
                for (long i = 0; i < (long)M * K; i++) fA[i] = __bfloat162float(b16A[i]);
                #pragma omp parallel for schedule(static)
                for (long i = 0; i < (long)K * N; i++) fB[i] = __bfloat162float(b16B[i]);
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            M, N, K, 1.0f, fA.data(), K, fB.data(), N, 0.0f, fC.data(), N);
                __nv_bfloat16 *b16C = (__nv_bfloat16 *)bC.host_ptr;
                #pragma omp parallel for schedule(static)
                for (long i = 0; i < (long)M * N; i++) b16C[i] = __float2bfloat16(fC[i]);
            }
        };

        double t0 = bench_get_time_ms();
        cpu_gemm_step();
        rec.t_kernel_cold_ms = bench_get_time_ms() - t0;

        for (int w = 0; w < warmup; w++) cpu_gemm_step();

        t0 = bench_get_time_ms();
        for (int it = 0; it < iters; it++) cpu_gemm_step();
        rec.t_kernel_warm_ms = (bench_get_time_ms() - t0) / iters;
        rec.t_h2d_ms = 0.0;
        rec.t_d2h_ms = 0.0;
        rec.t_e2e_ms = rec.t_kernel_warm_ms;
        rec.gflops = num_flops / (rec.t_kernel_warm_ms * 1e6);
        rec.eff_bw_gbps = (double)rec.working_set_bytes / (rec.t_e2e_ms * 1e6);
        rec.c2c_bw_gbps = 0.0;
        rec.pct_peak = (rec.gflops / (72.0 * 3.1 * 32.0)) * 100.0;
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
            CUDA_CHECK(cudaMemcpyAsync(bA.dev_ptr, bA.host_ptr, bytes_A, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(bB.dev_ptr, bB.host_ptr, bytes_B, cudaMemcpyHostToDevice, stream));
            cudaEventRecord(ev_h2d1, stream);
            cudaEventSynchronize(ev_h2d1);
            float h2d_ms = 0;
            cudaEventElapsedTime(&h2d_ms, ev_h2d0, ev_h2d1);
            rec.t_h2d_ms = (double)h2d_ms;
        } else {
            rec.t_h2d_ms = 0.0;
        }

        void *dA = (residency == RESIDENCY_GPU_EXPLICIT_COPY) ? bA.dev_ptr : bA.host_ptr;
        void *dB = (residency == RESIDENCY_GPU_EXPLICIT_COPY) ? bB.dev_ptr : bB.host_ptr;
        void *dC = (residency == RESIDENCY_GPU_EXPLICIT_COPY) ? bC.dev_ptr : bC.host_ptr;

        float alpha_f = 1.0f, beta_f = 0.0f;
        double alpha_d = 1.0, beta_d = 0.0;
        void *p_alpha = (strcmp(prec, "fp64") == 0) ? (void *)&alpha_d : (void *)&alpha_f;
        void *p_beta  = (strcmp(prec, "fp64") == 0) ? (void *)&beta_d  : (void *)&beta_f;

        /* Measure Cold Launch */
        cudaEventRecord(ev_k0, stream);
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     N, M, K,
                     p_alpha,
                     dB, cuda_in_type, N,
                     dA, cuda_in_type, K,
                     p_beta,
                     dC, cuda_out_type, N,
                     cuda_comp_type, CUBLAS_GEMM_DEFAULT);
        cudaEventRecord(ev_k1, stream);
        cudaEventSynchronize(ev_k1);
        float cold_ms = 0;
        cudaEventElapsedTime(&cold_ms, ev_k0, ev_k1);
        rec.t_kernel_cold_ms = (double)cold_ms;

        /* Warmup */
        for (int w = 0; w < warmup; w++) {
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         N, M, K,
                         p_alpha,
                         dB, cuda_in_type, N,
                         dA, cuda_in_type, K,
                         p_beta,
                         dC, cuda_out_type, N,
                         cuda_comp_type, CUBLAS_GEMM_DEFAULT);
        }
        cudaStreamSynchronize(stream);

        /* Timed Warm Loop */
        cudaEventRecord(ev_k0, stream);
        for (int it = 0; it < iters; it++) {
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         N, M, K,
                         p_alpha,
                         dB, cuda_in_type, N,
                         dA, cuda_in_type, K,
                         p_beta,
                         dC, cuda_out_type, N,
                         cuda_comp_type, CUBLAS_GEMM_DEFAULT);
        }
        cudaEventRecord(ev_k1, stream);
        cudaEventSynchronize(ev_k1);
        float warm_ms = 0;
        cudaEventElapsedTime(&warm_ms, ev_k0, ev_k1);
        rec.t_kernel_warm_ms = (double)warm_ms / iters;

        /* Explicit D2H */
        if (residency == RESIDENCY_GPU_EXPLICIT_COPY) {
            cudaEventRecord(ev_d2h0, stream);
            CUDA_CHECK(cudaMemcpyAsync(bC.host_ptr, bC.dev_ptr, bytes_C, cudaMemcpyDeviceToHost, stream));
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
            rec.c2c_bw_gbps = (rec.t_kernel_cold_ms > 0) ? ((double)(bytes_A + bytes_B) / (rec.t_kernel_cold_ms * 1e6)) : 0.0;
        } else if (residency == RESIDENCY_GPU_ZERO_COPY) {
            rec.t_e2e_ms = rec.t_kernel_warm_ms;
            rec.c2c_bw_gbps = (rec.t_kernel_warm_ms > 0) ? ((double)rec.working_set_bytes / (rec.t_kernel_warm_ms * 1e6)) : 0.0;
        } else {
            rec.t_e2e_ms = rec.t_h2d_ms + rec.t_kernel_warm_ms + rec.t_d2h_ms;
            rec.c2c_bw_gbps = (double)(bytes_A + bytes_B) / (rec.t_h2d_ms * 1e6 + 1e-9);
        }

        rec.gflops = num_flops / (rec.t_kernel_warm_ms * 1e6);
        rec.eff_bw_gbps = (double)rec.working_set_bytes / (rec.t_e2e_ms * 1e6);

        double peak_tflops = spec.peak_fp32_tflops;
        if (strcmp(prec, "fp64") == 0) peak_tflops = spec.peak_fp64_tensor_tflops;
        else if (strcmp(prec, "bf16") == 0) peak_tflops = spec.peak_bf16_tflops;
        else if (strcmp(prec, "fp16") == 0) peak_tflops = spec.peak_fp16_tflops;
        else if (strcmp(prec, "tf32") == 0) peak_tflops = spec.peak_tf32_tflops;

        rec.pct_peak = ((rec.gflops / 1e3) / peak_tflops) * 100.0;

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
    printf("GEMM  [%-14s] %-4s M=%-5d N=%-5d K=%-5d: e2e=%7.2fms (kern=%7.2fms) | %8.1f GFLOPS (%5.1f%% pk)\n",
           residency_to_str(residency), prec, M, N, K, rec.t_e2e_ms, rec.t_kernel_warm_ms, rec.gflops, rec.pct_peak);

    bench_free_buf(&bA);
    bench_free_buf(&bB);
    bench_free_buf(&bC);
    return 0;
}

/* ── Benchmark Monte Carlo (Black-Scholes Option Pricing) ─────────────────── */
__device__ inline float cnd_cuda(float d) {
    const float A1 =  0.319381530f;
    const float A2 = -0.356563782f;
    const float A3 =  1.781477937f;
    const float A4 = -1.821255978f;
    const float A5 =  1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;
    float K = 1.0f / (1.0f + 0.2316419f * fabsf(d));
    float cnd = RSQRT2PI * expf(-0.5f * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
    return (d > 0.0f) ? (1.0f - cnd) : cnd;
}

__device__ inline double cnd_cuda_fp64(double d) {
    const double A1 =  0.319381530;
    const double A2 = -0.356563782;
    const double A3 =  1.781477937;
    const double A4 = -1.821255978;
    const double A5 =  1.330274429;
    const double RSQRT2PI = 0.39894228040143267793994605993438;
    double K = 1.0 / (1.0 + 0.2316419 * fabs(d));
    double cnd = RSQRT2PI * exp(-0.5 * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
    return (d > 0.0) ? (1.0 - cnd) : cnd;
}

__global__ void black_scholes_kernel(float *call, float *put, const float *S, const float *X,
                                     const float *T, float R, float V, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        float s = S[i], x = X[i], t = T[i];
        float sqrtT = sqrtf(t);
        float d1 = (logf(s / x) + (R + 0.5f * V * V) * t) / (V * sqrtT);
        float d2 = d1 - V * sqrtT;
        float cndD1 = cnd_cuda(d1);
        float cndD2 = cnd_cuda(d2);
        float expRT = expf(-R * t);
        call[i] = s * cndD1 - x * expRT * cndD2;
        put[i]  = x * expRT * (1.0f - cndD2) - s * (1.0f - cndD1);
    }
}

__global__ void black_scholes_kernel_fp64(double *call, double *put, const double *S, const double *X,
                                          const double *T, double R, double V, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        double s = S[i], x = X[i], t = T[i];
        double sqrtT = sqrt(t);
        double d1 = (log(s / x) + (R + 0.5 * V * V) * t) / (V * sqrtT);
        double d2 = d1 - V * sqrtT;
        double cndD1 = cnd_cuda_fp64(d1);
        double cndD2 = cnd_cuda_fp64(d2);
        double expRT = exp(-R * t);
        call[i] = s * cndD1 - x * expRT * cndD2;
        put[i]  = x * expRT * (1.0 - cndD2) - s * (1.0 - cndD1);
    }
}

int run_montecarlo_bench(FILE *csv_out, residency_mode_t residency, const char *prec, int N, int iters, int warmup) {
    bench_record_t rec;
    memset(&rec, 0, sizeof(rec));
    snprintf(rec.workload, sizeof(rec.workload), "montecarlo");
    rec.regime = REGIME_ARITHMETIC_COMPUTE;
    rec.residency = residency;
    snprintf(rec.precision, sizeof(rec.precision), "%s", prec);
    snprintf(rec.problem_size, sizeof(rec.problem_size), "N=%d", N);
    snprintf(rec.status, sizeof(rec.status), "ok");

    bool is_fp64 = (strcmp(prec, "fp64") == 0);
    size_t elem_size = is_fp64 ? sizeof(double) : sizeof(float);
    size_t vec_bytes = (size_t)N * elem_size;
    rec.working_set_bytes = 5 * vec_bytes; /* 3 inputs (S, X, T) + 2 outputs (call, put) */
    double num_flops = (double)N * 50.0;   /* ~50 FLOPs per Black-Scholes call/put evaluation */

    double t0_alloc = bench_get_time_ms();
    bench_buf_t bS = bench_alloc_buf(vec_bytes, residency);
    bench_buf_t bX = bench_alloc_buf(vec_bytes, residency);
    bench_buf_t bT = bench_alloc_buf(vec_bytes, residency);
    bench_buf_t bCall = bench_alloc_buf(vec_bytes, residency);
    bench_buf_t bPut = bench_alloc_buf(vec_bytes, residency);
    rec.t_alloc_ms = bench_get_time_ms() - t0_alloc;

    if (!bS.host_ptr || !bX.host_ptr || !bT.host_ptr || !bCall.host_ptr || !bPut.host_ptr) {
        snprintf(rec.status, sizeof(rec.status), "oom-skip");
        log_bench_record(csv_out, &rec);
        bench_free_buf(&bS); bench_free_buf(&bX); bench_free_buf(&bT);
        bench_free_buf(&bCall); bench_free_buf(&bPut);
        return 0;
    }

    if (is_fp64) {
        double *hS = (double *)bS.host_ptr;
        double *hX = (double *)bX.host_ptr;
        double *hT = (double *)bT.host_ptr;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            hS[i] = 100.0; hX[i] = 100.0; hT[i] = 1.0;
        }
    } else {
        float *hS = (float *)bS.host_ptr;
        float *hX = (float *)bX.host_ptr;
        float *hT = (float *)bT.host_ptr;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            hS[i] = 100.0f; hX[i] = 100.0f; hT[i] = 1.0f;
        }
    }

    if (residency == RESIDENCY_CPU_LOCAL) {
        rec.threads_or_sms = 72;
        if (is_fp64) {
            double *call = (double *)bCall.host_ptr;
            double *put = (double *)bPut.host_ptr;
            const double *hS = (const double *)bS.host_ptr;
            const double *hX = (const double *)bX.host_ptr;
            const double *hT = (const double *)bT.host_ptr;
            double R = 0.05, V = 0.2;

            auto cpu_mc_step = [&]() {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < N; i++) {
                    double s = hS[i], x = hX[i], t = hT[i];
                    double sqrtT = sqrt(t);
                    double d1 = (log(s / x) + (R + 0.5 * V * V) * t) / (V * sqrtT);
                    double d2 = d1 - V * sqrtT;
                    double cndD1 = 0.5 * (1.0 + erf(d1 * 0.7071067811865475));
                    double cndD2 = 0.5 * (1.0 + erf(d2 * 0.7071067811865475));
                    double expRT = exp(-R * t);
                    call[i] = s * cndD1 - x * expRT * cndD2;
                    put[i]  = x * expRT * (1.0 - cndD2) - s * (1.0 - cndD1);
                }
            };

            double t0 = bench_get_time_ms();
            cpu_mc_step();
            rec.t_kernel_cold_ms = bench_get_time_ms() - t0;

            for (int w = 0; w < warmup; w++) cpu_mc_step();

            t0 = bench_get_time_ms();
            for (int it = 0; it < iters; it++) cpu_mc_step();
            rec.t_kernel_warm_ms = (bench_get_time_ms() - t0) / iters;
        } else {
            float *call = (float *)bCall.host_ptr;
            float *put = (float *)bPut.host_ptr;
            const float *hS = (const float *)bS.host_ptr;
            const float *hX = (const float *)bX.host_ptr;
            const float *hT = (const float *)bT.host_ptr;
            float R = 0.05f, V = 0.2f;

            auto cpu_mc_step = [&]() {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < N; i++) {
                    float s = hS[i], x = hX[i], t = hT[i];
                    float sqrtT = sqrtf(t);
                    float d1 = (logf(s / x) + (R + 0.5f * V * V) * t) / (V * sqrtT);
                    float d2 = d1 - V * sqrtT;
                    float cndD1 = 0.5f * (1.0f + erff(d1 * 0.70710678f));
                    float cndD2 = 0.5f * (1.0f + erff(d2 * 0.70710678f));
                    float expRT = expf(-R * t);
                    call[i] = s * cndD1 - x * expRT * cndD2;
                    put[i]  = x * expRT * (1.0f - cndD2) - s * (1.0f - cndD1);
                }
            };

            double t0 = bench_get_time_ms();
            cpu_mc_step();
            rec.t_kernel_cold_ms = bench_get_time_ms() - t0;

            for (int w = 0; w < warmup; w++) cpu_mc_step();

            t0 = bench_get_time_ms();
            for (int it = 0; it < iters; it++) cpu_mc_step();
            rec.t_kernel_warm_ms = (bench_get_time_ms() - t0) / iters;
        }
        rec.t_h2d_ms = 0.0;
        rec.t_d2h_ms = 0.0;
        rec.t_e2e_ms = rec.t_kernel_warm_ms;
        rec.gflops = num_flops / (rec.t_kernel_warm_ms * 1e6);
        rec.eff_bw_gbps = (double)rec.working_set_bytes / (rec.t_e2e_ms * 1e6);
        rec.pct_peak = (rec.gflops / (72.0 * 2.6 * (is_fp64 ? 16.0 : 32.0))) * 100.0;
    } else {
        rec.threads_or_sms = 132;
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        cudaEvent_t ev_h2d0, ev_h2d1, ev_k0, ev_k1, ev_d2h0, ev_d2h1;
        CUDA_CHECK(cudaEventCreate(&ev_h2d0));
        CUDA_CHECK(cudaEventCreate(&ev_h2d1));
        CUDA_CHECK(cudaEventCreate(&ev_k0));
        CUDA_CHECK(cudaEventCreate(&ev_k1));
        CUDA_CHECK(cudaEventCreate(&ev_d2h0));
        CUDA_CHECK(cudaEventCreate(&ev_d2h1));

        if (residency == RESIDENCY_GPU_EXPLICIT_COPY) {
            cudaEventRecord(ev_h2d0, stream);
            CUDA_CHECK(cudaMemcpyAsync(bS.dev_ptr, bS.host_ptr, vec_bytes, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(bX.dev_ptr, bX.host_ptr, vec_bytes, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(bT.dev_ptr, bT.host_ptr, vec_bytes, cudaMemcpyHostToDevice, stream));
            cudaEventRecord(ev_h2d1, stream);
            cudaEventSynchronize(ev_h2d1);
            float h2d_ms = 0;
            cudaEventElapsedTime(&h2d_ms, ev_h2d0, ev_h2d1);
            rec.t_h2d_ms = (double)h2d_ms;
        } else {
            rec.t_h2d_ms = 0.0;
        }

        void *dS = (residency == RESIDENCY_GPU_EXPLICIT_COPY) ? bS.dev_ptr : bS.host_ptr;
        void *dX = (residency == RESIDENCY_GPU_EXPLICIT_COPY) ? bX.dev_ptr : bX.host_ptr;
        void *dT = (residency == RESIDENCY_GPU_EXPLICIT_COPY) ? bT.dev_ptr : bT.host_ptr;
        void *dCall = (residency == RESIDENCY_GPU_EXPLICIT_COPY) ? bCall.dev_ptr : bCall.host_ptr;
        void *dPut = (residency == RESIDENCY_GPU_EXPLICIT_COPY) ? bPut.dev_ptr : bPut.host_ptr;

        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        auto gpu_mc_launch = [&]() {
            if (is_fp64) {
                black_scholes_kernel_fp64<<<blocks, threads, 0, stream>>>((double *)dCall, (double *)dPut, (const double *)dS, (const double *)dX, (const double *)dT, 0.05, 0.2, N);
            } else {
                black_scholes_kernel<<<blocks, threads, 0, stream>>>((float *)dCall, (float *)dPut, (const float *)dS, (const float *)dX, (const float *)dT, 0.05f, 0.2f, N);
            }
        };

        cudaEventRecord(ev_k0, stream);
        gpu_mc_launch();
        cudaEventRecord(ev_k1, stream);
        cudaEventSynchronize(ev_k1);
        float cold_ms = 0;
        cudaEventElapsedTime(&cold_ms, ev_k0, ev_k1);
        rec.t_kernel_cold_ms = (double)cold_ms;

        for (int w = 0; w < warmup; w++) gpu_mc_launch();
        cudaStreamSynchronize(stream);

        cudaEventRecord(ev_k0, stream);
        for (int it = 0; it < iters; it++) gpu_mc_launch();
        cudaEventRecord(ev_k1, stream);
        cudaEventSynchronize(ev_k1);
        float warm_ms = 0;
        cudaEventElapsedTime(&warm_ms, ev_k0, ev_k1);
        rec.t_kernel_warm_ms = (double)warm_ms / iters;

        if (residency == RESIDENCY_GPU_EXPLICIT_COPY) {
            cudaEventRecord(ev_d2h0, stream);
            CUDA_CHECK(cudaMemcpyAsync(bCall.host_ptr, bCall.dev_ptr, vec_bytes, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(bPut.host_ptr, bPut.dev_ptr, vec_bytes, cudaMemcpyDeviceToHost, stream));
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
            rec.c2c_bw_gbps = (((double)vec_bytes * 5.0) / (rec.t_h2d_ms * 1e6 + 1e-9));
        }

        rec.gflops = num_flops / (rec.t_kernel_warm_ms * 1e6);
        rec.eff_bw_gbps = (double)rec.working_set_bytes / (rec.t_e2e_ms * 1e6);

        gpu_spec_t spec = get_gpu_spec_for_device(0);
        double peak_tflops = is_fp64 ? spec.peak_fp64_cuda_tflops : spec.peak_fp32_tflops;
        rec.pct_peak = ((rec.gflops / 1e3) / peak_tflops) * 100.0;

        cudaStreamDestroy(stream);
        cudaEventDestroy(ev_h2d0); cudaEventDestroy(ev_h2d1);
        cudaEventDestroy(ev_k0);   cudaEventDestroy(ev_k1);
        cudaEventDestroy(ev_d2h0); cudaEventDestroy(ev_d2h1);
    }

    log_bench_record(csv_out, &rec);
    printf("MC    [%-14s] %-4s N=%-10d: e2e=%7.2fms (kern=%7.2fms) | %8.1f GFLOPS (%5.1f%% pk)\n",
           residency_to_str(residency), prec, N, rec.t_e2e_ms, rec.t_kernel_warm_ms, rec.gflops, rec.pct_peak);

    bench_free_buf(&bS); bench_free_buf(&bX); bench_free_buf(&bT);
    bench_free_buf(&bCall); bench_free_buf(&bPut);
    return 0;
}

int main(int argc, char **argv) {
    const char *csv_file = "residency_compute_sweep.csv";
    int iters = 5;
    int warmup = 2;
    int cpu_threads = 72;
    const char *workload = "all";
    const char *sizes_arg = NULL;
    const char *precs_arg = "fp32,fp64,bf16,fp16,tf32";
    const char *modes_arg = "all";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv_file = argv[++i];
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) cpu_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--workload") && i + 1 < argc) workload = argv[++i];
        else if (!strcmp(argv[i], "--sizes") && i + 1 < argc) sizes_arg = argv[++i];
        else if (!strcmp(argv[i], "--precs") && i + 1 < argc) precs_arg = argv[++i];
        else if (!strcmp(argv[i], "--modes") && i + 1 < argc) modes_arg = argv[++i];
    }

    openblas_set_num_threads(cpu_threads);
    omp_set_num_threads(cpu_threads);

    FILE *fout = fopen(csv_file, "a+");
    if (!fout) { perror("fopen csv"); return 1; }
    fseek(fout, 0, SEEK_END);
    if (ftell(fout) == 0) print_bench_csv_header(fout);

    printf("========================================================================================\n");
    printf(" NVIDIA Grace Hopper (GH200): Regime 1 Arithmetic Compute Across 4 Coherent NUMA Modes\n");
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

    /* 1. Monte Carlo Black-Scholes */
    if (!strcmp(workload, "all") || !strcmp(workload, "montecarlo")) {
        int mc_sizes[] = {1000000, 8000000, 32000000, 64000000, 134217728};
        int n_mc = sizeof(mc_sizes) / sizeof(mc_sizes[0]);

        printf("── [Part 1] Monte Carlo (Black-Scholes Vector Math, Regime 1A) ───────────────────────\n");
        for (int s = 0; s < n_mc; s++) {
            int N = mc_sizes[s];
            for (int m = 0; m < n_modes; m++) {
                if (strstr(precs_arg, "fp32")) run_montecarlo_bench(fout, modes[m], "fp32", N, iters, warmup);
                if (strstr(precs_arg, "fp64")) run_montecarlo_bench(fout, modes[m], "fp64", N, iters, warmup);
            }
            printf("----------------------------------------------------------------------------------------\n");
        }
    }

    /* 2. GEMM */
    if (!strcmp(workload, "all") || !strcmp(workload, "gemm")) {
        int gemm_sizes[32];
        int n_gemm = 0;
        if (sizes_arg) {
            char buf[512]; snprintf(buf, sizeof(buf), "%s", sizes_arg);
            for (char *tok = strtok(buf, ","); tok && n_gemm < 32; tok = strtok(NULL, ",")) {
                gemm_sizes[n_gemm++] = atoi(tok);
            }
        } else {
            int default_sizes[] = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 49152};
            n_gemm = sizeof(default_sizes) / sizeof(default_sizes[0]);
            for (int i = 0; i < n_gemm; i++) gemm_sizes[i] = default_sizes[i];
        }

        printf("\n── [Part 2] GEMM (Multi-Pass Matrix Compute, Testing C2C Eviction Penalty) ───────────\n");
        for (int s = 0; s < n_gemm; s++) {
            int S = gemm_sizes[s];
            for (int m = 0; m < n_modes; m++) {
                if (strstr(precs_arg, "fp32")) run_gemm_bench(fout, modes[m], "fp32", S, S, S, iters, warmup);
                if (strstr(precs_arg, "fp64")) run_gemm_bench(fout, modes[m], "fp64", S, S, S, iters, warmup);
                if (strstr(precs_arg, "bf16")) run_gemm_bench(fout, modes[m], "bf16", S, S, S, iters, warmup);
                if (strstr(precs_arg, "fp16")) run_gemm_bench(fout, modes[m], "fp16", S, S, S, iters, warmup);
                if (strstr(precs_arg, "tf32")) run_gemm_bench(fout, modes[m], "tf32", S, S, S, iters, warmup);
            }
            printf("----------------------------------------------------------------------------------------\n");
        }
    }

    fclose(fout);
    printf("\nCompute benchmark complete! Results recorded to %s\n", csv_file);
    return 0;
}
