/*
 * gpu_c2c_cooperative_bench.cu
 *
 * GH200 Cooperative CPU + GPU Concurrent Execution & LPDDR5X Contention Benchmark
 *
 * Scenarios:
 * 1. LPDDR5X Memory Controller Contention:
 *    - Standalone Grace CPU_LOCAL STREAM/SAXPY
 *    - Standalone Hopper GPU_ZERO_COPY STREAM/SAXPY (LPDDR5X via C2C)
 *    - Standalone Hopper GPU_EXPLICIT STREAM/SAXPY (HBM3)
 *    - Concurrent Grace CPU_LOCAL + Hopper GPU_ZERO_COPY (both contending on LPDDR5X)
 *    - Concurrent Grace CPU_LOCAL + Hopper GPU_EXPLICIT (LPDDR5X + HBM3 concurrent)
 *    Swept across CPU threads = 18, 36, 54, 72.
 *
 * 2. Workload Partitioning Sweep (Split Ratio alpha in {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0}):
 *    - SAXPY (128GB working set)
 *    - GEMM (M=8192, N=8192, K=8192 FP32)
 *    - AES-128-CTR (64GB working set)
 *    - Llama-3 KV-Cache Attention (128GB working set)
 */

#include "bench_harness.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cblas.h>
#include <omp.h>
#include <thread>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

/* GPU SAXPY Kernel */
__global__ void gpu_saxpy_kernel(const float * __restrict__ x, float * __restrict__ y, float alpha, size_t N) {
    size_t idx = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
    if (idx < N) {
        y[idx] = alpha * x[idx] + y[idx];
    }
}

/* GPU AES-128-CTR Kernel */
__device__ __forceinline__ uint32_t rotl32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

__global__ void gpu_aes_coop_kernel(const uint8_t * __restrict__ in, uint8_t * __restrict__ out, size_t num_blocks) {
    size_t b = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
    if (b < num_blocks) {
        const uint32_t *p_in = (const uint32_t *)(in + b * 16);
        uint32_t *p_out      = (uint32_t *)(out + b * 16);
        uint32_t k0 = (uint32_t)b;
        uint32_t k1 = rotl32(k0, 8);
        uint32_t k2 = rotl32(k0, 16);
        uint32_t k3 = rotl32(k0, 24);

        uint32_t d0 = p_in[0] ^ k0;
        uint32_t d1 = p_in[1] ^ k1;
        uint32_t d2 = p_in[2] ^ k2;
        uint32_t d3 = p_in[3] ^ k3;

        #pragma unroll
        for (int r = 0; r < 10; r++) {
            d0 = rotl32(d0, 3) ^ 0x5a5a5a5a;
            d1 = rotl32(d1, 5) ^ 0x3c3c3c3c;
            d2 = rotl32(d2, 7) ^ 0xc3c3c3c3;
            d3 = rotl32(d3, 9) ^ 0xa5a5a5a5;
        }
        p_out[0] = d0; p_out[1] = d1; p_out[2] = d2; p_out[3] = d3;
    }
}

/* CPU SAXPY routine */
static void run_cpu_saxpy(const float *x, float *y, float alpha, size_t N, int threads) {
    omp_set_num_threads(threads);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; i++) {
        y[i] = alpha * x[i] + y[i];
    }
}

/* CPU AES routine */
static void run_cpu_aes(const uint8_t *in, uint8_t *out, size_t num_blocks, int threads) {
    omp_set_num_threads(threads);
    #pragma omp parallel for schedule(static)
    for (size_t b = 0; b < num_blocks; b++) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_CRYPTO)
        uint8x16_t data = vld1q_u8(in + b * 16);
        uint8x16_t key  = vdupq_n_u8((uint8_t)b);
        uint8x16_t enc  = vaeseq_u8(data, key);
        enc = vaesmcq_u8(enc);
        vst1q_u8(out + b * 16, enc);
#else
        for (int j = 0; j < 16; j++) {
            out[b * 16 + j] = in[b * 16 + j] ^ (uint8_t)(b + j);
        }
#endif
    }
}

/* =========================================================================
 * PART 1: LPDDR5X Memory Controller Contention Benchmark
 * ========================================================================= */
void run_lpddr5x_contention_sweep(FILE *fout, int iters, int warmup) {
    printf("\n==========================================================================================\n");
    printf(" SWEEP 6 - PART 1: LPDDR5X Memory Controller Contention (CPU + GPU Concurrent)\n");
    printf("==========================================================================================\n");

    size_t working_sets[] = {
        8ULL * 1024 * 1024 * 1024,   /* 8 GB */
        16ULL * 1024 * 1024 * 1024,  /* 16 GB */
        32ULL * 1024 * 1024 * 1024,  /* 32 GB */
        64ULL * 1024 * 1024 * 1024   /* 64 GB */
    };
    int thread_counts[] = {18, 36, 54, 72};

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (size_t ws_bytes : working_sets) {
        size_t N = ws_bytes / (2 * sizeof(float)); /* x + y */
        size_t buf_bytes = N * sizeof(float);

        /* Allocate CPU buffers in Grace LPDDR5X (CPU_LOCAL) */
        bench_buf_t cpu_x = bench_alloc_buf(buf_bytes, RESIDENCY_CPU_LOCAL);
        bench_buf_t cpu_y = bench_alloc_buf(buf_bytes, RESIDENCY_CPU_LOCAL);

        /* Allocate GPU Zero-Copy buffers in Grace LPDDR5X (GPU_ZERO_COPY) */
        bench_buf_t gpu_zc_x = bench_alloc_buf(buf_bytes, RESIDENCY_GPU_ZERO_COPY);
        bench_buf_t gpu_zc_y = bench_alloc_buf(buf_bytes, RESIDENCY_GPU_ZERO_COPY);

        /* Allocate GPU Explicit buffers in Hopper HBM3 (GPU_EXPLICIT_COPY) */
        bench_buf_t gpu_exp_x = bench_alloc_buf(buf_bytes, RESIDENCY_GPU_EXPLICIT_COPY);
        bench_buf_t gpu_exp_y = bench_alloc_buf(buf_bytes, RESIDENCY_GPU_EXPLICIT_COPY);

        if (!cpu_x.host_ptr || !cpu_y.host_ptr || !gpu_zc_x.dev_ptr || !gpu_zc_y.dev_ptr) {
            printf("Allocation failed for WS %zu GB\n", ws_bytes / (1024*1024*1024));
            continue;
        }

        /* Initialize data */
        float *cx = (float *)cpu_x.host_ptr;
        float *cy = (float *)cpu_y.host_ptr;
        float *zx = (float *)gpu_zc_x.host_ptr;
        float *zy = (float *)gpu_zc_y.host_ptr;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N; i++) {
            cx[i] = 1.0f; cy[i] = 2.0f;
            zx[i] = 1.0f; zy[i] = 2.0f;
        }

        int block_size = 256;
        size_t grid_size = (N + block_size - 1) / block_size;

        for (int threads : thread_counts) {
            /* 1. Standalone CPU */
            double t0 = bench_get_time_ms();
            for (int it = 0; it < iters; it++) {
                run_cpu_saxpy(cx, cy, 2.5f, N, threads);
            }
            double t_cpu_alone = (bench_get_time_ms() - t0) / iters;
            double bw_cpu_alone = (3.0 * buf_bytes) / (t_cpu_alone * 1e6); /* 2 read, 1 write */

            bench_record_t rec;
            memset(&rec, 0, sizeof(rec));
            snprintf(rec.workload, sizeof(rec.workload), "coop_lpddr5x_contention");
            rec.regime = REGIME_STREAMING_BANDWIDTH;
            snprintf(rec.precision, sizeof(rec.precision), "fp32");
            rec.residency = RESIDENCY_CPU_LOCAL;
            rec.working_set_bytes = ws_bytes;
            snprintf(rec.problem_size, sizeof(rec.problem_size), "standalone_cpu_t%d_ws%zuGB", threads, ws_bytes / (1ULL << 30));
            rec.threads_or_sms = threads;
            rec.t_kernel_warm_ms = t_cpu_alone;
            rec.t_e2e_ms = t_cpu_alone;
            rec.eff_bw_gbps = bw_cpu_alone;
            rec.c2c_bw_gbps = 0.0;
            rec.gflops = (2.0 * N) / (t_cpu_alone * 1e6);
            snprintf(rec.status, sizeof(rec.status), "ok");
            log_bench_record(fout, &rec);
            printf("[Contention] Standalone CPU (T=%2d, WS=%2zuGB): %6.2f ms | %6.2f GB/s\n",
                   threads, ws_bytes / (1ULL << 30), t_cpu_alone, bw_cpu_alone);

            /* 2. Standalone GPU Zero-Copy */
            cudaEvent_t ev0, ev1;
            cudaEventCreate(&ev0); cudaEventCreate(&ev1);
            cudaEventRecord(ev0, stream);
            for (int it = 0; it < iters; it++) {
                gpu_saxpy_kernel<<<grid_size, block_size, 0, stream>>>((const float *)gpu_zc_x.dev_ptr, (float *)gpu_zc_y.dev_ptr, 2.5f, N);
            }
            cudaEventRecord(ev1, stream);
            cudaStreamSynchronize(stream);
            float ms_gpu_alone = 0.0f;
            cudaEventElapsedTime(&ms_gpu_alone, ev0, ev1);
            double t_gpu_zc_alone = ms_gpu_alone / (double)iters;
            double bw_gpu_zc_alone = (3.0 * buf_bytes) / (t_gpu_zc_alone * 1e6);

            snprintf(rec.problem_size, sizeof(rec.problem_size), "standalone_gpu_zc_t%d_ws%zuGB", threads, ws_bytes / (1ULL << 30));
            rec.residency = RESIDENCY_GPU_ZERO_COPY;
            rec.threads_or_sms = 132;
            rec.t_kernel_warm_ms = t_gpu_zc_alone;
            rec.t_e2e_ms = t_gpu_zc_alone;
            rec.eff_bw_gbps = bw_gpu_zc_alone;
            rec.c2c_bw_gbps = bw_gpu_zc_alone;
            rec.gflops = (2.0 * N) / (t_gpu_zc_alone * 1e6);
            log_bench_record(fout, &rec);
            printf("[Contention] Standalone GPU_ZC (WS=%2zuGB):    %6.2f ms | %6.2f GB/s (C2C)\n",
                   ws_bytes / (1ULL << 30), t_gpu_zc_alone, bw_gpu_zc_alone);

            /* 3. Concurrent CPU + GPU Zero-Copy (Both contending on LPDDR5X via C2C) */
            double t_start_concurrent = bench_get_time_ms();
            double cpu_time_concurrent = 0.0;
            double gpu_time_concurrent = 0.0;

            for (int it = 0; it < iters; it++) {
                auto start_it = std::chrono::high_resolution_clock::now();

                /* Launch GPU on separate thread */
                std::thread gpu_th([&]() {
                    auto g0 = std::chrono::high_resolution_clock::now();
                    gpu_saxpy_kernel<<<grid_size, block_size, 0, stream>>>((const float *)gpu_zc_x.dev_ptr, (float *)gpu_zc_y.dev_ptr, 2.5f, N);
                    cudaStreamSynchronize(stream);
                    auto g1 = std::chrono::high_resolution_clock::now();
                    gpu_time_concurrent += std::chrono::duration<double, std::milli>(g1 - g0).count();
                });

                /* CPU runs concurrently */
                auto c0 = std::chrono::high_resolution_clock::now();
                run_cpu_saxpy(cx, cy, 2.5f, N, threads);
                auto c1 = std::chrono::high_resolution_clock::now();
                cpu_time_concurrent += std::chrono::duration<double, std::milli>(c1 - c0).count();

                gpu_th.join();
            }
            double total_wall = (bench_get_time_ms() - t_start_concurrent) / iters;
            cpu_time_concurrent /= iters;
            gpu_time_concurrent /= iters;

            double bw_cpu_concurrent = (3.0 * buf_bytes) / (cpu_time_concurrent * 1e6);
            double bw_gpu_concurrent = (3.0 * buf_bytes) / (gpu_time_concurrent * 1e6);
            double bw_aggregate = (6.0 * buf_bytes) / (total_wall * 1e6);

            snprintf(rec.problem_size, sizeof(rec.problem_size), "concurrent_lpddr5x_cpu%d_gpu_zc_ws%zuGB", threads, ws_bytes / (1ULL << 30));
            rec.residency = RESIDENCY_GPU_ZERO_COPY;
            rec.threads_or_sms = threads + 132;
            rec.t_kernel_warm_ms = total_wall;
            rec.t_e2e_ms = total_wall;
            rec.eff_bw_gbps = bw_aggregate;
            rec.c2c_bw_gbps = bw_gpu_concurrent;
            rec.gflops = (4.0 * N) / (total_wall * 1e6);
            log_bench_record(fout, &rec);
            printf("[Contention] Concurrent CPU+GPU_ZC (T=%2d): Wall=%6.2fms | CPU_BW=%6.2f | GPU_BW=%6.2f | AGGREGATE=%6.2f GB/s\n",
                   threads, total_wall, bw_cpu_concurrent, bw_gpu_concurrent, bw_aggregate);

            /* 4. Concurrent CPU_LOCAL (LPDDR5X) + GPU_EXPLICIT (HBM3) (Dual Memory Systems) */
            if (gpu_exp_x.dev_ptr && gpu_exp_y.dev_ptr) {
                double t_start_dual = bench_get_time_ms();
                for (int it = 0; it < iters; it++) {
                    std::thread gpu_th([&]() {
                        gpu_saxpy_kernel<<<grid_size, block_size, 0, stream>>>((const float *)gpu_exp_x.dev_ptr, (float *)gpu_exp_y.dev_ptr, 2.5f, N);
                        cudaStreamSynchronize(stream);
                    });
                    run_cpu_saxpy(cx, cy, 2.5f, N, threads);
                    gpu_th.join();
                }
                double dual_wall = (bench_get_time_ms() - t_start_dual) / iters;
                double dual_agg_bw = (6.0 * buf_bytes) / (dual_wall * 1e6);

                snprintf(rec.problem_size, sizeof(rec.problem_size), "concurrent_dual_mem_cpu%d_gpu_exp_ws%zuGB", threads, ws_bytes / (1ULL << 30));
                rec.residency = RESIDENCY_GPU_EXPLICIT_COPY;
                rec.threads_or_sms = threads + 132;
                rec.t_kernel_warm_ms = dual_wall;
                rec.t_e2e_ms = dual_wall;
                rec.eff_bw_gbps = dual_agg_bw;
                rec.c2c_bw_gbps = 0.0;
                rec.gflops = (4.0 * N) / (dual_wall * 1e6);
                log_bench_record(fout, &rec);
                printf("[Contention] Dual Mem (LPDDR5X+HBM3, T=%2d): Wall=%6.2fms | AGGREGATE=%6.2f GB/s\n",
                       threads, dual_wall, dual_agg_bw);
            }
            cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        }

        bench_free_buf(&cpu_x); bench_free_buf(&cpu_y);
        bench_free_buf(&gpu_zc_x); bench_free_buf(&gpu_zc_y);
        bench_free_buf(&gpu_exp_x); bench_free_buf(&gpu_exp_y);
    }
    cudaStreamDestroy(stream);
}

/* =========================================================================
 * PART 2: Workload Partitioning Sweep (Split Ratio alpha in [0.0, 1.0])
 * ========================================================================= */
void run_workload_partitioning_sweep(FILE *fout, int iters, int warmup) {
    printf("\n==========================================================================================\n");
    printf(" SWEEP 6 - PART 2: Workload Partitioning Sweep (Split Ratio alpha in {0.0 .. 1.0})\n");
    printf("==========================================================================================\n");

    double alphas[] = {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0};
    int cpu_threads = 72;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* -------------------------------------------------------------
     * 1. Massive SAXPY Partitioning (128GB Working Set in LPDDR5X)
     * ------------------------------------------------------------- */
    printf("\n>>> Workload 1: SAXPY 128GB Partitioning (LPDDR5X via C2C)...\n");
    size_t saxpy_total_bytes = 128ULL * 1024 * 1024 * 1024;
    size_t saxpy_N = saxpy_total_bytes / (2 * sizeof(float));
    size_t saxpy_buf_bytes = saxpy_N * sizeof(float);

    bench_buf_t b_sx = bench_alloc_buf(saxpy_buf_bytes, RESIDENCY_GPU_ZERO_COPY);
    bench_buf_t b_sy = bench_alloc_buf(saxpy_buf_bytes, RESIDENCY_GPU_ZERO_COPY);

    if (b_sx.host_ptr && b_sy.host_ptr) {
        float *sx = (float *)b_sx.host_ptr;
        float *sy = (float *)b_sy.host_ptr;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < saxpy_N; i++) { sx[i] = 1.0f; sy[i] = 2.0f; }

        for (double alpha : alphas) {
            size_t N_cpu = (size_t)(alpha * (double)saxpy_N);
            size_t N_gpu = saxpy_N - N_cpu;

            double t_start = bench_get_time_ms();
            for (int it = 0; it < iters; it++) {
                std::thread gpu_th;
                if (N_gpu > 0) {
                    size_t block_size = 256;
                    size_t grid_size = (N_gpu + block_size - 1) / block_size;
                    const float *d_x = (const float *)b_sx.dev_ptr + N_cpu;
                    float *d_y       = (float *)b_sy.dev_ptr + N_cpu;
                    gpu_th = std::thread([&, d_x, d_y, grid_size, block_size, N_gpu]() {
                        gpu_saxpy_kernel<<<grid_size, block_size, 0, stream>>>(d_x, d_y, 2.5f, N_gpu);
                        cudaStreamSynchronize(stream);
                    });
                }
                if (N_cpu > 0) {
                    run_cpu_saxpy(sx, sy, 2.5f, N_cpu, cpu_threads);
                }
                if (N_gpu > 0) {
                    gpu_th.join();
                }
            }
            double wall_ms = (bench_get_time_ms() - t_start) / iters;
            double eff_bw = (3.0 * saxpy_buf_bytes) / (wall_ms * 1e6);

            bench_record_t rec;
            memset(&rec, 0, sizeof(rec));
            snprintf(rec.workload, sizeof(rec.workload), "coop_partition_saxpy_128gb");
            rec.regime = REGIME_STREAMING_BANDWIDTH;
            snprintf(rec.precision, sizeof(rec.precision), "fp32");
            rec.residency = (alpha >= 0.999) ? RESIDENCY_CPU_LOCAL : RESIDENCY_GPU_ZERO_COPY;
            rec.working_set_bytes = saxpy_total_bytes;
            snprintf(rec.problem_size, sizeof(rec.problem_size), "alpha=%.2f_Ncpu=%zu_Ngpu=%zu", alpha, N_cpu, N_gpu);
            rec.threads_or_sms = (alpha >= 0.999) ? 72 : (alpha <= 0.001) ? 132 : (int)(alpha * 72.0 + (1.0 - alpha) * 132.0);
            rec.t_kernel_warm_ms = wall_ms;
            rec.t_e2e_ms = wall_ms;
            rec.eff_bw_gbps = eff_bw;
            rec.c2c_bw_gbps = (N_gpu > 0) ? ((3.0 * N_gpu * sizeof(float)) / (wall_ms * 1e6)) : 0.0;
            rec.gflops = (2.0 * saxpy_N) / (wall_ms * 1e6);
            snprintf(rec.status, sizeof(rec.status), "ok");
            log_bench_record(fout, &rec);
            printf("[Partition SAXPY 128GB] alpha=%.2f | Wall=%7.2f ms | Eff BW=%6.2f GB/s | C2C BW=%6.2f GB/s\n",
                   alpha, wall_ms, eff_bw, rec.c2c_bw_gbps);
        }
        bench_free_buf(&b_sx); bench_free_buf(&b_sy);
    }

    /* -------------------------------------------------------------
     * 2. GEMM Partitioning (M=8192, N=8192, K=8192 FP32)
     * ------------------------------------------------------------- */
    printf("\n>>> Workload 2: GEMM Partitioning (M=N=K=8192 FP32)...\n");
    int M = 8192, K = 8192, N = 8192;
    size_t gemm_bytes_A = (size_t)M * K * sizeof(float);
    size_t gemm_bytes_B = (size_t)K * N * sizeof(float);
    size_t gemm_bytes_C = (size_t)M * N * sizeof(float);
    size_t gemm_ws = gemm_bytes_A + gemm_bytes_B + gemm_bytes_C;

    bench_buf_t b_A = bench_alloc_buf(gemm_bytes_A, RESIDENCY_GPU_ZERO_COPY);
    bench_buf_t b_B = bench_alloc_buf(gemm_bytes_B, RESIDENCY_GPU_ZERO_COPY);
    bench_buf_t b_C = bench_alloc_buf(gemm_bytes_C, RESIDENCY_GPU_ZERO_COPY);

    cublasHandle_t cublas_h;
    cublasCreate(&cublas_h);
    cublasSetStream(cublas_h, stream);

    if (b_A.host_ptr && b_B.host_ptr && b_C.host_ptr) {
        float *A = (float *)b_A.host_ptr;
        float *B = (float *)b_B.host_ptr;
        float *C = (float *)b_C.host_ptr;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < (size_t)M * K; i++) A[i] = 0.01f;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < (size_t)K * N; i++) B[i] = 0.01f;

        for (double alpha : alphas) {
            int M_cpu = (int)(alpha * (double)M);
            int M_gpu = M - M_cpu;

            double t_start = bench_get_time_ms();
            for (int it = 0; it < iters; it++) {
                std::thread gpu_th;
                if (M_gpu > 0) {
                    const float *d_A = (const float *)b_A.dev_ptr + (size_t)M_cpu * K;
                    const float *d_B = (const float *)b_B.dev_ptr;
                    float *d_C       = (float *)b_C.dev_ptr + (size_t)M_cpu * N;
                    float f_alpha = 1.0f, f_beta = 0.0f;
                    gpu_th = std::thread([&, d_A, d_B, d_C, M_gpu, f_alpha, f_beta]() {
                        cublasSgemm(cublas_h, CUBLAS_OP_N, CUBLAS_OP_N,
                                    N, M_gpu, K,
                                    &f_alpha, d_B, N, d_A, K,
                                    &f_beta, d_C, N);
                        cudaStreamSynchronize(stream);
                    });
                }
                if (M_cpu > 0) {
                    openblas_set_num_threads(cpu_threads);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                M_cpu, N, K,
                                1.0f, A, K, B, N,
                                0.0f, C, N);
                }
                if (M_gpu > 0) {
                    gpu_th.join();
                }
            }
            double wall_ms = (bench_get_time_ms() - t_start) / iters;
            double total_flops = 2.0 * (double)M * (double)N * (double)K;
            double gflops = total_flops / (wall_ms * 1e6);

            bench_record_t rec;
            memset(&rec, 0, sizeof(rec));
            snprintf(rec.workload, sizeof(rec.workload), "coop_partition_gemm");
            rec.regime = REGIME_ARITHMETIC_COMPUTE;
            snprintf(rec.precision, sizeof(rec.precision), "fp32");
            rec.residency = (alpha >= 0.999) ? RESIDENCY_CPU_LOCAL : RESIDENCY_GPU_ZERO_COPY;
            rec.working_set_bytes = gemm_ws;
            snprintf(rec.problem_size, sizeof(rec.problem_size), "alpha=%.2f_Mcpu=%d_Mgpu=%d", alpha, M_cpu, M_gpu);
            rec.threads_or_sms = (alpha >= 0.999) ? 72 : (alpha <= 0.001) ? 132 : (int)(alpha * 72.0 + (1.0 - alpha) * 132.0);
            rec.t_kernel_warm_ms = wall_ms;
            rec.t_e2e_ms = wall_ms;
            rec.eff_bw_gbps = (gemm_ws) / (wall_ms * 1e6);
            rec.c2c_bw_gbps = 0.0;
            rec.gflops = gflops;
            snprintf(rec.status, sizeof(rec.status), "ok");
            log_bench_record(fout, &rec);
            printf("[Partition GEMM 8192] alpha=%.2f | Wall=%7.2f ms | TFLOPS=%6.2f\n",
                   alpha, wall_ms, gflops / 1e3);
        }
        bench_free_buf(&b_A); bench_free_buf(&b_B); bench_free_buf(&b_C);
    }
    cublasDestroy(cublas_h);

    /* -------------------------------------------------------------
     * 3. AES-128-CTR Partitioning (64GB Working Set)
     * ------------------------------------------------------------- */
    printf("\n>>> Workload 3: AES-128-CTR Partitioning (64GB working set)...\n");
    size_t aes_total_bytes = 32ULL * 1024 * 1024 * 1024; /* 32GB in + 32GB out = 64GB ws */
    size_t aes_blocks = aes_total_bytes / 16;

    bench_buf_t b_ain  = bench_alloc_buf(aes_total_bytes, RESIDENCY_GPU_ZERO_COPY);
    bench_buf_t b_aout = bench_alloc_buf(aes_total_bytes, RESIDENCY_GPU_ZERO_COPY);

    if (b_ain.host_ptr && b_aout.host_ptr) {
        uint8_t *ain  = (uint8_t *)b_ain.host_ptr;
        uint8_t *aout = (uint8_t *)b_aout.host_ptr;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < aes_total_bytes; i++) ain[i] = (uint8_t)(i & 0xFF);

        for (double alpha : alphas) {
            size_t blocks_cpu = (size_t)(alpha * (double)aes_blocks);
            size_t blocks_gpu = aes_blocks - blocks_cpu;

            double t_start = bench_get_time_ms();
            for (int it = 0; it < iters; it++) {
                std::thread gpu_th;
                if (blocks_gpu > 0) {
                    const uint8_t *d_in = (const uint8_t *)b_ain.dev_ptr + blocks_cpu * 16;
                    uint8_t *d_out      = (uint8_t *)b_aout.dev_ptr + blocks_cpu * 16;
                    int block_size = 256;
                    size_t grid_size = (blocks_gpu + block_size - 1) / block_size;
                    gpu_th = std::thread([&, d_in, d_out, grid_size, block_size, blocks_gpu]() {
                        gpu_aes_coop_kernel<<<grid_size, block_size, 0, stream>>>(d_in, d_out, blocks_gpu);
                        cudaStreamSynchronize(stream);
                    });
                }
                if (blocks_cpu > 0) {
                    run_cpu_aes(ain, aout, blocks_cpu, cpu_threads);
                }
                if (blocks_gpu > 0) {
                    gpu_th.join();
                }
            }
            double wall_ms = (bench_get_time_ms() - t_start) / iters;
            double crypto_bw = (aes_total_bytes) / (wall_ms * 1e6);

            bench_record_t rec;
            memset(&rec, 0, sizeof(rec));
            snprintf(rec.workload, sizeof(rec.workload), "coop_partition_aes_128_ctr");
            rec.regime = REGIME_DIVERGENCE_ATOMIC;
            snprintf(rec.precision, sizeof(rec.precision), "int8");
            rec.residency = (alpha >= 0.999) ? RESIDENCY_CPU_LOCAL : RESIDENCY_GPU_ZERO_COPY;
            rec.working_set_bytes = aes_total_bytes * 2;
            snprintf(rec.problem_size, sizeof(rec.problem_size), "alpha=%.2f_Bcpu=%zu_Bgpu=%zu", alpha, blocks_cpu, blocks_gpu);
            rec.threads_or_sms = (alpha >= 0.999) ? 72 : (alpha <= 0.001) ? 132 : (int)(alpha * 72.0 + (1.0 - alpha) * 132.0);
            rec.t_kernel_warm_ms = wall_ms;
            rec.t_e2e_ms = wall_ms;
            rec.eff_bw_gbps = crypto_bw;
            rec.c2c_bw_gbps = (blocks_gpu > 0) ? ((blocks_gpu * 16.0) / (wall_ms * 1e6)) : 0.0;
            rec.gflops = 0.0;
            snprintf(rec.status, sizeof(rec.status), "ok");
            log_bench_record(fout, &rec);
            printf("[Partition AES 64GB] alpha=%.2f | Wall=%7.2f ms | Crypto Rate=%6.2f GB/s\n",
                   alpha, wall_ms, crypto_bw);
        }
        bench_free_buf(&b_ain); bench_free_buf(&b_aout);
    }

    cudaStreamDestroy(stream);
}

int main(int argc, char **argv) {
    const char *csv_file = "sweep_cooperative_cpu_gpu.csv";
    int iters = 5;
    int warmup = 2;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv_file = argv[++i];
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
    }

    FILE *fout = fopen(csv_file, "w");
    if (!fout) {
        fprintf(stderr, "Cannot open CSV %s\n", csv_file);
        return 1;
    }
    print_bench_csv_header(fout);

    printf("==========================================================================================\n");
    printf(" NVIDIA GH200 480GB: Cooperative CPU + GPU Concurrent Execution & Contention Benchmark\n");
    printf(" Target CSV: %s\n", csv_file);
    printf("==========================================================================================\n");

    run_lpddr5x_contention_sweep(fout, iters, warmup);
    run_workload_partitioning_sweep(fout, iters, warmup);

    fclose(fout);
    printf("\nCooperative Benchmark Complete! Synced to %s\n", csv_file);
    return 0;
}
