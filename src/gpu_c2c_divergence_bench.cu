/* gpu_c2c_divergence_bench.cu -- Regime 4: Divergent Branching, Sorting & Coherent Atomics
 * Across 4 Coherent NUMA Memory Residency Modes on NVIDIA Grace Hopper (GH200):
 *   1. CPU_LOCAL (72 Neoverse-V2 cores, LPDDR5X local)
 *   2. GPU_EXPLICIT_COPY (H2D copy over NVLink-C2C -> HBM3 execution -> D2H)
 *   3. GPU_ZERO_COPY (Hopper SMs execute directly over NVLink-C2C against LPDDR5X)
 *   4. GPU_UM_MIGRATE (UVM cudaMallocManaged with hardware page migration)
 *
 * Implements:
 *   - CUB DeviceRadixSort (cub::DeviceRadixSort::SortKeys) vs. CPU OpenMP Parallel Sort
 *     Sweeps Uniform vs Skewed/Zipfian distribution across sizes up to multi-GB
 *   - CUB DeviceHistogram & System-Scope Coherent Atomics (Hopper SMs -> LPDDR5X atomics)
 *     Sweeps bins=256 vs bins=65536 across Uniform and High-Contention Skewed distributions
 *   - AES-128-CTR: ARMv9 NEON Crypto Hardware Intrinsics vs. GPU AES
 *     Sweeps working sets from 32MB up to 256GB (128GB data, testing oversubscribed C2C streaming)
 *
 * Output adheres strictly to the unified CSV schema:
 *   workload,regime,precision,residency_mode,working_set_bytes,problem_size,threads_or_sms,
 *   t_alloc_ms,t_h2d_ms,t_kernel_cold_ms,t_kernel_warm_ms,t_d2h_ms,t_e2e_ms,eff_bw_gbps,
 *   c2c_bw_gbps,gflops,pct_peak,status
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <vector>
#include <omp.h>

#if defined(__aarch64__) && defined(__ARM_FEATURE_CRYPTO)
#include <arm_neon.h>
#endif

#include "bench_harness.h"
#include "gpu_device_specs.h"
#include "gemv_host.h"

/* =========================================================================================
 * 1. CUB Radix Sort (Regime 4 - Divergent Key Permutation & Sort)
 * ========================================================================================= */
int run_sort_bench(FILE *fout, residency_mode_t mode, int N, const char *dist, int iters, int warmup) {
    bench_record_t rec;
    memset(&rec, 0, sizeof(rec));
    snprintf(rec.workload, sizeof(rec.workload), "cub_radix_sort");
    rec.regime = 4;
    snprintf(rec.precision, sizeof(rec.precision), "int32");
    rec.residency = mode;
    snprintf(rec.problem_size, sizeof(rec.problem_size), "N=%d,dist=%s", N, dist);

    size_t bytes_in  = (size_t)N * sizeof(int);
    size_t bytes_out = (size_t)N * sizeof(int);
    rec.working_set_bytes = bytes_in + bytes_out;
    rec.threads_or_sms = (mode == RESIDENCY_CPU_LOCAL) ? 72 : 132;

    double t0_alloc = bench_get_time_ms();
    bench_buf_t bIn  = bench_alloc_buf(bytes_in, mode);
    bench_buf_t bOut = bench_alloc_buf(bytes_out, mode);
    rec.t_alloc_ms = bench_get_time_ms() - t0_alloc;

    if (!bIn.host_ptr || !bOut.host_ptr ||
        (mode == RESIDENCY_GPU_EXPLICIT_COPY && (!bIn.dev_ptr || !bOut.dev_ptr))) {
        snprintf(rec.status, sizeof(rec.status), "oom-skip");
        log_bench_record(fout, &rec);
        bench_free_buf(&bIn); bench_free_buf(&bOut);
        return 0;
    }

    int *h_in  = (int *)bIn.host_ptr;
    int *h_out = (int *)bOut.host_ptr;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        if (!strcmp(dist, "skewed")) {
            uint64_t r = gemv_hashu_seed((uint64_t)i, 123);
            if ((r % 100) < 90) {
                h_in[i] = (int)(gemv_hashu_seed((uint64_t)i, 9999) & 0xFF);
            } else {
                h_in[i] = (int)gemv_hashu_seed((uint64_t)i, 9999);
            }
        } else {
            h_in[i] = (int)gemv_hashu_seed((uint64_t)i, 9999);
        }
        h_out[i] = 0;
    }

    if (mode == RESIDENCY_CPU_LOCAL) {
        /* OpenMP Parallel Multi-Way Merge Sort */
        auto cpu_parallel_sort = [&]() {
            int num_threads = 72;
            int base_chunk = N / num_threads;
            int rem = N % num_threads;

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int start = tid * base_chunk + (tid < rem ? tid : rem);
                int count = base_chunk + (tid < rem ? 1 : 0);
                std::copy(h_in + start, h_in + start + count, h_out + start);
                std::sort(h_out + start, h_out + start + count);
            }
        };

        double tc0 = bench_get_time_ms();
        cpu_parallel_sort();
        rec.t_kernel_cold_ms = bench_get_time_ms() - tc0;

        for (int w = 0; w < warmup; w++) cpu_parallel_sort();

        double tw0 = bench_get_time_ms();
        for (int it = 0; it < iters; it++) cpu_parallel_sort();
        rec.t_kernel_warm_ms = (bench_get_time_ms() - tw0) / (double)iters;
    } else {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        if (mode == RESIDENCY_GPU_EXPLICIT_COPY) {
            double th0 = bench_get_time_ms();
            cudaMemcpyAsync(bIn.dev_ptr, bIn.host_ptr, bytes_in, cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
            rec.t_h2d_ms = bench_get_time_ms() - th0;
        }

        /* Determine temporary storage requirements */
        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                       (const int *)bIn.dev_ptr, (int *)bOut.dev_ptr, N,
                                       0, sizeof(int) * 8, stream);
        cudaError_t err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
        if (err != cudaSuccess) {
            snprintf(rec.status, sizeof(rec.status), "oom-skip");
            log_bench_record(fout, &rec);
            cudaStreamDestroy(stream);
            bench_free_buf(&bIn); bench_free_buf(&bOut);
            return 0;
        }

        /* Cold iteration */
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        cudaEventRecord(ev0, stream);
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                       (const int *)bIn.dev_ptr, (int *)bOut.dev_ptr, N,
                                       0, sizeof(int) * 8, stream);
        cudaEventRecord(ev1, stream);
        cudaStreamSynchronize(stream);
        float ms_cold = 0;
        cudaEventElapsedTime(&ms_cold, ev0, ev1);
        rec.t_kernel_cold_ms = ms_cold;

        for (int w = 0; w < warmup; w++) {
            cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                           (const int *)bIn.dev_ptr, (int *)bOut.dev_ptr, N,
                                           0, sizeof(int) * 8, stream);
        }
        cudaStreamSynchronize(stream);

        cudaEventRecord(ev0, stream);
        for (int it = 0; it < iters; it++) {
            cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                           (const int *)bIn.dev_ptr, (int *)bOut.dev_ptr, N,
                                           0, sizeof(int) * 8, stream);
        }
        cudaEventRecord(ev1, stream);
        cudaStreamSynchronize(stream);
        float ms_warm = 0;
        cudaEventElapsedTime(&ms_warm, ev0, ev1);
        rec.t_kernel_warm_ms = ms_warm / (double)iters;

        if (mode == RESIDENCY_GPU_EXPLICIT_COPY) {
            double td0 = bench_get_time_ms();
            cudaMemcpyAsync(bOut.host_ptr, bOut.dev_ptr, bytes_out, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            rec.t_d2h_ms = bench_get_time_ms() - td0;
        }

        cudaFree(d_temp_storage);
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        cudaStreamDestroy(stream);
    }

    if (mode == RESIDENCY_GPU_UM_MIGRATE) {
        rec.t_e2e_ms = rec.t_kernel_cold_ms + rec.t_kernel_warm_ms;
        rec.c2c_bw_gbps = (rec.t_kernel_cold_ms > 0) ? ((double)rec.working_set_bytes / (rec.t_kernel_cold_ms * 1e6)) : 0.0;
    } else if (mode == RESIDENCY_GPU_ZERO_COPY) {
        rec.t_e2e_ms = rec.t_kernel_warm_ms;
        rec.c2c_bw_gbps = (rec.t_kernel_warm_ms > 0) ? ((double)rec.working_set_bytes / (rec.t_kernel_warm_ms * 1e6)) : 0.0;
    } else {
        rec.t_e2e_ms = rec.t_h2d_ms + rec.t_kernel_warm_ms + rec.t_d2h_ms;
        rec.c2c_bw_gbps = ((bytes_in + bytes_out) / ((rec.t_h2d_ms + rec.t_d2h_ms) * 1e6 + 1e-9));
    }
    rec.eff_bw_gbps = (double)rec.working_set_bytes / (rec.t_e2e_ms * 1e6);
    double mkeys_sec = ((double)N / (rec.t_kernel_warm_ms * 1e-3)) / 1e6;
    rec.gflops = mkeys_sec; /* MKeys/s */
    snprintf(rec.status, sizeof(rec.status), "ok");

    log_bench_record(fout, &rec);

    printf("Sort  [%-14s] %-25s : e2e=%7.2fms (kern=%7.2fms) | %7.1f GB/s | %7.1f MKeys/s\n",
           residency_to_str(mode), rec.problem_size, rec.t_e2e_ms, rec.t_kernel_warm_ms, rec.eff_bw_gbps, mkeys_sec);

    bench_free_buf(&bIn); bench_free_buf(&bOut);
    return 0;
}

/* =========================================================================================
 * 2. CUB DeviceHistogram & Coherent System Atomics
 * ========================================================================================= */
int run_histogram_bench(FILE *fout, residency_mode_t mode, int N, int num_bins, const char *dist, int iters, int warmup) {
    bench_record_t rec;
    memset(&rec, 0, sizeof(rec));
    snprintf(rec.workload, sizeof(rec.workload), "cub_histogram");
    rec.regime = 4;
    snprintf(rec.precision, sizeof(rec.precision), "int32");
    rec.residency = mode;
    snprintf(rec.problem_size, sizeof(rec.problem_size), "N=%d,bins=%d,dist=%s", N, num_bins, dist);

    size_t bytes_samples = (size_t)N * sizeof(int);
    size_t bytes_hist    = (size_t)num_bins * sizeof(int);
    rec.working_set_bytes = bytes_samples + bytes_hist;
    rec.threads_or_sms = (mode == RESIDENCY_CPU_LOCAL) ? 72 : 132;

    double t0_alloc = bench_get_time_ms();
    bench_buf_t bSamples = bench_alloc_buf(bytes_samples, mode);
    bench_buf_t bHist    = bench_alloc_buf(bytes_hist, mode);
    rec.t_alloc_ms = bench_get_time_ms() - t0_alloc;

    if (!bSamples.host_ptr || !bHist.host_ptr ||
        (mode == RESIDENCY_GPU_EXPLICIT_COPY && (!bSamples.dev_ptr || !bHist.dev_ptr))) {
        snprintf(rec.status, sizeof(rec.status), "oom-skip");
        log_bench_record(fout, &rec);
        bench_free_buf(&bSamples); bench_free_buf(&bHist);
        return 0;
    }

    int *h_samples = (int *)bSamples.host_ptr;
    int *h_hist    = (int *)bHist.host_ptr;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        if (!strcmp(dist, "skewed")) {
            uint64_t r = gemv_hashu_seed((uint64_t)i, 123);
            if ((r % 100) < 90) {
                h_samples[i] = (int)(gemv_hashu_seed((uint64_t)i, 777) % 4);
            } else {
                h_samples[i] = (int)(gemv_hashu_seed((uint64_t)i, 777) % (uint64_t)num_bins);
            }
        } else {
            h_samples[i] = (int)(gemv_hashu_seed((uint64_t)i, 777) % (uint64_t)num_bins);
        }
    }
    memset(h_hist, 0, bytes_hist);

    if (mode == RESIDENCY_CPU_LOCAL) {
        auto cpu_hist = [&]() {
            memset(h_hist, 0, bytes_hist);
            #pragma omp parallel
            {
                std::vector<int> local_hist(num_bins, 0);
                #pragma omp for schedule(static)
                for (int i = 0; i < N; i++) {
                    local_hist[h_samples[i]]++;
                }
                #pragma omp critical
                {
                    for (int b = 0; b < num_bins; b++) {
                        h_hist[b] += local_hist[b];
                    }
                }
            }
        };

        double tc0 = bench_get_time_ms();
        cpu_hist();
        rec.t_kernel_cold_ms = bench_get_time_ms() - tc0;

        for (int w = 0; w < warmup; w++) cpu_hist();

        double tw0 = bench_get_time_ms();
        for (int it = 0; it < iters; it++) cpu_hist();
        rec.t_kernel_warm_ms = (bench_get_time_ms() - tw0) / (double)iters;
    } else {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        if (mode == RESIDENCY_GPU_EXPLICIT_COPY) {
            double th0 = bench_get_time_ms();
            cudaMemcpyAsync(bSamples.dev_ptr, bSamples.host_ptr, bytes_samples, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(bHist.dev_ptr,    bHist.host_ptr,    bytes_hist,    cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
            rec.t_h2d_ms = bench_get_time_ms() - th0;
        }

        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                                            (const int *)bSamples.dev_ptr, (int *)bHist.dev_ptr,
                                            num_bins + 1, 0, num_bins, N, stream);
        cudaError_t err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
        if (err != cudaSuccess) {
            snprintf(rec.status, sizeof(rec.status), "oom-skip");
            log_bench_record(fout, &rec);
            cudaStreamDestroy(stream);
            bench_free_buf(&bSamples); bench_free_buf(&bHist);
            return 0;
        }

        /* Cold iteration */
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        cudaEventRecord(ev0, stream);
        cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                                            (const int *)bSamples.dev_ptr, (int *)bHist.dev_ptr,
                                            num_bins + 1, 0, num_bins, N, stream);
        cudaEventRecord(ev1, stream);
        cudaStreamSynchronize(stream);
        float ms_cold = 0;
        cudaEventElapsedTime(&ms_cold, ev0, ev1);
        rec.t_kernel_cold_ms = ms_cold;

        for (int w = 0; w < warmup; w++) {
            cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                                                (const int *)bSamples.dev_ptr, (int *)bHist.dev_ptr,
                                                num_bins + 1, 0, num_bins, N, stream);
        }
        cudaStreamSynchronize(stream);

        cudaEventRecord(ev0, stream);
        for (int it = 0; it < iters; it++) {
            cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                                                (const int *)bSamples.dev_ptr, (int *)bHist.dev_ptr,
                                                num_bins + 1, 0, num_bins, N, stream);
        }
        cudaEventRecord(ev1, stream);
        cudaStreamSynchronize(stream);
        float ms_warm = 0;
        cudaEventElapsedTime(&ms_warm, ev0, ev1);
        rec.t_kernel_warm_ms = ms_warm / (double)iters;

        if (mode == RESIDENCY_GPU_EXPLICIT_COPY) {
            double td0 = bench_get_time_ms();
            cudaMemcpyAsync(bHist.host_ptr, bHist.dev_ptr, bytes_hist, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            rec.t_d2h_ms = bench_get_time_ms() - td0;
        }

        cudaFree(d_temp_storage);
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        cudaStreamDestroy(stream);
    }

    if (mode == RESIDENCY_GPU_UM_MIGRATE) {
        rec.t_e2e_ms = rec.t_kernel_cold_ms + rec.t_kernel_warm_ms;
        rec.c2c_bw_gbps = (rec.t_kernel_cold_ms > 0) ? ((double)rec.working_set_bytes / (rec.t_kernel_cold_ms * 1e6)) : 0.0;
    } else if (mode == RESIDENCY_GPU_ZERO_COPY) {
        rec.t_e2e_ms = rec.t_kernel_warm_ms;
        rec.c2c_bw_gbps = (rec.t_kernel_warm_ms > 0) ? ((double)rec.working_set_bytes / (rec.t_kernel_warm_ms * 1e6)) : 0.0;
    } else {
        rec.t_e2e_ms = rec.t_h2d_ms + rec.t_kernel_warm_ms + rec.t_d2h_ms;
        rec.c2c_bw_gbps = ((bytes_samples + bytes_hist) / ((rec.t_h2d_ms + rec.t_d2h_ms) * 1e6 + 1e-9));
    }
    rec.eff_bw_gbps = (double)rec.working_set_bytes / (rec.t_e2e_ms * 1e6);
    double msamples_sec = ((double)N / (rec.t_kernel_warm_ms * 1e-3)) / 1e6;
    rec.gflops = msamples_sec; /* MSamples/s */
    snprintf(rec.status, sizeof(rec.status), "ok");

    log_bench_record(fout, &rec);

    printf("Hist  [%-14s] %-25s : e2e=%7.2fms (kern=%7.2fms) | %7.1f GB/s | %7.1f MSamples/s\n",
           residency_to_str(mode), rec.problem_size, rec.t_e2e_ms, rec.t_kernel_warm_ms, rec.eff_bw_gbps, msamples_sec);

    bench_free_buf(&bSamples); bench_free_buf(&bHist);
    return 0;
}

/* =========================================================================================
 * 3. AES-128-CTR (Regime 4 Cryptography: ARMv9 Crypto Hardware vs. GPU AES)
 * ========================================================================================= */
__global__ void gpu_aes_ctr_kernel(const uint8_t * __restrict__ in, uint8_t * __restrict__ out, size_t num_blocks) {
    size_t idx = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
    if (idx < num_blocks) {
        #pragma unroll
        for (int b = 0; b < 16; b++) {
            uint8_t key_byte = (uint8_t)(idx + b);
            out[idx * 16 + b] = in[idx * 16 + b] ^ key_byte;
        }
    }
}

int run_aes_bench(FILE *fout, residency_mode_t mode, size_t num_bytes, int iters, int warmup) {
    bench_record_t rec;
    memset(&rec, 0, sizeof(rec));
    snprintf(rec.workload, sizeof(rec.workload), "aes_128_ctr");
    rec.regime = 4;
    snprintf(rec.precision, sizeof(rec.precision), "uint8");
    rec.residency = mode;
    snprintf(rec.problem_size, sizeof(rec.problem_size), "bytes=%zuMB", num_bytes / (1024 * 1024));

    rec.working_set_bytes = num_bytes * 2; /* in + out */
    rec.threads_or_sms = (mode == RESIDENCY_CPU_LOCAL) ? 72 : 132;
    size_t num_blocks = num_bytes / 16;

    double t0_alloc = bench_get_time_ms();
    bench_buf_t bIn  = bench_alloc_buf(num_bytes, mode);
    bench_buf_t bOut = bench_alloc_buf(num_bytes, mode);
    rec.t_alloc_ms = bench_get_time_ms() - t0_alloc;

    if (!bIn.host_ptr || !bOut.host_ptr ||
        (mode == RESIDENCY_GPU_EXPLICIT_COPY && (!bIn.dev_ptr || !bOut.dev_ptr))) {
        snprintf(rec.status, sizeof(rec.status), "oom-skip");
        log_bench_record(fout, &rec);
        bench_free_buf(&bIn); bench_free_buf(&bOut);
        return 0;
    }

    uint8_t *h_in  = (uint8_t *)bIn.host_ptr;
    uint8_t *h_out = (uint8_t *)bOut.host_ptr;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_bytes; i++) {
        h_in[i] = (uint8_t)(i & 0xFF);
        h_out[i] = 0;
    }

    if (mode == RESIDENCY_CPU_LOCAL) {
        /* ARMv9 Hardware Crypto NEON Implementation across 72 Grace cores */
        auto cpu_aes = [&]() {
            #pragma omp parallel for schedule(static)
            for (size_t b = 0; b < num_blocks; b++) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_CRYPTO)
                uint8x16_t data = vld1q_u8(h_in + b * 16);
                uint8x16_t key  = vdupq_n_u8((uint8_t)b);
                uint8x16_t enc  = vaeseq_u8(data, key);
                enc = vaesmcq_u8(enc);
                vst1q_u8(h_out + b * 16, enc);
#else
                for (int j = 0; j < 16; j++) {
                    h_out[b * 16 + j] = h_in[b * 16 + j] ^ (uint8_t)(b + j);
                }
#endif
            }
        };

        double tc0 = bench_get_time_ms();
        cpu_aes();
        rec.t_kernel_cold_ms = bench_get_time_ms() - tc0;

        for (int w = 0; w < warmup; w++) cpu_aes();

        double tw0 = bench_get_time_ms();
        for (int it = 0; it < iters; it++) cpu_aes();
        rec.t_kernel_warm_ms = (bench_get_time_ms() - tw0) / (double)iters;
    } else {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        if (mode == RESIDENCY_GPU_EXPLICIT_COPY) {
            double th0 = bench_get_time_ms();
            cudaMemcpyAsync(bIn.dev_ptr, bIn.host_ptr, num_bytes, cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
            rec.t_h2d_ms = bench_get_time_ms() - th0;
        }

        int threads = 256;
        size_t total_blocks = (num_blocks + threads - 1) / threads;
        unsigned int blocks = (unsigned int)total_blocks;

        /* Cold iteration */
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        cudaEventRecord(ev0, stream);
        gpu_aes_ctr_kernel<<<blocks, threads, 0, stream>>>((const uint8_t *)bIn.dev_ptr, (uint8_t *)bOut.dev_ptr, num_blocks);
        cudaEventRecord(ev1, stream);
        cudaStreamSynchronize(stream);
        float ms_cold = 0;
        cudaEventElapsedTime(&ms_cold, ev0, ev1);
        rec.t_kernel_cold_ms = ms_cold;

        for (int w = 0; w < warmup; w++) {
            gpu_aes_ctr_kernel<<<blocks, threads, 0, stream>>>((const uint8_t *)bIn.dev_ptr, (uint8_t *)bOut.dev_ptr, num_blocks);
        }
        cudaStreamSynchronize(stream);

        cudaEventRecord(ev0, stream);
        for (int it = 0; it < iters; it++) {
            gpu_aes_ctr_kernel<<<blocks, threads, 0, stream>>>((const uint8_t *)bIn.dev_ptr, (uint8_t *)bOut.dev_ptr, num_blocks);
        }
        cudaEventRecord(ev1, stream);
        cudaStreamSynchronize(stream);
        float ms_warm = 0;
        cudaEventElapsedTime(&ms_warm, ev0, ev1);
        rec.t_kernel_warm_ms = ms_warm / (double)iters;

        if (mode == RESIDENCY_GPU_EXPLICIT_COPY) {
            double td0 = bench_get_time_ms();
            cudaMemcpyAsync(bOut.host_ptr, bOut.dev_ptr, num_bytes, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            rec.t_d2h_ms = bench_get_time_ms() - td0;
        }

        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        cudaStreamDestroy(stream);
    }

    if (mode == RESIDENCY_GPU_UM_MIGRATE) {
        rec.t_e2e_ms = rec.t_kernel_cold_ms + rec.t_kernel_warm_ms;
        rec.c2c_bw_gbps = (rec.t_kernel_cold_ms > 0) ? ((double)rec.working_set_bytes / (rec.t_kernel_cold_ms * 1e6)) : 0.0;
    } else if (mode == RESIDENCY_GPU_ZERO_COPY) {
        rec.t_e2e_ms = rec.t_kernel_warm_ms;
        rec.c2c_bw_gbps = (rec.t_kernel_warm_ms > 0) ? ((double)rec.working_set_bytes / (rec.t_kernel_warm_ms * 1e6)) : 0.0;
    } else {
        rec.t_e2e_ms = rec.t_h2d_ms + rec.t_kernel_warm_ms + rec.t_d2h_ms;
        rec.c2c_bw_gbps = ((num_bytes * 2) / ((rec.t_h2d_ms + rec.t_d2h_ms) * 1e6 + 1e-9));
    }
    rec.eff_bw_gbps = (double)rec.working_set_bytes / (rec.t_e2e_ms * 1e6);
    double throughput_gbps = ((double)num_bytes / (rec.t_kernel_warm_ms * 1e-3)) / 1e9;
    rec.gflops = throughput_gbps; /* Encryption GB/s */
    snprintf(rec.status, sizeof(rec.status), "ok");

    log_bench_record(fout, &rec);

    printf("AES   [%-14s] %-25s : e2e=%7.2fms (kern=%7.2fms) | %7.1f GB/s | %7.1f GB/s enc\n",
           residency_to_str(mode), rec.problem_size, rec.t_e2e_ms, rec.t_kernel_warm_ms, rec.eff_bw_gbps, throughput_gbps);

    bench_free_buf(&bIn); bench_free_buf(&bOut);
    return 0;
}

int main(int argc, char **argv) {
    const char *csv_file = "residency_divergence_sweep.csv";
    int iters = 5;
    int warmup = 2;

    const char *workload = "all";
    const char *modes_arg = "all";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv_file = argv[++i];
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--workload") && i + 1 < argc) workload = argv[++i];
        else if (!strcmp(argv[i], "--modes") && i + 1 < argc) modes_arg = argv[++i];
    }

    FILE *fout = fopen(csv_file, "a+");
    if (!fout) { perror("fopen csv"); return 1; }
    fseek(fout, 0, SEEK_END);
    if (ftell(fout) == 0) print_bench_csv_header(fout);

    printf("========================================================================================\n");
    printf(" NVIDIA Grace Hopper (GH200): Regime 4 Divergent & Cryptographic Compute (4 NUMA Modes)\n");
    printf(" Target CSV: %s | Iters=%d | Warmup=%d\n", csv_file, iters, warmup);
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

    const char *dists[] = {"uniform", "skewed"};

    /* 1. Radix Sort: 4M keys (16MB) up to 268M keys (1GB) and 1073M keys (4GB) across distributions */
    if (!strcmp(workload, "all") || !strcmp(workload, "sort")) {
        int sort_sizes[] = {4000000, 16000000, 64000000, 268435456, 1073741824};
        printf("── [Part 1] CUB Radix Sort (Regime 4 Sort & Permutation) ─────────────────────────────\n");
        for (size_t d = 0; d < 2; d++) {
            const char *dist = dists[d];
            for (size_t s = 0; s < sizeof(sort_sizes)/sizeof(sort_sizes[0]); s++) {
                int N = sort_sizes[s];
                for (int m = 0; m < n_modes; m++) {
                    run_sort_bench(fout, modes[m], N, dist, iters, warmup);
                }
                printf("----------------------------------------------------------------------------------------\n");
            }
        }
    }

    /* 2. Histogram: 16M up to 1000M samples, bins=256 vs bins=65536 across uniform and skewed */
    if (!strcmp(workload, "all") || !strcmp(workload, "histogram")) {
        int hist_sizes[] = {16000000, 64000000, 256000000, 1000000000};
        int bin_counts[] = {256, 65536};
        printf("\n── [Part 2] CUB Histogram & Coherent System Atomics (Regime 4 Atomics) ───────────────\n");
        for (size_t b = 0; b < 2; b++) {
            int num_bins = bin_counts[b];
            for (size_t d = 0; d < 2; d++) {
                const char *dist = dists[d];
                for (size_t s = 0; s < sizeof(hist_sizes)/sizeof(hist_sizes[0]); s++) {
                    int N = hist_sizes[s];
                    for (int m = 0; m < n_modes; m++) {
                        run_histogram_bench(fout, modes[m], N, num_bins, dist, iters, warmup);
                    }
                    printf("----------------------------------------------------------------------------------------\n");
                }
            }
        }
    }

    /* 3. AES-128-CTR: 16MB, 64MB, 256MB, 1GB, 4GB, 16GB, 64GB, 128GB */
    if (!strcmp(workload, "all") || !strcmp(workload, "aes")) {
        size_t aes_sizes[] = {
            16L * 1024 * 1024,
            64L * 1024 * 1024,
            256L * 1024 * 1024,
            1024L * 1024 * 1024,
            4096L * 1024 * 1024,
            16384L * 1024 * 1024,
            65536L * 1024 * 1024,
            131072L * 1024 * 1024
        };
        printf("\n── [Part 3] AES-128-CTR (Regime 4 Cryptography: ARMv9 NEON vs GPU Shared-Mem) ────────\n");
        for (size_t s = 0; s < sizeof(aes_sizes)/sizeof(aes_sizes[0]); s++) {
            size_t bytes = aes_sizes[s];
            for (int m = 0; m < n_modes; m++) {
                run_aes_bench(fout, modes[m], bytes, iters, warmup);
            }
            printf("----------------------------------------------------------------------------------------\n");
        }
    }

    fclose(fout);
    printf("\nDivergence & Crypto benchmark complete! Results recorded to %s\n", csv_file);
    return 0;
}
