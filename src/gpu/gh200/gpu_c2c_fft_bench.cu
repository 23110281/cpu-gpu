/*
 * gpu_c2c_fft_bench.cu
 *
 * Regime 1C / Extended Compute: 1D & 2D Fast Fourier Transform (FFT)
 * across 4 Coherent NUMA Residency Modes on NVIDIA Grace Hopper (GH200):
 *   1. CPU_LOCAL         (72 Grace Neoverse-V2 cores, FFTW3 multi-threaded)
 *   2. GPU_EXPLICIT_COPY (Pinned LPDDR5X -> HBM3 -> cuFFT -> LPDDR5X)
 *   3. GPU_ZERO_COPY     (Mapped LPDDR5X streamed directly over NVLink-C2C)
 *   4. GPU_UM_MIGRATE    (cudaMallocManaged with cold ATS faulting vs warm)
 *
 * Completes Lee et al. (ISCA 2010) Table 1 FFT benchmark.
 */

#include "bench_harness.h"
#include "gpu_device_specs.h"
#include <cufft.h>
#include <fftw3.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUFFT_CHECK(expr) do { \
    cufftResult st = (expr); \
    if (st != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error %d at line %d\n", (int)st, __LINE__); \
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

static int g_cpu_threads = 72;

/* ── Benchmark 1D FFT ─────────────────────────────────────────────────────── */
int run_fft_1d_bench(FILE *csv_out, residency_mode_t residency, long N, int iters, int warmup) {
    bench_record_t rec;
    memset(&rec, 0, sizeof(rec));
    snprintf(rec.workload, sizeof(rec.workload), "cufft_1d");
    rec.regime = REGIME_ARITHMETIC_COMPUTE;
    rec.residency = residency;
    snprintf(rec.precision, sizeof(rec.precision), "fp32_complex");
    snprintf(rec.problem_size, sizeof(rec.problem_size), "N=%ld", N);
    snprintf(rec.status, sizeof(rec.status), "ok");

    size_t bytes = (size_t)N * sizeof(cufftComplex);
    rec.working_set_bytes = bytes * 2; /* In + Out */
    double flops = 5.0 * (double)N * (log((double)N) / log(2.0));

    double t0_alloc = bench_get_time_ms();
    bench_buf_t bIn = bench_alloc_buf(bytes, residency);
    bench_buf_t bOut = bench_alloc_buf(bytes, residency);
    rec.t_alloc_ms = bench_get_time_ms() - t0_alloc;

    if (!bIn.host_ptr || !bOut.host_ptr ||
        (residency == RESIDENCY_GPU_EXPLICIT_COPY && (!bIn.dev_ptr || !bOut.dev_ptr))) {
        snprintf(rec.status, sizeof(rec.status), "oom-skip");
        log_bench_record(csv_out, &rec);
        bench_free_buf(&bIn); bench_free_buf(&bOut);
        return 0;
    }

    cufftComplex *hin = (cufftComplex *)bIn.host_ptr;
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < N; i++) {
        hin[i].x = 1.0f;
        hin[i].y = 0.5f;
    }

    if (residency == RESIDENCY_CPU_LOCAL) {
        rec.threads_or_sms = g_cpu_threads;
        fftwf_init_threads();
        fftwf_plan_with_nthreads(g_cpu_threads);

        fftwf_complex *in = (fftwf_complex *)bIn.host_ptr;
        fftwf_complex *out = (fftwf_complex *)bOut.host_ptr;
        fftwf_plan plan = fftwf_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

        double t0 = bench_get_time_ms();
        fftwf_execute(plan);
        rec.t_kernel_cold_ms = bench_get_time_ms() - t0;

        for (int w = 0; w < warmup; w++) fftwf_execute(plan);

        t0 = bench_get_time_ms();
        for (int it = 0; it < iters; it++) fftwf_execute(plan);
        rec.t_kernel_warm_ms = (bench_get_time_ms() - t0) / iters;

        fftwf_destroy_plan(plan);

        rec.t_h2d_ms = 0.0;
        rec.t_d2h_ms = 0.0;
        rec.t_e2e_ms = rec.t_kernel_warm_ms;
        rec.eff_bw_gbps = (double)rec.working_set_bytes / (rec.t_e2e_ms * 1e6);
        rec.c2c_bw_gbps = 0.0;
        rec.gflops = flops / (rec.t_kernel_warm_ms * 1e6);
        rec.pct_peak = (rec.gflops / 5000.0) * 100.0;
    } else {
        rec.threads_or_sms = 132;
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        cudaEvent_t ev0, ev1;
        CUDA_CHECK(cudaEventCreate(&ev0)); CUDA_CHECK(cudaEventCreate(&ev1));

        if (residency == RESIDENCY_GPU_EXPLICIT_COPY) {
            double th0 = bench_get_time_ms();
            cudaMemcpyAsync(bIn.dev_ptr, bIn.host_ptr, bytes, cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
            rec.t_h2d_ms = bench_get_time_ms() - th0;
        }

        cufftHandle plan;
        CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_C2C, 1));
        CUFFT_CHECK(cufftSetStream(plan, stream));

        cufftComplex *din = (cufftComplex *)((residency == RESIDENCY_GPU_EXPLICIT_COPY) ? bIn.dev_ptr : bIn.host_ptr);
        cufftComplex *dout = (cufftComplex *)((residency == RESIDENCY_GPU_EXPLICIT_COPY) ? bOut.dev_ptr : bOut.host_ptr);

        cudaEventRecord(ev0, stream);
        CUFFT_CHECK(cufftExecC2C(plan, din, dout, CUFFT_FORWARD));
        cudaEventRecord(ev1, stream);
        cudaStreamSynchronize(stream);
        float ms_cold = 0;
        cudaEventElapsedTime(&ms_cold, ev0, ev1);
        rec.t_kernel_cold_ms = ms_cold;

        for (int w = 0; w < warmup; w++) {
            cufftExecC2C(plan, din, dout, CUFFT_FORWARD);
        }
        cudaStreamSynchronize(stream);

        cudaEventRecord(ev0, stream);
        for (int it = 0; it < iters; it++) {
            cufftExecC2C(plan, din, dout, CUFFT_FORWARD);
        }
        cudaEventRecord(ev1, stream);
        cudaStreamSynchronize(stream);
        float ms_warm = 0;
        cudaEventElapsedTime(&ms_warm, ev0, ev1);
        rec.t_kernel_warm_ms = ms_warm / iters;

        if (residency == RESIDENCY_GPU_EXPLICIT_COPY) {
            double td0 = bench_get_time_ms();
            cudaMemcpyAsync(bOut.host_ptr, bOut.dev_ptr, bytes, cudaMemcpyDeviceToHost, stream);
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
            rec.c2c_bw_gbps = (double)bytes / (rec.t_h2d_ms * 1e6 + 1e-9);
        }

        rec.eff_bw_gbps = (double)rec.working_set_bytes / (rec.t_e2e_ms * 1e6);
        rec.gflops = flops / (rec.t_kernel_warm_ms * 1e6);
        rec.pct_peak = (rec.gflops / (get_gpu_spec_for_device(0).peak_fp32_tflops * 1e3)) * 100.0;

        cufftDestroy(plan);
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        cudaStreamDestroy(stream);
    }

    log_bench_record(csv_out, &rec);
    printf("FFT1D [%-14s] N=%-10ld : e2e=%7.2fms (kern=%7.2fms) | EffBW=%7.1f GB/s | %7.1f GFLOPS\n",
           residency_to_str(residency), N, rec.t_e2e_ms, rec.t_kernel_warm_ms, rec.eff_bw_gbps, rec.gflops);

    bench_free_buf(&bIn); bench_free_buf(&bOut);
    return 0;
}

/* ── Benchmark 2D FFT ─────────────────────────────────────────────────────── */
int run_fft_2d_bench(FILE *csv_out, residency_mode_t residency, int S, int iters, int warmup) {
    bench_record_t rec;
    memset(&rec, 0, sizeof(rec));
    snprintf(rec.workload, sizeof(rec.workload), "cufft_2d");
    rec.regime = REGIME_ARITHMETIC_COMPUTE;
    rec.residency = residency;
    snprintf(rec.precision, sizeof(rec.precision), "fp32_complex");
    snprintf(rec.problem_size, sizeof(rec.problem_size), "S=%d", S);
    snprintf(rec.status, sizeof(rec.status), "ok");

    long N = (long)S * S;
    size_t bytes = (size_t)N * sizeof(cufftComplex);
    rec.working_set_bytes = bytes * 2;
    double flops = 5.0 * (double)N * (log((double)N) / log(2.0));

    double t0_alloc = bench_get_time_ms();
    bench_buf_t bIn = bench_alloc_buf(bytes, residency);
    bench_buf_t bOut = bench_alloc_buf(bytes, residency);
    rec.t_alloc_ms = bench_get_time_ms() - t0_alloc;

    if (!bIn.host_ptr || !bOut.host_ptr ||
        (residency == RESIDENCY_GPU_EXPLICIT_COPY && (!bIn.dev_ptr || !bOut.dev_ptr))) {
        snprintf(rec.status, sizeof(rec.status), "oom-skip");
        log_bench_record(csv_out, &rec);
        bench_free_buf(&bIn); bench_free_buf(&bOut);
        return 0;
    }

    cufftComplex *hin = (cufftComplex *)bIn.host_ptr;
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < N; i++) {
        hin[i].x = 1.0f;
        hin[i].y = 0.5f;
    }

    if (residency == RESIDENCY_CPU_LOCAL) {
        rec.threads_or_sms = g_cpu_threads;
        fftwf_init_threads();
        fftwf_plan_with_nthreads(g_cpu_threads);

        fftwf_complex *in = (fftwf_complex *)bIn.host_ptr;
        fftwf_complex *out = (fftwf_complex *)bOut.host_ptr;
        fftwf_plan plan = fftwf_plan_dft_2d(S, S, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

        double t0 = bench_get_time_ms();
        fftwf_execute(plan);
        rec.t_kernel_cold_ms = bench_get_time_ms() - t0;

        for (int w = 0; w < warmup; w++) fftwf_execute(plan);

        t0 = bench_get_time_ms();
        for (int it = 0; it < iters; it++) fftwf_execute(plan);
        rec.t_kernel_warm_ms = (bench_get_time_ms() - t0) / iters;

        fftwf_destroy_plan(plan);

        rec.t_h2d_ms = 0.0;
        rec.t_d2h_ms = 0.0;
        rec.t_e2e_ms = rec.t_kernel_warm_ms;
        rec.eff_bw_gbps = (double)rec.working_set_bytes / (rec.t_e2e_ms * 1e6);
        rec.c2c_bw_gbps = 0.0;
        rec.gflops = flops / (rec.t_kernel_warm_ms * 1e6);
        rec.pct_peak = (rec.gflops / 5000.0) * 100.0;
    } else {
        rec.threads_or_sms = 132;
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        cudaEvent_t ev0, ev1;
        CUDA_CHECK(cudaEventCreate(&ev0)); CUDA_CHECK(cudaEventCreate(&ev1));

        if (residency == RESIDENCY_GPU_EXPLICIT_COPY) {
            double th0 = bench_get_time_ms();
            cudaMemcpyAsync(bIn.dev_ptr, bIn.host_ptr, bytes, cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
            rec.t_h2d_ms = bench_get_time_ms() - th0;
        }

        cufftHandle plan;
        CUFFT_CHECK(cufftPlan2d(&plan, S, S, CUFFT_C2C));
        CUFFT_CHECK(cufftSetStream(plan, stream));

        cufftComplex *din = (cufftComplex *)((residency == RESIDENCY_GPU_EXPLICIT_COPY) ? bIn.dev_ptr : bIn.host_ptr);
        cufftComplex *dout = (cufftComplex *)((residency == RESIDENCY_GPU_EXPLICIT_COPY) ? bOut.dev_ptr : bOut.host_ptr);

        cudaEventRecord(ev0, stream);
        CUFFT_CHECK(cufftExecC2C(plan, din, dout, CUFFT_FORWARD));
        cudaEventRecord(ev1, stream);
        cudaStreamSynchronize(stream);
        float ms_cold = 0;
        cudaEventElapsedTime(&ms_cold, ev0, ev1);
        rec.t_kernel_cold_ms = ms_cold;

        for (int w = 0; w < warmup; w++) {
            cufftExecC2C(plan, din, dout, CUFFT_FORWARD);
        }
        cudaStreamSynchronize(stream);

        cudaEventRecord(ev0, stream);
        for (int it = 0; it < iters; it++) {
            cufftExecC2C(plan, din, dout, CUFFT_FORWARD);
        }
        cudaEventRecord(ev1, stream);
        cudaStreamSynchronize(stream);
        float ms_warm = 0;
        cudaEventElapsedTime(&ms_warm, ev0, ev1);
        rec.t_kernel_warm_ms = ms_warm / iters;

        if (residency == RESIDENCY_GPU_EXPLICIT_COPY) {
            double td0 = bench_get_time_ms();
            cudaMemcpyAsync(bOut.host_ptr, bOut.dev_ptr, bytes, cudaMemcpyDeviceToHost, stream);
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
            rec.c2c_bw_gbps = (double)bytes / (rec.t_h2d_ms * 1e6 + 1e-9);
        }

        rec.eff_bw_gbps = (double)rec.working_set_bytes / (rec.t_e2e_ms * 1e6);
        rec.gflops = flops / (rec.t_kernel_warm_ms * 1e6);
        rec.pct_peak = (rec.gflops / (get_gpu_spec_for_device(0).peak_fp32_tflops * 1e3)) * 100.0;

        cufftDestroy(plan);
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        cudaStreamDestroy(stream);
    }

    log_bench_record(csv_out, &rec);
    printf("FFT2D [%-14s] S=%-6d     : e2e=%7.2fms (kern=%7.2fms) | EffBW=%7.1f GB/s | %7.1f GFLOPS\n",
           residency_to_str(residency), S, rec.t_e2e_ms, rec.t_kernel_warm_ms, rec.eff_bw_gbps, rec.gflops);

    bench_free_buf(&bIn); bench_free_buf(&bOut);
    return 0;
}

/* ── Benchmark Conv2D (Lee et al. Table 1) ─────────────────────────────────── */
__global__ void conv2d_kernel(float *out, const float *in, const float *filter,
                             int H, int W, int K) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int k_rad = K / 2;

    if (y < H && x < W) {
        float sum = 0.0f;
        for (int ky = -k_rad; ky <= k_rad; ky++) {
            int iy = min(max(y + ky, 0), H - 1);
            for (int kx = -k_rad; kx <= k_rad; kx++) {
                int ix = min(max(x + kx, 0), W - 1);
                float f = filter[(ky + k_rad) * K + (kx + k_rad)];
                sum += in[iy * W + ix] * f;
            }
        }
        out[y * W + x] = sum;
    }
}

int run_conv2d_bench(FILE *csv_out, residency_mode_t residency, int H, int W, int K, int iters, int warmup) {
    bench_record_t rec;
    memset(&rec, 0, sizeof(rec));
    snprintf(rec.workload, sizeof(rec.workload), "conv2d");
    rec.regime = REGIME_ARITHMETIC_COMPUTE;
    rec.residency = residency;
    snprintf(rec.precision, sizeof(rec.precision), "fp32");
    snprintf(rec.problem_size, sizeof(rec.problem_size), "H=%d,W=%d,K=%d", H, W, K);
    snprintf(rec.status, sizeof(rec.status), "ok");

    size_t bytes_in = (size_t)H * W * sizeof(float);
    size_t bytes_out = bytes_in;
    size_t bytes_filter = (size_t)K * K * sizeof(float);
    rec.working_set_bytes = bytes_in + bytes_out + bytes_filter;
    double flops = 2.0 * (double)H * (double)W * (double)K * (double)K;

    double t0_alloc = bench_get_time_ms();
    bench_buf_t bIn = bench_alloc_buf(bytes_in, residency);
    bench_buf_t bOut = bench_alloc_buf(bytes_out, residency);
    bench_buf_t bFilter = bench_alloc_buf(bytes_filter, residency);
    rec.t_alloc_ms = bench_get_time_ms() - t0_alloc;

    if (!bIn.host_ptr || !bOut.host_ptr || !bFilter.host_ptr ||
        (residency == RESIDENCY_GPU_EXPLICIT_COPY && (!bIn.dev_ptr || !bOut.dev_ptr || !bFilter.dev_ptr))) {
        snprintf(rec.status, sizeof(rec.status), "oom-skip");
        log_bench_record(csv_out, &rec);
        bench_free_buf(&bIn); bench_free_buf(&bOut); bench_free_buf(&bFilter);
        return 0;
    }

    float *hin = (float *)bIn.host_ptr;
    float *hfilt = (float *)bFilter.host_ptr;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < (size_t)H * W; i++) hin[i] = 1.0f;
    for (int i = 0; i < K * K; i++) hfilt[i] = 1.0f / (float)(K * K);

    if (residency == RESIDENCY_CPU_LOCAL) {
        rec.threads_or_sms = g_cpu_threads;
        float *out = (float *)bOut.host_ptr;
        const float *in = (const float *)bIn.host_ptr;
        const float *filt = (const float *)bFilter.host_ptr;
        int k_rad = K / 2;

        auto cpu_conv_step = [&]() {
            #pragma omp parallel for schedule(static)
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    float sum = 0.0f;
                    for (int ky = -k_rad; ky <= k_rad; ky++) {
                        int iy = (y + ky < 0) ? 0 : (y + ky >= H ? H - 1 : y + ky);
                        for (int kx = -k_rad; kx <= k_rad; kx++) {
                            int ix = (x + kx < 0) ? 0 : (x + kx >= W ? W - 1 : x + kx);
                            sum += in[iy * W + ix] * filt[(ky + k_rad) * K + (kx + k_rad)];
                        }
                    }
                    out[y * W + x] = sum;
                }
            }
        };

        double t0 = bench_get_time_ms();
        cpu_conv_step();
        rec.t_kernel_cold_ms = bench_get_time_ms() - t0;

        for (int w = 0; w < warmup; w++) cpu_conv_step();

        t0 = bench_get_time_ms();
        for (int it = 0; it < iters; it++) cpu_conv_step();
        rec.t_kernel_warm_ms = (bench_get_time_ms() - t0) / iters;

        rec.t_h2d_ms = 0.0;
        rec.t_d2h_ms = 0.0;
        rec.t_e2e_ms = rec.t_kernel_warm_ms;
        rec.c2c_bw_gbps = 0.0;
        rec.eff_bw_gbps = (double)rec.working_set_bytes / (rec.t_e2e_ms * 1e6);
        rec.gflops = flops / (rec.t_kernel_warm_ms * 1e6);
        rec.pct_peak = (rec.gflops / (72.0 * 3.1 * 32.0)) * 100.0;
    } else {
        rec.threads_or_sms = 132;
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        cudaEvent_t ev0, ev1, ev_h2d0, ev_h2d1, ev_d2h0, ev_d2h1;
        CUDA_CHECK(cudaEventCreate(&ev0)); CUDA_CHECK(cudaEventCreate(&ev1));
        CUDA_CHECK(cudaEventCreate(&ev_h2d0)); CUDA_CHECK(cudaEventCreate(&ev_h2d1));
        CUDA_CHECK(cudaEventCreate(&ev_d2h0)); CUDA_CHECK(cudaEventCreate(&ev_d2h1));

        float *dIn = (residency == RESIDENCY_GPU_EXPLICIT_COPY) ? (float *)bIn.dev_ptr : (float *)bIn.host_ptr;
        float *dOut = (residency == RESIDENCY_GPU_EXPLICIT_COPY) ? (float *)bOut.dev_ptr : (float *)bOut.host_ptr;
        float *dFilter = (residency == RESIDENCY_GPU_EXPLICIT_COPY) ? (float *)bFilter.dev_ptr : (float *)bFilter.host_ptr;

        if (residency == RESIDENCY_GPU_EXPLICIT_COPY) {
            cudaEventRecord(ev_h2d0, stream);
            CUDA_CHECK(cudaMemcpyAsync(bIn.dev_ptr, bIn.host_ptr, bytes_in, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(bFilter.dev_ptr, bFilter.host_ptr, bytes_filter, cudaMemcpyHostToDevice, stream));
            cudaEventRecord(ev_h2d1, stream);
            cudaEventSynchronize(ev_h2d1);
            float h2d_ms = 0;
            cudaEventElapsedTime(&h2d_ms, ev_h2d0, ev_h2d1);
            rec.t_h2d_ms = (double)h2d_ms;
        } else {
            rec.t_h2d_ms = 0.0;
        }

        dim3 block(16, 16);
        dim3 grid((W + 15) / 16, (H + 15) / 16);

        cudaEventRecord(ev0, stream);
        conv2d_kernel<<<grid, block, 0, stream>>>(dOut, dIn, dFilter, H, W, K);
        cudaEventRecord(ev1, stream);
        cudaEventSynchronize(ev1);
        float cold_ms = 0;
        cudaEventElapsedTime(&cold_ms, ev0, ev1);
        rec.t_kernel_cold_ms = (double)cold_ms;

        for (int w = 0; w < warmup; w++) {
            conv2d_kernel<<<grid, block, 0, stream>>>(dOut, dIn, dFilter, H, W, K);
        }
        cudaStreamSynchronize(stream);

        cudaEventRecord(ev0, stream);
        for (int it = 0; it < iters; it++) {
            conv2d_kernel<<<grid, block, 0, stream>>>(dOut, dIn, dFilter, H, W, K);
        }
        cudaEventRecord(ev1, stream);
        cudaEventSynchronize(ev1);
        float warm_ms = 0;
        cudaEventElapsedTime(&warm_ms, ev0, ev1);
        rec.t_kernel_warm_ms = (double)warm_ms / iters;

        if (residency == RESIDENCY_GPU_EXPLICIT_COPY) {
            cudaEventRecord(ev_d2h0, stream);
            CUDA_CHECK(cudaMemcpyAsync(bOut.host_ptr, bOut.dev_ptr, bytes_out, cudaMemcpyDeviceToHost, stream));
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
            rec.c2c_bw_gbps = (double)bytes_in / (rec.t_h2d_ms * 1e6 + 1e-9);
        }

        rec.eff_bw_gbps = (double)rec.working_set_bytes / (rec.t_e2e_ms * 1e6);
        rec.gflops = flops / (rec.t_kernel_warm_ms * 1e6);
        rec.pct_peak = (rec.gflops / (get_gpu_spec_for_device(0).peak_fp32_tflops * 1e3)) * 100.0;

        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        cudaEventDestroy(ev_h2d0); cudaEventDestroy(ev_h2d1);
        cudaEventDestroy(ev_d2h0); cudaEventDestroy(ev_d2h1);
        cudaStreamDestroy(stream);
    }

    log_bench_record(csv_out, &rec);
    printf("Conv2D[%-14s] H=%d W=%d K=%d : e2e=%7.2fms (kern=%7.2fms) | EffBW=%7.1f GB/s | %7.1f GFLOPS\n",
           residency_to_str(residency), H, W, K, rec.t_e2e_ms, rec.t_kernel_warm_ms, rec.eff_bw_gbps, rec.gflops);

    bench_free_buf(&bIn); bench_free_buf(&bOut); bench_free_buf(&bFilter);
    return 0;
}

int main(int argc, char **argv) {
    const char *csv_file = "residency_fft_sweep.csv";
    int iters = 10;
    int warmup = 3;
    int cpu_threads = 72;
    const char *modes_arg = "all";

    const char *workload = "all";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv_file = argv[++i];
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) cpu_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--modes") && i + 1 < argc) modes_arg = argv[++i];
        else if (!strcmp(argv[i], "--workload") && i + 1 < argc) workload = argv[++i];
    }

    g_cpu_threads = cpu_threads;

    FILE *fout = fopen(csv_file, "a+");
    if (!fout) { perror("fopen csv"); return 1; }
    fseek(fout, 0, SEEK_END);
    if (ftell(fout) == 0) print_bench_csv_header(fout);

    printf("========================================================================================\n");
    printf(" NVIDIA Grace Hopper (GH200): Regime 1C FFT & Conv2D (cuFFT, FFTW3, Spatial Conv)\n");
    printf(" Target CSV: %s | Iters=%d | Warmup=%d | CPU Threads=%d | Workload=%s\n", csv_file, iters, warmup, cpu_threads, workload);
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

    /* 1D FFT: 1M up to 64M elements */
    if (!strcmp(workload, "all") || !strcmp(workload, "fft") || !strcmp(workload, "fft1d")) {
        long fft_1d_sizes[] = {1048576L, 4194304L, 16777216L, 67108864L};
        int n_1d = sizeof(fft_1d_sizes) / sizeof(fft_1d_sizes[0]);

        printf("── [Part 1] 1D FFT (cuFFT vs FFTW3 Across 4 NUMA Modes) ────────────────────────────────\n");
        for (int s = 0; s < n_1d; s++) {
            long N = fft_1d_sizes[s];
            for (int m = 0; m < n_modes; m++) {
                run_fft_1d_bench(fout, modes[m], N, iters, warmup);
            }
            printf("----------------------------------------------------------------------------------------\n");
        }
    }

    /* 2D FFT: 1024 to 8192 */
    if (!strcmp(workload, "all") || !strcmp(workload, "fft") || !strcmp(workload, "fft2d")) {
        int fft_2d_sizes[] = {1024, 2048, 4096, 8192};
        int n_2d = sizeof(fft_2d_sizes) / sizeof(fft_2d_sizes[0]);

        printf("\n── [Part 2] 2D FFT (cuFFT vs FFTW3 Across 4 NUMA Modes) ────────────────────────────────\n");
        for (int s = 0; s < n_2d; s++) {
            int S = fft_2d_sizes[s];
            for (int m = 0; m < n_modes; m++) {
                run_fft_2d_bench(fout, modes[m], S, iters, warmup);
            }
            printf("----------------------------------------------------------------------------------------\n");
        }
    }

    /* 3. Conv2D: 1024 to 16384 */
    if (!strcmp(workload, "all") || !strcmp(workload, "conv") || !strcmp(workload, "conv2d")) {
        int conv_sizes[] = {1024, 2048, 4096, 8192, 16384};
        int n_conv = sizeof(conv_sizes) / sizeof(conv_sizes[0]);

        printf("\n── [Part 3] Conv2D 5x5 Spatial Filter Across 4 NUMA Modes ───────────────────────────────\n");
        for (int s = 0; s < n_conv; s++) {
            int S = conv_sizes[s];
            for (int m = 0; m < n_modes; m++) {
                run_conv2d_bench(fout, modes[m], S, S, 5, iters, warmup);
            }
            printf("----------------------------------------------------------------------------------------\n");
        }
    }

    fclose(fout);
    printf("\nFFT & Conv2D benchmark complete! Results recorded to %s\n", csv_file);
    return 0;
}
