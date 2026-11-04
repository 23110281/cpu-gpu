/* gpu_c2c_irregular_bench.cu -- Regime 3: Irregular Latency & Sparse Compute
 * Across 4 Coherent NUMA Memory Residency Modes on NVIDIA Grace Hopper (GH200):
 *   1. CPU_LOCAL (72 Neoverse-V2 cores, LPDDR5X local)
 *   2. GPU_EXPLICIT_COPY (H2D copy over NVLink-C2C -> HBM3 execution -> D2H)
 *   3. GPU_ZERO_COPY (Hopper SMs execute directly over NVLink-C2C against LPDDR5X)
 *   4. GPU_UM_MIGRATE (UVM cudaMallocManaged with hardware page migration)
 *
 * Implements:
 *   - cuSPARSE SpMV (Sparse Matrix-Vector Multiply)
 *   - Thrust Binary Search (thrust::lower_bound over pointer types)
 *   - cuSPARSE SpGEMM (Sparse Matrix-Matrix Multiply)
 *
 * Output adheres strictly to the unified CSV schema:
 *   workload,regime,precision,residency_mode,working_set_bytes,problem_size,threads_or_sms,
 *   t_alloc_ms,t_h2d_ms,t_kernel_cold_ms,t_kernel_warm_ms,t_d2h_ms,t_e2e_ms,eff_bw_gbps,
 *   c2c_bw_gbps,gflops,pct_peak,status
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/device_ptr.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <algorithm>
#include <vector>
#include <omp.h>

#include "bench_harness.h"
#include "gpu_device_specs.h"
#include "gemv_host.h"

#define CHECK_CUSPARSE(call) do { \
    cusparseStatus_t status = (call); \
    if (status != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSPARSE Error at %s:%d: status=%d\n", __FILE__, __LINE__, (int)status); \
        return -1; \
    } \
} while (0)

/* =========================================================================================
 * 1. cuSPARSE SpMV (Sparse Matrix-Vector Multiplication)
 * ========================================================================================= */
int run_spmv_bench(FILE *fout, residency_mode_t mode, int S, double density, int iters, int warmup) {
    bench_record_t rec;
    memset(&rec, 0, sizeof(rec));
    snprintf(rec.workload, sizeof(rec.workload), "spmv");
    rec.regime = 3;
    snprintf(rec.precision, sizeof(rec.precision), "fp32");
    rec.residency = mode;
    snprintf(rec.problem_size, sizeof(rec.problem_size), "S=%d,density=%.4f", S, density);

    int npr = (int)llround(density * (double)S);
    if (npr < 1) npr = 1;
    if (npr > S) npr = S;
    long nnz = (long)npr * S;
    if (nnz >= 2147483647L) {
        printf("SpMV %s: NNZ exceeds int32 limits. Skipping.\n", rec.problem_size);
        return 0;
    }

    size_t bytes_rowOff = (S + 1) * sizeof(int);
    size_t bytes_colInd = nnz * sizeof(int);
    size_t bytes_vals   = nnz * sizeof(float);
    size_t bytes_x      = S * sizeof(float);
    size_t bytes_y      = S * sizeof(float);
    rec.working_set_bytes = bytes_rowOff + bytes_colInd + bytes_vals + bytes_x + bytes_y;

    gpu_spec_t spec = get_gpu_spec_for_device(0);
    rec.threads_or_sms = (mode == RESIDENCY_CPU_LOCAL) ? 72 : 132;
    double peak_flops = (mode == RESIDENCY_CPU_LOCAL) ? (72.0 * 3.1 * 32.0 * 1e9) : (spec.peak_fp32_tflops * 1e12);
    double flops = 2.0 * (double)nnz;

    double t0_alloc = bench_get_time_ms();
    bench_buf_t bRowOff = bench_alloc_buf(bytes_rowOff, mode);
    bench_buf_t bColInd = bench_alloc_buf(bytes_colInd, mode);
    bench_buf_t bVals   = bench_alloc_buf(bytes_vals, mode);
    bench_buf_t bX      = bench_alloc_buf(bytes_x, mode);
    bench_buf_t bY      = bench_alloc_buf(bytes_y, mode);
    rec.t_alloc_ms = bench_get_time_ms() - t0_alloc;

    if (!bRowOff.host_ptr || !bColInd.host_ptr || !bVals.host_ptr || !bX.host_ptr || !bY.host_ptr ||
        (mode == RESIDENCY_GPU_EXPLICIT_COPY && (!bRowOff.dev_ptr || !bColInd.dev_ptr || !bVals.dev_ptr || !bX.dev_ptr || !bY.dev_ptr))) {
        snprintf(rec.status, sizeof(rec.status), "oom-skip");
        log_bench_record(fout, &rec);
        bench_free_buf(&bRowOff); bench_free_buf(&bColInd); bench_free_buf(&bVals);
        bench_free_buf(&bX); bench_free_buf(&bY);
        return 0;
    }

    /* Initialize CSR matrix and vectors on CPU */
    int *h_rowOff = (int *)bRowOff.host_ptr;
    int *h_colInd = (int *)bColInd.host_ptr;
    float *h_vals = (float *)bVals.host_ptr;
    float *h_x    = (float *)bX.host_ptr;
    float *h_y    = (float *)bY.host_ptr;

    long stride = S / npr; if (stride < 1) stride = 1;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < S; r++) {
        h_rowOff[r] = r * npr;
        for (int jj = 0; jj < npr; jj++) {
            uint64_t base = ((uint64_t)r * 1469598103934665603ULL) ^ ((uint64_t)jj * 1099511628211ULL);
            uint64_t x = gemv_hashu_seed(base, 42);
            long jit = stride > 1 ? (long)(x % (uint64_t)stride) : 0;
            long col = (long)jj * stride + jit;
            if (col >= S) col = S - 1;
            size_t idx = (size_t)r * (size_t)npr + (size_t)jj;
            h_colInd[idx] = (int)col;
            h_vals[idx] = 1.0f;
        }
        h_x[r] = 1.0f;
        h_y[r] = 0.0f;
    }
    h_rowOff[S] = (int)nnz;

    if (mode == RESIDENCY_CPU_LOCAL) {
        /* Cold iteration */
        double tc0 = bench_get_time_ms();
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < S; i++) {
            float sum = 0.0f;
            int start = h_rowOff[i];
            int end = h_rowOff[i + 1];
            for (int j = start; j < end; j++) {
                sum += h_vals[j] * h_x[h_colInd[j]];
            }
            h_y[i] = sum;
        }
        rec.t_kernel_cold_ms = bench_get_time_ms() - tc0;

        for (int w = 0; w < warmup; w++) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < S; i++) {
                float sum = 0.0f;
                int start = h_rowOff[i];
                int end = h_rowOff[i + 1];
                for (int j = start; j < end; j++) sum += h_vals[j] * h_x[h_colInd[j]];
                h_y[i] = sum;
            }
        }

        double tw0 = bench_get_time_ms();
        for (int it = 0; it < iters; it++) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < S; i++) {
                float sum = 0.0f;
                int start = h_rowOff[i];
                int end = h_rowOff[i + 1];
                for (int j = start; j < end; j++) sum += h_vals[j] * h_x[h_colInd[j]];
                h_y[i] = sum;
            }
        }
        rec.t_kernel_warm_ms = (bench_get_time_ms() - tw0) / (double)iters;
    } else {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cusparseHandle_t handle;
        cusparseCreate(&handle);
        cusparseSetStream(handle, stream);

        if (mode == RESIDENCY_GPU_EXPLICIT_COPY) {
            double th0 = bench_get_time_ms();
            cudaMemcpyAsync(bRowOff.dev_ptr, bRowOff.host_ptr, bytes_rowOff, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(bColInd.dev_ptr, bColInd.host_ptr, bytes_colInd, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(bVals.dev_ptr,   bVals.host_ptr,   bytes_vals,   cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(bX.dev_ptr,      bX.host_ptr,      bytes_x,      cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
            rec.t_h2d_ms = bench_get_time_ms() - th0;
        }

        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t vecX, vecY;
        CHECK_CUSPARSE(cusparseCreateCsr(&matA, S, S, nnz,
                                         bRowOff.dev_ptr, bColInd.dev_ptr, bVals.dev_ptr,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, S, bX.dev_ptr, CUDA_R_32F));
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, S, bY.dev_ptr, CUDA_R_32F));

        float alpha = 1.0f, beta = 0.0f;
        size_t bufferSize = 0;
        CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                               CUSPARSE_SPMV_CSR_ALG2, &bufferSize));
        void *dBuffer = NULL;
        cudaMalloc(&dBuffer, bufferSize);

        /* Cold iteration */
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        cudaEventRecord(ev0, stream);
        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                    CUSPARSE_SPMV_CSR_ALG2, dBuffer));
        cudaEventRecord(ev1, stream);
        cudaStreamSynchronize(stream);
        float ms_cold = 0;
        cudaEventElapsedTime(&ms_cold, ev0, ev1);
        rec.t_kernel_cold_ms = ms_cold;

        for (int w = 0; w < warmup; w++) {
            cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                         CUSPARSE_SPMV_CSR_ALG2, dBuffer);
        }
        cudaStreamSynchronize(stream);

        cudaEventRecord(ev0, stream);
        for (int it = 0; it < iters; it++) {
            cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                         CUSPARSE_SPMV_CSR_ALG2, dBuffer);
        }
        cudaEventRecord(ev1, stream);
        cudaStreamSynchronize(stream);
        float ms_warm = 0;
        cudaEventElapsedTime(&ms_warm, ev0, ev1);
        rec.t_kernel_warm_ms = ms_warm / (double)iters;

        if (mode == RESIDENCY_GPU_EXPLICIT_COPY) {
            double td0 = bench_get_time_ms();
            cudaMemcpyAsync(bY.host_ptr, bY.dev_ptr, bytes_y, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            rec.t_d2h_ms = bench_get_time_ms() - td0;
        }

        cudaFree(dBuffer);
        cusparseDestroySpMat(matA);
        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecY);
        cusparseDestroy(handle);
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        cudaStreamDestroy(stream);
    }

    rec.t_e2e_ms = rec.t_h2d_ms + rec.t_kernel_warm_ms + rec.t_d2h_ms;
    rec.eff_bw_gbps = (double)rec.working_set_bytes / (rec.t_e2e_ms * 1e6);
    rec.c2c_bw_gbps = (mode == RESIDENCY_GPU_EXPLICIT_COPY) ? ((bytes_rowOff + bytes_colInd + bytes_vals + bytes_x + bytes_y) / ((rec.t_h2d_ms + rec.t_d2h_ms) * 1e6)) :
                      (mode == RESIDENCY_GPU_ZERO_COPY) ? rec.eff_bw_gbps : 0.0;
    rec.gflops = (flops / (rec.t_kernel_warm_ms * 1e6));
    rec.pct_peak = (flops / (rec.t_kernel_warm_ms * 1e-3)) / peak_flops * 100.0;
    snprintf(rec.status, sizeof(rec.status), "ok");

    log_bench_record(fout, &rec);

    printf("SpMV  [%-14s] %-20s : e2e=%7.2fms (kern=%7.2fms) | %7.1f GB/s | %7.1f GFLOPS (%5.1f%% pk)\n",
           residency_to_str(mode), rec.problem_size, rec.t_e2e_ms, rec.t_kernel_warm_ms, rec.eff_bw_gbps, rec.gflops, rec.pct_peak);

    bench_free_buf(&bRowOff); bench_free_buf(&bColInd); bench_free_buf(&bVals);
    bench_free_buf(&bX); bench_free_buf(&bY);
    return 0;
}

/* =========================================================================================
 * 2. Thrust / Binary Search (Regime 3 - Irregular Key Traversal)
 * ========================================================================================= */
__global__ void binary_search_kernel(int64_t *results, const int64_t *queries, const int64_t *keys, long N_keys, int M_queries) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < M_queries) {
        int64_t q = queries[idx];
        long left = 0;
        long right = N_keys;
        while (left < right) {
            long mid = left + (right - left) / 2;
            if (keys[mid] < q) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        results[idx] = left;
    }
}
int run_thrust_search_bench(FILE *fout, residency_mode_t mode, long N_keys, int M_queries, int iters, int warmup) {
    bench_record_t rec;
    memset(&rec, 0, sizeof(rec));
    snprintf(rec.workload, sizeof(rec.workload), "thrust_search");
    rec.regime = 3;
    snprintf(rec.precision, sizeof(rec.precision), "int64");
    rec.residency = mode;
    snprintf(rec.problem_size, sizeof(rec.problem_size), "N=%ld,M=%d", N_keys, M_queries);

    size_t bytes_keys    = (size_t)N_keys * sizeof(int64_t);
    size_t bytes_queries = (size_t)M_queries * sizeof(int64_t);
    size_t bytes_results = (size_t)M_queries * sizeof(int64_t);
    rec.working_set_bytes = bytes_keys + bytes_queries + bytes_results;

    gpu_spec_t spec = get_gpu_spec_for_device(0);
    rec.threads_or_sms = (mode == RESIDENCY_CPU_LOCAL) ? 72 : 132;
    double lookups = (double)M_queries * log2((double)N_keys);

    double t0_alloc = bench_get_time_ms();
    bench_buf_t bKeys    = bench_alloc_buf(bytes_keys, mode);
    bench_buf_t bQueries = bench_alloc_buf(bytes_queries, mode);
    bench_buf_t bResults = bench_alloc_buf(bytes_results, mode);
    rec.t_alloc_ms = bench_get_time_ms() - t0_alloc;

    if (!bKeys.host_ptr || !bQueries.host_ptr || !bResults.host_ptr ||
        (mode == RESIDENCY_GPU_EXPLICIT_COPY && (!bKeys.dev_ptr || !bQueries.dev_ptr || !bResults.dev_ptr))) {
        snprintf(rec.status, sizeof(rec.status), "oom-skip");
        log_bench_record(fout, &rec);
        bench_free_buf(&bKeys); bench_free_buf(&bQueries); bench_free_buf(&bResults);
        return 0;
    }

    int64_t *h_keys    = (int64_t *)bKeys.host_ptr;
    int64_t *h_queries = (int64_t *)bQueries.host_ptr;
    int64_t *h_results = (int64_t *)bResults.host_ptr;

    /* Fill strictly sorted 64-bit keys and random queries */
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < N_keys; i++) {
        h_keys[i] = (int64_t)i * 2;
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M_queries; i++) {
        uint64_t x = gemv_hashu_seed((uint64_t)i, 1337);
        h_queries[i] = (int64_t)(x % ((uint64_t)N_keys * 2ULL));
        h_results[i] = 0;
    }

    if (mode == RESIDENCY_CPU_LOCAL) {
        double tc0 = bench_get_time_ms();
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < M_queries; i++) {
            const int64_t *pos = std::lower_bound(h_keys, h_keys + N_keys, h_queries[i]);
            h_results[i] = (int64_t)(pos - h_keys);
        }
        rec.t_kernel_cold_ms = bench_get_time_ms() - tc0;

        for (int w = 0; w < warmup; w++) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < M_queries; i++) {
                const int64_t *pos = std::lower_bound(h_keys, h_keys + N_keys, h_queries[i]);
                h_results[i] = (int64_t)(pos - h_keys);
            }
        }

        double tw0 = bench_get_time_ms();
        for (int it = 0; it < iters; it++) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < M_queries; i++) {
                const int64_t *pos = std::lower_bound(h_keys, h_keys + N_keys, h_queries[i]);
                h_results[i] = (int64_t)(pos - h_keys);
            }
        }
        rec.t_kernel_warm_ms = (bench_get_time_ms() - tw0) / (double)iters;
    } else {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        if (mode == RESIDENCY_GPU_EXPLICIT_COPY) {
            double th0 = bench_get_time_ms();
            cudaMemcpyAsync(bKeys.dev_ptr,    bKeys.host_ptr,    bytes_keys,    cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(bQueries.dev_ptr, bQueries.host_ptr, bytes_queries, cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
            rec.t_h2d_ms = bench_get_time_ms() - th0;
        }

        int threads = 256;
        int blocks = (M_queries + threads - 1) / threads;
        const int64_t *d_keys = (const int64_t *)bKeys.dev_ptr;
        const int64_t *d_queries = (const int64_t *)bQueries.dev_ptr;
        int64_t *d_results = (int64_t *)bResults.dev_ptr;

        /* Cold iteration */
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        cudaEventRecord(ev0, stream);
        binary_search_kernel<<<blocks, threads, 0, stream>>>(d_results, d_queries, d_keys, N_keys, M_queries);
        cudaEventRecord(ev1, stream);
        cudaStreamSynchronize(stream);
        float ms_cold = 0;
        cudaEventElapsedTime(&ms_cold, ev0, ev1);
        rec.t_kernel_cold_ms = ms_cold;

        for (int w = 0; w < warmup; w++) {
            binary_search_kernel<<<blocks, threads, 0, stream>>>(d_results, d_queries, d_keys, N_keys, M_queries);
        }
        cudaStreamSynchronize(stream);

        cudaEventRecord(ev0, stream);
        for (int it = 0; it < iters; it++) {
            binary_search_kernel<<<blocks, threads, 0, stream>>>(d_results, d_queries, d_keys, N_keys, M_queries);
        }
        cudaEventRecord(ev1, stream);
        cudaStreamSynchronize(stream);
        float ms_warm = 0;
        cudaEventElapsedTime(&ms_warm, ev0, ev1);
        rec.t_kernel_warm_ms = ms_warm / (double)iters;

        if (mode == RESIDENCY_GPU_EXPLICIT_COPY) {
            double td0 = bench_get_time_ms();
            cudaMemcpyAsync(bResults.host_ptr, bResults.dev_ptr, bytes_results, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            rec.t_d2h_ms = bench_get_time_ms() - td0;
        }

        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        cudaStreamDestroy(stream);
    }

    double logN = log2((double)N_keys);
    if (logN < 1.0) logN = 1.0;
    size_t steps_per_query = (size_t)ceil(logN);
    size_t bytes_accessed = (size_t)M_queries * steps_per_query * 64ULL;

    if (mode == RESIDENCY_GPU_UM_MIGRATE) {
        rec.t_e2e_ms = rec.t_kernel_cold_ms + rec.t_kernel_warm_ms;
        rec.c2c_bw_gbps = (rec.t_kernel_cold_ms > 0) ? ((double)bytes_accessed / (rec.t_kernel_cold_ms * 1e6)) : 0.0;
    } else if (mode == RESIDENCY_GPU_ZERO_COPY) {
        rec.t_e2e_ms = rec.t_kernel_warm_ms;
        rec.c2c_bw_gbps = (rec.t_kernel_warm_ms > 0) ? ((double)bytes_accessed / (rec.t_kernel_warm_ms * 1e6)) : 0.0;
    } else {
        rec.t_e2e_ms = rec.t_h2d_ms + rec.t_kernel_warm_ms + rec.t_d2h_ms;
        rec.c2c_bw_gbps = (mode == RESIDENCY_GPU_EXPLICIT_COPY) ? ((bytes_keys + bytes_queries + bytes_results) / ((rec.t_h2d_ms + rec.t_d2h_ms) * 1e6 + 1e-9)) : 0.0;
    }
    rec.eff_bw_gbps = (double)bytes_accessed / (rec.t_e2e_ms * 1e6);
    rec.gflops = (lookups / (rec.t_kernel_warm_ms * 1e-3)) / 1e6; /* MLookups/s */
    snprintf(rec.status, sizeof(rec.status), "ok");

    log_bench_record(fout, &rec);

    printf("Search[%-14s] %-20s : e2e=%7.2fms (kern=%7.2fms) | %7.1f GB/s | %7.1f MLookups/s\n",
           residency_to_str(mode), rec.problem_size, rec.t_e2e_ms, rec.t_kernel_warm_ms, rec.eff_bw_gbps, rec.gflops);

    bench_free_buf(&bKeys); bench_free_buf(&bQueries); bench_free_buf(&bResults);
    return 0;
}

/* =========================================================================================
 * 3. cuSPARSE SpGEMM (Sparse Matrix-Matrix Multiplication: C = A * B)
 * ========================================================================================= */
int run_spgemm_bench(FILE *fout, residency_mode_t mode, int S, double density, int iters, int warmup) {
    bench_record_t rec;
    memset(&rec, 0, sizeof(rec));
    snprintf(rec.workload, sizeof(rec.workload), "spgemm");
    rec.regime = 3;
    snprintf(rec.precision, sizeof(rec.precision), "fp32");
    rec.residency = mode;
    snprintf(rec.problem_size, sizeof(rec.problem_size), "S=%d,density=%.4f", S, density);

    int npr = (int)llround(density * (double)S);
    if (npr < 1) npr = 1;
    if (npr > S) npr = S;
    long nnz = (long)npr * S;

    size_t bytes_rowOff = (S + 1) * sizeof(int);
    size_t bytes_colInd = nnz * sizeof(int);
    size_t bytes_vals   = nnz * sizeof(float);
    rec.working_set_bytes = (bytes_rowOff + bytes_colInd + bytes_vals) * 2; /* for A and B */

    gpu_spec_t spec = get_gpu_spec_for_device(0);
    rec.threads_or_sms = (mode == RESIDENCY_CPU_LOCAL) ? 72 : 132;
    double flops = 2.0 * density * density * (double)S * (double)S * (double)S;

    double t0_alloc = bench_get_time_ms();
    bench_buf_t bRowOffA = bench_alloc_buf(bytes_rowOff, mode);
    bench_buf_t bColIndA = bench_alloc_buf(bytes_colInd, mode);
    bench_buf_t bValsA   = bench_alloc_buf(bytes_vals, mode);
    bench_buf_t bRowOffB = bench_alloc_buf(bytes_rowOff, mode);
    bench_buf_t bColIndB = bench_alloc_buf(bytes_colInd, mode);
    bench_buf_t bValsB   = bench_alloc_buf(bytes_vals, mode);
    rec.t_alloc_ms = bench_get_time_ms() - t0_alloc;

    if (!bRowOffA.host_ptr || !bColIndA.host_ptr || !bValsA.host_ptr ||
        !bRowOffB.host_ptr || !bColIndB.host_ptr || !bValsB.host_ptr ||
        (mode == RESIDENCY_GPU_EXPLICIT_COPY && (!bRowOffA.dev_ptr || !bColIndA.dev_ptr || !bValsA.dev_ptr ||
                                                !bRowOffB.dev_ptr || !bColIndB.dev_ptr || !bValsB.dev_ptr))) {
        snprintf(rec.status, sizeof(rec.status), "oom-skip");
        log_bench_record(fout, &rec);
        bench_free_buf(&bRowOffA); bench_free_buf(&bColIndA); bench_free_buf(&bValsA);
        bench_free_buf(&bRowOffB); bench_free_buf(&bColIndB); bench_free_buf(&bValsB);
        return 0;
    }

    int *h_rowOffA = (int *)bRowOffA.host_ptr;
    int *h_colIndA = (int *)bColIndA.host_ptr;
    float *h_valsA = (float *)bValsA.host_ptr;
    int *h_rowOffB = (int *)bRowOffB.host_ptr;
    int *h_colIndB = (int *)bColIndB.host_ptr;
    float *h_valsB = (float *)bValsB.host_ptr;

    long stride = S / npr; if (stride < 1) stride = 1;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < S; r++) {
        h_rowOffA[r] = r * npr;
        h_rowOffB[r] = r * npr;
        for (int jj = 0; jj < npr; jj++) {
            uint64_t baseA = ((uint64_t)r * 1469598103934665603ULL) ^ ((uint64_t)jj * 1099511628211ULL);
            uint64_t xA = gemv_hashu_seed(baseA, 101);
            long jitA = stride > 1 ? (long)(xA % (uint64_t)stride) : 0;
            long colA = (long)jj * stride + jitA;
            if (colA >= S) colA = S - 1;
            size_t idxA = (size_t)r * (size_t)npr + (size_t)jj;
            h_colIndA[idxA] = (int)colA;
            h_valsA[idxA] = 1.0f;

            uint64_t baseB = ((uint64_t)r * 1099511628211ULL) ^ ((uint64_t)jj * 1469598103934665603ULL);
            uint64_t xB = gemv_hashu_seed(baseB, 202);
            long jitB = stride > 1 ? (long)(xB % (uint64_t)stride) : 0;
            long colB = (long)jj * stride + jitB;
            if (colB >= S) colB = S - 1;
            h_colIndB[idxA] = (int)colB;
            h_valsB[idxA] = 1.0f;
        }

        /* Sort and deduplicate row columns to ensure cuSPARSE compliance */
        int start = r * npr;
        std::sort(&h_colIndA[start], &h_colIndA[start + npr]);
        std::sort(&h_colIndB[start], &h_colIndB[start + npr]);
        for (int jj = 1; jj < npr; jj++) {
            if (h_colIndA[start + jj] <= h_colIndA[start + jj - 1]) {
                h_colIndA[start + jj] = (h_colIndA[start + jj - 1] + 1) % S;
            }
            if (h_colIndB[start + jj] <= h_colIndB[start + jj - 1]) {
                h_colIndB[start + jj] = (h_colIndB[start + jj - 1] + 1) % S;
            }
        }
        std::sort(&h_colIndA[start], &h_colIndA[start + npr]);
        std::sort(&h_colIndB[start], &h_colIndB[start + npr]);
    }
    h_rowOffA[S] = (int)nnz;
    h_rowOffB[S] = (int)nnz;

    if (mode == RESIDENCY_CPU_LOCAL) {
        /* OpenMP Row-Accumulator CSR SpGEMM */
        auto compute_cpu_spgemm = [&]() {
            #pragma omp parallel
            {
                std::vector<float> dense_row(S, 0.0f);
                std::vector<int> nonzeros;
                nonzeros.reserve(S);

                #pragma omp for schedule(dynamic, 16)
                for (int i = 0; i < S; i++) {
                    int startA = h_rowOffA[i];
                    int endA   = h_rowOffA[i + 1];
                    for (int jA = startA; jA < endA; jA++) {
                        int k = h_colIndA[jA];
                        float valA = h_valsA[jA];
                        int startB = h_rowOffB[k];
                        int endB   = h_rowOffB[k + 1];
                        for (int jB = startB; jB < endB; jB++) {
                            int j = h_colIndB[jB];
                            if (dense_row[j] == 0.0f) {
                                nonzeros.push_back(j);
                            }
                            dense_row[j] += valA * h_valsB[jB];
                        }
                    }
                    /* Reset row accumulator */
                    for (int col : nonzeros) {
                        dense_row[col] = 0.0f;
                    }
                    nonzeros.clear();
                }
            }
        };

        double tc0 = bench_get_time_ms();
        compute_cpu_spgemm();
        rec.t_kernel_cold_ms = bench_get_time_ms() - tc0;

        for (int w = 0; w < warmup; w++) compute_cpu_spgemm();

        double tw0 = bench_get_time_ms();
        for (int it = 0; it < iters; it++) compute_cpu_spgemm();
        rec.t_kernel_warm_ms = (bench_get_time_ms() - tw0) / (double)iters;
    } else {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cusparseHandle_t handle;
        cusparseCreate(&handle);
        cusparseSetStream(handle, stream);

        if (mode == RESIDENCY_GPU_EXPLICIT_COPY) {
            double th0 = bench_get_time_ms();
            cudaMemcpyAsync(bRowOffA.dev_ptr, bRowOffA.host_ptr, bytes_rowOff, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(bColIndA.dev_ptr, bColIndA.host_ptr, bytes_colInd, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(bValsA.dev_ptr,   bValsA.host_ptr,   bytes_vals,   cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(bRowOffB.dev_ptr, bRowOffB.host_ptr, bytes_rowOff, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(bColIndB.dev_ptr, bColIndB.host_ptr, bytes_colInd, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(bValsB.dev_ptr,   bValsB.host_ptr,   bytes_vals,   cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
            rec.t_h2d_ms = bench_get_time_ms() - th0;
        }

        cusparseSpMatDescr_t matA = NULL, matB = NULL, matC = NULL;
        cusparseSpGEMMDescr_t spgemmDesc = NULL;
        bench_buf_t bRowOffC = {NULL, NULL, 0, mode};
        bench_buf_t bColIndC = {NULL, NULL, 0, mode};
        bench_buf_t bValsC   = {NULL, NULL, 0, mode};
        void *dBuffer1 = NULL, *dBuffer2 = NULL;
        cudaEvent_t ev0 = NULL, ev1 = NULL;
        int64_t rowsC = 0, colsC = 0, nnzC = 0;
        float ms_cold = 0.0f, ms_warm = 0.0f;
        float alpha = 1.0f, beta = 0.0f;
        size_t bufferSize1 = 0, bufferSize2 = 0;
        bool spgemm_ok = true;

        #define SAFE_CUSPARSE(call) do { \
            cusparseStatus_t st = (call); \
            if (st != CUSPARSE_STATUS_SUCCESS) { \
                fprintf(stderr, "cuSPARSE notice: status=%d at %s:%d\n", (int)st, __FILE__, __LINE__); \
                spgemm_ok = false; \
                goto spgemm_cleanup; \
            } \
        } while (0)

        #define SAFE_CUDA(call) do { \
            cudaError_t er = (call); \
            if (er != cudaSuccess) { \
                fprintf(stderr, "CUDA notice: %s at %s:%d\n", cudaGetErrorString(er), __FILE__, __LINE__); \
                cudaGetLastError(); \
                spgemm_ok = false; \
                goto spgemm_cleanup; \
            } \
        } while (0)

        SAFE_CUSPARSE(cusparseCreateCsr(&matA, S, S, nnz,
                                        bRowOffA.dev_ptr, bColIndA.dev_ptr, bValsA.dev_ptr,
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
        SAFE_CUSPARSE(cusparseCreateCsr(&matB, S, S, nnz,
                                        bRowOffB.dev_ptr, bColIndB.dev_ptr, bValsB.dev_ptr,
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
        
        /* C matrix descriptors */
        bRowOffC = bench_alloc_buf(bytes_rowOff, mode);
        if (!bRowOffC.host_ptr || (mode == RESIDENCY_GPU_EXPLICIT_COPY && !bRowOffC.dev_ptr)) {
            spgemm_ok = false;
            goto spgemm_cleanup;
        }
        SAFE_CUSPARSE(cusparseCreateCsr(&matC, S, S, 0,
                                        bRowOffC.dev_ptr, NULL, NULL,
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

        SAFE_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc));

        /* Step 1: Work estimation (symbolic) */
        SAFE_CUSPARSE(cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                     &alpha, matA, matB, &beta, matC,
                                                     CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                                     spgemmDesc, &bufferSize1, NULL));
        SAFE_CUDA(cudaMalloc(&dBuffer1, bufferSize1 ? bufferSize1 : 64));
        SAFE_CUSPARSE(cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                     &alpha, matA, matB, &beta, matC,
                                                     CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                                     spgemmDesc, &bufferSize1, dBuffer1));

        /* Step 2: Compute buffers */
        SAFE_CUSPARSE(cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              &alpha, matA, matB, &beta, matC,
                                              CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                              spgemmDesc, &bufferSize2, NULL));
        SAFE_CUDA(cudaMalloc(&dBuffer2, bufferSize2 ? bufferSize2 : 64));
        SAFE_CUSPARSE(cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              &alpha, matA, matB, &beta, matC,
                                              CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                              spgemmDesc, &bufferSize2, dBuffer2));

        rowsC = 0; colsC = 0; nnzC = 0;
        SAFE_CUSPARSE(cusparseSpMatGetSize(matC, &rowsC, &colsC, &nnzC));

        if (nnzC < 0) {
            spgemm_ok = false;
            goto spgemm_cleanup;
        }

        {
            size_t alloc_nnz = (nnzC > 0) ? (size_t)nnzC : 1;
            bColIndC = bench_alloc_buf(alloc_nnz * sizeof(int), mode);
            bValsC   = bench_alloc_buf(alloc_nnz * sizeof(float), mode);
            if (!bColIndC.host_ptr || !bValsC.host_ptr ||
                (mode == RESIDENCY_GPU_EXPLICIT_COPY && (!bColIndC.dev_ptr || !bValsC.dev_ptr))) {
                spgemm_ok = false;
                goto spgemm_cleanup;
            }
        }
        SAFE_CUSPARSE(cusparseCsrSetPointers(matC, bRowOffC.dev_ptr, bColIndC.dev_ptr, bValsC.dev_ptr));

        /* Timed compute + copy loop */
        SAFE_CUDA(cudaEventCreate(&ev0));
        SAFE_CUDA(cudaEventCreate(&ev1));
        SAFE_CUDA(cudaEventRecord(ev0, stream));
        SAFE_CUSPARSE(cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              &alpha, matA, matB, &beta, matC,
                                              CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                              spgemmDesc, &bufferSize2, dBuffer2));
        SAFE_CUSPARSE(cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matA, matB, &beta, matC,
                                           CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDesc));
        SAFE_CUDA(cudaEventRecord(ev1, stream));
        SAFE_CUDA(cudaStreamSynchronize(stream));
        ms_cold = 0.0f;
        cudaEventElapsedTime(&ms_cold, ev0, ev1);
        rec.t_kernel_cold_ms = ms_cold;

        for (int w = 0; w < warmup; w++) {
            cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   &alpha, matA, matB, &beta, matC,
                                   CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                   spgemmDesc, &bufferSize2, dBuffer2);
            cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC,
                                CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                spgemmDesc);
        }
        cudaStreamSynchronize(stream);

        cudaEventRecord(ev0, stream);
        for (int it = 0; it < iters; it++) {
            cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   &alpha, matA, matB, &beta, matC,
                                   CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                   spgemmDesc, &bufferSize2, dBuffer2);
            cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC,
                                CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                spgemmDesc);
        }
        cudaEventRecord(ev1, stream);
        cudaStreamSynchronize(stream);
        ms_warm = 0.0f;
        cudaEventElapsedTime(&ms_warm, ev0, ev1);
        rec.t_kernel_warm_ms = ms_warm / (double)iters;

        if (mode == RESIDENCY_GPU_EXPLICIT_COPY) {
            double td0 = bench_get_time_ms();
            cudaMemcpyAsync(bRowOffC.host_ptr, bRowOffC.dev_ptr, bytes_rowOff, cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(bColIndC.host_ptr, bColIndC.dev_ptr, (size_t)nnzC * sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(bValsC.host_ptr,   bValsC.dev_ptr,   (size_t)nnzC * sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            rec.t_d2h_ms = bench_get_time_ms() - td0;
        }

spgemm_cleanup:
        #undef SAFE_CUSPARSE
        #undef SAFE_CUDA
        if (dBuffer1) cudaFree(dBuffer1);
        if (dBuffer2) cudaFree(dBuffer2);
        if (spgemmDesc) cusparseSpGEMM_destroyDescr(spgemmDesc);
        if (matA) cusparseDestroySpMat(matA);
        if (matB) cusparseDestroySpMat(matB);
        if (matC) cusparseDestroySpMat(matC);
        cusparseDestroy(handle);
        if (ev0) cudaEventDestroy(ev0);
        if (ev1) cudaEventDestroy(ev1);
        cudaStreamDestroy(stream);
        bench_free_buf(&bRowOffC); bench_free_buf(&bColIndC); bench_free_buf(&bValsC);

        if (!spgemm_ok) {
            snprintf(rec.status, sizeof(rec.status), "oom-skip");
            log_bench_record(fout, &rec);
            printf("SpGEMM [%-14s] %-20s : OOM/Skip\n", residency_to_str(mode), rec.problem_size);
            bench_free_buf(&bRowOffA); bench_free_buf(&bColIndA); bench_free_buf(&bValsA);
            bench_free_buf(&bRowOffB); bench_free_buf(&bColIndB); bench_free_buf(&bValsB);
            return 0;
        }
    }

    if (mode == RESIDENCY_GPU_UM_MIGRATE) {
        rec.t_e2e_ms = rec.t_kernel_cold_ms + rec.t_kernel_warm_ms;
        rec.c2c_bw_gbps = (rec.t_kernel_cold_ms > 0) ? ((double)rec.working_set_bytes / (rec.t_kernel_cold_ms * 1e6)) : 0.0;
    } else if (mode == RESIDENCY_GPU_ZERO_COPY) {
        rec.t_e2e_ms = rec.t_kernel_warm_ms;
        rec.c2c_bw_gbps = (rec.t_kernel_warm_ms > 0) ? ((double)rec.working_set_bytes / (rec.t_kernel_warm_ms * 1e6)) : 0.0;
    } else {
        rec.t_e2e_ms = rec.t_h2d_ms + rec.t_kernel_warm_ms + rec.t_d2h_ms;
        rec.c2c_bw_gbps = (rec.working_set_bytes / ((rec.t_h2d_ms + rec.t_d2h_ms) * 1e6 + 1e-9));
    }
    rec.eff_bw_gbps = (double)rec.working_set_bytes / (rec.t_e2e_ms * 1e6);
    rec.gflops = (flops / (rec.t_kernel_warm_ms * 1e6));
    snprintf(rec.status, sizeof(rec.status), "ok");

    log_bench_record(fout, &rec);

    printf("SpGEMM[%-14s] %-20s : e2e=%7.2fms (kern=%7.2fms) | %7.1f GB/s | %7.1f GFLOPS\n",
           residency_to_str(mode), rec.problem_size, rec.t_e2e_ms, rec.t_kernel_warm_ms, rec.eff_bw_gbps, rec.gflops);

    bench_free_buf(&bRowOffA); bench_free_buf(&bColIndA); bench_free_buf(&bValsA);
    bench_free_buf(&bRowOffB); bench_free_buf(&bColIndB); bench_free_buf(&bValsB);
    return 0;
}

int main(int argc, char **argv) {
    const char *csv_file = "residency_irregular_sweep.csv";
    int iters = 5;
    int warmup = 2;
    const char *workload = "all";
    const char *sizes_arg = NULL;
    const char *densities_arg = NULL;
    const char *modes_arg = "all";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv_file = argv[++i];
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--workload") && i + 1 < argc) workload = argv[++i];
        else if (!strcmp(argv[i], "--sizes") && i + 1 < argc) sizes_arg = argv[++i];
        else if (!strcmp(argv[i], "--densities") && i + 1 < argc) densities_arg = argv[++i];
        else if (!strcmp(argv[i], "--modes") && i + 1 < argc) modes_arg = argv[++i];
    }

    FILE *fout = fopen(csv_file, "a+");
    if (!fout) { perror("fopen csv"); return 1; }
    fseek(fout, 0, SEEK_END);
    if (ftell(fout) == 0) print_bench_csv_header(fout);

    printf("========================================================================================\n");
    printf(" NVIDIA Grace Hopper (GH200): Regime 3 Irregular & Sparse Across 4 Coherent NUMA Modes\n");
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

    double densities[8];
    int n_dens = 0;
    if (densities_arg) {
        char buf[256]; snprintf(buf, sizeof(buf), "%s", densities_arg);
        for (char *tok = strtok(buf, ","); tok && n_dens < 8; tok = strtok(NULL, ",")) {
            densities[n_dens++] = atof(tok);
        }
    } else {
        densities[0] = 0.001; densities[1] = 0.005; densities[2] = 0.01; densities[3] = 0.05;
        n_dens = 4;
    }

    /* 1. SpMV: S=1024 to 262144 */
    if (!strcmp(workload, "all") || !strcmp(workload, "spmv")) {
        int spmv_sizes[32];
        int n_spmv = 0;
        if (sizes_arg) {
            char buf[512]; snprintf(buf, sizeof(buf), "%s", sizes_arg);
            for (char *tok = strtok(buf, ","); tok && n_spmv < 32; tok = strtok(NULL, ",")) {
                spmv_sizes[n_spmv++] = atoi(tok);
            }
        } else {
            int default_spmv[] = {1024, 4096, 16384, 65536, 131072, 262144};
            n_spmv = sizeof(default_spmv) / sizeof(default_spmv[0]);
            for (int i = 0; i < n_spmv; i++) spmv_sizes[i] = default_spmv[i];
        }

        printf("── [Part 1] cuSPARSE SpMV (Sparse Matrix-Vector Multiply, Regime 3) ──────────────────\n");
        for (int d = 0; d < n_dens; d++) {
            double dens = densities[d];
            for (int s = 0; s < n_spmv; s++) {
                int S = spmv_sizes[s];
                for (int m = 0; m < n_modes; m++) {
                    run_spmv_bench(fout, modes[m], S, dens, iters, warmup);
                }
                printf("----------------------------------------------------------------------------------------\n");
            }
        }
    }

    /* 2. Thrust Binary Search: 1MB to 128GB */
    if (!strcmp(workload, "all") || !strcmp(workload, "thrust_search")) {
        long search_keys[] = {
            262144L,       /* 1 MB (L2 cache) */
            16777216L,     /* 64 MB (SLC cache) */
            268435456L,    /* 1 GB (HBM3) */
            4294967296L,   /* 16 GB */
            17179869184L,  /* 64 GB */
            34359738368L   /* 128 GB (oversubscribed > 96GB) */
        };
        int n_search = sizeof(search_keys) / sizeof(search_keys[0]);

        printf("\n── [Part 2] Thrust Binary Search (Regime 3 Irregular Address Traversal) ───────────────\n");
        for (int s = 0; s < n_search; s++) {
            long N = search_keys[s];
            int M = 4000000;
            for (int m = 0; m < n_modes; m++) {
                run_thrust_search_bench(fout, modes[m], N, M, iters, warmup);
            }
            printf("----------------------------------------------------------------------------------------\n");
        }
    }

    /* 3. cuSPARSE SpGEMM: S=1024, 2048, 4096, 8192 across densities */
    if (!strcmp(workload, "all") || !strcmp(workload, "spgemm")) {
        int spgemm_sizes[32];
        int n_spgemm = 0;
        if (sizes_arg) {
            char buf[512]; snprintf(buf, sizeof(buf), "%s", sizes_arg);
            for (char *tok = strtok(buf, ","); tok && n_spgemm < 32; tok = strtok(NULL, ",")) {
                spgemm_sizes[n_spgemm++] = atoi(tok);
            }
        } else {
            int default_spgemm[] = {1024, 2048, 4096, 8192};
            n_spgemm = sizeof(default_spgemm) / sizeof(default_spgemm[0]);
            for (int i = 0; i < n_spgemm; i++) spgemm_sizes[i] = default_spgemm[i];
        }

        printf("\n── [Part 3] cuSPARSE SpGEMM (Sparse Matrix-Matrix Multiply, Fixed Buffers) ───────────\n");
        for (int d = 0; d < n_dens; d++) {
            double dens = densities[d];
            for (int s = 0; s < n_spgemm; s++) {
                int S = spgemm_sizes[s];
                for (int m = 0; m < n_modes; m++) {
                    run_spgemm_bench(fout, modes[m], S, dens, iters, warmup);
                }
                printf("----------------------------------------------------------------------------------------\n");
            }
        }
    }

    fclose(fout);
    printf("\nIrregular benchmark complete! Results recorded to %s\n", csv_file);
    return 0;
}
