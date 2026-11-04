/*
 * gpu_c2c_uvm_advise_bench.cu
 *
 * GH200 Fine-Grained UVM cudaMemAdvise & cudaMemPrefetchAsync Cliff Sweep Benchmark
 *
 * Evaluates performance around the 96GB HBM3 physical capacity cliff:
 * Working sets: 64GB, 80GB, 90GB, 94GB, 96GB, 98GB, 104GB, 112GB, 128GB, 160GB
 *
 * Workloads:
 * 1. saxpy (FP32 vector addition)
 * 2. gemv  (cuBLAS cublasSgemv_64, M=32768)
 * 3. llama3_kv_offload (BF16 decode attention streaming over K and V)
 *
 * UVM / C2C Policies:
 * 1. gpu_zero_copy:                  cudaHostAllocMapped
 * 2. uvm_demand:                     cudaMallocManaged (default demand faulting)
 * 3. uvm_prefetch_gpu:               cudaMallocManaged + cudaMemPrefetchAsync to GPU (catches OOM on >96GB)
 * 4. uvm_preferred_cpu_accessed_by_gpu: cudaMemAdviseSetPreferredLocation(CPU) + cudaMemAdviseSetAccessedBy(GPU)
 * 5. uvm_preferred_gpu_accessed_by_cpu: cudaMemAdviseSetPreferredLocation(GPU) + cudaMemAdviseSetAccessedBy(CPU)
 */

#include "bench_harness.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>

enum uvm_policy_t {
    POLICY_ZERO_COPY = 0,
    POLICY_UVM_DEMAND = 1,
    POLICY_UVM_PREFETCH_GPU = 2,
    POLICY_UVM_PREF_CPU_ACC_GPU = 3,
    POLICY_UVM_PREF_GPU_ACC_CPU = 4
};

static const char* policy_to_str(uvm_policy_t p) {
    switch (p) {
        case POLICY_ZERO_COPY: return "gpu_zero_copy";
        case POLICY_UVM_DEMAND: return "uvm_demand";
        case POLICY_UVM_PREFETCH_GPU: return "uvm_prefetch_gpu";
        case POLICY_UVM_PREF_CPU_ACC_GPU: return "uvm_pref_cpu_acc_gpu";
        case POLICY_UVM_PREF_GPU_ACC_CPU: return "uvm_pref_gpu_acc_cpu";
        default: return "unknown";
    }
}

static inline void log_uvm_record(FILE *f, const char *workload, int regime, const char *prec,
                                  const char *policy_str, size_t ws_bytes, const char *p_size,
                                  int threads_or_sms, double t_warm_ms, double t_e2e_ms,
                                  double eff_bw, double c2c_bw, double gflops, const char *status)
{
    fprintf(f, "%s,%d,%s,%s,%zu,\"%s\",%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%.1f,%s\n",
            workload, regime, prec, policy_str, ws_bytes, p_size, threads_or_sms,
            0.0, 0.0, 0.0, t_warm_ms, 0.0, t_e2e_ms,
            eff_bw, c2c_bw, gflops, 0.0, status);
    fflush(f);
}

/* Vectorized float4 SAXPY kernel */
__global__ void saxpy_vec4_kernel(const float4 * __restrict__ x, float4 * __restrict__ y, float alpha, size_t N4) {
    size_t idx = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
    size_t stride = (size_t)gridDim.x * (size_t)blockDim.x;
    for (size_t i = idx; i < N4; i += stride) {
        float4 xi = x[i];
        float4 yi = y[i];
        yi.x = alpha * xi.x + yi.x;
        yi.y = alpha * xi.y + yi.y;
        yi.z = alpha * xi.z + yi.z;
        yi.w = alpha * xi.w + yi.w;
        y[i] = yi;
    }
}

/* Llama-3 KV-Cache Attention Streaming Kernel */
__global__ void llama3_kv_stream_kernel(const float4 * __restrict__ k, const float4 * __restrict__ v, float * __restrict__ dummy_out, size_t num_vec4) {
    size_t idx = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
    size_t stride = (size_t)gridDim.x * (size_t)blockDim.x;
    float sum = 0.0f;
    for (size_t i = idx; i < num_vec4; i += stride) {
        float4 ki = k[i];
        float4 vi = v[i];
        sum += (ki.x + vi.x) + (ki.y + vi.y) + (ki.z + vi.z) + (ki.w + vi.w);
    }
    if (idx == 0 && sum > 1e30f) {
        *dummy_out = sum;
    }
}

/* Memory Allocation & Setup Helper based on Policy */
static cudaError_t alloc_buffer(void **ptr, size_t bytes, uvm_policy_t policy, cudaStream_t stream) {
    cudaError_t err = cudaSuccess;
    if (policy == POLICY_ZERO_COPY) {
        void *h_ptr = NULL;
        err = cudaHostAlloc(&h_ptr, bytes, cudaHostAllocMapped);
        if (err != cudaSuccess) return err;
        err = cudaHostGetDevicePointer(ptr, h_ptr, 0);
        return err;
    }

    /* All other policies allocate via Managed Memory */
    err = cudaMallocManaged(ptr, bytes, cudaMemAttachGlobal);
    if (err != cudaSuccess) return err;

    if (policy == POLICY_UVM_PREFETCH_GPU) {
        err = cudaMemPrefetchAsync(*ptr, bytes, 0, stream);
        if (err != cudaSuccess) {
            cudaFree(*ptr);
            *ptr = NULL;
            return err;
        }
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            cudaFree(*ptr);
            *ptr = NULL;
            return err;
        }
    } else if (policy == POLICY_UVM_PREF_CPU_ACC_GPU) {
        err = cudaMemAdvise(*ptr, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
        if (err == cudaSuccess) {
            err = cudaMemAdvise(*ptr, bytes, cudaMemAdviseSetAccessedBy, 0);
        }
    } else if (policy == POLICY_UVM_PREF_GPU_ACC_CPU) {
        err = cudaMemAdvise(*ptr, bytes, cudaMemAdviseSetPreferredLocation, 0);
        if (err == cudaSuccess) {
            err = cudaMemAdvise(*ptr, bytes, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
        }
    }
    return err;
}

static void free_buffer(void *ptr, uvm_policy_t policy) {
    if (!ptr) return;
    if (policy == POLICY_ZERO_COPY) {
        cudaFreeHost(ptr);
    } else {
        cudaFree(ptr);
    }
}

/* ── Benchmark Runner for SAXPY ─────────────────────────────────────────── */
void run_cliff_saxpy(FILE *fout, size_t ws_bytes, uvm_policy_t policy, int iters, int warmup) {
    size_t total_floats = ws_bytes / sizeof(float);
    size_t N = total_floats / 2; // X and Y vectors
    size_t vec4_N = N / 4;
    size_t actual_ws_bytes = 2 * N * sizeof(float);
    size_t traffic_bytes = 3 * N * sizeof(float);

    char p_size[128];
    snprintf(p_size, sizeof(p_size), "saxpy_cliff_WS_%zuGB_N_%zu", ws_bytes / (1024*1024*1024), N);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float4 *d_x = NULL, *d_y = NULL;
    cudaError_t err_x = alloc_buffer((void**)&d_x, N * sizeof(float), policy, stream);
    cudaError_t err_y = (err_x == cudaSuccess) ? alloc_buffer((void**)&d_y, N * sizeof(float), policy, stream) : err_x;

    if (err_x != cudaSuccess || err_y != cudaSuccess) {
        if (d_x) free_buffer(d_x, policy);
        if (d_y) free_buffer(d_y, policy);
        cudaStreamDestroy(stream);
        log_uvm_record(fout, "saxpy_uvm_cliff", 2, "fp32", policy_to_str(policy), actual_ws_bytes, p_size,
                       132, 0.0, 0.0, 0.0, 0.0, 0.0, "oom-skip");
        return;
    }

    int block = 256;
    int grid = 132 * 8; // 132 SMs

    for (int w = 0; w < warmup; w++) {
        saxpy_vec4_kernel<<<grid, block, 0, stream>>>(d_x, d_y, 2.0f, vec4_N);
    }
    cudaStreamSynchronize(stream);

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    cudaEventRecord(ev_start, stream);
    for (int i = 0; i < iters; i++) {
        saxpy_vec4_kernel<<<grid, block, 0, stream>>>(d_x, d_y, 2.0f, vec4_N);
    }
    cudaEventRecord(ev_stop, stream);
    cudaStreamSynchronize(stream);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, ev_start, ev_stop);
    double avg_kernel_ms = (double)total_ms / iters;

    double eff_bw = (traffic_bytes / (avg_kernel_ms * 1e6));
    double gflops = (2.0 * N / (avg_kernel_ms * 1e6));

    log_uvm_record(fout, "saxpy_uvm_cliff", 2, "fp32", policy_to_str(policy), actual_ws_bytes, p_size,
                   132, avg_kernel_ms, avg_kernel_ms, eff_bw, eff_bw, gflops, "ok");

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    free_buffer(d_x, policy);
    free_buffer(d_y, policy);
    cudaStreamDestroy(stream);
}

/* ── Benchmark Runner for GEMV using cuBLAS 64-bit API ──────────────────── */
void run_cliff_gemv(FILE *fout, size_t ws_bytes, uvm_policy_t policy, int iters, int warmup) {
    int64_t M = 32768;
    int64_t total_elements = ws_bytes / sizeof(float);
    int64_t N = total_elements / M;
    N = (N / 4) * 4; // Align to 4
    size_t matrix_bytes = (size_t)M * N * sizeof(float);
    size_t traffic_bytes = matrix_bytes + ((size_t)M + N) * sizeof(float);

    char p_size[128];
    snprintf(p_size, sizeof(p_size), "gemv_cliff_WS_%zuGB_M_%ld_N_%ld", ws_bytes / (1024*1024*1024), (long)M, (long)N);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        log_uvm_record(fout, "gemv_uvm_cliff", 2, "fp32", policy_to_str(policy), matrix_bytes, p_size,
                       132, 0.0, 0.0, 0.0, 0.0, 0.0, "oom-skip");
        cudaStreamDestroy(stream);
        return;
    }
    cublasSetStream(handle, stream);

    float *d_A = NULL, *d_x = NULL, *d_y = NULL;
    cudaError_t err_A = alloc_buffer((void**)&d_A, matrix_bytes, policy, stream);
    cudaError_t err_x = (err_A == cudaSuccess) ? alloc_buffer((void**)&d_x, N * sizeof(float), policy, stream) : err_A;
    cudaError_t err_y = (err_x == cudaSuccess) ? alloc_buffer((void**)&d_y, M * sizeof(float), policy, stream) : err_x;

    if (err_A != cudaSuccess || err_x != cudaSuccess || err_y != cudaSuccess) {
        if (d_A) free_buffer(d_A, policy);
        if (d_x) free_buffer(d_x, policy);
        if (d_y) free_buffer(d_y, policy);
        cublasDestroy(handle);
        cudaStreamDestroy(stream);
        log_uvm_record(fout, "gemv_uvm_cliff", 2, "fp32", policy_to_str(policy), matrix_bytes, p_size,
                       132, 0.0, 0.0, 0.0, 0.0, 0.0, "oom-skip");
        return;
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    for (int w = 0; w < warmup; w++) {
        cublasSgemv_64(handle, CUBLAS_OP_N, M, N, &alpha, d_A, M, d_x, 1, &beta, d_y, 1);
    }
    cudaStreamSynchronize(stream);

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    cudaEventRecord(ev_start, stream);
    for (int i = 0; i < iters; i++) {
        cublasSgemv_64(handle, CUBLAS_OP_N, M, N, &alpha, d_A, M, d_x, 1, &beta, d_y, 1);
    }
    cudaEventRecord(ev_stop, stream);
    cudaStreamSynchronize(stream);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, ev_start, ev_stop);
    double avg_kernel_ms = (double)total_ms / iters;

    double eff_bw = (traffic_bytes / (avg_kernel_ms * 1e6));
    double gflops = (2.0 * M * N / (avg_kernel_ms * 1e6));

    log_uvm_record(fout, "gemv_uvm_cliff", 2, "fp32", policy_to_str(policy), matrix_bytes, p_size,
                   132, avg_kernel_ms, avg_kernel_ms, eff_bw, eff_bw, gflops, "ok");

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cublasDestroy(handle);
    free_buffer(d_A, policy);
    free_buffer(d_x, policy);
    free_buffer(d_y, policy);
    cudaStreamDestroy(stream);
}

/* ── Benchmark Runner for Llama-3 KV Offload ────────────────────────────── */
void run_cliff_kv(FILE *fout, size_t ws_bytes, uvm_policy_t policy, int iters, int warmup) {
    size_t bytes_per_token = 4096; // 8 heads * 128 dim * 2 bytes * 2 (K and V)
    size_t S = ws_bytes / bytes_per_token;
    size_t num_vec4_total = ws_bytes / sizeof(float4);
    size_t half_vec4 = num_vec4_total / 2;

    char p_size[128];
    snprintf(p_size, sizeof(p_size), "llama3_kv_cliff_WS_%zuGB_S_%zu", ws_bytes / (1024*1024*1024), S);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float4 *d_k = NULL, *d_v = NULL;
    float *d_dummy = NULL;
    cudaError_t err_k = alloc_buffer((void**)&d_k, ws_bytes / 2, policy, stream);
    cudaError_t err_v = (err_k == cudaSuccess) ? alloc_buffer((void**)&d_v, ws_bytes / 2, policy, stream) : err_k;
    cudaError_t err_d = (err_v == cudaSuccess) ? cudaMalloc(&d_dummy, sizeof(float)) : err_v;

    if (err_k != cudaSuccess || err_v != cudaSuccess || err_d != cudaSuccess) {
        if (d_k) free_buffer(d_k, policy);
        if (d_v) free_buffer(d_v, policy);
        if (d_dummy) cudaFree(d_dummy);
        cudaStreamDestroy(stream);
        log_uvm_record(fout, "llama3_kv_uvm_cliff", 2, "bf16", policy_to_str(policy), ws_bytes, p_size,
                       132, 0.0, 0.0, 0.0, 0.0, 0.0, "oom-skip");
        return;
    }

    int block = 256;
    int grid = 132 * 8;

    for (int w = 0; w < warmup; w++) {
        llama3_kv_stream_kernel<<<grid, block, 0, stream>>>(d_k, d_v, d_dummy, half_vec4);
    }
    cudaStreamSynchronize(stream);

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    cudaEventRecord(ev_start, stream);
    for (int i = 0; i < iters; i++) {
        llama3_kv_stream_kernel<<<grid, block, 0, stream>>>(d_k, d_v, d_dummy, half_vec4);
    }
    cudaEventRecord(ev_stop, stream);
    cudaStreamSynchronize(stream);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, ev_start, ev_stop);
    double avg_kernel_ms = (double)total_ms / iters;

    double eff_bw = (ws_bytes / (avg_kernel_ms * 1e6));
    double gflops = (4.0 * 32 * S * 128 / (avg_kernel_ms * 1e6));

    log_uvm_record(fout, "llama3_kv_uvm_cliff", 2, "bf16", policy_to_str(policy), ws_bytes, p_size,
                   132, avg_kernel_ms, avg_kernel_ms, eff_bw, eff_bw, gflops, "ok");

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    free_buffer(d_k, policy);
    free_buffer(d_v, policy);
    cudaFree(d_dummy);
    cudaStreamDestroy(stream);
}

int main(int argc, char **argv) {
    const char *csv_path = "sweep_uvm_advise_cliff.csv";
    int iters = 5;
    int warmup = 2;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        }
    }

    FILE *fout = fopen(csv_path, "a+");
    if (!fout) {
        fprintf(stderr, "Failed to open CSV: %s\n", csv_path);
        return 1;
    }

    // Print header if file is empty
    fseek(fout, 0, SEEK_END);
    if (ftell(fout) == 0) {
        print_bench_csv_header(fout);
        fflush(fout);
    }

    printf("================================================================================\n");
    printf(" GH200 Fine-Grained UVM Advise & Prefetch 96GB Cliff Sweep\n");
    printf(" CSV Destination: %s\n", csv_path);
    printf("================================================================================\n");

    const size_t GB = 1024ULL * 1024ULL * 1024ULL;
    std::vector<size_t> cliff_sizes = {
        64 * GB, 80 * GB, 90 * GB, 94 * GB, 96 * GB,
        98 * GB, 104 * GB, 112 * GB, 128 * GB, 160 * GB
    };

    std::vector<uvm_policy_t> policies = {
        POLICY_ZERO_COPY,
        POLICY_UVM_DEMAND,
        POLICY_UVM_PREFETCH_GPU,
        POLICY_UVM_PREF_CPU_ACC_GPU,
        POLICY_UVM_PREF_GPU_ACC_CPU
    };

    for (size_t ws : cliff_sizes) {
        for (uvm_policy_t pol : policies) {
            printf("[Cliff Sweep] saxpy | %3zu GB | Policy: %-22s ... ", ws / GB, policy_to_str(pol));
            fflush(stdout);
            run_cliff_saxpy(fout, ws, pol, iters, warmup);
            printf("done\n");

            printf("[Cliff Sweep] gemv  | %3zu GB | Policy: %-22s ... ", ws / GB, policy_to_str(pol));
            fflush(stdout);
            run_cliff_gemv(fout, ws, pol, iters, warmup);
            printf("done\n");

            printf("[Cliff Sweep] kv    | %3zu GB | Policy: %-22s ... ", ws / GB, policy_to_str(pol));
            fflush(stdout);
            run_cliff_kv(fout, ws, pol, iters, warmup);
            printf("done\n");
        }
    }

    fclose(fout);
    printf("\nAll Cliff Sweeps successfully completed!\n");
    return 0;
}
