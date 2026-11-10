#ifndef GPU_DEVICE_SPECS_H
#define GPU_DEVICE_SPECS_H

#include <cuda_runtime.h>
#include <string.h>

typedef struct {
    char name[128];
    double peak_hbm_gbps;
    double peak_fp32_tflops;
    double peak_fp64_cuda_tflops;
    double peak_fp64_tensor_tflops;
    double peak_bf16_tflops;
    double peak_fp16_tflops;
    double peak_tf32_tflops;
    double pcie_or_nvlink_gbps;
    const char *interconnect_name;
} gpu_spec_t;

static inline gpu_spec_t get_gpu_spec_for_device(int dev_id) {
    gpu_spec_t s;
    memset(&s, 0, sizeof(s));
    s.peak_hbm_gbps = 1555.0;
    s.peak_fp32_tflops = 19.5;
    s.peak_fp64_cuda_tflops = 9.7;
    s.peak_fp64_tensor_tflops = 19.5;
    s.peak_bf16_tflops = 312.0;
    s.peak_fp16_tflops = 312.0;
    s.peak_tf32_tflops = 156.0;
    s.pcie_or_nvlink_gbps = 24.0;
    s.interconnect_name = "PCIe Gen4";

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev_id) != cudaSuccess) {
        strcpy(s.name, "Default_GPU");
        return s;
    }
    strncpy(s.name, prop.name, sizeof(s.name) - 1);

    // GH200 / Hopper Superchip (GH100 GPU + NVLink-C2C)
    if (strstr(prop.name, "GH200") != NULL) {
        s.peak_hbm_gbps = 4000.0;
        s.peak_fp32_tflops = 67.0;
        s.peak_fp64_cuda_tflops = 34.0;
        s.peak_fp64_tensor_tflops = 67.0;
        s.peak_bf16_tflops = 989.0;
        s.peak_fp16_tflops = 989.0;
        s.peak_tf32_tflops = 494.5;
        s.pcie_or_nvlink_gbps = 900.0;
        s.interconnect_name = "NVLink-C2C";
        return s;
    }
    // H200 SXM
    if (strstr(prop.name, "H200") != NULL) {
        s.peak_hbm_gbps = 4800.0;
        s.peak_fp32_tflops = 67.0;
        s.peak_fp64_cuda_tflops = 34.0;
        s.peak_fp64_tensor_tflops = 67.0;
        s.peak_bf16_tflops = 989.0;
        s.pcie_or_nvlink_gbps = 900.0;
        s.interconnect_name = "NVLink";
        return s;
    }
    // H100 SXM / PCIe
    if (strstr(prop.name, "H100") != NULL) {
        if (strstr(prop.name, "PCIe") != NULL) {
            s.peak_hbm_gbps = 2000.0;
            s.peak_fp32_tflops = 51.0;
            s.peak_fp64_cuda_tflops = 26.0;
            s.peak_fp64_tensor_tflops = 51.0;
            s.peak_bf16_tflops = 756.0;
            s.pcie_or_nvlink_gbps = 64.0;
            s.interconnect_name = "PCIe Gen5";
        } else {
            s.peak_hbm_gbps = 3350.0;
            s.peak_fp32_tflops = 67.0;
            s.peak_fp64_cuda_tflops = 34.0;
            s.peak_fp64_tensor_tflops = 67.0;
            s.peak_bf16_tflops = 989.0;
            s.pcie_or_nvlink_gbps = 900.0;
            s.interconnect_name = "NVLink";
        }
        return s;
    }
    // L40S / L40 (Ada Lovelace)
    if (strstr(prop.name, "L40S") != NULL || strstr(prop.name, "L40") != NULL) {
        s.peak_hbm_gbps = 864.0;
        s.peak_fp32_tflops = 91.6;
        s.peak_fp64_cuda_tflops = 1.4;
        s.peak_fp64_tensor_tflops = 1.4;
        s.peak_bf16_tflops = 366.0;
        s.pcie_or_nvlink_gbps = 24.0;
        s.interconnect_name = "PCIe Gen4";
        return s;
    }
    // A100
    if (strstr(prop.name, "A100") != NULL) {
        s.peak_hbm_gbps = (prop.totalGlobalMem > (70ULL << 30)) ? 2039.0 : 1555.0;
        s.peak_fp32_tflops = 19.5;
        s.peak_fp64_cuda_tflops = 9.7;
        s.peak_fp64_tensor_tflops = 19.5;
        s.peak_bf16_tflops = 312.0;
        s.pcie_or_nvlink_gbps = (strstr(prop.name, "PCIe") != NULL) ? 24.0 : 600.0;
        s.interconnect_name = (strstr(prop.name, "PCIe") != NULL) ? "PCIe Gen4" : "NVLink";
        return s;
    }

    // Architecture-level fallback:
    if (prop.major == 9) { // Hopper
        s.peak_hbm_gbps = 4000.0;
        s.peak_fp32_tflops = 67.0;
        s.peak_fp64_cuda_tflops = 34.0;
        s.peak_fp64_tensor_tflops = 67.0;
        s.peak_bf16_tflops = 989.0;
        s.pcie_or_nvlink_gbps = 900.0;
        s.interconnect_name = "NVLink-C2C";
    } else if (prop.major == 8 && prop.minor == 9) { // Ada
        s.peak_hbm_gbps = 864.0;
        s.peak_fp32_tflops = 91.6;
        s.peak_fp64_cuda_tflops = 1.4;
        s.peak_fp64_tensor_tflops = 1.4;
        s.peak_bf16_tflops = 366.0;
        s.pcie_or_nvlink_gbps = 24.0;
        s.interconnect_name = "PCIe Gen4";
    } else if (prop.major == 8 && prop.minor == 0) { // Ampere A100
        s.peak_hbm_gbps = 1555.0;
        s.peak_fp32_tflops = 19.5;
        s.peak_fp64_cuda_tflops = 9.7;
        s.peak_fp64_tensor_tflops = 19.5;
        s.peak_bf16_tflops = 312.0;
        s.pcie_or_nvlink_gbps = 24.0;
        s.interconnect_name = "PCIe Gen4";
    }

    return s;
}

#endif /* GPU_DEVICE_SPECS_H */
