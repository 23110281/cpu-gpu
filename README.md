# gpu_bench — Multi-GPU cuBLAS GEMM Benchmark

Multi-GPU GEMM benchmark harness for A100 GPUs (sweeps matrix size × precision × GPU count). Measures achieved TFLOPS, live hardware telemetry (NVML), and PCIe transfer bandwidth.

## Build & Run

Ensure the NVIDIA driver and CUDA toolkit are loaded, then build:

```bash
# Default (CUDA C++17 build via nvcc, pinned memory, CUDA event timers)
make                               # → bin/gpu_gemm_bench
make run                           # smoke test (1 GPU, validated)
./scripts/run_sweep.sh             # full sweep → results/sweep_<stamp>.csv

# Fallback (C11 build via gcc, pinned memory, host-clock timers)
make bench_gcc                     # → bin/gpu_gemm_bench_gcc
make run_gcc                       # smoke test (1 GPU, validated)
./scripts/run_sweep_gcc.sh         # full sweep → results/sweep_gcc_<stamp>.csv

# Clean
make clean                         # removes bin/
make both                          # builds both versions
```

## Precision Routing

Precision conversion (FP32 to BF16/FP64) is handled entirely **host-side (CPU)** prior to transfer. Timed H2D copies measure the transfer of native-size data over PCIe.

| Label | Precision | Default Execution Unit | DMMA Opt-in (`--fp64-tensor`) |
|---|---|---|---|
| `fp32` | FP32 | **CUDA Cores** (TF32 disabled) | N/A |
| `fp64` | FP64 | **CUDA Cores** (cuBLASLt non-tensor filter) | **DMMA Tensor Cores** (explicit routing) |
| `bf16` | BF16 | **Tensor Cores** (FP32 accumulation) | N/A |

## CLI Options

Common CLI options for `bin/gpu_gemm_bench`:
* `--sizes M,N,K` or `S` : Matrix dimensions (default: 256 to 65536)
* `--precisions prec` : `fp32`, `fp64`, `bf16` (comma-separated list)
* `--gpus G` : Comma-separated list of active GPU IDs
* `--mode split|replicas` : Column-split distribution vs full replica per GPU
* `--engine dense|sparse` : cuBLAS dense GEMM vs cuSPARSE SpMM
* `--density d` : Nonzero density (for sparse engine sweeps)
* `--validate` : Verify numerical correctness against CPU reference on sample coordinates

## Profiling & Clock Control

```bash
# Profile with Nsight Compute (checks FMA vs DMMA cycles)
ncu --metrics sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active ./bin/gpu_gemm_bench --sizes 8192 --precisions fp64 --gpus 1 --no-nvml

# Capture multi-GPU timeline with Nsight Systems
./scripts/profile_nsys.sh 16384 bf16 4

# Pin/unlock GPU clocks for reproducible benchmarking
./scripts/lock_clocks.sh lock
./scripts/lock_clocks.sh unlock
```

## Directory Structure

* `src/gpu_gemm_bench.cu` : CUDA C++17 version (default)
* `src/gpu_gemm_bench.c` : Legacy C11 version (fallback)
* `src/bf16_cvt.h` : Host-side FP32↔BF16 conversion
* `scripts/run_sweep.sh` : Default CUDA benchmark sweep execution
* `scripts/run_sweep_gcc.sh` : Legacy C benchmark sweep execution
* `Makefile` : Self-contained build script
