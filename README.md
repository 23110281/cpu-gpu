# gpu_bench — Multi-GPU Benchmarks (A100 & L40S)

Four standalone harnesses for benchmarking GPUs (tested heavily on NVIDIA A100 and L40S), sweeping size × precision × GPU count with NVML telemetry and PCIe transfer timing:

* **`gpu_gemm_bench`** (Dense GEMM): Matrix-matrix (`C = A*B`). Compute-bound at scale → headline metric is achieved **TFLOPS**.
* **`gpu_spgemm_bench`** (Sparse GEMM): Sparse matrix-matrix (`C = A*B`). Uses cuSPARSE. Headline metric is **TFLOPS**.
* **`gpu_gemv_bench`** (GEMV/SpMV): Matrix-vector (`y = A*x`). Constant arithmetic intensity → **bandwidth-bound** → headline metric is achieved **GB/s** (memory-bus peak).
* **`gpu_saxpy_bench`** (SAXPY): Vector addition (`y = a*x + y`). **PCIe / Memory bandwidth-bound** → headline metric is achieved **GB/s**.

## Build & Run

| Workload | Build | Smoke test | Full sweep | Plots |
|---|---|---|---|---|
| **Dense GEMM** | `make` | `make run` | `./scripts/run_sweep.sh` | `python3 results/generate_plots.py` |
| **SpGEMM** | `make spgemm` | `make run_spgemm` | `./scripts/run_sweep_spgemm.sh` | `python3 results/generate_spgemm_plots.py` |
| **GEMV** | `make gemv` | `make run_gemv` | `./scripts/run_sweep_gemv.sh` | `python3 results/generate_gemv_plots.py` |
| **SAXPY** | `make saxpy` | `make run_saxpy` | `./scripts/run_sweep_saxpy.sh` | `python3 results/generate_saxpy_plots.py` |

**Note**: Sweeps dynamically query the GPU name (via `cudaGetDeviceProperties` and `nvidia-smi`) and output CSVs to `results/data/<GPU_NAME>/`. Plot scripts process these directories and output aesthetic, data-to-viz compliant charts to `results/plots/<GPU_NAME>/`.

`make clean` removes the `bin/` directory. 
Plot scripts require `pandas`, `matplotlib`, `seaborn`, `squarify`, and `scipy` (e.g., using `uv pip install -r requirements.txt`).

## CLI Options (Shared shape)

* `--sizes ...` : GEMM/SpGEMM/GEMV = `S` (NxN or SxS matrices); SAXPY = `N` (Vector size).
* `--precisions fp32,fp64,bf16` : Precision conversion is host-side, pre-transfer. (Note: GEMV bf16 has no native cuBLAS Level-2 gemv, so it's routed via `cublasGemmEx` n=1).
* `--gpus G,..` : Active GPU IDs (e.g. `1,2,4`).
* `--mode split|replicas` : Row/column split across GPUs vs full replica per GPU.
* `--density d,..` : Matrix density for SpGEMM and Sparse GEMV (e.g. `0.01,0.1`).
* `--engine dense|sparse` : Engine routing (e.g. Dense vs cuSPARSELt 2:4 structured sparsity).
* `--validate`, `--iters`, `--warmup`, `--no-nvml`, `--no-transfers`, `--csv`, `--tag`, `--help`

## Profiling & Clock Control

```bash
# Profile with Nsight Compute (checks FMA vs DMMA cycles) — GEMM only
ncu --metrics sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active ./bin/gpu_gemm_bench --sizes 8192 --precisions fp64 --gpus 1 --no-nvml

# Pin/unlock GPU clocks for reproducible benchmarking
./scripts/lock_clocks.sh lock
./scripts/lock_clocks.sh unlock
```

## Directory Structure

* `src/` : CUDA C++17 source files for the benchmarks (`gpu_gemm_bench.cu`, `gpu_spgemm_bench.cu`, `gpu_gemv_bench.cu`, `gpu_saxpy_bench.cu`).
* `scripts/` : Bash drivers to automate execution sweeps (`run_sweep*.sh`) and clock locking.
* `results/data/<GPU_NAME>/` : CSV output directories grouped by hardware environment.
* `results/plots/<GPU_NAME>/` : Visualization output directories.
* `results/generate_*_plots.py` : Python plotting scripts.
* `Makefile` : Centralized build system.
