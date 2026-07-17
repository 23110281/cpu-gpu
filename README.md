# gpu_bench — Multi-GPU GEMM & GEMV/SpMV Benchmarks (A100)

Two standalone harnesses for A100 GPUs, both sweeping size × precision × GPU
count with NVML telemetry and PCIe transfer timing:

* **`gpu_gemm_bench`** — matrix-matrix (`C = A*B`). Compute-bound at scale →
  headline metric is achieved **TFLOPS**.
* **`gpu_gemv_bench`** — matrix-vector (`y = A*x`). Constant arithmetic
  intensity (`2/elemsize` FLOPs/byte) → always **bandwidth-bound** → headline
  metric is achieved **GB/s** / % of HBM peak (~1555 GB/s on A100-SXM4-40GB).
  `agg_tflops` is logged too (verified == `agg_gbps*(2/elemsize)`, i.e. a
  derived cross-reference, not an independent measurement) but tops out
  around 0.4–1.5 TFLOPS — far below GEMM's TFLOPS, because it's a different
  op class that can never become compute-bound, not a bug. **Don't compare
  the two benchmarks' "% of peak" columns directly** — one is % of compute
  peak, the other % of memory-bus peak.

## Build & Run

| | GEMM | GEMV/SpMV |
|---|---|---|
| Build | `make` → `bin/gpu_gemm_bench` | `make gemv` → `bin/gpu_gemv_bench` |
| Smoke test | `make run` | `make run_gemv` |
| Full sweep | `./scripts/run_sweep.sh` → `results/sweep_<stamp>.csv` | `./scripts/run_sweep_gemv.sh` → `results/sweep_gemv_<stamp>.csv` |
| Plots | `python3 results/generate_plots.py` → `results/plots/gemm/` | `python3 results/generate_gemv_plots.py` → `results/plots/gemv/` |
| Legacy fallback | `make bench_gcc` / `make run_gcc` / `./scripts/run_sweep_gcc.sh` (C11+gcc) | — |

`make clean` removes `bin/` for both. Plot scripts need `pandas`/`matplotlib`/`seaborn` (e.g. a local venv: `python3 -m venv .venv && .venv/bin/pip install pandas matplotlib seaborn`).

## CLI Options (shared shape, both binaries)

* `--sizes ...` : GEMM = `M,N,K` or `S`; GEMV = `S` (A is SxS)
* `--precisions fp32,fp64,bf16` : precision conversion is host-side, pre-transfer. GEMV bf16 has no native cuBLAS Level-2 gemv, so it's routed via `cublasGemmEx` (n=1)
* `--gpus G,..` : active GPU IDs
* `--mode split|replicas` : row/column split vs full replica per GPU
* `--engine dense|sparse` : GEMM sparse = cuSPARSELt (structured 2:4, fixed 50% density); GEMV sparse = plain cuSPARSE CSR SpMV (`--density d,..` is a real sweep dimension here)
* `--validate`, `--iters`, `--warmup`, `--no-nvml`, `--no-transfers`, `--mem-frac`, `--csv`, `--tag`, `--help`

## Precision Routing (GEMM)

| Label | Precision | Default Execution Unit | DMMA Opt-in (`--fp64-tensor`) |
|---|---|---|---|
| `fp32` | FP32 | **CUDA Cores** (TF32 disabled) | N/A |
| `fp64` | FP64 | **CUDA Cores** (cuBLASLt non-tensor filter) | **DMMA Tensor Cores** (explicit routing) |
| `bf16` | BF16 | **Tensor Cores** (FP32 accumulation) | N/A |

GEMV TFLOPS ceiling (bandwidth-bound, `peak_HBM_GBps * 2/elemsize`): fp32 ≈0.78, fp64 ≈0.39, bf16 ≈1.56 TFLOPS — matches ~0.72/0.37/1.43 achieved at large sizes (90%+ of peak bandwidth).

## Profiling & Clock Control

```bash
# Profile with Nsight Compute (checks FMA vs DMMA cycles) — GEMM only
ncu --metrics sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active ./bin/gpu_gemm_bench --sizes 8192 --precisions fp64 --gpus 1 --no-nvml

# Capture multi-GPU timeline with Nsight Systems — GEMM only
./scripts/profile_nsys.sh 16384 bf16 4

# Pin/unlock GPU clocks for reproducible benchmarking — either binary
./scripts/lock_clocks.sh lock
./scripts/lock_clocks.sh unlock
```

## Directory Structure

* `src/gpu_gemm_bench.cu` / `.c` : GEMM, CUDA C++17 default / C11 fallback
* `src/gpu_gemv_bench.cu` : GEMV/SpMV, CUDA C++17
* `src/gemv_host.h` : GEMV pure-C host math (CSR gen, memory footprint, CLI parsing) — CUDA-free, unit-tested (`tests/test_gemv_host.c`)
* `src/bf16_cvt.h` : shared host-side FP32↔BF16 conversion
* `scripts/run_sweep.sh` / `run_sweep_gcc.sh` / `run_sweep_gemv.sh` : sweep drivers
* `results/generate_plots.py` / `generate_gemv_plots.py` : plot generation (`tests/test_generate_gemv_plots.py` covers the latter)
* `Makefile` : builds everything
