# studies/gh200_c2c/ — GH200 CPU–GPU shared-memory study (not fixed yet)

## What the study asks

On an NVIDIA GH200, the Grace CPU (72 Arm cores, LPDDR5X memory) and the Hopper GPU (HBM3 memory) are joined by NVLink-C2C. Either processor can read the other's memory directly. The study asks: **where should the data live, and which processor should run each kernel?**

Every kernel is run in four "residency modes":

| Mode | Meaning |
|---|---|
| `CPU_LOCAL` | the Grace cores compute on their own LPDDR5X memory |
| `GPU_EXPLICIT_COPY` | copy the data to the GPU's HBM3, compute, copy the result back |
| `GPU_ZERO_COPY` | the GPU computes directly on CPU memory over NVLink-C2C, with no copy |
| `GPU_UM_MIGRATE` | unified memory (`cudaMallocManaged`): pages move to the GPU on first touch |

This is a **separate, system-level question**, not part of the CPU-vs-GPU comparison in the rest of the repo. Its data is never mixed into that comparison.

## Status: not research grade, not fixed

- The code below has known defects, listed next. **No number from this folder should be published until they are fixed.**
- Fixing and re-running it needs a GH200 node.
- Whether the study stays in the paper at all is an open decision.
- Everything is preserved as it was, except that paths now point inside this folder. The git history has the original locations: `src/gpu/gh200/`, `scripts/gpu/gh200/`, `results/plotting/gpu/gh200/`, `results/raw/`, and `results/data/legacy/gpu/NVIDIA_GH200_480GB/`.

## Known defects (fix before using any number)

| # | Defect | Where |
|---|---|---|
| 1 | The GPU "AES-128-CTR" kernel is a byte XOR, not AES; the cooperative bench's GPU "AES" is rotate/XOR | `src/gpu_c2c_divergence_bench.cu:382-391`, `src/gpu_c2c_cooperative_bench.cu:51-75` |
| 2 | The Grace AES path runs 1 AES round instead of 10 (and falls back to XOR without crypto extensions) | `src/gpu_c2c_divergence_bench.cu:433-442`, `src/gpu_c2c_cooperative_bench.cu:91-96` |
| 3 | "montecarlo" evaluates the closed-form Black-Scholes formula; there are no random paths, and CPU and GPU use different maths | `src/gpu_c2c_compute_bench.cu:388-418` (GPU), `480-525` (CPU) |
| 4 | The Grace sort sorts each thread's chunk and never merges them; it also assumes exactly 72 threads | `src/gpu_c2c_divergence_bench.cu:91-103` |
| 5 | "thrust_search" is a hand-written kernel, not Thrust | `src/gpu_c2c_irregular_bench.cu:261-277` |
| 6 | None of the C2C benches validates its results | all `src/gpu_c2c_*.cu` |
| 7 | No single end-to-end time definition: it differs per mode, and unified-memory cold time is dropped in places | `src/bench_harness.h`, `src/gpu_c2c_irregular_bench.cu:240-243` |
| 8 | `CPU_LOCAL` buffers are described as "pinned" but are pageable `aligned_alloc`; zero-copy silently falls back to the host pointer | `src/bench_harness.h:82-83, 100` |
| 9 | The UVM-cliff bench never initialises its buffers on the host and warms up before timing, so it never measures page migration | `src/gpu_c2c_uvm_advise_bench.cu:160-187` |
| 10 | A "Llama-3 KV partition" is promised but not implemented, yet `coop_partition_kv_offload` rows exist in the data | `src/gpu_c2c_cooperative_bench.cu:19` |
| 11 | The protocol suite prints "Correctness: PASS" without checking (GEMM, Eytzinger search), and has no build target | `src/protocol_benchmark_suite.cu:470, 659-662`; `Makefile` |
| 12 | The power script's "aes_128_ctr" is `torch.bitwise_xor` | `scripts/bench_power_energy_telemetry.py:168` |
| 13 | Several figure and table scripts contain hard-coded "measured" numbers | `plotting/generate_protocol_and_sweep_plots.py` (fig5 b–f, fig7b), `plotting/generate_all_workload_and_coherence_plots.py:274,367-369,392-393`, `plotting/generate_protocol_verified_table.py` |
| 14 | `clean_and_verify_sweeps.py` verifies nothing, and labels Llama model sizes by row position | `plotting/clean_and_verify_sweeps.py:231-249` |
| 15 | In `protocol_rerun_20261030_224002`, the workload-1 launch logs do not match the run summary (different command and values) | `data/protocol_reruns/protocol_rerun_20261030_224002/` |

## Every file

### Benchmarks (`src/`)

Shared timing convention: one cold run, warm-ups, then N timed runs, with the **mean** reported (GPU: CUDA events; CPU: `CLOCK_MONOTONIC`). All benches write one CSV schema (`workload, regime, precision, residency_mode, working_set_bytes, problem_size, …, t_kernel_warm_ms, t_e2e_ms, eff_bw_gbps, c2c_bw_gbps, gflops, pct_peak, status`). The `gflops` column is reused for other units in some benches.

| File | What it measures | Kernels inside | Modes | Defects |
|---|---|---|---|---|
| `gpu_c2c_streaming_bench.cu` | unit-stride bandwidth | SAXPY fp32/fp64 (4 MB–256 GB), GEMV (OpenBLAS on Grace, cuBLAS), RMSNorm | all 4 | CPU peak hard-coded (500 GB/s); explicit-mode C2C bandwidth uses only the H2D time |
| `gpu_c2c_compute_bench.cu` | compute-bound | GEMM (`cublasGemmEx` fp32/fp64/bf16/fp16/tf32; OpenBLAS on Grace), "Monte Carlo" | all 4 | #3; CPU bf16/fp16 converted inside the timed step; CPU peaks hard-coded inconsistently |
| `gpu_c2c_fft_bench.cu` | spectral | cuFFT vs FFTW3 (1-D, 2-D), 5×5 convolution | all 4 | FFTW plans with `FFTW_ESTIMATE`; arbitrary CPU FFT peak (5000 GFLOP/s) |
| `gpu_c2c_irregular_bench.cu` | irregular / sparse | SpMV (cuSPARSE vs OpenMP CSR), "thrust_search", SpGEMM | all 4 | #5, #7; CPU SpGEMM discards its result |
| `gpu_c2c_divergence_bench.cu` | branching, sorting, atomics | CUB radix sort, CUB HistogramEven (256 / 65536 bins), "AES-128-CTR" | all 4 | #1, #2, #4 |
| `gpu_c2c_cooperative_bench.cu` | CPU + GPU at the same time | memory contention (CPU SAXPY + GPU zero-copy / HBM), α-split partitioning of SAXPY / GEMM / "AES" | per scenario | #1, #2, #10; CSV overwritten on each run |
| `gpu_c2c_uvm_advise_bench.cu` | the 96 GB HBM3 capacity cliff | float4 SAXPY, `cublasSgemv_64`, a KV-cache streaming sum; 64–160 GB | 5 `cudaMemAdvise` / prefetch policies | #9; several columns written as 0 |
| `protocol_benchmark_suite.cu` | the 5-workload "mandatory protocol" | Grace STREAM, 65k-bin histogram (global atomics vs cluster shared memory), binary search (std vs Eytzinger), bf16 GEMM zero-copy vs staged, Llama-3.2-1B decode attention (unsplit vs split-KV) | its own | #11; reports median/min/max/CV (better than the others) |
| `torch_c2c_allocator.cu` | PyTorch pluggable allocator placing tensors in zero-copy or managed memory | — | — | never returns memory; zero-copy returns the host pointer |
| `bench_harness.h` | shared allocation per mode, timing and CSV helpers | — | — | #8 |
| `gemv_host.h`, `gpu_device_specs.h` | copies of the legacy helpers (CSR generator; GPU peak table) | — | — | see `legacy/README.md` |

### Scripts (`scripts/`)

| File | What it does | Reads → writes | Defects |
|---|---|---|---|
| `run_exploration_sweeps.sh` | fixed GEMM/GEMV/SAXPY/SpGEMM sweeps | → `results/data/native/gpu/<GPU>/exploration/` | calls the **legacy** binaries (`make -C legacy` first), which this Makefile does not build |
| `run_protocol_five_workloads.py` | hardware snapshot, 100 ms `nvidia-smi` telemetry, 5 launches × 5 workloads, then Nsight Compute | `./bin/protocol_benchmark_suite` → `data/protocol_reruns/protocol_rerun_<ts>/` | ncu path hard-coded to CUDA 12.8; profiles a binary not in the repo (`/tmp/test_zc_gemm_16k`) |
| `bench_power_energy_telemetry.py` | PyTorch steady-state loops per residency mode, power sampled every ~80 ms | → `--csv` | #12; CPU power = module − GPU |

### Analysis and figures (`plotting/`)

| File | Reads → writes | Notes |
|---|---|---|
| `clean_and_verify_sweeps.py` | `data/residency_sweeps/*` → `data/cleaned_for_paper/*` (**overwrites them**) | #14 |
| `analyze_residency_sweeps.py` | `data/residency_sweeps/sweep_cycle*.csv` → a combined taxonomy CSV | run from this folder |
| `generate_paper_figures.py` | `data/cleaned_for_paper/` → `plots/` (tables 1, figs 1–4) | several hard-coded annotations |
| `generate_protocol_and_sweep_plots.py` | cleaned data + `protocol_rerun_20261030_224002` → `plots/` (figs 5–7) | #13 |
| `generate_all_workload_and_coherence_plots.py` | master CSV → `plots/<workload>/`, `plots/cpu_gpu_*` | #13 |
| `generate_protocol_verified_table.py` | a protocol-rerun folder → `protocol_verified_summary.json` in it | #13: never reads `peaks.json` |

### Data (`data/`)

| Folder | Contents | Produced by → used by |
|---|---|---|
| `residency_sweeps/` | 22 raw CSVs from the C2C benches (cycles 1–5, streaming grid, compute/FFT, divergence, irregular, Llama-3 KV, cooperative, UVM, power) | the benches, run by hand → `clean_and_verify_sweeps.py`, `analyze_residency_sweeps.py`. The Llama-3 and `coop_partition_kv_offload` rows have no producing code in the repo |
| `cleaned_for_paper/` | 10 cleaned CSVs incl. `gh200_paper_master_clean.csv` (3488 rows) | `clean_and_verify_sweeps.py` → the figure scripts, and the dashboard's GH200/Grace "prelim" / ⚠ cells |
| `hardware_specs/` | captures from the node: `lscpu`, `lstopo`, `dmidecode`, `numactl`, `nvidia-smi -q`, `/proc/cpuinfo`, … | no script reads them |
| `exploration/full_fixed_sweeps/` | 4 legacy-format sweeps (gemm, gemv, saxpy, spgemm) | `run_exploration_sweeps.sh` → nothing |
| `protocol_reruns/` | three runs from 2026-10-30: `202052` (aborted, environment log only), `202123` (25 launch logs + telemetry), `224002` (25 launch logs, telemetry, 4 Nsight logs, verified summary; see #15) | `run_protocol_five_workloads.py` → the protocol figure and table scripts |
| `calibration/` | `peaks.json`, `nvbandwidth.log`, reference microbenchmarks | no producer or reader in the repo |

## How to build and run (on a GH200)

```bash
make -C studies/gh200_c2c                 # all C2C benches + the allocator .so -> studies/gh200_c2c/bin/
cd studies/gh200_c2c
./bin/gpu_c2c_streaming_bench --help      # each bench prints its options
python3 scripts/run_protocol_five_workloads.py   # needs ./bin/protocol_benchmark_suite (no target yet, see #11:
                                                 #   nvcc -arch=sm_90 -O3 src/protocol_benchmark_suite.cu -lcublas -Xcompiler -fopenmp)
python3 plotting/generate_paper_figures.py       # -> studies/gh200_c2c/plots/
```

You need: the CUDA toolkit (sm_90), cuBLAS/cuSPARSE/cuFFT, OpenBLAS and FFTW3 (float, threaded) on Grace, and pandas/matplotlib for the plots.

## To revive it

1. Fix defects 1–5 and 9–12, the ones that compute the wrong thing.
2. Add validation to every bench (6), and use one end-to-end time definition across modes (7, 8).
3. Replace every hard-coded number in the figure scripts with values read from data (13), and make `clean_and_verify_sweeps.py` actually verify (14).
4. Re-collect all data, then follow the main harness's rules (`docs/methodology.md`): validate first, medians with confidence intervals, 3 process-isolated trials, NVML power.

Ideally the study moves into the main harness as a GH200-specific "residency" option on the existing workloads, rather than staying as separate programs.
