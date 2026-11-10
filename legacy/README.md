# legacy/ — the first-generation benchmarks (frozen)

## What this folder is

This is the project's first benchmark pipeline: native C/CUDA programs, shell sweep scripts and plot scripts, run on CloudLab machines in Sept–Oct 2026.

- It is **frozen**: nothing here is maintained.
- It is **never used in the paper**.
- It is kept for three reasons: as a reference; because its data fills the dashboard's grey **"prelim"** cells until our current harness ([`bench/`](../bench/)) has measured that machine; and to show how the methodology changed.

Everything is preserved as it was, except that paths were adjusted so it still runs from this folder. The git history has the original locations: before November 2026 these files lived under `src/`, `scripts/`, `results/plotting/` and `results/data/legacy/`.

## Why these numbers are not research grade

An audit found these problems. Each was fixed in the current harness, which is why it replaced this code.

| Problem | Example from this folder |
|---|---|
| Timing is the **mean** of 5–20 iterations in one process: no median, no confidence interval, no repeated trials | every `*_bench.c/.cu`: one clock around N iterations, `comp_ms` = mean |
| Results marked valid without being checked | GPU SpGEMM writes `max_rel_err=0` and takes `validated` from the command-line flag (`src/gpu/gpu_spgemm_bench.cu:593,746`); CPU SpGEMM never validates (`src/cpu/cpu_spgemm_bench.c:250`) |
| A fabricated result | the AMD c6525-100g SpGEMM data: all 192 rows take 1.002–1.006 ms whatever the size. The old OpenBLAS build timed a 10 ms `nanosleep` instead of SpGEMM (comment at `cpu_spgemm_bench.c:212-214`) |
| Wrong library label | AMD rows say `mkl_version=oneMKL` although that build used OpenBLAS |
| Cache-resident runs reported as memory bandwidth | GH200 SAXPY rows up to 235% of "peak": every one is flagged `l2-bound` |
| % of peak computed against the **A100's** peaks on other GPUs | `src/gpu/gpu_gemm_bench.c:93-95` hard-codes A100 peaks; `gpu_device_specs.h` leaves H100/H200/L40S fp16/tf32 at A100 values; L40S fp32 shows 271% of "peak" |
| bf16 "measured" on GPUs without bf16 hardware | V100 bf16 GEMM at 11 TFLOP/s (emulated) |
| Slower libraries than the best available | old CPU GEMM 1.48 vs 2.0 TFLOP/s with the fastest library |
| Unequal work between CPU and GPU | Llama fp32 on CPU vs bf16 on GPU (deleted scripts) |
| Power read wrongly | wrong AMD power-register (MSR) addresses in early runs; 0 W written when power was unreadable |
| Sparse GEMM "density sweep" that changes nothing | `gpu_gemm_bench.cu` ignores `--density` (always 2:4 structured sparsity) but records 0.5 (lines 1064, 1480-1497) |

Fixed later in these native benches, too: the SpGEMM `nanosleep` was removed (a build without MKL now prints "unsupported"), MSR addresses were corrected, and status is taken from validation.

## What replaced each benchmark

| Legacy benchmark | Now | Notes |
|---|---|---|
| `cpu_gemm_bench.c`, `gpu_gemm_bench.cu` / `.c` | C1 GEMM | the C GEMM and SAXPY live on in [`crosscheck/`](../crosscheck/) as independent checks of the harness |
| `cpu_gemv_bench.c`, `gpu_gemv_bench.cu` (dense GEMV + SpMV) | B3 SpMV; dense GEMV is the core of A1 decode | |
| `cpu_saxpy_bench.c`, `gpu_saxpy_bench.cu` | B1 SAXPY | |
| `cpu_spgemm_bench.c`, `gpu_spgemm_bench.cu` | B4 SpGEMM | |
| `cpu_montecarlo_bench.c` | C2 Monte Carlo | the new one uses the same Philox random stream on CPU and GPU |
| (dropped) PCA, k-means, SHA-512 | — | redundant with GEMM, histogram and SHA-256 |
| (dropped) GJK collision detection, ray casting | — | not primarily bandwidth-, compute- or synchronisation-bound |

## Every file

### Native CPU benches (`src/cpu/`, `src/common/`)

| File | What it does | Reads → writes | Known problems |
|---|---|---|---|
| `cpu_gemm_bench.c` | square GEMM in fp32/fp64/bf16 (`cblas_?gemm`), size × precision × threads sweep | CLI → `sweep_gemm_*.csv` | mean timing; without MKL, bf16 converts to fp32 with a `malloc` inside every timed call; prints "Intel oneMKL / AMX" whatever the library; validates 32 sampled entries. *Live copy in `crosscheck/src/`.* |
| `cpu_gemv_bench.c` | dense GEMV (`cblas_?gemv`) and CSR SpMV (MKL sparse; a hand-written OpenMP loop under OpenBLAS) | CLI → `sweep_gemv_*.csv` | under OpenBLAS the sparse path is not a library call; dense validation only |
| `cpu_saxpy_bench.c` | `cblas_saxpy/daxpy` bandwidth sweep | CLI → `sweep_saxpy_*.csv` | mean timing; validation is a separate call after timing. *Live copy in `crosscheck/src/`.* |
| `cpu_spgemm_bench.c` | sparse × sparse via `mkl_sparse_spmm` | CLI → `sweep_spgemm_*.csv` | never validates; `mkl_sparse_destroy` inside the timed loop; theoretical FLOPs; the "unsupported" row drops a column |
| `cpu_montecarlo_bench.c` | real Monte Carlo option pricing (xorshift128+, Box-Muller) checked against Black-Scholes (2%) | CLI → `sweep_montecarlo_*.csv` | same seeds every iteration; fixed 24 FLOP/path; no CSV of it exists in `data/` |
| `cpu_telemetry.h` | power / frequency / temperature reader (RAPL, AMD MSR, hwmon, cpufreq) | sysfs, `/dev/cpu/*/msr` | frequency and temperature are one snapshot at the end, not an average |
| `common/gemv_host.h` | random-number hash, CSR matrix generator, helpers | — | `gemv_peak_gbps()` hard-coded to 1555 GB/s (A100) |
| `common/bf16_cvt.h` | fp32 ↔ bf16 conversion (round-to-nearest-even) | — | — |

### Native GPU benches (`src/gpu/`)

| File | What it does | Reads → writes | Known problems |
|---|---|---|---|
| `gpu_gemm_bench.cu` | multi-GPU cuBLAS/cuBLASLt GEMM (fp32/fp64/bf16, fp64 tensor cores), split/replica modes, cuSPARSELt 2:4 sparse, NVML power | CLI → `sweep_*.csv` | mean timing; sparse density ignored (always 2:4); peaks from `gpu_device_specs.h` |
| `gpu_gemm_bench.c` | older pure-C (gcc) version: cuBLAS GEMM + cuSPARSE SpMM | CLI → `sweep_*.csv` | A100 peaks hard-coded |
| `gpu_gemv_bench.cu` | multi-GPU cuBLAS GEMV and cuSPARSE SpMV | CLI → `sweep_gemv_*.csv` | sparse path never validates |
| `gpu_saxpy_bench.cu` | multi-GPU `cublasSaxpy/Daxpy` | CLI → `sweep_saxpy_*.csv` | marks `l2-bound` when above an (A100) peak |
| `gpu_spgemm_bench.cu` | multi-GPU cuSPARSE SpGEMM | CLI → `sweep_spgemm_*.csv` | "validated" without any check (see the table above) |
| `gpu_device_specs.h` | peak table matched by GPU name (GH200, H200, H100, L40S, A100; A100 default) | — | no V100/P100/A30 entries; several values left at A100 defaults |

### Sweep and setup scripts (`scripts/`)

| File | What it does | Known problems |
|---|---|---|
| `cpu/run_{gemm,gemv,saxpy,spgemm,montecarlo}_sweep.sh` | build if needed, sweep sizes × threads, write `results/data/native/cpu/<model>/sweep_<bench>_<ts>.csv` | `--mode split` passed on CPU does nothing; SpGEMM `--validate` is never honoured |
| `cpu/run_all_intel.sh`, `cpu/run_all_amd.sh` | build, smoke-test, run the 5 CPU sweeps, then call `backup_results.sh` | banner says "8 sweeps", 5 run |
| `cpu/backup_results.sh` | pushed this machine's CSVs to a personal GitHub backup repo (`KushalMehta17/cpu-gpu-proj-course-backup`) using a token from `$BACKUP_GH_TOKEN` or `~/.git_backup_token` | **do not use**: it puts the token in a clone URL; results are now copied back with `rsync` (docs/RUNNING.md) |
| `cpu/setup_cloudlab_xeon.sh`, `cpu/setup_cloudlab_epyc.sh` | old node setup (Intel oneAPI MKL on Xeon; OpenBLAS + AMD power modules on EPYC) | replaced by `scripts/setup_node.sh` |
| `gpu/run_sweep.sh`, `gpu/run_sweep_{gemv,saxpy,spgemm}.sh` | GPU sweeps (split/replicas, sparse) → `results/data/native/gpu/<GPU>/` | the sparse "density sweep" gives identical work |
| `gpu/run_sweep_gcc.sh` | meant to sweep the gcc build | actually runs the nvcc binary |
| `gpu/lock_clocks.sh` | persistence mode + lock the SM clock | hard-coded to 1410 MHz (A100) |
| `gpu/profile_ncu.sh`, `gpu/profile_nsys.sh` | Nsight Compute / Systems profiles of one GEMM | A100-specific metric list |
| `gpu/setup_gpu_node.sh` | old GPU node setup | broken: calls a script that no longer exists; use `scripts/setup_node.sh` |

### Plot scripts (`plotting/`)

All read `data/<cpu|gpu>/*/sweep_<bench>_*.csv` and write `plots/<device>/<bench>/` (git-ignored). They need `pip install -r requirements.txt -r legacy/requirements.txt`.

| File | Figures |
|---|---|
| `cpu/generate_cpu_gemm_plots.py` | TFLOP/s vs size, by threads, thread scaling, speed-up, heatmap, power, energy efficiency, fp32/fp64 ratio, frequency/temperature |
| `cpu/generate_cpu_gemv_plots.py` | dense bandwidth vs size, dense vs sparse, scaling, SpMV density scaling, heatmap, power, energy, frequency/temperature |
| `cpu/generate_cpu_saxpy_plots.py` | bandwidth vs N with a DRAM peak line (peaks hard-coded by CPU name), threads, scaling, heatmap, power, energy, fp32/fp64, frequency/temperature |
| `cpu/generate_cpu_spgemm_plots.py` | GFLOP/s vs size, bandwidth, scaling, density impact, heatmap, power efficiency, frequency/temperature |
| `gpu/generate_plots.py` | GEMM: effective/absolute TFLOP/s heatmaps, compute vs effective, power/utilisation, time split, bandwidth, treemap (+ sparse) |
| `gpu/generate_execution_time_plot.py` | GEMM execution-time comparison |
| `gpu/generate_gemv_plots.py` | GEMV/SpMV bandwidth heatmaps and curves, power, TFLOP/s |
| `gpu/generate_saxpy_plots.py` | SAXPY heatmap, bandwidth vs size, power, PCIe bandwidth, time split |
| `gpu/generate_spgemm_plots.py` | SpGEMM status heatmap, density frontier, TFLOP/s vs density and size, time split, efficiency, power |
| `gpu/gpu_specs.py` | peak reference lines (GH200, A100, L40S only; others fall back to 1000 GB/s) |

### Tests (`tests/`) and docs (`docs/`)

| File | Notes |
|---|---|
| `tests/common/test_gemv_host.c` | `make -C legacy test_gemv_host`: CSR generator checks (passes) |
| `tests/cpu/test_cpu_telemetry.c` | `make -C legacy test_telemetry`: never fails even without RAPL |
| `tests/gpu/test_generate_gemv_plots.py` (+ `smoke_a100_sparse_gemm.csv`) | plot-script test on synthetic data; the CSV is unused |
| `docs/walkthrough.md` | the old CloudLab guide; the current one is [`docs/RUNNING.md`](../docs/RUNNING.md) |

## The data (`data/`)

All of it is shown on the dashboard only as "prelim" (or ⚠ where known to be invalid), and only until our harness measures that machine.

| Device | CloudLab node | Collected | Files (rows) |
|---|---|---|---|
| `cpu/AMD_C6525_100G` | c6525-100g (EPYC 7402P, 24 cores) | 2026-10-01 | gemm (129), gemv (272), saxpy (96), spgemm (192: **fabricated**, see above) |
| `cpu/INTEL_C220G5` | c220g5 (2× Xeon Silver 4114, 20 cores) | 2026-09-17 | gemm (86), gemv (272), saxpy (96), spgemm (192) |
| `cpu/INTEL_C6220` | c6220 (Xeon E5-2650 v2, 16 cores) | 2026-09-17 | gemm (66), gemv (204), saxpy (72), spgemm (144) |
| `cpu/INTEL_C6620` | c6620 (Xeon Gold 5512U, 28 cores) | 2026-09-29 | gemm (129, incl. bf16), gemv (272), saxpy (96), spgemm (192) |
| `cpu/INTEL_M510` | m510 (Xeon D-1548, 8 cores) | 2026-09-17 | gemm (56), gemv (170), saxpy (60), spgemm (120) |
| `gpu/NVIDIA_A100` | 1–4× A100 | 2026-06/07 | gemm (380), gemv (352) |
| `gpu/NVIDIA_A30` | A30 | 2026-09-11 | gemm (24), gemv (88), saxpy (24), spgemm (120; 40 error) |
| `gpu/NVIDIA_L40S` | 1–2× L40S | 2026-07-27 | gemm (142), gemv (176), saxpy (48), spgemm (240; 83 error) |
| `gpu/NVIDIA_GH200_480GB` | GH200 (top-level sweeps only; the residency study is in `studies/gh200_c2c/`) | 2026-10-26 | gemm (71), gemv (88), saxpy (24, all `l2-bound`), spgemm (120; 44 error) |
| `gpu/Tesla_P100-PCIE-12GB` | P100 | 2026-09-08 | gemm (71), gemv (88), saxpy (24), spgemm (120) |
| `gpu/Tesla_V100-SXM2-16GB` | 1–4× V100 (now replaced on the dashboard by `results/data/gpu/NVIDIA_V100`) | 2026-09-08 | gemm (73 + 284), gemv (352), saxpy (96), spgemm (480; 146 error) |

Each benchmark has its own CSV columns, listed in the header comment of its source file. They are not the harness's schema and cannot be mixed with it.

## How to run it, if you ever need to

From the repo root (output is not comparable with our data):

```bash
make -C legacy cpu_all BLAS=openblas      # CPU benches (or BLAS=mkl with Intel oneAPI)
make -C legacy all ARCH=sm_80             # GPU benches (needs CUDA; cuSPARSELt for the sparse GEMM)
bash legacy/scripts/cpu/run_gemm_sweep.sh # one CPU sweep -> legacy/results/data/native/cpu/...
make -C legacy test_gemv_host test_telemetry
python3 legacy/plotting/cpu/generate_cpu_gemm_plots.py   # -> legacy/plots/
```
