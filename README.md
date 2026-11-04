# cpu-gpu — CPU vs GPU Throughput Benchmarks

A modern re-run of Lee et al., *"Debunking the 100X GPU vs. CPU Myth"* (ISCA'10). Throughput kernels are grouped by what limits them: **bandwidth**, **compute** or **synchronization**. The aim is to benchmark each kernel with a tuned CPU implementation and a tuned GPU implementation on a common set of devices. Then the GPU/CPU ratio can be compared against each device pair's peak bandwidth and peak FLOP ratios.

**Live dashboard:** <https://23110281.github.io/cpu-gpu/>
- **Workload Status & TODO** (landing page): which workloads are implemented, and which devices still need data.
- **Per-device plot galleries** for every CPU and GPU.

This repo holds both halves of the project:
- **CPU sweeps:** CloudLab Intel Xeon and AMD EPYC nodes. These were merged in from the former `cpu-gpu-kushal` repo.
- **GPU sweeps:** A100, L40S, V100, P100, A30 and GH200. The GH200 runs include the Grace CPU and the NVLink-C2C coherence experiments.

## Final workload suite

Sixteen workloads, chosen from the paper's compute-, bandwidth- and synchronization-bound kernels, plus LLM inference and cryptography:

| Bound | Workloads |
|---|---|
| Bandwidth (B) | SAXPY, LBM D3Q19, SpMV, SpGEMM |
| Compute (C) | GEMM, Monte Carlo (fp64), 2D convolution, FFT, bilateral filter, radix sort, index search, AES-128-CTR, SHA-256 |
| Synchronization (S) | Histogram, constraint solver |
| Application | Llama-3 inference (prefill ≈ C, decode ≈ B) |

**Core device matrix:**
- CPUs: Intel Xeon Sapphire Rapids (c6620), AMD EPYC (c6525-100g), NVIDIA Grace.
- GPUs: A100, L40S, H100 (GH200).
- Older devices (Xeon D-1548 / E5-2650 / Silver 4114, A30, V100, P100) are kept as legacy data.

**Dropped from the suite:**
- GJK and ray casting: their results are not primarily compute-, bandwidth- or sync-bound.
- PCA, k-means and SHA-512: redundant with other workloads.

**Where status lives:** [`results/workloads.json`](results/workloads.json) holds each workload's bound class, field and tags, its CPU and GPU implementation status, and its TODO list. The dashboard's ✓ / ⚠ / TODO data cells are *computed* from the CSVs in `results/data/` each time the page is built. They update automatically when new data is committed.

> **Known-invalid data — do not use for comparisons:**
> - **GH200 `aes_128_ctr`:** the GPU kernel is a byte XOR, and the Grace path runs 1 AES round instead of 10.
> - **GH200 `montecarlo`:** evaluates the closed-form Black-Scholes formula, not Monte Carlo paths.
>
> Both are marked ⚠ on the dashboard and are due to be re-implemented.

## Directory layout

```
src/
  common/      bf16_cvt.h, gemv_host.h                 shared by CPU + GPU harnesses
  cpu/         cpu_*_bench.c, cpu_telemetry.h          C/OpenMP + MKL/OpenBLAS, RAPL/MSR power telemetry
  gpu/         gpu_{gemm,gemv,saxpy,spgemm}_bench.cu   cuBLAS / cuSPARSE / cuSPARSELt, multi-GPU, NVML telemetry
  gpu/gh200/   gpu_c2c_*_bench.cu, protocol_benchmark_suite.cu, torch_c2c_allocator.cu, bench_harness.h
                                                       Grace Hopper residency / NVLink-C2C benchmarks
  torch/       torch_*_bench.py, torch_telemetry.py, bench_llama32_layers.py
                                                       PyTorch harnesses (--device cpu|cuda)
scripts/
  cpu/         run_*_sweep.sh, run_all_{intel,amd,torch}.sh, setup_cloudlab_{xeon,epyc}.sh,
               backup_results.sh, walkthrough.md       CloudLab CPU node workflow
  gpu/         run_sweep*.sh, lock_clocks.sh, profile_{ncu,nsys}.sh, install_cuda.sh
  gpu/gh200/   run_exploration_sweeps.sh, run_protocol_five_workloads.py, bench_power_energy_telemetry.py
results/
  data/cpu/<CPU>/        CPU CSVs: AMD_C6525_100G, INTEL_C6620, INTEL_C6220, INTEL_C220G5, INTEL_M510
  data/gpu/<GPU>/        GPU CSVs: NVIDIA_A100, NVIDIA_L40S, NVIDIA_GH200_480GB, NVIDIA_A30, Tesla_V100-*, Tesla_P100-*
                         (NVIDIA_GH200_480GB also holds the Grace CPU rows: residency_mode=cpu_local)
  raw/                   GH200 calibration (peaks.json) + protocol re-run logs
  plotting/cpu/          generate_cpu_{gemm,gemv,saxpy,spgemm}_plots.py
  plotting/gpu/          generate_{plots,gemv_plots,saxpy_plots,spgemm_plots,execution_time_plot}.py, gpu_specs.py
  plotting/gpu/gh200/    GH200 analysis, sweep cleaning and paper-figure scripts
  workloads.json         workload suite, implementation status, TODOs
  index.html             dashboard template (rendered by generate_index.py)
  generate_index.py      renders results/plots/index.html from the plots, workloads.json and the data tree
  plots/                 generated output (git-ignored)
tests/
  common/  cpu/  gpu/
```

CPU and GPU data are kept in separate trees, `results/data/cpu/` and `results/data/gpu/`, because the two sides use the same CSV names (`sweep_gemv_*.csv` and so on) with different columns. Each plotting script reads only its own side.

## Build & run

### GPU

You need the CUDA toolkit. `ARCH` defaults to `sm_80`, or `sm_90` on aarch64 (GH200); override it with `make ARCH=sm_89`, for example.

| Workload | Build | Smoke test | Full sweep |
|---|---|---|---|
| Dense GEMM | `make` | `make run` | `./scripts/gpu/run_sweep.sh` |
| SpGEMM | `make spgemm` | `make run_spgemm` | `./scripts/gpu/run_sweep_spgemm.sh` |
| GEMV / SpMV | `make gemv` | `make run_gemv` | `./scripts/gpu/run_sweep_gemv.sh` |
| SAXPY | `make saxpy` | `make run_saxpy` | `./scripts/gpu/run_sweep_saxpy.sh` |

The sweeps take environment-variable overrides, e.g. `SIZES=512,1024 PRECS=fp32 ./scripts/gpu/run_sweep.sh`. Output goes to `results/data/gpu/<GPU_NAME>/`.

**GH200 only:**
- Build: `make c2c_streaming c2c_compute c2c_fft c2c_irregular c2c_divergence c2c_cooperative c2c_uvm_advise c2c_allocator`
- Fixed sweeps: `./scripts/gpu/gh200/run_exploration_sweeps.sh`
- Protocol suite: `python3 scripts/gpu/gh200/run_protocol_five_workloads.py`

### CPU

Pick the BLAS with `make BLAS=mkl|openblas|auto`. The default, `auto`, uses oneMKL if `/opt/intel/oneapi/mkl/latest` exists, and OpenBLAS otherwise. CloudLab node setup is described in [`scripts/cpu/walkthrough.md`](scripts/cpu/walkthrough.md).

| Workload | Build | Smoke test | Full sweep |
|---|---|---|---|
| GEMM | `make cpu_gemm` | `make run_cpu_gemm` | `make sweep_cpu_gemm` |
| GEMV / SpMV | `make cpu_gemv` | `make run_cpu_gemv` | `make sweep_cpu_gemv` |
| SAXPY | `make cpu_saxpy` | `make run_cpu_saxpy` | `make sweep_cpu_saxpy` |
| SpGEMM | `make cpu_spgemm` | `make run_cpu_spgemm` | `make sweep_cpu_spgemm` |
| Monte Carlo | `make cpu_montecarlo` | `make run_cpu_montecarlo` | `make sweep_cpu_montecarlo` |
| SHA-256 (+ SHA-512) | `make cpu_cryptography` | `make run_cpu_cryptography` | `make sweep_cpu_cryptography` |
| Whole suite, in background | `make cpu_all` | — | `make intel_suite_bg` or `make amd_suite_bg` |
| PyTorch CPU suite | — | `make run_torch_gemm` (also `_gemv`, `_saxpy`, `_spgemm`) | `make torch_suite_bg` |

CPU output goes to `results/data/cpu/<CPU_MODEL>/`. The background suites log to `results/logs/`.

### Plots, dashboard and tests

| Task | Command |
|---|---|
| All CPU + GPU plots, then the dashboard | `make plots` |
| Dashboard only (e.g. after editing `results/workloads.json`) | `make index` |
| View locally | `python3 -m http.server -d results/plots 8765`, then open <http://localhost:8765/> |
| Unit tests | `make test_gemv_host`, `make test_telemetry` |

`make venv` uses [uv](https://github.com/astral-sh/uv) and `requirements.txt`. The pinned versions need Python ≥ 3.11; CI uses 3.12. Benchmark-node extras, such as torch for `src/torch/`, are listed separately in `requirements-bench.txt`.

### CI / published site

On every push to `main` that touches the data, plotting code, dashboard, `workloads.json`, the Makefile or the requirements, [`.github/workflows/plots.yml`](.github/workflows/plots.yml) runs:
- `make plots` when data or plotting code changed;
- only `generate_index.py` when just the dashboard or `workloads.json` changed.

The workflow then publishes `results/plots/` to the `plots` branch, which GitHub Pages serves at <https://23110281.github.io/cpu-gpu/>.

## GPU CLI options (shared shape)

* `--sizes ...` : `S` for GEMM / SpGEMM / GEMV (S×S matrices); `N` for SAXPY (vector length).
* `--precisions fp32,fp64,bf16` : precision conversion is host-side, before transfer. GEMV in bf16 has no native cuBLAS Level-2 gemv, so it is routed through `cublasGemmEx` with n=1.
* `--gpus G,..`, `--mode split|replicas`, `--density d,..`, `--engine dense|sparse`
* `--validate`, `--iters`, `--warmup`, `--no-nvml`, `--no-transfers`, `--csv`, `--tag`, `--help`

## Profiling & clock control

```bash
./scripts/gpu/profile_ncu.sh 8192 fp64 fp64-tensor   # Nsight Compute: FMA vs DMMA pipe utilisation (GEMM)
./scripts/gpu/profile_nsys.sh 16384 bf16 4           # Nsight Systems timeline: <size> <precision> <ngpus>
./scripts/gpu/lock_clocks.sh lock                    # also: max, reset  (lock target is currently A100-specific: 1410 MHz)
```
