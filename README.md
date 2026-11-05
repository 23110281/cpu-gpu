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

**Where status lives:** [`results/workloads.json`](results/workloads.json) holds each workload's bound class, field and tags, its CPU and GPU implementation status, and its TODO list. The dashboard's data cells are *computed* from `results/data/` each time the page is built, so they update automatically when new data is committed:
- ✓ means v2 protocol data exists;
- prelim means only legacy data exists;
- ⚠ means the legacy data is invalid;
- TODO means there is no data.

> **All data collected before the v2 harness is preliminary.** It is frozen in `results/data/legacy/` and never used in the paper; an audit found it does not meet the protocol (mean-only timing, no pinning, unequal CPU/GPU work, partial validation). The dashboard shows these cells as *prelim*.
>
> Some legacy data is outright invalid and shows as ⚠:
> - **GH200 `aes_128_ctr`:** the GPU kernel is a byte XOR, and the Grace path runs 1 AES round instead of 10.
> - **GH200 `montecarlo`:** evaluates closed-form Black-Scholes, not Monte Carlo paths.
> - **EPYC SpGEMM:** the OpenBLAS build timed a 10 ms `nanosleep` instead of SpGEMM.

## Publication harness (v2)

Every number in the paper comes from `src/torch/bench/`, which enforces the protocol in [`docs/methodology.md`](docs/methodology.md).

**What the protocol fixes:**
- **Comparison unit:** one socket against one GPU, with pinned threads and parallel first touch.
- **Timing:** kernel-only, with adaptive warmup, at least 30 samples and at least 1 s per configuration, and 3 independent process trials.
- **Statistics:** median, IQR and a bootstrap 95% CI.
- **Validation:** at full size before timing, with principled tolerances.
- **Equal work:** CPU and GPU produce the same output, each using its fastest well-known library.
- **Telemetry:** a sampled telemetry thread.
- **Provenance:** an environment record and git SHA for every row.

**Status:** all 16 workloads are implemented in the v2 harness, and the CPU paths are validated on an EPYC 7402P. The CUDA paths are written but have not yet run on a GPU: the first session on each GPU type is a validation run.

**Node setup:**
- CPU: `scripts/cpu/setup_cloudlab_epyc.sh` or `scripts/cpu/setup_cloudlab_xeon.sh`.
- GPU: `scripts/gpu/setup_gpu_node.sh`.
- All three share `scripts/setup_v2_harness.sh`.
- Calibration: `python3 scripts/calibrate.py --device-config configs/devices/<DEVICE>.json`.
- Native cross-check: `python3 scripts/crosscheck_native.py --device-id <DEVICE>`.

```bash
make bench_list                          # registered workloads
make bench_smoke                         # every workload, smallest configs, short timing (-> results/data/smoke, git-ignored)
make bench_smoke_cuda                    # same on a GPU node
make bench_suite_dry DEVICE=INTEL_C6620  # show the job list for a device (configs/devices/<DEVICE>.json)
make bench_suite     DEVICE=INTEL_C6620  # full collection: every config x placement x 3 trials -> results/data/v2/
make test_bench                          # harness unit tests (statistics, validators, schema, placement, telemetry)
# one configuration by hand:
PYTHONPATH=src/torch python3 -m bench run --workload C1 --device cuda --params '{"precision":"fp32","variant":"base","n":8192}'
```

**What a sweep writes, per device, under `results/data/v2/<cpu|gpu>/<DEVICE>/`:**
- one long-format CSV per workload (schema: `src/torch/bench/schema.py`);
- raw per-sample timings in `samples/`;
- environment records in `env/`.

Sweeps can be resumed by passing `--campaign <id>`. The native C/CUDA harnesses below are kept as independent **cross-checks** (torch vs native medians must agree within 3%), plus the GH200 residency study. They write to `results/data/native/`.

## Directory layout

```
src/
  common/      bf16_cvt.h, gemv_host.h                 shared by CPU + GPU harnesses
  cpu/         cpu_*_bench.c, cpu_telemetry.h          C/OpenMP + MKL/OpenBLAS, RAPL/MSR power telemetry
  gpu/         gpu_{gemm,gemv,saxpy,spgemm}_bench.cu   cuBLAS / cuSPARSE / cuSPARSELt, multi-GPU, NVML telemetry
  gpu/gh200/   gpu_c2c_*_bench.cu, protocol_benchmark_suite.cu, torch_c2c_allocator.cu, bench_harness.h
                                                       Grace Hopper residency / NVLink-C2C benchmarks
  torch/bench/ v2 publication harness: core (timing/stats), devices (placement), validate, telemetry,
               env (provenance), schema, sweep; workloads/<id>_<name>.py

scripts/
  cpu/         run_*_sweep.sh, run_all_{intel,amd,torch}.sh, setup_cloudlab_{xeon,epyc}.sh,
               backup_results.sh, walkthrough.md       CloudLab CPU node workflow
  gpu/         run_sweep*.sh, lock_clocks.sh, profile_{ncu,nsys}.sh, install_cuda.sh
  gpu/gh200/   run_exploration_sweeps.sh, run_protocol_five_workloads.py, bench_power_energy_telemetry.py
configs/devices/         one JSON per core device (placements, scaling sweep; memory auto-detected)
docs/methodology.md      the measurement protocol
results/
  data/v2/<cpu|gpu>/<DEVICE>/      publication data (v2 harness): <ID>_<workload>.csv, samples/, env/
  data/legacy/cpu/<CPU>/           FROZEN pre-protocol CPU CSVs (AMD_C6525_100G, INTEL_*)
  data/legacy/gpu/<GPU>/           FROZEN pre-protocol GPU CSVs (NVIDIA_*, Tesla_*); NVIDIA_GH200_480GB
                                   also holds the Grace CPU rows (residency_mode=cpu_local)
  data/native/<cpu|gpu>/<DEVICE>/  new output of the native C/CUDA cross-check harnesses
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

CPU and GPU data are kept in separate `cpu/` and `gpu/` trees, because the legacy harnesses use the same CSV names (`sweep_gemv_*.csv` and so on) with different columns. The legacy plotting scripts read only `results/data/legacy/`.

## Build & run

### Native GPU harnesses (cross-check)

You need the CUDA toolkit. `ARCH` defaults to `sm_80`, or `sm_90` on aarch64 (GH200); override it with `make ARCH=sm_89`, for example.

| Workload | Build | Smoke test | Full sweep |
|---|---|---|---|
| Dense GEMM | `make` | `make run` | `./scripts/gpu/run_sweep.sh` |
| SpGEMM | `make spgemm` | `make run_spgemm` | `./scripts/gpu/run_sweep_spgemm.sh` |
| GEMV / SpMV | `make gemv` | `make run_gemv` | `./scripts/gpu/run_sweep_gemv.sh` |
| SAXPY | `make saxpy` | `make run_saxpy` | `./scripts/gpu/run_sweep_saxpy.sh` |

The sweeps take environment-variable overrides, e.g. `SIZES=512,1024 PRECS=fp32 ./scripts/gpu/run_sweep.sh`. Output goes to `results/data/native/gpu/<GPU_NAME>/`.

**GH200 only:**
- Build: `make c2c_streaming c2c_compute c2c_fft c2c_irregular c2c_divergence c2c_cooperative c2c_uvm_advise c2c_allocator`
- Fixed sweeps: `./scripts/gpu/gh200/run_exploration_sweeps.sh`
- Protocol suite: `python3 scripts/gpu/gh200/run_protocol_five_workloads.py`

### Native CPU harnesses (cross-check)

Pick the BLAS with `make BLAS=mkl|openblas|auto`. The default, `auto`, uses oneMKL if `/opt/intel/oneapi/mkl/latest` exists, and OpenBLAS otherwise. CloudLab node setup is described in [`scripts/cpu/walkthrough.md`](scripts/cpu/walkthrough.md).

| Workload | Build | Smoke test | Full sweep |
|---|---|---|---|
| GEMM | `make cpu_gemm` | `make run_cpu_gemm` | `make sweep_cpu_gemm` |
| GEMV / SpMV | `make cpu_gemv` | `make run_cpu_gemv` | `make sweep_cpu_gemv` |
| SAXPY | `make cpu_saxpy` | `make run_cpu_saxpy` | `make sweep_cpu_saxpy` |
| SpGEMM | `make cpu_spgemm` | `make run_cpu_spgemm` | `make sweep_cpu_spgemm` |
| Monte Carlo | `make cpu_montecarlo` | `make run_cpu_montecarlo` | `make sweep_cpu_montecarlo` |
| Whole suite, in background | `make cpu_all` | — | `make intel_suite_bg` or `make amd_suite_bg` |
| v2 harness suite (any device) | — | `make bench_smoke` | `make torch_suite_bg DEVICE=<id>` |

CPU output goes to `results/data/native/cpu/<CPU_MODEL>/`. The background suites log to `results/logs/`.

### Plots, dashboard and tests

| Task | Command |
|---|---|
| All CPU + GPU plots, then the dashboard | `make plots` |
| Dashboard only (e.g. after editing `results/workloads.json`) | `make index` |
| View locally | `python3 -m http.server -d results/plots 8765`, then open <http://localhost:8765/> |
| Unit tests | `make test_bench` (v2 harness), `make test_gemv_host`, `make test_telemetry` |

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
