# CPU vs GPU: a modern re-run of "Debunking the 100X GPU vs. CPU Myth"

In 2010, Lee et al. (ISCA'10) showed that GPU speed-ups over CPUs shrink from "100×" to about 2.5× once **both** sides run well-optimised code. This project repeats that study on today's hardware. Each workload runs on a CPU socket and on a GPU, both using the fastest well-known library (or a tuned kernel of our own), with identical inputs, and every result is checked against a reference before it is timed.

**Dashboard:** <https://23110281.github.io/cpu-gpu/>. It has three parts:
- workload status;
- one page per machine, with results, findings and figures;
- a library comparison across machines.

## Workloads

Each workload is labelled by what limits it: **B** = memory bandwidth, **C** = compute, **S** = synchronisation, **A** = a whole application. "native" means a kernel we wrote ourselves; it is compiled on the machine at run time by PyTorch (`torch.utils.cpp_extension`). Where several libraries are listed, all of them are timed and the fastest one is kept ("fastest kept").

| ID | Workload | What it does | What it measures (bound) | CPU libraries / APIs | GPU libraries / APIs |
|---|---|---|---|---|---|
| B1 | SAXPY | y ← a·x + y on long vectors | memory bandwidth, GB/s (B) | PyTorch (ATen vectorised kernel, OpenMP threads) | PyTorch (ATen CUDA kernel) |
| B2 | LBM D3Q19 | one time step of a 3-D fluid simulation (lattice Boltzmann) | lattice updates/s, MLUPS (B) | native C++ (OpenMP + SIMD, AVX streaming stores) | native CUDA |
| B3 | SpMV | sparse matrix × vector, CSR format | GB/s (B) | PyTorch → Intel oneMKL sparse BLAS | PyTorch → cuSPARSE |
| B4 | SpGEMM | sparse × sparse matrix product | GFLOP/s (B) | PyTorch → oneMKL sparse BLAS | PyTorch → cuSPARSE |
| C1 | GEMM | dense matrix multiply | GFLOP/s (C) | PyTorch (oneMKL), OpenBLAS, BLIS (fastest kept) | PyTorch → cuBLAS |
| C2 | Monte Carlo | option pricing with random paths, fp64 | paths/s (C) | native C++ (OpenMP + SIMD, glibc libmvec, Philox random numbers) | native CUDA (same Philox stream) |
| C3 | 2-D convolution | one k×k filter over an image | Mpixels/s (C) | PyTorch `conv2d` (→ oneDNN), OpenCV `filter2D`, native C++ (fastest kept) | PyTorch → cuDNN, native CUDA (fastest kept) |
| C4 | FFT | 1-D and 2-D complex Fourier transforms | GFLOP/s (C) | PyTorch → oneMKL FFT (pocketfft on Arm CPUs) | PyTorch → cuFFT |
| C5 | Bilateral filter | edge-preserving image smoothing | Mpixels/s (C) | OpenCV `bilateralFilter`, native C++ (fastest kept) | native CUDA (plus a fast-`exp` variant) |
| C6 | Sort | stable sort of key-value pairs | Mpairs/s (C) | PyTorch sort, GNU parallel mergesort (libstdc++), oneTBB `parallel_sort` (fastest kept) | PyTorch → CUB radix sort |
| C7 | Index search | binary search of many queries in sorted keys | Mqueries/s (C) | PyTorch `searchsorted` | PyTorch `searchsorted` (CUDA) |
| C8 | AES-128-CTR | encryption | GB/s (C) | OpenSSL (AES-NI), plus a hardware-off variant | native CUDA (T-table AES) |
| C9 | SHA-256 | hashing many independent messages | GB/s (C) | OpenSSL (SHA-NI), plus a hardware-off variant | native CUDA |
| S1 | Histogram | count values into bins | Msamples/s (S) | PyTorch `histc`, `bincount` (fastest kept) | PyTorch `histc`, `bincount` (atomics; fastest kept) |
| S2 | Constraint solver | batched Gauss-Seidel with a barrier between batches | iterations/s (S) | native C++ (OpenMP barriers) | native CUDA: CUDA graph vs cooperative launch (fastest kept) |
| A1 | Llama-3 inference | a full Llama-3 model forward pass (1B / 3B / 8B shapes), prompt + token generation | tokens/s (A) | PyTorch (oneMKL SGEMM), OpenBLAS, BLIS for the linear layers (fastest kept) | PyTorch → cuBLAS; generation also replayed as a CUDA graph |

How results are checked before timing:
- bandwidth and compute workloads are compared with a float64 reference (GEMM with a randomised Freivalds check plus exactly recomputed entries);
- sort, search, histogram and the crypto workloads must match exactly (crypto: official NIST/FIPS test vectors and the full output against OpenSSL);
- Monte Carlo must land within 4 standard errors of the exact Black-Scholes price.

Details per workload: [`docs/methodology.md`](docs/methodology.md) and the docstring at the top of each file in [`bench/workloads/`](bench/workloads/).

## Library index

| Library | What it is for (plain words) | Open / closed source | Used by |
|---|---|---|---|
| PyTorch | the framework that runs most workloads on both CPU and GPU, and compiles our native kernels | open (BSD-3) | all |
| Intel oneMKL | Intel's maths library: matrix multiply, sparse maths, FFT; built into PyTorch on x86 | closed, free to use (Intel Simplified Software License) | B3, B4, C1, C4, A1 (CPU) |
| oneDNN | Intel's neural-network kernel library, built into PyTorch | open (Apache-2.0) | C3 (CPU: PyTorch `conv2d` runs oneDNN's depthwise kernel) |
| pocketfft | small FFT library PyTorch uses on CPUs without oneMKL (Arm) | open (BSD-3) | C4 on Arm CPUs |
| OpenBLAS | open-source BLAS (matrix maths), OpenMP build | open (BSD-3) | C1, A1 (CPU); native cross-check |
| BLIS | open-source BLAS framework (basis of AMD's AOCL); Ubuntu's 0.8.1, or upstream 2.1 built by setup on CPUs it doesn't know (Zen 4) | open (BSD-3) | C1, A1 (CPU) |
| OpenCV | computer-vision library with fast image filters | open (Apache-2.0 for 4.x) | C3, C5 (CPU) |
| oneTBB | Intel's parallel-algorithms library | open (Apache-2.0) | C6 (CPU) |
| GNU libstdc++ parallel mode | GCC's parallel versions of standard algorithms (`sort`) | open (GPL-3 with runtime exception) | C6 (CPU) |
| OpenMP (libgomp) | runs our native CPU kernels on all cores | open (GPL-3 with runtime exception) | B2, C2, C3, C5, C8, C9, S2 (CPU) |
| glibc libmvec | vectorised `exp`/`log`/`cos` used inside SIMD loops | open (LGPL-2.1+) | C2, C5 (CPU) |
| OpenSSL | cryptography library using the CPU's AES/SHA instructions | open (Apache-2.0 for 3.x) | C8, C9 (CPU) |
| CUDA toolkit and runtime | NVIDIA's compiler and runtime for our GPU kernels, CUDA graphs, cooperative launch | closed (NVIDIA EULA) | all GPU workloads |
| cuBLAS | NVIDIA's matrix-multiply library | closed (NVIDIA EULA) | C1, A1 (GPU) |
| cuSPARSE | NVIDIA's sparse-matrix library | closed (NVIDIA EULA) | B3, B4 (GPU) |
| cuFFT | NVIDIA's FFT library | closed (NVIDIA EULA) | C4 (GPU) |
| cuDNN | NVIDIA's neural-network kernel library | closed (NVIDIA cuDNN licence) | C3 (GPU) |
| CUB (NVIDIA CCCL) | GPU building blocks such as radix sort, used inside PyTorch | open (BSD-3) | C6 (GPU) |
| NVML | NVIDIA's management library, read for GPU power and clocks | closed (part of the driver) | telemetry |
| BabelStream | standard memory-bandwidth benchmark, used to measure each machine's real bandwidth ceiling | source available, custom licence (results must follow its run rules) | calibration |
| NumPy / SciPy | reference answers for validation and test matrices (not timed) | open (BSD-3) | validation |

## Fastest library per machine

Only machines with a full campaign appear here; this table grows as new machines are run. The same data, per problem size, is on the dashboard's **Library analysis** page.

<!-- library-table:start -->
_Generated by `make readme` from the campaign data (`analysis/library_report.py`); do not edit by hand._
Each cell: **fastest implementation**, sizes it won / sizes compared, then the others at their median speed
relative to the winner (1.00× = as fast). ‘tie’ = the 95% confidence intervals overlap.

| Workload | NVIDIA V100 |
|---|---|
| C3 2D Convolution (fp32) | **native kernel** 18/18; PyTorch 0.42× [1] |
| S1 Histogram (fp32/int32) | **torch.histc** 15/18; torch.bincount 0.66× |
| S2 Constraint solver (batched PGS) (fp32) | **CUDA graph launch** 4/4; cooperative launch 0.90× |
| A1 Llama-3 inference (fp32) | **PyTorch + CUDA graph** 2/2; PyTorch 0.44× |

Variants (not library choices):
- NVIDIA V100, C5 — fast-math / fixed-function variant (C5: __expf instead of expf): 1.18–1.50× the base implementation (9 sizes).

Notes:
1. The native direct convolution (filter size a compile-time constant, all K*K taps unrolled, vectorised x loop) wins every size: neural-network convolution libraries are not built for one small filter over a large image.
<!-- library-table:end -->

## How a number is produced

- **Fair comparison unit:** one CPU socket (threads pinned, memory on that socket) against one GPU.
- **Validated first:** every configuration is checked at full size before it is timed. A wrong answer is recorded as `validation_failed` and never timed.
- **Timing:** only the kernel is timed, with adaptive warm-up, at least 30 samples and at least 1 s per configuration, repeated in 3 separate processes (trials).
- **Statistics:** the median with a bootstrap 95% confidence interval; results with more than 5% variation are flagged `unstable`.
- **Fastest library kept:** every candidate library is timed on the same inputs, and the headline uses the fastest.
- **Measured ceilings:** each machine's real bandwidth (BabelStream) and peak matrix-multiply speed are measured, so every result can be given as a "% of what this machine can do".

The full protocol is in [`docs/methodology.md`](docs/methodology.md).

## Run it

On a CloudLab node, after `git clone`:

```bash
sudo bash scripts/install_cuda.sh && sudo reboot   # GPU nodes only, once
bash scripts/setup_node.sh                         # installs everything (CPU or GPU, auto-detected)
bash scripts/run_campaign.sh start <DEVICE>        # tests -> smoke -> calibration -> full sweep, in the background
bash scripts/run_campaign.sh status <DEVICE>       # progress
```

Then copy the results back and run `make site` on your laptop. The full step-by-step guide, with troubleshooting, is [`docs/RUNNING.md`](docs/RUNNING.md). Other targets: `make test`, `make smoke`, `make calibrate DEVICE=…`, `make sweep DEVICE=…`, `make analysis`, `make readme`, `make list`.

## Repository layout

```
bench/                    the benchmark harness (python3 -m bench ...)
  core.py                   timing policy, validate-then-time, one result row per run
  devices.py                CPU topology, thread/memory placement, GPU setup
  sweep.py                  full sweeps: one process per (config, trial), resumable
  validate.py  stats.py     correctness checks; median, confidence intervals
  telemetry.py env.py       power, clocks, temperature; provenance (versions, git commit)
  schema.py                 the result CSV format
  workloads/                one file per workload (b1_saxpy.py ... a1_llama.py) + shared helpers
analysis/                 turns results into tables, figures and the dashboard
  analyze.py                headline numbers, library comparison, bound checks, figures
  library_report.py         which library wins where (also writes the README table)
  dashboard_summary.py      per-machine summary pages
  figures.py                figure style
  build_dashboard.py        builds results/site/index.html from dashboard.html
  dashboard.html            the dashboard page template
configs/
  devices/<DEVICE>.json     one file per machine: placements, datasheet ceilings
  workloads.json            the workload list, implementation status and TODOs
  library_notes.json        curated "why this library wins" notes
  dashboard_notes.json      curated findings shown on each machine's page
scripts/
  setup_node.sh             node setup (CPU or GPU)
  install_cuda.sh           NVIDIA driver + CUDA toolkit (GPU nodes)
  run_campaign.sh           one-command campaign: start / status / resume / stop
  calibrate.py              measures bandwidth and peak GEMM ceilings
tests/test_bench.py       harness unit tests (make test)
docs/
  RUNNING.md                how to run on a new machine
  methodology.md            the measurement protocol
  HANDOVER.md               current state of the project (for the next work session)
results/
  data/<cpu|gpu>/<DEVICE>/  campaign results: one CSV per workload, raw samples, environment records
  data/sessions/            machine test sessions (kept for the record; never analysed)
  calibration/              measured ceilings per machine
  crosscheck/               native cross-check results
  analysis/                 generated tables (rebuilt by make analysis; sessions/ holds snapshots)
  site/                     generated dashboard (rebuilt by make site; not committed)
crosscheck/               independent native C benchmarks (GEMM, SAXPY) that confirm the harness's numbers
studies/gh200_c2c/        separate GH200 CPU-GPU shared-memory study (not fixed yet; see its README)
legacy/                   the frozen first-generation benchmarks and data (never used in the paper; see its README)
Makefile  requirements.txt  requirements-node.txt
```

## Legacy code and the GH200 study

- [`legacy/`](legacy/README.md) holds the project's first benchmark pipeline: native C/CUDA programs, sweep scripts, plots and their data from Sept–Oct 2026. It is frozen and is **not** used in the paper; its README explains why. Its data only fills "prelim" placeholder cells on the dashboard until our harness covers that machine.
- [`studies/gh200_c2c/`](studies/gh200_c2c/README.md) is a separate system-level study of the GH200's shared CPU–GPU memory: where data should live, and which processor should run each kernel. It still has known defects, listed in its README, and its data is never mixed into the CPU-vs-GPU comparison.
- [`crosscheck/`](crosscheck/) compares the harness against two independent native C benchmarks (`python3 crosscheck/crosscheck.py --device-id <DEVICE>`); they must agree within 3%.
