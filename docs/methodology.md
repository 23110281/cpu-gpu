# Measurement Methodology (v2)

This document is the measurement protocol for every number this project publishes. The harness in `src/torch/bench/` enforces it, and each rule below names the code that implements it. Data collected any other way, including everything under `results/data/legacy/`, is **preliminary** and is never used in the paper.

The protocol follows the comparison principle of Lee et al. (ISCA'10): compare *optimized against optimized*, chip against chip, with the GPU measured kernel-only. On top of that it adds the statistical and provenance requirements of current artifact-evaluation practice.

## 1. Unit of comparison
- **CPU headline: one socket.** Threads are pinned to socket 0's cores, and memory is bound to socket 0's NUMA node with `numactl --membind` (or `sched_setaffinity`, if numactl is absent). The whole node is a secondary placement (`node`).
- **GPU headline: one GPU** (`--gpu N`). Multi-GPU scaling is out of scope for the headline comparison.

## 2. Platform configuration
**Run as shipped, and record everything** (`bench/env.py`):
- CPU:
  - Governor `performance`; turbo at the vendor default.
  - SMT, THP and NUMA-balancing state recorded.
- GPU:
  - Persistence mode on; default application clocks.
  - ECC state and power limit recorded.

**Clock sensitivity:** SAXPY, GEMM and Histogram are repeated with clocks locked (`scripts/gpu/lock_clocks.sh`), and reported separately.

## 3. Threads and memory placement (`bench/devices.py`)
- **Pinning.** `OMP_PLACES=cores`, `OMP_PROC_BIND=close`, `MKL_DYNAMIC=FALSE`, and `OMP_NUM_THREADS` = the thread count. `torch.set_num_threads` / `set_num_interop_threads(1)` are each called exactly **once per process**. Each configuration runs in its own process (`bench/sweep.py`).
- **Placements:**
  - `socket<S>`: one thread per physical core of socket S.
  - `socket<S>-smt`: every hardware thread of socket S.
  - `node`: one thread per physical core of the whole node.
  - `node-smt`: every hardware thread of the node.
  - `socket<S>:<T>`: the first T cores of socket S (scaling sweep).
- **Headline thread count:** the faster of `socket0` and `socket0-smt`, per workload and device. Both are always recorded. The paper also used each platform's best configuration.
- **Parallel first touch.** Inputs are allocated uninitialised and first written by a parallel torch op under the final thread placement, so pages land on the bound node (STREAM practice).

## 4. Timed region (`bench/core.py`)
- **Kernel only.** These are **never** timed: allocation, input generation, host↔device transfers, plan creation, autotuning, inspector-executor analysis and `torch.compile` compilation.
- **Setup time** is recorded as `setup_ms`.
- **GPU transfers:** `h2d_ms` / `d2h_ms` (pinned buffers) and `e2e_ms = h2d + kernel + d2h` are separate columns.

## 5. Repetition and statistics (`bench/core.py`, `bench/stats.py`)
| Parameter | Value |
|---|---|
| Warmup | ≥ 5 calls and ≥ 0.5 s, then continue until the median of the last 5 calls changes by < 2% (cap 10 s). Kernels whose calls take ≥ 1 s need only 2 warmup calls: the first absorbs lazy initialisation, the second confirms the steady state. |
| Measurement | ≥ 30 samples and ≥ 1 s of measured time. For long kernels the run stops after 30 s once ≥ 10 samples exist. Caps: 1000 samples, 900 s. |
| Sample | CPU: `perf_counter_ns` around k calls. GPU: one CUDA event pair around k calls. k = 1 unless one call is < 50 µs, in which case k is chosen so a sample is ≥ 1 ms; the sample is divided by k. |
| Trials | 3 independent process launches per configuration. The configuration order is shuffled per trial with a recorded seed. |
| Reported per row (one trial) | median, IQR, min, coefficient of variation, 95% bootstrap CI of the median (2000 resamples, fixed seed) |
| Stability flag | `flag=unstable` if CV > 5% |
| Analysis | Rows from all trials are pooled using their raw samples (`samples/*.csv.gz`). GPU/CPU ratios are ratios of medians, with a bootstrap CI. |

## 6. Working set (`bench/devices.py`, workload `working_set_bytes`)
- Every row records `working_set_bytes` and `ws_over_llc`.
  - LLC is the L3 of one socket, or the whole node under `node` placements.
  - GPU LLC is the device L2.
- **Bandwidth claims** use only rows with `ws_over_llc ≥ 4` (`regime=dram`). Smaller rows are `regime=cache`.
- **Size grids are absolute and identical on all devices.** A size that does not fit a device is recorded as `status=oom`. GPU/CPU ratios use only sizes present on both devices.

## 7. Equal work
- CPU and GPU produce the **same, fully materialised output**: same precision, same algorithm class and the same inputs (same seed).
- Each side uses its fastest well-known library.
- Each workload's module documents its exact output contract.

## 8. Validation (`bench/validate.py`)
- **Runs on every configuration, at full size, before timing.** Rows that fail are written with `status=validation_failed` and excluded from analysis.

| Kind | Check | Tolerance |
|---|---|---|
| Element-wise / BLAS-1 | normwise ‖y − y_ref‖₂ / ‖y_ref‖₂, reference in fp64 | `c · ε(dtype)`, c = 8 |
| GEMM-like (k-term reductions) | `validate.gemm_check`, two independent checks, both at full size in fp64. (a) **Row-scaled Freivalds:** max_i \|(C r − A(B r))_i\| / (‖A_i,:‖ · ‖colnorm(B)∘r‖), Gaussian r, 2 trials. (b) **2048 sampled entries** recomputed exactly and checked componentwise against c·(ε_out\|ref\| + ε_acc√k‖A_i‖‖B_j‖). `max_err` is reported in units of tolerance (`tol = 1`). | (a) `c · (ε_out + ε_acc · √k)`; (b) as stated; c = 8 |
| Statistical (Monte Carlo) | \|price − analytic\| ≤ 4 · standard error | — |
| Exact (sort, histogram, search, crypto) | bit-exact match against the reference / official test vectors | 0 |

`ε` is the unit roundoff of the *compute* dtype; bf16 ε = 2⁻⁸. TF32 is disabled, so fp32 means fp32.

**What the GEMM check guarantees.** The unit tests (`tests/torch/test_bench.py`) show that it rejects each of the following, while correct results pass:
- a missing or uninitialised tile;
- one row wrong by 0.1%;
- transposed output;
- reduced-precision inputs: bf16-rounded operands, which is what silent TF32 produces.

**What it cannot guarantee.** No O(n²) check can reliably find *one* slightly wrong element in an n×n result. The bounds are the principled probabilistic ones, √k (Higham & Mary 2019), and are not tuned to observed errors.

## 9. Precision
- Headline precision is fp32; Monte Carlo is fp64, as in the paper.
- Reduced-precision or fixed-function variants (bf16 tensor cores, AMX, TF32) are separate rows with `variant=ff`. They run only where the hardware supports them natively (`devices.native_bf16`); elsewhere the row is `status=unsupported` and emulated results are never reported.

## 10. Metrics (`bench/workloads/*`, `bench/metrics.py`)
- Each workload defines its FLOP count, its **compulsory** bytes and one headline metric in paper units. A column never carries a unit other than its name.
- `pct_measured_peak` is relative to the device's calibrated ceilings in `results/calibration/<device>.json`:
  - Bandwidth workloads: BabelStream Triad.
  - Compute workloads: measured GEMM peak for the same precision.
- `pct_datasheet_peak` is reported separately.
- **Bandwidth ceiling convention.** Workloads count compulsory bytes, i.e. real DRAM traffic. STREAM and BabelStream count Triad as 3 arrays, but a normal CPU store first reads the target line (write-allocate), so Triad really moves 4 arrays and Copy 3.
  - CPU ceiling = max(Dot, Triad·4/3, Copy·3/2, …). On EPYC 7402P: Triad 91.2 → 121.6, Copy 83.4 → 125.0, Dot 123.8 GB/s, consistent with SAXPY's 121 GB/s. SAXPY needs no correction because its output array is also an input.
  - GPU ceiling = reported Triad.
  - Raw per-kernel rates are kept in the calibration file.

## 11. Telemetry (`bench/telemetry.py`)
- A background sampler runs during the measured window:
  - CPU: RAPL package + DRAM energy every 50 ms (powercap, or the AMD MSRs `0xC001029B` / unit `0xC0010299`), with counter wraparound handled, plus frequency and temperature.
  - GPU: NVML power, SM clock, temperature and throttle reasons every 10 ms (idle bit masked), using the NVML energy counter where available.
- `energy_j_per_op` = window energy / calls executed in the window.
- `power_valid=0` when the window is shorter than 100 ms or a counter is unreadable. Unreadable values are NaN, never 0.

## 12. Provenance (`bench/env.py`, `bench/schema.py`)
- Each process writes `env/<run_id>.json`, and every CSV row carries its `run_id` and `git_sha`. The record covers:
  - host, CPU / GPU model, microcode, kernel, driver;
  - CUDA, cuDNN, torch configuration and BLAS backend, library versions;
  - SMT, governor, turbo, THP, NUMA balancing;
  - git SHA plus dirty flag, the full command line, and the OMP / MKL / KMP environment.
- A dirty working tree is flagged (`flag=dirty_tree`); data for the paper must come from a clean tree.

## 13. Output
- **Rows:** one per (configuration, trial), in `results/data/v2/<cpu|gpu>/<DEVICE>/<ID>_<workload>.csv`. The schema is `bench/schema.py::COLUMNS`.
- **Raw samples:** `results/data/v2/<cpu|gpu>/<DEVICE>/samples/<run_id>_<cfg>.csv.gz`.
- **Environment records:** `results/data/v2/<cpu|gpu>/<DEVICE>/env/<run_id>.json`.

## Deviations
Any deviation from this protocol, such as a library that cannot exclude its own planning time, is recorded in the workload module's docstring and in the table below.

| Workload | Deviation | Reason |
|---|---|---|
| C4 FFT (CPU) | MKL DFTI descriptors are created inside every `torch.fft` call, so CPU samples include planning time. | ATen does not expose plan reuse. pyFFTW with reusable plans is the planned alternative CPU library. |
| B4 SpGEMM | The symbolic (structure) phase and output allocation are inside every call, on both devices. | `torch.sparse.mm` has no separable phases. It is identical on CPU and GPU. |
| A1 Llama-3 | Exact Llama-3.2-1B/3B and Llama-3-8B architectures with **random** weights (N(0, 0.02)). | The official checkpoints are gated by licence and token. Dense fp32/bf16 kernel time does not depend on weight values. Correctness is checked against fp64 on a 2-layer instance. End-to-end llama.cpp/vLLM rows with real weights are future work. |
| B2 LBM, C2 MC, C5 bilateral, S2 solver | Native kernels (C++/OpenMP and CUDA), one algorithm for both devices, instead of a library. | No well-known library implements these exact kernels for both devices. Each is validated against an fp64 reference, and its vectorization was verified with compiler reports. |
| C5 bilateral (OpenCV) | OpenCV approximates the range weights with a lookup table, so it is validated with tolerance 1e-2 instead of 1e-4. | It is a library approximation. Native exact kernels are 7× faster on EPYC anyway. |
| Size grids | GEMM ≤ 8192, conv ≤ 8192², sort ≤ 2²⁷ pairs; the thread-scaling sweep uses the median size. | The larger sizes cost hours per device without new information: the GEMM peak is already reached at 8192, and every bandwidth workload still has rows ≥ 4× LLC. |
| C3 conv2d (torch) | `F.conv2d` allocates its output on every call. | It has no `out=` variant. The GPU caching allocator reuses the block. On the CPU, glibc vs jemalloc (verified loaded) measured 1552 vs 1552 ms (1 channel) and 6143 vs 6306 ms (3 channels) at 16K², so allocation inside the call is not a bottleneck. Default: glibc; jemalloc is opt-in (`BENCH_ALLOCATOR=jemalloc`, row flag `alloc=jemalloc`). |

## Platform pitfalls found during validation (EPYC 7402P, c6525-100g)

Each of these would have silently produced a weak CPU baseline. The harness or setup now prevents them.

1. **pthread OpenBLAS under OpenMP binding runs on one core.**
   - Cause: `OMP_PROC_BIND` pins the calling thread to a single core, and pthread workers inherit that mask.
   - Effect: 80 vs 1760 GFLOP/s at N=4096 fp32.
   - Fix: the setup script selects the OpenMP OpenBLAS build.
   - The same issue hit oneTBB sort (10 vs 94 Mpairs/s); `_cpusort.WidenAffinity` widens the caller's mask to the placement for the duration of the call.
2. **torch's default CPU BLAS (MKL) is not the fastest on AMD.** At N=4096: BLIS 1966 / OpenBLAS ~1760 / MKL ~1700 GFLOP/s in fp32; BLIS 916 / OpenBLAS ~890 / MKL ~854 in fp64. C1 therefore times every library on the node.
3. **ATen's CPU sort scales only 5× on 24 cores** (100 Mpairs/s). libstdc++ parallel mode reaches 170–203 Mpairs/s.
4. **torch `conv2d` (oneDNN) is 3.4× slower than OpenCV `filter2D` for image filtering** (1521 vs 444 ms, 16K², k=5).
5. **`torch.bincount` scales only 1.5× on 24 cores; `torch.histc` scales 17×.** histc is therefore run in exact 2²⁴-sample chunks.
6. **File capabilities disable `LD_PRELOAD`.** Granting `CAP_SYS_RAWIO` to Python puts ld.so in secure mode, which silently drops `LD_PRELOAD=/path/libjemalloc.so.2`.
   - This invalidated a first glibc-vs-jemalloc comparison, which turned out to be glibc vs glibc.
   - Fix: preload by bare name with the set-user-ID bit on the library.
   - Whether jemalloc is active is now read from `/proc/self/maps`, not from the environment variable.
7. **Monte Carlo was not vectorized** (0.72 → 2.39 Gpaths/s once fixed). Three things each blocked AVX2 vectorization of the path loop, found with `-fopt-info-vec` and by checking libmvec `_ZGV*` symbols:
   - GCC fuses `sin(θ)`/`cos(θ)` into a scalar `sincos`; the fix computes the second normal as `cos(θ − π/2)`;
   - an array-based Philox state;
   - `omp parallel for simd` per-lane reduction arrays.
8. **OpenSSL 3 one-shot `SHA256()` has per-call provider overhead and global locks.** It reached 82 MB/s per core with 24 threads. The low-level API gives 35 GB/s (1 KB messages) and 13.3 GB/s (64 B), a 17–100× difference.
9. **Root-only counters.** RAPL `energy_uj` is root-only and `/sys/class/powercap` entries are symlinks, so `chmod -R` misses them. APERF/MPERF MSRs need `CAP_SYS_RAWIO`. Both are fixed in the setup script.

## Open items
- **Native C GEMM fp64 at N=8192:** 1600 ms native vs 1237 ms through the harness, with the same OpenBLAS `.so`, the same clock (turbostat) and the same power.
  - Ruled out: alignment, transparent huge pages, the timed loop, OpenMP binding.
  - The harness value is fully validated and cannot be an under-measurement, because the call is synchronous. To be explained with perf counters before publication.
  - Native fp32 at N=4096 is 9.7% faster than the harness's OpenBLAS (78 vs 86 ms); other cases agree within 3%.
