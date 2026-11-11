# Measurement methodology

This document is the measurement protocol for every number this project publishes. The harness in `bench/` enforces it, and each rule below names the code that implements it. Data collected any other way, including everything under `legacy/data/`, is **preliminary** and is never used in the paper.

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

**Clock sensitivity (planned, not yet collected):** SAXPY, GEMM and Histogram repeated with clocks locked (`scripts/gpu/lock_clocks.sh`, still to be parameterised) and reported separately. Until collected, all results are at default clocks only.

## 3. Threads and memory placement (`bench/devices.py`)
- **Pinning.** `OMP_PLACES` lists the exact CPUs of the placement, one place each (`{c0},{c1},…`), with `OMP_PROC_BIND=close`, `OMP_DYNAMIC=FALSE`, `MKL_DYNAMIC=FALSE`, `KMP_AFFINITY=disabled` (no double binding), and `OMP_NUM_THREADS` = `MKL_NUM_THREADS` = `OPENBLAS_NUM_THREADS` = `BLIS_NUM_THREADS` = the thread count (`bench/devices.py:placement_env`). `torch.set_num_threads` / `set_num_interop_threads(1)` are each called exactly **once per process**. Each configuration runs in its own process (`bench/sweep.py`).
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
| Analysis | Rows from all trials are pooled using their raw samples (`samples/*.csv.gz` on the node, packed into `samples.tar` after copy-back). GPU/CPU ratios are ratios of medians, with a bootstrap CI. |

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
| Exact (sort, histogram, search, crypto) | bit-exact match against the reference: official test vectors (NIST AES, FIPS SHA-256) plus the full output vs OpenSSL / hashlib for crypto; numpy (`bincount`, `searchsorted`) or structural checks (sortedness, permutation checksums, stability) for the others | 0 |

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

## 10. Metrics (`bench/workloads/*`, `Workload.work()`)
- Each workload defines its FLOP count, its **compulsory** bytes and one headline metric in paper units. A column never carries a unit other than its name.
- `pct_measured_peak` is relative to the device's calibrated ceilings in `results/calibration/<device>.json`:
  - Bandwidth workloads: BabelStream Triad.
  - Compute workloads: measured GEMM peak for the same precision.
- `pct_datasheet_peak` is reported separately.
- **Bandwidth ceiling convention.** Workloads count compulsory bytes, i.e. real DRAM traffic. STREAM and BabelStream count Triad as 3 arrays, but a normal CPU store first reads the target line (write-allocate), so Triad really moves 4 arrays and Copy 3.
  - CPU ceiling = max(Dot, Triad·4/3, Copy·3/2, …). On EPYC 7402P: Triad 91.2 → 121.6, Copy 83.4 → 125.0, Dot 123.8 GB/s, consistent with SAXPY's 121 GB/s. SAXPY needs no correction because its output array is also an input.
  - **The correction is applied only if it is consistent with Dot** (`WA_MAX_OVER_DOT` = 1.05 in `scripts/calibrate.py`). A read+write stream cannot move more DRAM traffic than a pure-read stream, so a corrected value more than 5% above Dot means that CPU did not read the written lines. Intel Sapphire/Emerald Rapids can skip that read for full-line writes (SpecI2M). On the Xeon Gold 5512U (c6620), Copy 181 → 272 corrected vs Dot 214: the correction is rejected and the ceiling is Dot, 213.6 GB/s (70% of the 307 GB/s datasheet). It was first computed as 273.5 and recomputed from the stored rates on 2026-11-11 with `calibrate.py --recompute-bandwidth`; the CSV column `pct_measured_peak` of those rows still uses 273.5, but the analysis recomputes every percentage from the calibration file.
  - GPU ceiling = reported Triad.
  - Raw per-kernel rates are kept in the calibration file.
  - **BabelStream licence (publication rule).** BabelStream (v5.0, unmodified, from github.com/UoB-HPC/BabelStream) allows publishing results, but only results that follow its Run Rules may be called "BabelStream benchmark results"; anything else must be labelled as derived. Our runs follow both rules (arrays ≫ last-level cache, 61× on V100; each kernel runs for milliseconds). When publishing, name the kernel ("BabelStream Triad, 845 GB/s"). The CPU ceiling above is **derived from** BabelStream (write-allocate correction), so it must be labelled "derived from BabelStream", never "BabelStream result".

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
- **Rows:** one per (configuration, trial), in `results/data/<cpu|gpu>/<DEVICE>/<ID>_<workload>.csv`. The schema is `bench/schema.py::COLUMNS`.
- **Raw samples:** `results/data/<cpu|gpu>/<DEVICE>/samples/<run_id>_<cfg>.csv.gz`.
- **Environment records:** `env/<run_id>.json` per job on the node; after copy-back `scripts/pack_results.py` reduces them to one `results/data/<cpu|gpu>/<DEVICE>/machine.json` (fields constant across jobs, per-placement thread settings, campaigns, git commits, count of dirty-tree jobs; anything that changed mid-campaign is listed under `changed_during_campaign`).

## Deviations
Any deviation from this protocol, such as a library that cannot exclude its own planning time, is recorded in the workload module's docstring and in the table below.

| Workload | Deviation | Reason |
|---|---|---|
| C4 FFT (CPU) | MKL DFTI descriptors are created inside every `torch.fft` call, so CPU samples include planning time. | ATen does not expose plan reuse. pyFFTW with reusable plans is the planned alternative CPU library. |
| B4 SpGEMM | The symbolic (structure) phase and output allocation are inside every call, on both devices. | `torch.sparse.mm` has no separable phases. It is identical on CPU and GPU. |
| A1 Llama-3 | Exact Llama-3.2-1B/3B and Llama-3-8B architectures with **random** weights (N(0, 0.02)). | The official checkpoints are gated by licence and token. Dense fp32/bf16 kernel time does not depend on weight values. Correctness is checked against fp64 on a 2-layer instance. End-to-end llama.cpp/vLLM rows with real weights are future work. |
| B2 LBM, C2 MC, C3 conv (`lib=native`), C5 bilateral, S2 solver | Native kernels (C++/OpenMP and CUDA), one algorithm for both devices, instead of a library. | No well-known library implements these exact kernels for both devices. Each is validated against an fp64 reference, and its vectorization was verified with compiler reports. |
| S1 Histogram (CPU) | Measured with PyTorch's built-in `histc` / `bincount` only (library-level), not a hand-tuned kernel. | Decision of 2026-11-11: the workload table names the library used. A plain OpenMP per-thread-private histogram probe on the c6620 reached 25.3 vs 4.0 Gsamples/s for `histc` (1G values, 256 bins), so CPU-vs-GPU histogram ratios measure PyTorch's CPU histogram, not the hardware. A native kernel is a Todo item. |
| A1 Llama-3 (CPU, fp32) | Linear layers are also timed through OpenBLAS and BLIS (`lib` key, as C1); the fastest library is kept. Batch-1 decode calls `?gemv` directly. | Batch-1 decode is pure GEMV; on EPYC torch/MKL streamed ~20 GB/s of a 125 GB/s socket (pitfall 11). |
| A1 Llama-3 decode (all devices) | Decode attention is two batched matmuls (q·Kᵀ, softmax, ·V) with GQA as a reshape, and the new token's K/V are written in place into a preallocated cache, instead of `torch.cat` + `scaled_dot_product_attention`. Prefill still uses SDPA. The old path is the fp64 validation reference. | SDPA's fused kernels tile over queries; with one query token most of the tile is idle. On the V100 SDPA took 47% (B=1) / 64% (B=8) of the decode step while the weight GEMVs ran at ~800 of 845 GB/s (pitfall 12). |
| A1 Llama-3 decode (GPU) | Also timed replayed from a CUDA graph (`lib=cudagraph`); the faster of eager / graph is the headline. The graph output is validated against the eager step. | Eager decode issues ~500 kernels per step from Python; on a slow host this, not the GPU, sets the time (pitfall 13). |
| B2 LBM (CPU) | Non-temporal (streaming) stores on x86/AVX; normal stores elsewhere (Grace). | The 19 destination arrays are write-only; a normal store reads each line first (write-allocate), +50% DRAM traffic. |
| C5 bilateral (OpenCV) | OpenCV approximates the range weights with a lookup table, so it is validated with tolerance 1e-2 instead of 1e-4. | It is a library approximation. (An earlier "native is 7× faster than OpenCV" measurement on EPYC 7402P is invalid: OpenCV ran on one core before commit d51fa7e; pitfall in `results/library_notes.json`.) |
| Size grids | GEMM ≤ 8192, conv ≤ 8192², sort ≤ 2²⁷ pairs; the thread-scaling sweep uses the median size. | The larger sizes cost hours per device without new information: the GEMM peak is already reached at 8192, and every bandwidth workload still has rows ≥ 4× LLC. |
| C3 conv2d (torch) | `F.conv2d` allocates its output on every call. | It has no `out=` variant. The GPU caching allocator reuses the block. On the CPU, glibc vs jemalloc (verified loaded) measured 1552 vs 1552 ms (1 channel) and 6143 vs 6306 ms (3 channels) at 16K², so allocation inside the call is not a bottleneck. Default: glibc; jemalloc is opt-in (`BENCH_ALLOCATOR=jemalloc`, row flag `alloc=jemalloc`). |

## Platform pitfalls found during validation (EPYC 7402P, c6525-100g)

Each of these would have silently produced a weak CPU baseline. The harness or setup now prevents them.

1. **pthread OpenBLAS under OpenMP binding runs on one core.**
   - Cause: `OMP_PROC_BIND` pins the calling thread to a single core, and pthread workers inherit that mask.
   - Effect: 80 vs 1760 GFLOP/s at N=4096 fp32.
   - Fix: the setup script selects the OpenMP OpenBLAS build.
   - The same issue hit oneTBB sort (10 vs 94 Mpairs/s); `_cpusort.WidenAffinity` widens the caller's mask to the placement for the duration of the call.
2. **torch's default CPU BLAS (MKL) is not the fastest on AMD.** At N=4096: BLIS 1966 / OpenBLAS ~1760 / MKL ~1700 GFLOP/s in fp32; BLIS 916 / OpenBLAS ~890 / MKL ~854 in fp64 (EPYC 7402P, Zen 2; the raw rows were lost with that campaign, so re-measure). On EPYC 9354P (Zen 4) the order changes again: OpenBLAS wins fp32 (3495 vs MKL 3066), MKL wins fp64 (1643 vs OpenBLAS 1542), and Ubuntu 22.04's BLIS 0.8.1 is ~3.5× slower (generic kernels; see pitfall 14 for the BLIS 2.1 fix). C1 therefore times every library on the node.
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

10. **LBM was not vectorised** (110–118 MLUPS, 27% of bandwidth). The x loop kept the 19 populations in an `f[19]` array read through a table of row pointers; GCC vectorised only the 19-direction inner loops. The kernel is now generated as 19 named scalars with constants folded (x loop vectorised, checked with `-fopt-info-vec`) and writes through streaming stores. Laptop check (4 threads, AVX2): 56 → 198 MLUPS fp32, 57 → 99 fp64; to be re-measured on EPYC.
11. **LLM decode on EPYC ran at single-core bandwidth.** Every A1 decode row streamed 13–22 GB/s (10–18% of the socket), even at batch 1, where the step is almost pure GEMV over the weights; the same torch/MKL GEMV runs at the full read rate on Intel. Suspected cause: MKL's GEMV path on AMD. A1 now also times OpenBLAS/BLIS; confirm the cause on the next AMD node. (Confirmed on Zen 4, r6615: OpenBLAS 9.3 vs torch/MKL 0.9 tok/s at 8B batch 1.)
12. **Decode attention dominated the LLM step.** `F.scaled_dot_product_attention` with a single query token (decode) picked a kernel that tiles over queries; on the V100 it took 47–64% of the step for a KV cache that is 3–18% of the bytes, plus ~8% for `torch.cat` re-copying the cache each step. Profile the decode step (torch profiler) on every new device type.
13. **Eager GPU decode can be host-launch-bound.** One decode step issues ~500 small kernels from Python. On the V100 node (Xeon E5-2667 v4, 2016) issuing them took 16 ms vs 8.5 ms of GPU work, and because each step must finish before the next token exists, the per-step time was set by the host CPU (llama3.2-1b batch 1: 44 tok/s eager vs 118 replayed from a CUDA graph). GPU decode is therefore also timed as a CUDA graph (`lib=cudagraph`, standard practice in vLLM / TensorRT-LLM); eager rows are kept and the faster is the headline. The CPU side pays the same Python per-op cost (an estimated 10–20% of a 1B batch-1 step); `torch.compile` for both sides is future work.

14. **Distribution BLAS builds may not recognise new CPUs.** Ubuntu 22.04's OpenBLAS 0.3.20 detects Emerald Rapids (c6620) as Prescott, a generic 2004 kernel: 848 vs 3479 GFLOP/s fp32. `_blas.py` sets `OPENBLAS_CORETYPE` (SkylakeX/Haswell) only when detection falls back to a generic core. Its BLIS 0.8.1 runs Zen 4 (r6615) on 'generic' kernels (960 vs 2923 GFLOP/s), and its `BLIS_ARCH_TYPE` override aborts, so `scripts/setup_node.sh` builds the pinned upstream BLIS 2.1 into `~/.local/blis` only in that case. 2.1 has no Zen 4 (AVX-512) kernel set and uses its Zen 3 kernels. Every row records the kernel that ran (`core=…` for OpenBLAS, `arch=…` for BLIS). Check both strings on every new CPU.

## Open items
- **Thread-scaling evidence for S rows:** S1 and S2 are now in every CPU config's scaling list (before, only B1 and C1 were, so the S-bound check had no data).
- **Native C GEMM fp64 at N=8192:** 1600 ms native vs 1237 ms through the harness, with the same OpenBLAS `.so`, the same clock (turbostat) and the same power.
  - Ruled out: alignment, transparent huge pages, the timed loop, OpenMP binding.
  - The harness value is fully validated and cannot be an under-measurement, because the call is synchronous. To be explained with perf counters before publication.
  - Native fp32 at N=4096 is 9.7% faster than the harness's OpenBLAS (78 vs 86 ms); other cases agree within 3%.
