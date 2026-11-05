# Handover: CPU vs GPU benchmark project (state as of 2026-11-06)

This is for an AI coding session picking up the work. **Verify every claim here against the code, git log and data before acting on it.** If something here disagrees with the repo, the repo wins; note the discrepancy.

## 1. The project in one paragraph
We are replicating ISCA'10 *"Debunking the 100X GPU vs. CPU Myth"* (Lee et al.) on modern hardware. That paper showed that GPU-vs-CPU speedups shrink to about 2.5× once **both** sides run properly optimized code. Our suite has **16 workloads**, each tagged by what limits it:
- B = bandwidth
- C = compute
- S = synchronization
- A = application

We run it on **6 core devices**:
- CPUs: Intel Xeon SPR (CloudLab c6620), AMD EPYC 7402P (CloudLab c6525-100g), NVIDIA Grace
- GPUs: A100, L40S, H100 (on GH200)

Repo: `/home/kushal/cpu gpu project/cpu-gpu`. GitHub: `https://github.com/23110281/cpu-gpu` (branch `main`, GitHub Pages dashboard built from `results/`).

## 2. Ground rules from the user
- **Commit locally only. Do not push** until the new suite has been tested end to end on one CPU node AND one GPU node, and the user asks. See §7 for unpushed commits.
- Narrate progress step by step in plain language. The user wants to follow along.
- Prefer PyTorch with the fastest well-known library on each side. Keep a publication standard.
- Re-collect all data. Old data under `results/data/legacy/` is frozen, shown as "preliminary", and **never used in the paper**.
- Keep the native C/CUDA benches only as cross-checks.
- Ask before anything hard to reverse: deleting data, pushing, releasing the CloudLab node.

## 3. Architecture (what replaced what)
**Primary data source:** the v2 harness in `src/torch/bench/` (~4.4k lines). Run with `PYTHONPATH=src/torch python3 -m bench ...`.

| File | Role |
|---|---|
| `core.py` | `TimingPolicy`, adaptive warmup, per-sample timing (CPU `perf_counter_ns`, GPU CUDA events), small-call batching, `Workload` base class, `run_one()` (validates first, then times, then writes the row) |
| `devices.py` | lscpu topology; placements `socketS[-smt][:T]` / `node`; OMP pinning env; numactl prefix; LLC size; bf16 nativeness; TF32 off; opt-in jemalloc |
| `env.py` | provenance JSON per run: versions, clocks, governor, git sha, `dirty_tree` |
| `schema.py` | the single long CSV schema (`COLUMNS`), flock append, raw samples as `.csv.gz` |
| `stats.py` | median, IQR, CV, bootstrap 95% CI of the median, ratio CI |
| `validate.py` | normwise checks; row-scaled Freivalds plus sampled entries (GEMM); exact match; 4·SE statistical check |
| `telemetry.py` | RAPL powercap, AMD MSR (0xC001029B / 0xC0010299), APERF/MPERF frequency, NVML; NaN, never 0, when unreadable |
| `sweep.py` | one **subprocess per (config, placement, trial)**; shuffled trial-major order; resumable by `--campaign`; preflight memory check; `--workloads B2,C3` subsets |
| `workloads/` | one module per workload: `b1_saxpy, b2_lbm, b3_spmv, b4_spgemm, c1_gemm, c2_montecarlo, c3_conv, c4_fft, c5_bilateral, c6_sort, c7_search, c8_aes, c9_sha256, s1_histogram, s2_solver, a1_llama`. Helpers: `_common, _blas, _cpusort, _sparse, _mc_ext, _crypto_ext`. |

**Same module for CPU and GPU.** Each workload is a single file and the device is a runtime argument (`--device cpu|cuda`, or `device_class` in `configs/devices/<ID>.json`). There are no separate CPU/GPU folders in v2. Within a module:
- `configs(device_class)` adds CPU-only library variants: BLIS/OpenBLAS (C1), OpenCV (C3, C5), GNU-parallel/TBB sort (C6), OpenSSL hw-off variants (C8, C9). It also adds GPU-only variants: S2 CUDA graph vs cooperative launch, C5 `__expf`.
- Native kernels (`load_inline`) have a C++/OpenMP version and a CUDA version of the **same algorithm**. This covers B2, C2, C5, C8 (GPU T-table AES; CPU uses OpenSSL EVP), C9 (CPU OpenSSL low-level, GPU custom) and S2.
- Validation, the work model, metrics, the CSV schema, statistics and the timing policy are shared across devices.

**Supporting files:**
- `docs/methodology.md`: the written protocol. It contains a deviations table, the platform pitfalls found on EPYC, and the open items. **Read it fully.**
- `configs/devices/*.json`: 6 device configs (AMD has datasheet ceilings). GPU configs are minimal; memory is detected at run time.
- `scripts/setup_v2_harness.sh` (shared), `scripts/cpu/setup_cloudlab_{epyc,xeon}.sh`, `scripts/gpu/setup_gpu_node.sh`: node setup.
- `scripts/calibrate.py`: BabelStream v5.0 plus best-library GEMM, written to `results/calibration/<ID>.json`.
- `scripts/crosscheck_native.py`: harness vs native C benches, written to `results/crosscheck/<ID>.json`.
- `results/plotting/v2/analyze.py` (`make analysis`) writes:
  - `results/analysis/<ID>_{headline,libraries,bounds}.csv`
  - `results/analysis/ratios.csv`
  - `results/plots/v2/<ID>/*.png`
- `tests/torch/test_bench.py`: 38 tests. On the laptop 23 pass and 15 skip (GCC 9 cannot build torch's C++20 extensions; no OpenBLAS/BLIS; broken system pandas). All must pass on the next CPU node before the sweep.
- `Makefile` targets:
  - `bench_smoke`, `bench_smoke_cuda`, `bench_list`
  - `torch_suite DEVICE=<ID>` / `torch_suite_bg`, `bench_suite`
  - `analysis`, `index` (dashboard), `test_bench`
  - `cpu_all` (native cross-check binaries)

**Legacy code still in the repo:**
- `src/cpu/cpu_{gemm,gemv,saxpy,spgemm,montecarlo}_bench.c` and `src/gpu/gpu_*_bench.cu`: cross-checks only.
- `src/gpu/gh200/*`: C2C residency study (not fixed, see §6).
- Old per-workload sweep shell scripts in `scripts/cpu/` and `scripts/gpu/`: legacy drivers, not used by v2.

Deleted: PCA, k-means, `cpu_cryptography_bench.c`, the old `torch_*_bench.py` scripts, torch_telemetry, bench_llama32_layers.

## 4. Measurement protocol (summary; full version in `docs/methodology.md`)
- **Unit of comparison:** one socket (numactl membind, OMP_PLACES pinning, parallel first touch) against one GPU.
- **Placements:**
  - Headline: the best of `socket0` and `socket0-smt`.
  - Thread scaling: at the median size only.
- **Timed region:** kernel only. GPU H2D/D2H times are recorded separately (`h2d_ms`, `d2h_ms`, `e2e_ms`).
- **Samples:** adaptive warmup. At least 30 samples and at least 1 s; for calls of 1 s or more, at least 10 samples.
- **Trials:** 3 process-isolated trials.
- **Reported statistics:** median, IQR, bootstrap CI. `flag=unstable` if CV > 5%.
- **Validation:** full size, before timing, on every config. Status is one of `ok | validation_failed | oom | unsupported | error`.
- **Precision:**
  - fp32 headline; Monte Carlo in fp64.
  - TF32 off.
  - bf16 only where native; EPYC Zen 2 records `unsupported`.
- **Bandwidth ceiling:** DRAM traffic including write-allocate, `max(Dot, Triad·4/3, Copy·3/2)`. On EPYC this is 125.1 GB/s.
- **Bound-label check:**
  - B rows must reach ≥ 60% of the measured bandwidth.
  - C rows are compared with the measured peak (the C1 best).
  - S rows: scaling efficiency, and S2 batch sensitivity.

## 5. Status per workload (from the lost EPYC campaign; no GPU path has ever run)
Rows marked *fixed 2026-11-06* were changed after that campaign and need re-measuring.

| ID | Implementation | EPYC status |
|---|---|---|
| B1 SAXPY | torch `add_` | 96.7% of bandwidth ✓ |
| B2 LBM D3Q19 | native C++/CUDA | was 27% of bandwidth. *Fixed 2026-11-06:* CPU kernel generated as 19 scalars (x loop now vectorised) + streaming stores; laptop 3.5x fp32 / 1.7x fp64. |
| B3 SpMV | torch sparse CSR (MKL/cuSPARSE) | 62% of bandwidth ✓ (synthetic matrices; SuiteSparse still TODO) |
| B4 SpGEMM | torch.sparse.mm | ok; symbolic phase inside the timer (documented) |
| C1 GEMM | torch/MKL, OpenBLAS, BLIS on CPU; cuBLAS on GPU | at peak: BLIS 1997 fp32 / ~954 fp64 GFLOP/s ✓ |
| C2 Monte Carlo | native Philox4x32-10 (KAT-tested), libmvec | 2.48 Gpaths/s ✓ |
| C3 conv2d | torch/oneDNN, OpenCV, native | was ~2% of peak. *Fixed 2026-11-06:* `lib=native` direct conv (CPU OpenMP+SIMD, CUDA smem tiles), K unrolled; laptop 5–6x torch. A 4-row register-blocked variant was tried and was slower (GCC). |
| C4 FFT | torch.fft (MKL/cuFFT) | ok; CPU timing includes plan creation (deviation); pyFFTW variant TODO |
| C5 bilateral | OpenCV + native exact; CUDA expf / `__expf` | ✓ (native 7× faster than OpenCV) |
| C6 sort | ATen, GNU parallel, oneTBB | ✓ |
| C7 index search | torch.searchsorted | ✓ after the int32 overflow fix in 8ebecfa; the one errored 2^29 config must be re-run |
| C8 AES-128-CTR | OpenSSL EVP on CPU (plus hw-off variant); CUDA T-table | ✓ NIST vectors; CPU and GPU ciphertext must match byte for byte (GPU side untested) |
| C9 SHA-256 | OpenSSL low-level on CPU; custom CUDA | ✓ |
| S1 histogram | histc (chunked to 2^24 samples), bincount | ✓ |
| S2 PGS solver | native OpenMP barrier per batch; CUDA graph / cooperative grid.sync | ✓ CPU. *Fixed 2026-11-06:* removed `-rdc=true -lcudadevrt` (torch's load_inline has no CUDA device-link step, so the module would not load). |
| A1 Llama-3 | exact 3.2-1B / 3.2-3B / 3-8B architectures, random weights, eager + SDPA; CPU fp32 linears also via OpenBLAS/BLIS | was 10–18% of bandwidth on every decode row, even B=1 (8B B=1: 0.72 tok/s vs ~3.9 bound; B=8: 4.0 vs ~31). KV `cat` is ~3% of traffic, so not the cause. MKL GEMV runs at full read rate on Intel, so suspect MKL GEMV on AMD. *Fixed 2026-11-06:* `lib` variants openblas/blis (gemv at B=1). Confirm on the next AMD node. |

## 6. Old (legacy) code and data
- **Old data** (`results/data/legacy/`): 5 CPUs and 6 GPUs, including GH200 residency sweeps. It is not research grade:
  - no or partial validation (old AMD SpGEMM 0/192 rows validated but marked ok; old GEMM 19/43 unvalidated);
  - cache-resident SAXPY reported as "1007 GB/s bandwidth";
  - a slower library (old GEMM 1.48 vs new 2.0 TFLOP/s fp32);
  - means of 5–20 iterations with no CI;
  - fake kernels (GPU AES = XOR, Grace AES 1 round, GH200 "Monte Carlo" = closed form, Grace sort never merges);
  - unequal work between CPU and GPU (Llama fp32 vs bf16);
  - wrong AMD MSR addresses; 0 W when power was unreadable.
- **Native cross-check benches, fixed:** SpGEMM `nanosleep` fabrication removed (now `unsupported` without MKL); MSR addresses; real OpenBLAS label; status taken from validation.
  - **Not done:** per-iteration samples and median in the native benches.
- **GH200 C2C/residency benches: not fixed.** They need one `t_e2e` definition, NVML, validation, removal of the cpu_local stub paths, and the Makefile build of `protocol_benchmark_suite.cu`. They need a GH200 to test.
  - This is a separate system-axis study. Its old data must not be mixed into the main comparison.
  - Whether to keep it at all is the user's call.

## 7. Unpushed commits
`git log origin/main..main`. GitHub `main` is still `82dd457`. The 7 commits from 2026-11-05 (`45a8e8c` … `8ebecfa`) plus the 2026-11-06 fix commits (S1/S2 scaling, analysis fixes, preflight provenance, S2 rdc, B2/C3/A1 baselines, docs).

## 8. The EPYC campaign of 2026-11-05 is lost
- Campaign `c20261105T022449Z_node0` on `amd249.utah.cloudlab.us` (c6525-100g) finished ~10:40 UTC, but the experiment ended before the data was copied back; the node is gone (ssh times out).
- Only a 03:22 UTC snapshot was ever pulled (181 of 1557 rows, trial 1). The user chose not to keep it. The native cross-check result is in `results/crosscheck/AMD_C6525_100G.json`.
- **Never again:** the next run must copy results back during the run (planned "group A": laptop-side rsync loop + a committed launcher with a fixed campaign id). The old launcher `/tmp/final_run.sh` began with `rm -rf results/data/v2`: never re-run anything like it to resume; resume with `bench sweep --campaign <id>`.
- Resume skips only rows with a terminal status (`ok|validation_failed|oom|unsupported`); `error` rows are re-run and the analysis ignores them, so they never need deleting.

## 9. Next work, in priority order
1. **CPU node test session** (user will book): setup, `pytest` (all 38 must pass, including the new B2/C3/A1 native and BLAS tests), smoke, calibration, full sweep, pull, `make analysis`, review `bounds.csv` (B2 >= 60% bandwidth? C3 % of peak? A1 decode GB/s by library? S1/S2 scaling).
2. **GPU node test session**: same, plus `make bench_smoke_cuda`. First hour = fix any CUDA build/launch issue (none of B2, C2, C3, C5, C8, C9, S2 CUDA code has ever been compiled). Check C8/C9 CPU vs GPU byte-identical output.
3. Group A (user deferred): automated run + periodic copy-back.
4. GEMM cross-check outliers (perf counters); native benches per-iteration samples; `lock_clocks.sh` (decide keep/drop); GH200 C2C study (decide keep/drop).
5. SuiteSparse for B3/B4; pyFFTW for C4; llama.cpp rows.
6. Intel SPR, Grace, remaining GPUs; cross-device analysis. Push only after step 1 and 2 pass and the user asks.

## 10. Known pitfalls (do not rediscover them)
- **Thread affinity:**
  - pthread OpenBLAS and oneTBB under `OMP_PROC_BIND` run on one core. Use the OpenMP OpenBLAS build (update-alternatives); `_cpusort.WidenAffinity` handles TBB.
- **Library choice:**
  - MKL is not the fastest BLAS on AMD; BLIS is. That is why C1 times every library.
  - ATen sort scales poorly. torch.bincount scales only 1.5×, so histc is chunked to exact 2^24 samples.
  - oneDNN conv2d is 3.4× slower than OpenCV for image filtering.
- **Native kernels:**
  - GCC fuses sin/cos into `sincos`, which blocks vectorization. Use `cos(θ−π/2)`. Check with `-fopt-info-vec` and the libmvec `_ZGV*` symbols.
  - OpenSSL 3 one-shot `SHA256()` has lock contention. Use the low-level API.
- **Permissions and environment:**
  - setcap (CAP_SYS_RAWIO for MSR) puts ld.so in secure mode and drops `LD_PRELOAD`. jemalloc is detected via `/proc/self/maps`.
  - RAPL `energy_uj` is root-only and the powercap entries are symlinks. chmod the resolved paths (the setup script does this).
- **Harness design:**
  - `torch.set_num_interop_threads` can only be set once per process. That is why there is one subprocess per config.
- **Laptop limits:**
  - GCC is too old for C++20 extensions, and there is only 3 GB RAM. Test native extensions on the node.
  - The laptop's system pandas is broken (numpy ABI). Use the scratchpad venv or the node.
- **Remote commands:**
  - Complex quoting over ssh breaks. Write helper scripts and heredocs.
- **Analysis:** fixed 2026-11-06: labels come from the `library` column when there is no `lib` key; scaling efficiency is per (problem, library) with the full-socket row as the top point (it used to median across libraries/precisions).
