# Handover: CPU vs GPU benchmark project (state as of 2026-11-11)

This is for an AI coding session picking up the work. **Verify every claim here against the code, git log and data before acting on it.** If something here disagrees with the repo, the repo wins; note the discrepancy. Keep this file current: update it at every milestone and delete what is no longer true (the user's standing request).

## 1. The project in one paragraph
We are replicating ISCA'10 *"Debunking the 100X GPU vs. CPU Myth"* (Lee et al.) on modern hardware. That paper showed that GPU-vs-CPU speedups shrink to about 2.5× once **both** sides run properly optimized code. Our suite has **16 workloads**, each tagged by what limits it: B = bandwidth, C = compute, S = synchronization, A = application.

Target devices:
- **Core:** Intel Xeon (c6620), AMD EPYC (c6525-100g Zen 2; r6615 Zen 4), NVIDIA Grace; A100, L40S, H100/GH200.
- **Extra:** V100 (c4130).

Repo: `/home/kushal/cpu gpu project/cpu-gpu`. GitHub: `https://github.com/23110281/cpu-gpu` (branch `main`; GitHub Pages dashboard built from `results/` by `.github/workflows/plots.yml`).

## 2. Ground rules from the user
- **Push only when the user asks.** Fixes stay as local commits until they have been tested on real hardware. See §8.
- Narrate progress step by step in plain language. The user wants to follow along, so take things slowly and explain.
- Prefer PyTorch with the fastest well-known library on each side. Keep a publication standard.
- Old data under `results/data/legacy/` is frozen, shown as "preliminary", and **never used in the paper**.
- Ask before anything hard to reverse: deleting data, pushing, releasing a CloudLab node, or a code change that alters measured results.
- Long runs on a node: `setsid -f nohup <launcher> >> log 2>&1 < /dev/null` (survives the laptop sleeping). The user backs up results when they choose.

## 3. Architecture
**Primary data source:** the v2 harness in `src/torch/bench/`. Run with `PYTHONPATH=src/torch python3 -m bench ...`.

| File | Role |
|---|---|
| `core.py` | `TimingPolicy`, adaptive warmup, per-sample timing (CPU `perf_counter_ns`, GPU CUDA events), small-call batching, `Workload` base, `run_one()` (validate, then time, then write the row) |
| `devices.py` | lscpu topology; placements `socketS[-smt][:T]` / `node`; OMP pinning env; numactl membind over all NUMA nodes of the placement (NPS4-safe); LLC summed over instances; bf16 nativeness (CPU flags, or GPU capability ≥ 8.0); TF32 off |
| `env.py` | provenance per run: versions, clocks, governor, git sha, `dirty_tree` (src/configs/scripts/Makefile only) |
| `schema.py`, `stats.py`, `validate.py`, `telemetry.py` | CSV schema; median, IQR, bootstrap CI; validation; RAPL/MSR/NVML (NaN when unreadable, never 0) |
| `sweep.py` | one subprocess per (config, placement, trial); resumable with `--campaign <id>` (skips `ok, validation_failed, oom, unsupported`; re-runs `error`); preflight OOM when ws × 2.5 > memory (row carries git sha) |
| `workloads/` | `b1_saxpy b2_lbm b3_spmv b4_spgemm c1_gemm c2_montecarlo c3_conv c4_fft c5_bilateral c6_sort c7_search c8_aes c9_sha256 s1_histogram s2_solver a1_llama`. Helpers: `_common` (incl. `widened_affinity`, `cuda_declarations`, `build_error_summary`), `_blas`, `_cpusort`, `_sparse`, `_mc_ext`, `_crypto_ext` |

Each workload is **one module for CPU and GPU**; the device is a runtime argument. Native kernels (`torch.utils.cpp_extension.load_inline`) have a C++/OpenMP version and a CUDA version of the same algorithm.

**Fastest-library rule:** library choice is a config key `lib`. The analysis defines a problem as params minus `lib`, and the headline is the best (lib, placement) per problem.

**Supporting files:**
- `docs/methodology.md`: the written protocol, deviations table, platform pitfalls, open items. **Read it fully.**
- `configs/devices/*.json`: `AMD_C6525_100G, AMD_R6615, INTEL_C6620, NVIDIA_GRACE, NVIDIA_A100, NVIDIA_L40S, NVIDIA_GH200_480GB, NVIDIA_V100`. Each config holds the placements, the thread-scaling list (B1, C1, S1, S2) and datasheet ceilings. It is a label and a run plan; the code is not tuned per machine.
- Setup:
  - `scripts/setup_v2_harness.sh [cpu|gpu]`. GPU: picks the torch wheel matching nvcc's major version **and** containing kernels for the GPU's compute capability.
  - `scripts/gpu/install_cuda.sh`: driver 580 on Volta/Pascal/Maxwell; override with `DRIVER_BRANCH`.
- `scripts/calibrate.py`: BabelStream v5.0 (built for the GPU's own sm_XY) plus best-library GEMM, written to `results/calibration/<ID>.json`.
- Analysis:
  - `results/plotting/v2/analyze.py` writes `<ID>_{headline,libraries,bounds,scaling}.csv`, `ratios.csv` and plots.
  - `results/plotting/v2/library_report.py` writes `library_report.{csv,json,md}` and `library_summary.csv`, with notes from `results/library_notes.json` (curated: verified vs hypothesis).
  - Both run via `make analysis`. Env overrides: `ANALYSIS_DATA`, `ANALYSIS_CAL`, `ANALYSIS_OUT`, `ANALYSIS_PLOTS`.
- Dashboard: `results/generate_index.py` (`make index`). A (workload, device) cell is "collected" if v2 rows exist under `results/data/v2/{cpu,gpu}/<ID>/<WID>_*.csv`, else "preliminary" from legacy.
- `tests/torch/test_bench.py`: 29 test functions (41 cases). Laptop: 26 pass, 15 skip (old GCC). The r6615 and V100 nodes passed all of them.

Legacy code still in the repo: native `src/cpu/*.c` and `src/gpu/*.cu` benches (cross-checks only), `src/gpu/gh200/*` (C2C study, not fixed), and old sweep shell scripts.

## 4. Measurement protocol (summary)
- One socket (numactl membind, OMP pinning) vs one GPU. Headline = best of `socket0` and `socket0-smt`.
- Kernel-only timing. GPU H2D/D2H are recorded separately.
- At least 30 samples and 1 s per run, 3 process-isolated trials. `flag=unstable` if CV > 5%.
- Every config is validated at full size before timing.
- fp32 headline (Monte Carlo fp64); TF32 off; bf16 only where native (otherwise `unsupported`).
- Bandwidth ceiling:
  - CPU: DRAM traffic including write-allocate.
  - GPU: BabelStream Triad as reported.
- Bound check: B rows ≥ 60% of measured bandwidth; C rows as % of the measured GEMM peak.
- **Clocks are not locked.** This is now shown to matter (§6, V100 GEMM). Whether to lock is the user's decision.

## 5. Hardware sessions so far

### r6615 (Clemson, AMD EPYC 9354P Zen 4, 32C, NPS4), 2026-11-10, 4 h, node released
- **Data:** `results/data/test/r6615_ctest_20261110/` (README inside), committed and pushed. Test data: one trial, partial grid, **not paper data**.
- **Analysis:** `results/analysis/test/r6615_20261110/`.
- **Passed:** setup, all tests, smoke 35/35 (incl. bf16).
- **Calibration:** 364.1 GB/s; fp32 3495 (OpenBLAS), fp64 1643 (MKL) GFLOP/s.
- **Targeted sweep, 136/136 ok:**
  - B2: 80–84% of bandwidth (was 27% before the 11-06 fix).
  - C3 native: wins 18/18 at 15–45% of peak.
  - A1 decode: batch 1, OpenBLAS 9.31 vs torch/MKL 0.89 tok/s (8B), so MKL GEMV is slow on AMD. Batch 8: 9–13% of bandwidth in every library.
- **Found and fixed:** OpenCV ran on one core inside the harness. torch's OMP binding pins the main thread and OpenCV's pthreads inherit it; fixed with `_common.widened_affinity` (d51fa7e). All OpenCV rows (C3, C5) collected before d51fa7e are invalid.

### V100 (Wisconsin c4130, 4× V100-SXM2-16GB; we use GPU 0), reserved to ~2026-11-20
- **Node:** `ssh -i ~/.ssh/cloudlab rudra281@c4130-110133.wisc.cloudlab.us`, repo at `~/cpu-gpu`. Driver 580.196.07, nvcc 12.9, torch 2.14.1+cu126.
- **Smoke:** 23 ok + 2 bf16 unsupported (expected). C8/C9 GPU output byte-identical to the CPU reference.
- **Calibration:** Triad 845.2 GB/s (94% of 900), GEMM fp32 14.1, fp64 7.0 TFLOP/s.
- **Full sweep:** campaign `c_v100_20261110`, 3 trials, 621 jobs, done 2026-11-10 19:30 UTC.
  - 516 ok, 27 unsupported (bf16), 78 preflight OOM (16 GB), 0 errors, 8 unstable (all tiny problems).
  - 10 early rows from 2999f37, rest from 3b1a3f2.
- **Data:** copied to `results/data/test/v100_c4130_20261110/` (untracked; also still on the node).
- **Analysis:** `results/analysis/test/v100_20261110/`.
- **Found and fixed (local commits 0fb3d5b..3b1a3f2):**
  - Driver 620 has no Volta support: pin 580.
  - The cu128/cu129 torch wheels have no sm_70 code: the setup now checks the arch list.
  - Every CUDA extension failed with "'fn' was not declared in this scope", because load_inline's main.cpp never sees cuda_sources: declarations are now generated with `cuda_declarations`.
  - BabelStream was built for sm_80: it now builds for the GPU's own arch.
  - Build errors were truncated: `build_error_summary` now shows the compiler's error lines.

## 6. Results review (2026-11-11)

**V100, per workload, at the largest size:**

| ID | Result | Verdict |
|---|---|---|
| B1 SAXPY | 849 GB/s = 100% of Triad; matches legacy (844) | ✓ |
| B2 LBM | 95% of bandwidth (5330 MLUPS fp32) | ✓ |
| B3 SpMV | poisson3d 78% ✓; powerlaw falls to 9% (irregular gather, B label fails, as flagged) | ok, as measured |
| B4 SpGEMM | 2–9% of bandwidth (cuSPARSE via torch); B label fails | ok, as measured; the label is questionable |
| C1 GEMM | 96–100% of measured peak. Measured peak 14.1 vs legacy 15.3 TFLOP/s: the SM clock was 1392 vs 1520 MHz at the same 300 W cap (15.34 × 1392/1520 = 14.05) | ✓ ; clocks matter |
| C2 Monte Carlo | 45 Gpaths/s fp64 | ✓ plausible |
| C3 conv | native beats cuDNN 18/18 (cuDNN 0.42× median); k=3 is bandwidth-bound (~83% BW), k=7 at 27% of FMA peak | ✓ ; possible headroom at k=7 |
| C4 FFT | 1.65 TFLOP/s (5N log N) | ✓ plausible |
| C5 bilateral | ✓ (`__expf` variant 1.2× exact) | ✓ |
| C6 sort | 4.8 G pairs/s (CUB radix) | ✓ |
| C7 search | sorted queries 5–34 G/s, random 0.36–16 G/s (latency-bound at large key sets) | ✓ |
| C8 AES | 126–135 GB/s | ✓ |
| C9 SHA-256 | 52–105 GB/s | **hypothesis: below well-tuned GPU SHA-256** (per-thread message reads are uncoalesced); check before the paper |
| S1, S2 | ok; S2 CUDA graph beats cooperative launch 4/4 | ✓ |
| A1 Llama | **decode results not trustworthy (see below)**; prefill ~14 TFLOP/s ≈ peak ✓ | ✗ decode |

**A1 decode, new finding (2026-11-11, torch profiler on the V100, llama3.2-1b fp32):**
- Attention (`F.scaled_dot_product_attention`, the memory-efficient CUTLASS kernel on sm_70) takes **47% of GPU time at batch 1 and 64% at batch 8**. It reads only 0.13 / 1.07 GB of KV cache but runs ~40× slower than bandwidth: with one query token, a 64-row tile is mostly empty.
- The weight GEMVs alone run at ~800 GB/s, so the BLAS side is fine.
- The per-step `torch.cat` of the KV cache adds another ~8% at batch 8.
- **The same SDPA + cat code runs on the CPU**, so the "batch-8 decode at 9–13% of bandwidth" seen on r6615 is probably mostly this, not a "skinny GEMM" problem (that library note is now doubtful).
- **Proposed fix (not yet approved or made):**
  - For decode (S = 1), compute attention explicitly: GQA reshape, then q·Kᵀ, softmax, ·V (bandwidth-bound batched GEMV).
  - Preallocate the KV cache and write in place instead of `cat`.
  - Then re-run A1 on the V100 (while reserved) and on the next CPU.
- Diagnostic script (not in repo): run on the node with `python3 /tmp/prof_a1.py` from `~/cpu-gpu`.

**r6615 vs V100 (same problems):**
- B2 LBM: GPU 2.3–3.4× faster, roughly the bandwidth ratio 845/364 = 2.3.
- C3 native conv: 1.0–3.0× (k=3 at 4096²: 0.96×).
- A1: 1.6× (B=1) and 5.5× (B=8; skewed by the attention issue).
- These are about the paper's ~2.5×, not 100×.

**Dashboard:**
- Legacy V100 data exists as `results/data/legacy/gpu/Tesla_V100-SXM2-16GB` (Sept 2026; GEMM, GEMV, SAXPY, SpGEMM). The new v2 device id is `NVIDIA_V100`, which differs, so `workloads.json` needs mapping/renaming before the dashboard shows the v2 data in the V100 column.
- Legacy V100 GEMM also reports bf16 11 TFLOP/s on a GPU without bf16 hardware (emulated). That is another reason to replace it.
- `analyze.py` writes plots to `results/plots/v2/<ID>/`, but `generate_index.py` treats every first-level folder of `results/plots/` as a device, so "v2" would show as a device. Check before publishing v2 plots.

## 7. Open items
1. **A1 decode attention fix** (§6). Needs user approval; changes A1 results on every device.
2. Lock clocks or not (user decision). The V100 GEMM shows a 9% clock-driven difference between sessions.
3. C9 GPU SHA-256 efficiency check.
4. Freeze a release: tag one commit and collect all paper campaigns from it. CPU paths have not changed since r6615 (d51fa7e..HEAD only touched CUDA builds and error messages), but the A1 fix would change both.
5. Library provenance (BLIS arch, OpenBLAS core, MKL/oneDNN ISA) in rows. The BLIS 0.8.1-on-Zen-4 slowness is a hypothesis.
6. GEMM cross-check outliers; native benches per-iteration samples; GH200 C2C study (keep or drop: user's call).
7. SuiteSparse for B3/B4; pyFFTW for C4; llama.cpp rows.
8. Group A (deferred by user): committed launcher + periodic copy-back.
9. Next hardware: c6620 (Intel, AMX: first Intel test with the new scripts, so do the drill first), c6525-100g (Zen 2, close to r6615), r6615 again for a full campaign, then A100/A30/H100. Every new machine type: tests, smoke, calibration (~30 min) before the full sweep. First bf16-capable GPU = first ever run of the GPU bf16 paths and of sizes that were OOM on 16 GB.

## 8. Git state
- `origin/main` = `9798ed8` (pushed 2026-11-10 23:00 IST, after the r6615 session; includes the library report and r6615 data).
- **Unpushed local commits** (V100 fixes): `0fb3d5b` config, `79e2eb4` driver 580, `138a007` torch wheel arch check, `2999f37` CUDA declarations + error summary, `3b1a3f2` BabelStream arch.
- Untracked: `results/data/test/v100_c4130_20261110/`, `results/analysis/test/v100_20261110/`.

## 9. Known pitfalls (do not rediscover them)
- **Thread affinity:** pthread OpenBLAS, oneTBB and OpenCV under torch's `OMP_PROC_BIND` run on one core. Use OpenMP OpenBLAS (update-alternatives), `_cpusort.WidenAffinity` for TBB, `_common.widened_affinity` for OpenCV.
- **Library choice:**
  - MKL is not the fastest BLAS on AMD; the fastest depends on CPU and precision (Zen 2: BLIS; Zen 4: OpenBLAS fp32, MKL fp64).
  - MKL GEMV is slow on AMD.
  - Ubuntu 22.04 BLIS 0.8.1 is slow on Zen 4.
  - oneDNN depthwise conv is ~160 Mpx/s (slow path).
- **Native kernels:**
  - GCC fuses sin/cos into `sincos` (blocks vectorization).
  - OpenSSL 3 one-shot `SHA256()` has lock contention.
  - The LBM inner loop needs the 19-scalar form to vectorize.
- **CUDA builds (load_inline):**
  - Declare every exposed function in `cpp_sources` (`cuda_declarations`).
  - No device-link step, so no `-rdc`.
  - torch's CUDA major version must equal nvcc's.
  - Wheels for CUDA ≥ 12.8 have no sm_70: V100 needs cu126.
  - Bump the extension name (`_vN`) when the source changes.
- **Drivers:** branch 580 is the last with Volta/Pascal/Maxwell.
- **Permissions:** setcap puts ld.so in secure mode (drops `LD_PRELOAD`; jemalloc needs a setuid lib); RAPL `energy_uj` is root-only (chmod the resolved symlinks).
- **Harness:** one subprocess per config, because `torch.set_num_interop_threads` can only be set once per process.
- **Laptop:**
  - GCC too old for C++20 extensions; 3 GB RAM.
  - System pandas broken: use a venv (the requirements.txt pins need Python 3.11; the setup installs pandas/matplotlib unpinned).
  - Don't use `python3 -I` (hides user site-packages, so torch appears missing).
- **ssh:**
  - Always `< /dev/null`; start background jobs with `setsid -f nohup`, or the channel hangs.
  - Write helper scripts instead of nested quoting.
- **Data safety:** a CloudLab experiment's end deletes the node. The 2026-11-05 EPYC campaign was lost that way. Copy results back before the reservation ends; never use launchers that `rm -rf results/data/v2`; resume with `--campaign <id>`.
