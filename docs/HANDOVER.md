# Handover: CPU vs GPU benchmark project (state as of 2026-11-11 ~15:00 UTC: c6620 + c6525-100g analysed and on the dashboard; r6615 running)

This is for an AI coding session picking up the work. **Verify every claim here against the code, git log and data before acting on it.** If something here disagrees with the repo, the repo wins; note the discrepancy. Keep this file current: update it at every milestone and delete what is no longer true (the user's standing request).

## 1. The project in one paragraph
We are replicating ISCA'10 *"Debunking the 100X GPU vs. CPU Myth"* (Lee et al.) on modern hardware. That paper showed that GPU-vs-CPU speedups shrink to about 2.5× once **both** sides run properly optimized code. Our suite has **16 workloads**, each tagged by what limits it: B = bandwidth, C = compute, S = synchronization, A = application.

Target devices:
- **Core:** Intel Xeon (c6620), AMD EPYC (c6525-100g Zen 2; r6615 Zen 4), NVIDIA Grace; A100, L40S, H100/GH200.
- **Extra:** V100 (c4130).

Repo: `/home/kushal/cpu gpu project/cpu-gpu`. GitHub: `https://github.com/23110281/cpu-gpu` (branch `main`; GitHub Pages dashboard built by `.github/workflows/dashboard.yml`).

## 2. Ground rules from the user
- **Push only when the user asks.** Fixes stay as local commits until they have been tested on real hardware. See §8.
- Narrate progress step by step in plain language. The user wants to follow along, so take things slowly and explain.
- Prefer PyTorch with the fastest well-known library on each side. Keep a publication standard.
- Old data under `legacy/data/` is frozen, shown as "preliminary", and **never used in the paper**.
- Ask before anything hard to reverse: deleting data, pushing, releasing a CloudLab node, or a code change that alters measured results.
- Long runs on a node: `bash scripts/run_campaign.sh start <DEVICE>` (detaches with setsid + nohup; survives the laptop sleeping). The user backs up results when they choose (docs/RUNNING.md step 9).
- Keep the repo minimal and the naming plain ("the harness", not "v2"); legacy stays in `legacy/`.

## 3. Layout and tools (since the 2026-11-11 restructure)
The primary code is just "the harness"; never call it "v2" again (user's request).

| Path | Role |
|---|---|
| `bench/` | the harness. Run from the repo root: `PYTHONPATH=. python3 -m bench {run,sweep,smoke,list,describe}` |
| `bench/workloads/` | one module per workload (CPU + GPU in one file) + helpers `_common` (`widened_affinity`, `cuda_declarations`, `build_error_summary`, `cpu_backend`), `_blas`, `_cpusort`, `_sparse`, `_mc_ext`, `_crypto_ext` |
| `analysis/` | `analyze.py` → `library_report.py` → `dashboard_summary.py` → `build_dashboard.py` (+ `figures.py`, `dashboard.html` template) |
| `configs/` | `devices/<ID>.json` (placements, scaling, datasheet), `workloads.json` (suite, status, TODOs, legacy data globs relative to the repo root, device `aliases`), `library_notes.json`, `dashboard_notes.json` |
| `scripts/` | `setup_node.sh` (any node, auto CPU/GPU), `install_cuda.sh` (driver 580 for Volta/Pascal), `run_campaign.sh` (start/status/resume/stop), `calibrate.py` |
| `tests/test_bench.py` | `make test` (laptop: 31 pass / 18 skip; V100 node: 49/49 pass) |
| `results/data/<cpu\|gpu>/<ID>/` | campaign data (the only thing the analysis reads) |
| `results/data/sessions/` | test-session archives (r6615, V100 extras); never analysed |
| `results/calibration/`, `results/crosscheck/` | measured ceilings; native cross-check output |
| `results/analysis/` (gitignored, `sessions/` committed), `results/site/` (gitignored) | generated tables; generated dashboard |
| `crosscheck/` | native C GEMM/SAXPY cross-check + its Makefile (OpenBLAS default; builds on demand) |
| `studies/gh200_c2c/` | GH200 C2C residency study: NOT fixed; README lists 15 defects with file:line |
| `legacy/` | frozen first-generation pipeline + data (dashboard "prelim" cells only); README explains why not research grade |
| `docs/RUNNING.md` | the standard new-machine guide; `docs/methodology.md` the protocol |

Make targets: `setup test smoke calibrate sweep sweep_dry campaign analysis site readme list venv clean`. `make site PYTHON=<py>=3.11 venv>` builds the dashboard; `make readme` regenerates the README's library-winner block (between `<!-- library-table:start/end -->`).

**Harness facts:**
- `sweep.py`: one subprocess per (config, placement, trial); resumable by campaign id (skips ok/validation_failed/oom/unsupported, re-runs error); preflight OOM when ws × 2.5 > memory.
- Fastest-library rule: config key `lib`; the headline is the best (lib, placement) per problem. `library_report.COMPARE_IGNORING` lets S1 histc (fp32) and bincount (int32) compete; variants (`ff`, `sw`) are reported as variant effects, not wins.
- Library tags: `cpu_backend()` reports MKL only when torch has MKL; C3/A1 CPU tags no longer claim oneDNN (`aten-cpu`, `torch-blas`): which kernel torch uses is to be verified with ONEDNN_VERBOSE/MKL_VERBOSE on the next CPU node.
- `env.py` dirty-tree check covers bench, analysis, configs, scripts, crosscheck, Makefile.
- CI: `.github/workflows/dashboard.yml` runs `make venv && make site` on every relevant push and publishes `results/site` to the `plots` branch with `keep_files: false`.

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

### r6615 full campaign (Clemson `Kushal17@clnode373.clemson.cloudlab.us`, reserved 24 h from 2026-11-11 ~06:00 UTC)
- Campaign `c_amd_r6615_20261111T0632Z`, 3 trials × 630 jobs, code `1b1bff8`, sweep started 06:38 UTC (first ETA 14 h, expect less).
- Setup ok (no cpufreq driver, like c6525: frequency recorded per row via APERF/MPERF); 51 tests passed; smoke 35/35 ok.
- Calibration: 360.3 GB/s; fp32 3385 (OpenBLAS) / BLIS 3349 / MKL 3059; fp64 1600 (BLIS) / OpenBLAS 1598 / MKL 1582.
- **Fixed:** Ubuntu's BLIS 0.8.1 ran Zen 4 on 'generic' kernels (960 GFLOP/s), and its override aborts. setup_node.sh now builds pinned upstream BLIS 2.1 into ~/.local/blis only when the distro BLIS reports generic (user approved the GitHub build). Rows record `BLIS 2.1 arch=zen3`. Methodology pitfall 14.
- **After it finishes:** copy back, cross-check against the test session below, then `git rm` the test session (`results/data/sessions/r6615_ctest_20261110`, `results/analysis/sessions/r6615_20261110`) as the user asked, and repoint `configs/library_notes.json` evidence that cites `ctest_r6615` to the new campaign.

### r6615 test session (same node type), 2026-11-10, 4 h, node released (to be deleted after the cross-check)
- **Data:** `results/data/sessions/r6615_ctest_20261110/` (README inside), committed and pushed. Test data: one trial, partial grid, **not paper data**.
- **Analysis:** `results/analysis/sessions/r6615_20261110/`.
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
- **Data:**
  - Campaign: `results/data/gpu/NVIDIA_V100/`, calibration `results/calibration/NVIDIA_V100.json` (committed locally; on the dashboard).
  - Session files: `results/data/sessions/v100_c4130_20261110/` (README, smoke, logs, superseded A1 rows).
  - Everything is also still on the node (`~/cpu-gpu` there is still at 09d3e54 with the old `results/data/v2` layout; do not `git pull` into it, because untracked data would collide: clone fresh, as the launcher test did).
- **Launcher test 2026-11-11 (layout commit e90b0c8, fresh clone, scratch BENCH_OUT, since deleted):** `run_campaign.sh start NVIDIA_V100 --workloads B1 --trials 1` detached; 49 tests passed; smoke 24 ok + 3 unsupported; calibration skipped (exists); sweep 10 ok + 2 oom; `status`/`resume` (skipped all 12)/duplicate-start refusal worked; crosscheck and legacy CPU+GPU benches build.
- **Analysis:** `make analysis` regenerates `results/analysis/NVIDIA_V100_*` (gitignored build outputs).
- **Found and fixed (local commits 0fb3d5b..3b1a3f2):**
  - Driver 620 has no Volta support: pin 580.
  - The cu128/cu129 torch wheels have no sm_70 code: the setup now checks the arch list.
  - Every CUDA extension failed with "'fn' was not declared in this scope", because load_inline's main.cpp never sees cuda_sources: declarations are now generated with `cuda_declarations`.
  - BabelStream was built for sm_80: it now builds for the GPU's own arch.
  - Build errors were truncated: `build_error_summary` now shows the compiler's error lines.

### c6620 + c6525-100g full campaigns: DONE, copied back, analysed, on the dashboard (2026-11-11)
| | c6620 `Kushal17@er110.utah.cloudlab.us` | c6525-100g `Kushal17@amd248.utah.cloudlab.us` |
|---|---|---|
| CPU | Xeon Gold 5512U (Emerald Rapids), 28C/56T, 1 socket, 1 NUMA node, AMX + AVX-512 bf16 | EPYC 7402P (Zen 2), 24C/48T, 1 socket, 1 NUMA node |
| Memory | 8× DDR5 configured 4800 → datasheet 307.2 GB/s | 8× DDR4-3200 → 204.8 GB/s |
| Setup notes | governor performance, RAPL readable | **no cpufreq driver** (firmware-controlled); frequency per row from APERF/MPERF |
| Campaign | `c_intel_c6620_20261111T0036Z`, 1872 ok, 18 oom (8B fp32 batch 32), 0 errors, done 10:25 UTC | `c_amd_c6525_100g_20261111T0036Z`, 1743 ok, 129 unsupported (bf16), 18 oom, 0 errors, done 13:09 UTC |
| Ceilings | **213.6 GB/s** (Dot; was 273.5, see below); GEMM 3797 / 1852 (MKL) | 125.0 GB/s; GEMM 2030 / 969 (BLIS) |
| Code | `2d52043` | same |

Nodes are still reserved (~3 days from 2026-11-11), so C4/S1/C7/C6 re-runs are possible.

**Results (V100 / CPU, geomean over problems):** V100 = 5.1× c6620, 8.4× c6525-100g; c6620 = 1.62× c6525-100g. Well-tuned workloads track the hardware gap (V100/c6620: GEMM 3.6×, LBM 4.4×, Llama decode 4.4×, conv 2.6×; hardware ~4× BW, 3.7× FLOP). **Untuned CPU paths** inflate the gap: S1 histogram 11× (torch.histc ~4-6 Gsamples/s ≈ 10% of BW), C7 search 24×, C6 sort 14×, C4 FFT 9.7×, C5 bilateral 10×. On AMD, MKL runs 2^22-point 1-D FFTs single-threaded (74.6 ms with 24 threads vs 72.7 ms with 1; 6.2 vs 138 GFLOP/s on Intel). These are curated "hypothesis"/"verified" findings in `configs/dashboard_notes.json`, plus infra TODOs. Headline CIs are tight: only 2 per device have CI width > 5%.
- Bandwidth: Intel SAXPY 92%, LBM 84%, Llama-8B B=1 decode 94% of 213.6; AMD 96-98% of 125. A1 batch 8: BLIS wins on both (AMD 65%, Intel 66% of BW). C1: MKL (torch) wins on Intel, BLIS on Zen 2. C5: OpenCV beats native (native 0.19-0.22×). C3: native wins, torch/oneDNN at 0.01-0.02×.

**Bandwidth ceiling fix (scripts/calibrate.py):** a write-allocate-corrected BabelStream rate is used only if ≤ 1.05 × pure-read Dot. On the c6620 Copy 181 ×1.5 = 272 > Dot 214, so the CPU does not read written lines (SpecI2M) and the ceiling is Dot. Recompute stored files with `python3 scripts/calibrate.py --recompute-bandwidth results/calibration/<DEV>.json` (only the c6620 changed). The CSV column `pct_measured_peak` keeps the run-time value; the analysis recomputes from calibration.

**Data packing (`make pack`, scripts/pack_results.py):** after copy-back, per-job `samples/*.csv.gz` → `samples.tar` (lossless, analysis reads it) and per-job `env/*.json` → one `machine.json` (constant fields, per-placement thread settings, campaigns, git SHAs, dirty-job count; real mid-campaign changes are listed under `changed_during_campaign`). Per device: 16 CSVs + 2 files, ~4 MB. The user wants the repo minimal: never commit per-job files. Campaign logs are kept gzipped in `results/data/sessions/<DEV>_<date>/logs/`.

**Found and fixed during the CPU drill:**
- **OpenBLAS 0.3.20 misdetects Emerald Rapids as "Prescott"** (generic kernel): sgemm 848 vs 3381 GFLOP/s. `_blas.load` now probes the core and forces `OPENBLAS_CORETYPE=SkylakeX` only for a generic fallback; rows record `core=SkylakeX (forced; auto-detected Prescott)`. AMD detects Zen correctly (not forced). `_blas.version()` was also broken (returned the file name), so earlier OpenBLAS rows never recorded the config.
- **Verified torch CPU backends** (both CPUs): C3 conv2d = oneDNN `jit_dw` (avx512_core / avx2) + 3 full-image reorders per call; A1 linears = oneMKL SGEMM, batch-1 decode as N=1 SGEMM (MKL on AMD reports the generic "Intel Architecture processors" path). Tags/README updated (`dbac0ac`).
- c6620 config: label (Emerald Rapids, not Sapphire Rapids) and datasheet bandwidth; datasheet FLOP peak left out (Intel ARK unreachable to verify AVX-512 FMA units; measured 3797 fp32 ≈ 28 × 2.1 GHz × 64 suggests 2 units).
- A1 decode on AMD after the attention fix (llama3.2-1b fp32, 1 process): B=1 OpenBLAS 17.7 tok/s = 90 GB/s (72%); B=8 BLIS 87.9 tok/s = 66 GB/s (53%; r6615 was 9–13% before the fix); torch/MKL 4.2 / 26.4 tok/s.

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
| A1 Llama | decode B=1 117.6 / B=8 678.5 tok/s (71% / 60% of bandwidth) after the 11-11 fixes (below); prefill ~14 TFLOP/s ≈ peak | ✓ (re-run with 09d3e54) |

**A1 decode: found and fixed 2026-11-11 (commits 72ab4db, 09d3e54; V100 A1 re-run `c_v100_a1graph_20261111`):**
1. **Attention.** SDPA's fused kernel (mem-efficient, sm_70) tiles over queries. With one query token it took 47% (B=1) / 64% (B=8) of the step, and `torch.cat` of the KV cache another ~8%.
   - Fix: decode attention = GQA reshape + q·Kᵀ / softmax / ·V (`decode_attention`), with the new K/V written in place into a preallocated cache slot.
   - Prefill keeps SDPA. The old cat + SDPA path is the fp64 validation reference.
2. **Host launch rate.** An eager step issues ~500 kernels. This node's Xeon E5-2667 v4 needs 16 ms to issue a step that takes the GPU 8.5 ms, and the harness rightly syncs per step (each token depends on the last).
   - Fix: GPU decode is also timed replayed from a CUDA graph (`lib=cudagraph`, standard in vLLM/TensorRT-LLM). Eager rows are kept; the analysis keeps the faster.

V100 llama3.2-1b fp32, tokens/s:

| | Before | Eager, new attention | CUDA graph |
|---|---|---|---|
| B=1 | 66.8 | 43.9 | **117.6** (597 GB/s, 71%) |
| B=8 | 251 | 347.5 | **678.5** (510 GB/s, 60%) |

**The same attention change applies to CPUs.** The r6615 A1 rows (before 72ab4db) are outdated: re-run A1 on every CPU and profile batch-8 decode there.
- The CPU still pays Python per-op overhead (an estimated 10–20% at 1B, batch 1).
- `torch.compile` for both sides is future work.

**r6615 vs V100 (same problems):**
- B2 LBM: GPU 2.3–3.4× faster, roughly the bandwidth ratio 845/364 = 2.3.
- C3 native conv: 1.0–3.0× (k=3 at 4096²: 0.96×).
- A1: 1.6× (B=1) and 5.5× (B=8; skewed by the attention issue).
- These are about the paper's ~2.5×, not 100×.

**Dashboard (built by `make site`):**
- Device pages (only machines with campaign data; V100 now): measured vs datasheet tiles, generated statements, findings (`configs/dashboard_notes.json`), per-workload table, overview + 16 figures.
- **Library analysis** page (`#library`): machine × workload matrix of the fastest implementation, per-size detail on click, variant effects, curated notes (shown only for libraries compared on that machine). V100: C3 native 18/18 (cuDNN 0.42×), S1 histc 15/18 (bincount 0.66×), S2 CUDA graph 4/4 (coop 0.90×), A1 decode CUDA graph 2/2 (eager 0.44×); C5 `__expf` variant 1.18–1.50×.
- Status matrix keeps "prelim" cells from `legacy/data/` and the GH200 study's cleaned data; V100 column maps the legacy `Tesla_V100-SXM2-16GB` dir via `aliases`.

## 7. Open items
1. **Push** (user's decision): the rule (tested on one CPU + one GPU node) is met; nothing pushed yet (§8).
2. **r6615 campaign** running (§5): copy back (RUNNING.md step 9), `make pack`, recompute its calibration bandwidth (`--recompute-bandwidth`; expected unchanged, Dot is max), analyse, cross-check against the 2026-11-10 test session, then `git rm` the test session and repoint `library_notes.json` evidence citing it.
3. **Tuned CPU baselines** (biggest research item): native per-thread-private histogram (S1), native search (C7), native sort (C6), FFTW as a 2nd CPU FFT library (C4); then re-run those workloads on all CPUs. Until then, S1/C7/C6/C4 CPU-vs-GPU ratios reflect library quality.
4. Lock clocks or not (user decision). The V100 GEMM shows a 9% clock-driven difference between sessions.
5. Intel datasheet FLOP peak after checking Intel ARK (AVX-512 FMA units). AMD c6525-100g bandwidth 61% of datasheet: check DIMM population/NPS with dmidecode.
6. C9 GPU SHA-256 efficiency check (Nsight Compute). B4 SpGEMM is labelled bandwidth-bound but reaches ~1% of BW on CPUs: revisit its bound class/byte model.
7. Freeze a release: tag one commit and collect all paper campaigns from it.
8. **Publication rules:** BabelStream results must follow its run rules; the CPU ceiling is "derived from BabelStream" (methodology §10). The GCC runtime-exception text (libgomp/libstdc++) could not be fetched (rate limit).
9. Sweep preflight margin: 2.5× working set skips problems that likely fit (V100: 8–9 GB problems).
10. GH200 study: keep or drop (user's call); if kept, fix the 15 defects in `studies/gh200_c2c/README.md` on a GH200.
11. GEMM cross-check outliers; native benches per-iteration samples; SuiteSparse for B3/B4; llama.cpp rows; `torch.compile` for decode on both sides.
12. "CPU/GPU ratio ready" tile counts only core GPUs (A100/L40S/H100), so it stays 0 with V100 data; V100 ratios are in `results/analysis/ratios.csv`. Next hardware: A100/A30/H100 (first run of GPU bf16 paths).

## 8. Git state
- `origin/main` = `9798ed8` (pushed 2026-11-10 after the r6615 session).
- **Unpushed local commits** (oldest first):
  - V100 fixes: `0fb3d5b` `79e2eb4` `138a007` `2999f37` `3b1a3f2`; `379a063` handover.
  - Llama decode: `72ab4db` attention, `09d3e54` CUDA graph.
  - `4aa688b` dashboard device pages; `abe1ec3` V100 data; `d3af208` handover.
  - Cleanup (2026-11-11): `3125e64` library facts, `974261c` layout restructure, `360a2f7` setup_node.sh, `90b271f` library analysis, `e234efb` run_campaign.sh + bench describe + RUNNING.md, `e90b0c8` README, `2d98edd` legacy/study READMEs, `8e3f4aa` handover.
  - CPU drill (2026-11-11): `019e74a` c6620 config, `dbac0ac` verified C3/A1 backends, `2d52043` OpenBLAS core fix, `801230e` handover.
  - Later 2026-11-11: `9ea86e1` per-library absolute numbers in the library report, `3e93943`/`1b1bff8` BLIS arch + BLIS 2.1 build on Zen 4, `aeac393` docs, `6d901da` packing, `9a186cf` c6620 + c6525-100g data, then the analysis commit (ceiling rule, findings, notes).
- Test unpushed code on a node with a git bundle (docs/RUNNING.md step 3).

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
- **LLM decode:**
  - Don't use SDPA for one-token decode (it tiles over queries).
  - Don't `torch.cat` the KV cache every step.
  - Eager GPU decode can be host-launch-bound: time it from a CUDA graph too.
  - Profile the decode step on every new device type.
- **Dashboard:** device pages come only from `results/site/devices/`; CI publishes with `keep_files: false`, so nothing stale survives on the `plots` branch.
- **Permissions:** setcap puts ld.so in secure mode (drops `LD_PRELOAD`; jemalloc needs a setuid lib); RAPL `energy_uj` is root-only (chmod the resolved symlinks).
- **Harness:** one subprocess per config, because `torch.set_num_interop_threads` can only be set once per process.
- **Laptop:**
  - GCC too old for C++20 extensions; 3 GB RAM.
  - System pandas broken: use a venv (the requirements.txt pins need Python 3.11; the setup installs pandas/matplotlib unpinned).
  - Don't use `python3 -I` (hides user site-packages, so torch appears missing).
- **ssh:**
  - Always `< /dev/null`; start background jobs with `setsid -f nohup`, or the channel hangs.
  - Write helper scripts instead of nested quoting.
- **Data safety:** a CloudLab experiment's end deletes the node. The 2026-11-05 EPYC campaign was lost that way. Copy results back before the reservation ends (docs/RUNNING.md step 9); resume with `bash scripts/run_campaign.sh resume <DEVICE>`.
