# r6615 test session, 2026-11-10 (TEST data, not paper data)

Machine: CloudLab r6615 `clnode330.clemson.cloudlab.us`, AMD EPYC 9354P (Zen 4), 32C/64T, 1 socket,
NPS4 (4 NUMA nodes), 256 MiB L3 (8 x 32 MiB), 12 x 16 GB DDR5-4800, Ubuntu 22.04, GCC 11.4, torch 2.14.1+cpu.

Purpose: validate the benchmark harness end to end on real hardware before the paper campaigns.
Campaign `ctest_r6615`, ONE trial, partial grid -> do not mix with paper data (kept in results/data/sessions/, which the analysis never reads).

| Folder | Contents |
|---|---|
| `smoke/` | `make bench_smoke` (commit 25717c0): 35/35 code paths ok, incl. bf16 |
| `calibration/` | BabelStream + GEMM: bandwidth ceiling 364.1 GB/s; peak fp32 3495 (OpenBLAS), fp64 1643 (MKL) GFLOP/s |
| `data/` | commit d51fa7e: B2 and C3 full grids (124 jobs, killed at job 16 and resumed), A1 decode subset (12 jobs) -- 136/136 ok |
| `logs/` | setup, smoke, sweep, A1 subset logs |

Analysis outputs: `results/analysis/sessions/r6615_20261110/` (headline, libraries, bounds, library report, plots).

Found and fixed during this session: OpenCV ran on one core inside the harness (d51fa7e); analysis
deps could not install from requirements.txt on Python 3.10 (50b5a32).
