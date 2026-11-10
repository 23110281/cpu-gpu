# V100 session, 2026-11-10/11 (supporting files; the campaign data itself is in results/data/v2)

Machine: CloudLab c4130 `c4130-110133.wisc.cloudlab.us`, 4x Tesla V100-SXM2-16GB (GPU 0 used), host
2x Xeon E5-2667 v4, Ubuntu 22.04, driver 580.196.07, nvcc 12.9, torch 2.14.1+cu126.

| Where | Contents |
|---|---|
| `results/data/v2/gpu/NVIDIA_V100/` | **campaign data** (shown on the dashboard): `c_v100_20261110` (3 trials, all workloads, commits 2999f37 / 3b1a3f2) + A1 re-run `c_v100_a1graph_20261111` (3 trials, commit 09d3e54) |
| `results/calibration/NVIDIA_V100.json` | BabelStream Triad 845.2 GB/s; GEMM fp32 14.1 / fp64 7.0 TFLOP/s |
| `smoke/` | `make bench_smoke_cuda` after the CUDA-declaration fix: 23 ok + 2 bf16 unsupported |
| `logs/` | setup, smoke, full campaign, A1 re-runs |
| `v2_superseded/` | A1 rows replaced by the re-run, kept for the record: `pre72ab4db` = old cat + SDPA decode (batch 1: 66.8 tok/s); `72ab4db` = new decode attention, eager only (batch 1: 44 tok/s, host-launch-bound) |

Found and fixed during this session (see docs/HANDOVER.md): driver 580 for Volta; torch wheel with sm_70
kernels (cu126); CUDA-extension declarations (`cuda_declarations`); BabelStream built for the GPU's
own arch; A1 decode attention (batched matmuls + in-place KV cache); A1 GPU decode timed from a CUDA
graph (`lib=cudagraph`).
