"""Unit tests for the v2 harness (src/torch/bench). Run: make test_bench  (pytest)."""
import gzip
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "torch"))

from bench import devices, schema, stats, telemetry, validate  # noqa: E402
from bench.devices import HwThread  # noqa: E402
from bench.workloads import registry  # noqa: E402


# ── statistics ───────────────────────────────────────────────────────────────
def test_summarize_basic():
    s = stats.summarize([1.0, 2.0, 3.0, 4.0, 100.0])
    assert s["median_ms"] == 3.0 and s["min_ms"] == 1.0 and s["n_samples"] == 5
    assert s["ci95_lo_ms"] <= s["median_ms"] <= s["ci95_hi_ms"]


def test_bootstrap_ci_coverage():
    """95% CI of the median should cover the true median in roughly 95% of experiments."""
    rng = np.random.default_rng(0)
    hits = 0
    for i in range(200):
        x = rng.lognormal(0.0, 0.3, size=60)          # true median = 1.0
        lo, hi = stats.bootstrap_ci_median(x, seed=i)
        hits += lo <= 1.0 <= hi
    assert 0.88 <= hits / 200 <= 0.995


def test_ratio_ci():
    r, lo, hi = stats.ratio_ci([10.0] * 30, [2.0] * 30)
    assert r == pytest.approx(5.0) and lo == pytest.approx(5.0) and hi == pytest.approx(5.0)


# ── validation ───────────────────────────────────────────────────────────────
def test_normwise_passes_and_catches_corruption():
    ref = torch.randn(1000, dtype=torch.float64)
    y = ref.to(torch.float32)
    assert validate.normwise(y, ref, validate.tol_elementwise(torch.float32)).ok
    y[17] += 1.0
    assert not validate.normwise(y, ref, validate.tol_elementwise(torch.float32)).ok


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gemm_check_detects_realistic_faults(dtype):
    g = torch.Generator().manual_seed(0)
    n = 512
    A = torch.randn(n, n, dtype=dtype, generator=g)
    B = torch.randn(n, n, dtype=dtype, generator=g)
    C = A @ B
    assert validate.gemm_check(A, B, C, dtype).ok                       # correct result passes
    D = C.clone(); D[:32, :32] = 0                                       # missing / uninitialised tile
    assert not validate.gemm_check(A, B, D, dtype).ok
    D = C.clone(); D[7] *= 1 + 1e-3                                      # one wrong row (0.1%)
    assert not validate.gemm_check(A, B, D, dtype).ok
    D = C.T.contiguous()                                                 # transposed output
    assert not validate.gemm_check(A, B, D, dtype).ok
    if dtype == torch.float32:                                           # silent reduced precision
        lo = (A.to(torch.bfloat16).float() @ B.to(torch.bfloat16).float())
        assert not validate.gemm_check(A, B, lo, dtype).ok


def test_freivalds_row_scaling_beats_global_norm():
    """A single wrong element that a global-Frobenius normalisation dilutes below tolerance."""
    g = torch.Generator().manual_seed(0)
    n = 256
    A = torch.randn(n, n, generator=g); B = torch.randn(n, n, generator=g)
    C = A @ B
    C[3, 5] += 0.05 * float(C.abs().max())
    tol = validate.TOL_C * 2 * validate.unit_roundoff(torch.float32) * math.sqrt(n)
    assert not validate.freivalds(A, B, C, tol).ok


def test_statistical_and_exact():
    assert validate.statistical(10.01, 10.0, 0.01).ok
    assert not validate.statistical(10.1, 10.0, 0.01).ok
    assert validate.exact(True).ok and not validate.exact(False, 3).ok


# ── schema ───────────────────────────────────────────────────────────────────
def test_schema_roundtrip(tmp_path):
    p = tmp_path / "x.csv"
    schema.append_row(p, {"workload_id": "B1", "status": "ok", "median_ms": 1.5, "params_json": {"n": 4}})
    schema.append_row(p, {"workload_id": "B1", "status": "oom", "max_err": float("nan")})
    rows = schema.read_rows(p)
    assert len(rows) == 2 and list(rows[0]) == schema.COLUMNS
    assert rows[0]["params_json"] == '{"n":4}' and rows[1]["max_err"] == "nan"


def test_schema_rejects_unknown_column_and_status(tmp_path):
    with pytest.raises(KeyError):
        schema.append_row(tmp_path / "x.csv", {"status": "ok", "not_a_column": 1})
    with pytest.raises(ValueError):
        schema.append_row(tmp_path / "x.csv", {"status": "fine"})


def test_samples_roundtrip(tmp_path):
    p = tmp_path / "s.csv.gz"
    schema.write_samples(p, [1.25, 2.5], 4)
    assert schema.read_samples(p) == [1.25, 2.5]
    assert gzip.open(p, "rt").readline().startswith("# calls_per_sample=4")


def test_config_hash_stable_and_distinct():
    a = schema.config_hash("B1", {"n": 1, "precision": "fp32"}, "socket0", "D")
    assert a == schema.config_hash("B1", {"precision": "fp32", "n": 1}, "socket0", "D")
    assert a != schema.config_hash("B1", {"n": 1, "precision": "fp32"}, "socket0-smt", "D")


# ── placement on a synthetic 2-socket x 4-core x 2-SMT machine ───────────────
@pytest.fixture
def two_socket(monkeypatch):
    topo = []
    for s in range(2):
        for c in range(4):
            core = s * 4 + c
            topo.append(HwThread(cpu=core, core=core, socket=s, node=s))          # first SMT sibling
            topo.append(HwThread(cpu=core + 8, core=core, socket=s, node=s))      # second sibling
    monkeypatch.setattr(devices, "topology", lambda: topo)
    monkeypatch.setattr(devices, "sockets", lambda: [0, 1])


def test_placements(two_socket):
    p = devices.resolve_placement("socket0")
    assert p.cpus == [0, 1, 2, 3] and p.nodes == [0] and p.threads == 4
    assert devices.resolve_placement("socket1-smt").cpus == [4, 5, 6, 7, 12, 13, 14, 15]
    assert devices.resolve_placement("node").threads == 8
    assert devices.resolve_placement("socket0:2").cpus == [0, 1]
    with pytest.raises(ValueError):
        devices.resolve_placement("socket0:9")
    with pytest.raises(ValueError):
        devices.resolve_placement("socket7")
    env = devices.placement_env(devices.resolve_placement("socket1"))
    assert env["OMP_PLACES"] == "{4},{5},{6},{7}" and env["OMP_NUM_THREADS"] == "4"


# ── telemetry ────────────────────────────────────────────────────────────────
def test_counter_wraparound():
    vals = iter([90.0, 95.0, 3.0, 10.0])   # wraps at 100 J
    c = telemetry._Counter("t", lambda: next(vals), wrap_j=100.0)
    for _ in range(4):
        c.sample()
    assert c.total == pytest.approx(5.0 + 8.0 + 7.0)


def test_unreadable_power_is_nan_not_zero():
    s = telemetry.Sampler("cpu", cpus=[0], sockets={99})     # no such package -> no counters
    s.counters = []
    with s:
        pass
    r = s.result(calls=10)
    assert math.isnan(r["power_w"]) and r["power_valid"] == 0


# ── workload work models ─────────────────────────────────────────────────────
def test_work_models():
    reg = registry()
    w = reg["B1"].work({"precision": "fp32", "n": 1000})
    assert w["bytes"] == 3 * 1000 * 4 and w["flops"] == 2000
    g = reg["C1"].work({"precision": "fp64", "n": 100})
    assert g["flops"] == 2 * 100 ** 3 and g["metric_name"] == "GFLOP/s"
    assert reg["B1"].working_set_bytes({"precision": "fp64", "n": 10}) == 160


def test_config_grids_identical_across_devices():
    """Size grids are absolute (methodology §6): no device-dependent sizes in configs()."""
    for w in registry().values():
        assert w.configs() == w.configs()
        assert all("precision" in c and "variant" in c for c in w.configs())


# ── C2 Monte Carlo RNG: Philox4x32-10 known-answer tests (Random123 kat_vectors) ──
PHILOX_KAT = [
    ((0, 0, 0, 0), (0, 0), (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)),
    ((0xffffffff,) * 4, (0xffffffff, 0xffffffff), (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd)),
]


def _philox_py(c, k):
    """Independent pure-Python Philox4x32-10 (reference for the C++/CUDA kernels)."""
    M0, M1, W0, W1, m = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85, 0xFFFFFFFF
    c0, c1, c2, c3 = c
    k0, k1 = k
    for _ in range(10):
        p0, p1 = M0 * c0, M1 * c2
        c0, c1, c2, c3 = ((p1 >> 32) ^ c1 ^ k0) & m, p1 & m, ((p0 >> 32) ^ c3 ^ k1) & m, p0 & m
        k0, k1 = (k0 + W0) & m, (k1 + W1) & m
    return (c0, c1, c2, c3)


@pytest.mark.parametrize("ctr,key,expect", PHILOX_KAT)
def test_philox_python_reference_kat(ctr, key, expect):
    assert _philox_py(ctr, key) == expect


@pytest.mark.parametrize("ctr,key,expect", PHILOX_KAT)
def test_philox_cpu_kernel_kat(ctr, key, expect):
    from bench.core import Unsupported
    from bench.workloads import _mc_ext
    try:
        ext = _mc_ext.cpu()
    except Unsupported as e:
        pytest.skip(str(e))
    assert tuple(v & 0xFFFFFFFF for v in ext.philox_kat(*ctr, *key)) == expect


def test_black_scholes_reference():
    from bench.workloads.c2_montecarlo import black_scholes_call
    # textbook value: S=100, K=105, r=5%, sigma=20%, T=1  ->  8.0214 (Hull)
    assert black_scholes_call(100, 105, 0.05, 0.2, 1.0) == pytest.approx(8.02135, abs=1e-4)
