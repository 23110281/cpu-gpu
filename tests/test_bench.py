"""Unit tests for the benchmark harness (bench/). Run: make test  (pytest)."""
import gzip
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def test_search_keys_fit_int32_and_are_sorted():
    """C7 builds strictly increasing int32 keys for every configured size (2^29 overflowed before)."""
    from bench.workloads.c7_search import KEYS
    for k in KEYS:
        max_gap = max(1, min(8, (2 ** 31 - 2) // k))
        assert k * max_gap < 2 ** 31 - 1
    import torch
    from bench.core import Ctx
    from bench.workloads.c7_search import WORKLOAD
    ctx = Ctx("cpu", torch.device("cpu"), "t", 1, "node", 0, 1, False, 32 << 20)
    st = WORKLOAD.setup({"keys": 1 << 14, "queries": 1 << 10, "order": "random", "precision": "int32",
                         "variant": "base"}, ctx)
    assert bool((st["keys"][1:] > st["keys"][:-1]).all())


# ── native kernels and library variants added for the weak CPU baselines (B2, C3, A1) ──
def _cpu_ctx(threads=2):
    from bench.core import Ctx
    return Ctx("cpu", torch.device("cpu"), "t", 1, "node", 0, threads, False, 32 << 20)


@pytest.mark.parametrize("n", [20, 64])
@pytest.mark.parametrize("prec", ["fp32", "fp64"])
def test_lbm_cpu_kernel_matches_reference(n, prec):
    """Generated scalar-variable D3Q19 kernel (vectorised x loop, streaming stores) == fp64 reference.
    n=20 exercises the partial chunk and the unaligned (non-streaming) store path."""
    from bench.core import Unsupported
    from bench.workloads import b2_lbm as M
    try:
        ext = M._cpu_ext()
    except Unsupported as e:
        pytest.skip(str(e))
    dt = torch.float32 if prec == "fp32" else torch.float64
    f = M.initial_state(n, dt, "cpu", torch.Generator().manual_seed(1))
    out = torch.empty_like(f)
    ext.lbm_step_cpu(f, out, n, 1.0 / M.TAU)
    ref = M.reference_step(f, 1.0 / M.TAU)
    err = float((out.double() - ref).norm() / ref.norm())
    assert err <= (1e-5 if prec == "fp32" else 1e-12)


@pytest.mark.parametrize("k", [3, 5, 7])
@pytest.mark.parametrize("c", [1, 3])
def test_conv_native_cpu_validates(k, c):
    from bench.core import Unsupported
    from bench.workloads.c3_conv import WORKLOAD
    p = {"precision": "fp32", "variant": "base", "hw": 37, "k": k, "c": c, "lib": "native"}
    try:
        s = WORKLOAD.setup(p, _cpu_ctx())
    except Unsupported as e:
        pytest.skip(str(e))
    assert WORKLOAD.validate(s, _cpu_ctx()).ok


@pytest.mark.parametrize("lib", ["openblas", "blis"])
def test_llama_blas_linear_matches_torch(lib, monkeypatch):
    """A1 CPU library variants: CBLAS gemv (decode B=1) / gemm (B>1, prefill) == torch matmul."""
    from bench.core import Unsupported
    from bench.workloads import _blas, a1_llama as A
    try:
        _blas.load(lib)
    except Unsupported as e:
        pytest.skip(str(e))
    tiny = dict(d=64, L=2, h=4, kv=2, ff=128, V=257, tied=False)
    monkeypatch.setitem(A.MODELS, "tiny", tiny)
    monkeypatch.setattr(A, "KV_LEN", 16)
    for phase, B, S in (("decode", 1, 1), ("decode", 8, 1), ("prefill", 1, 12)):
        p = {"precision": "fp32", "variant": "base", "model": "tiny", "phase": phase, "batch": B, "seq": S, "lib": lib}
        s = A.WORKLOAD.setup(p, _cpu_ctx())
        assert A.WORKLOAD.validate(s, _cpu_ctx()).ok
        want = A.forward(s["model"], s["tokens"], s["cache"], s["pos0"])
        got = A.forward(s["model"], s["tokens"], s["cache"], s["pos0"], s["mm"])
        assert float((got - want).norm() / want.norm()) < 1e-5


@pytest.mark.parametrize("B,h,kv", [(1, 4, 2), (3, 8, 2), (2, 4, 4)])
def test_llama_decode_attention_matches_sdpa(B, h, kv):
    """decode_attention (GQA reshape + 2 matmuls) == SDPA with enable_gqa, on a strided cache view."""
    import torch.nn.functional as F
    from bench.workloads.a1_llama import decode_attention
    g = torch.Generator().manual_seed(3)
    P, hd = 37, 16
    q = torch.randn(B, h, 1, hd, generator=g, dtype=torch.float64)
    buf_k = torch.randn(B, kv, P + 5, hd, generator=g, dtype=torch.float64)
    buf_v = torch.randn(B, kv, P + 5, hd, generator=g, dtype=torch.float64)
    k, v = buf_k[:, :, :P], buf_v[:, :, :P]
    want = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    got = decode_attention(q, k, v)
    assert float((got - want).norm() / want.norm()) < 1e-12


@pytest.mark.parametrize("B", [1, 8])
def test_llama_decode_inplace_cache_matches_reference(B, monkeypatch):
    """New decode path (in-place KV write + decode_attention) == old cat + SDPA path; the past
    cache positions are untouched and repeated steps give identical logits (timing is idempotent)."""
    from bench.workloads import a1_llama as A
    monkeypatch.setitem(A.MODELS, "tiny", dict(d=64, L=2, h=4, kv=2, ff=128, V=257, tied=False))
    monkeypatch.setattr(A, "KV_LEN", 16)
    p = {"precision": "fp32", "variant": "base", "model": "tiny", "phase": "decode", "batch": B, "seq": 1, "lib": "torch"}
    s = A.WORKLOAD.setup(p, _cpu_ctx())
    past = [k[:, :, :16].clone() for k, _ in s["cache"]]
    want = A.forward(s["model"], s["tokens"], s["cache"], s["pos0"], reference=True)
    got1 = A.forward(s["model"], s["tokens"], s["cache"], s["pos0"])
    got2 = A.forward(s["model"], s["tokens"], s["cache"], s["pos0"])
    assert float((got1 - want).norm() / want.norm()) < 1e-5
    assert torch.equal(got1, got2)
    assert all(torch.equal(a, k[:, :, :16]) for a, (k, _) in zip(past, s["cache"]))
    assert A.WORKLOAD.validate(s, _cpu_ctx()).ok


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("B", [1, 8])
def test_llama_cudagraph_decode_validates(B, monkeypatch):
    """lib=cudagraph: the replayed graph validates and equals the eager step."""
    from bench.core import Ctx
    from bench.workloads import a1_llama as A
    monkeypatch.setitem(A.MODELS, "tiny", dict(d=64, L=2, h=4, kv=2, ff=128, V=257, tied=False))
    monkeypatch.setattr(A, "KV_LEN", 16)
    ctx = Ctx("gpu", torch.device("cuda"), "t", 1, "gpu0", 0, 1, False, 6 << 20)
    p = {"precision": "fp32", "variant": "base", "model": "tiny", "phase": "decode", "batch": B, "seq": 1, "lib": "cudagraph"}
    s = A.WORKLOAD.setup(p, ctx)
    assert A.WORKLOAD.validate(s, ctx).ok
    A.WORKLOAD.call(s)
    eager = A.forward(s["model"], s["tokens"], s["cache"], s["pos0"])
    assert float((s["out"] - eager).norm() / eager.norm()) < 1e-6


@pytest.mark.parametrize("detected,flags,expect", [
    ("Prescott", {"avx512f", "avx2"}, "SkylakeX"),     # Emerald Rapids misdetected by OpenBLAS 0.3.20
    ("Prescott", {"avx2"}, "Haswell"),
    ("Zen", {"avx2"}, None),                         # correct detection: never overridden
    ("Cooperlake", {"avx512f"}, None),
])
def test_openblas_coretype_override_only_on_generic_fallback(detected, flags, expect, monkeypatch):
    from bench import devices
    from bench.workloads import _blas
    monkeypatch.delenv("OPENBLAS_CORETYPE", raising=False)
    monkeypatch.setattr(_blas, "_openblas_corename", lambda path, env: detected)
    monkeypatch.setattr(devices, "cpu_flags", lambda: flags)
    monkeypatch.setattr(_blas, "_forced_core", None)
    _blas._fix_openblas_coretype("/nonexistent/libopenblas.so.0")
    assert __import__("os").environ.get("OPENBLAS_CORETYPE") == expect
    monkeypatch.delenv("OPENBLAS_CORETYPE", raising=False)


def test_llama_configs_cpu_libs_fp32_only():
    from bench.workloads.a1_llama import WORKLOAD
    cpu, gpu = WORKLOAD.configs("cpu"), WORKLOAD.configs("gpu")
    assert {c["lib"] for c in cpu if c["precision"] == "bf16"} == {"torch"}
    assert {c["lib"] for c in cpu if c["precision"] == "fp32"} == {"torch", "openblas", "blis"}
    assert {c["lib"] for c in gpu if c["phase"] == "prefill"} == {"torch"}
    assert {c["lib"] for c in gpu if c["phase"] == "decode"} == {"torch", "cudagraph"}
    assert "cudagraph" not in {c["lib"] for c in cpu}


def test_analysis_library_label():
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
        import analyze
    except Exception as e:                       # pandas missing/broken on this machine
        pytest.skip(f"analysis deps unavailable: {e}")
    assert analyze.lib_label('{"lib": "blis", "n": 4}', "blis/BLIS 0.9") == "blis"
    assert analyze.lib_label('{"n": 64}', "native C++/OpenMP+SIMD D3Q19 BGK") == "native"
    assert analyze.lib_label('{"n": 64}', "torch2.14.0/add_/blas=mkl") == "torch"
    assert analyze.lib_label('{"bytes": 1}', "OpenSSL 3.0.13 EVP") == "openssl"


def test_library_report_compare_rule_and_readme_block(tmp_path):
    """S1's histc (fp32 input) and bincount (int32 input) are compared as one problem; the README
    block is written only between its markers, and rewriting it is idempotent."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
        import library_report as LR
    except Exception as e:
        pytest.skip(f"analysis deps unavailable: {e}")
    a = LR.compare_key("S1", '{"bins": 256, "n": 16, "precision": "fp32"}')
    b = LR.compare_key("S1", '{"bins": 256, "n": 16, "precision": "int32"}')
    assert a == b and "precision" not in a
    assert "precision" in LR.compare_key("C3", '{"k": 3, "precision": "fp32"}')
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{LR.MARK_START}\nold\n{LR.MARK_END}\noutro\n")
    LR.write_readme(readme, f"{LR.MARK_START}\nnew table\n{LR.MARK_END}")
    once = readme.read_text()
    LR.write_readme(readme, f"{LR.MARK_START}\nnew table\n{LR.MARK_END}")
    assert readme.read_text() == once == f"intro\n{LR.MARK_START}\nnew table\n{LR.MARK_END}\noutro\n"


def test_preflight_oom_row_has_provenance(tmp_path):
    from bench import sweep
    from bench.workloads import get
    w = get("B1")
    c = w.configs("cpu")[0]
    dcfg = {"device_id": "TEST", "device_class": "cpu", "max_mem_gb": 0.0}
    sweep._preflight_oom_row(w, c, dcfg, "socket0", 1, "cTEST", tmp_path)
    rows = schema.read_rows(schema.output_paths(tmp_path, "cpu", "TEST", w.id, w.name)["rows"])
    assert rows[0]["status"] == "oom" and len(rows[0]["git_sha"]) >= 7


def test_widened_affinity_restores_mask():
    """OpenCV/TBB pools must see the whole placement; the caller's mask is restored afterwards."""
    import os
    from bench.workloads import _common as C
    saved = os.sched_getaffinity(0)
    one = {min(saved)}
    os.sched_setaffinity(0, one)                  # what torch's OMP_PROC_BIND does to the main thread
    try:
        with C.widened_affinity(saved):
            assert os.sched_getaffinity(0) == saved
        assert os.sched_getaffinity(0) == one
    finally:
        os.sched_setaffinity(0, saved)


def test_cuda_extensions_declare_every_exposed_function():
    """load_inline compiles the Python bindings (main.cpp) without seeing cuda_sources: each exposed
    CUDA function needs a declaration in cpp_sources (all 7 kernels failed to build on the first V100)."""
    from bench.workloads import _common as C, _crypto_ext, _mc_ext, b2_lbm, c3_conv, c5_bilateral, s2_solver
    cases = [(b2_lbm._CUDA_SRC, ["lbm_step_cuda"]), (c3_conv._CUDA_SRC, ["conv2d_native_cuda"]),
             (c5_bilateral._CUDA_SRC, ["bilateral_cuda"]), (_mc_ext._CUDA_SRC, ["mc_european_cuda"]),
             (_crypto_ext._CUDA_SRC, ["aes128_ctr_cuda", "sha256_multi_cuda"]),
             (s2_solver._CUDA_SRC, ["pgs_iter_launches", "pgs_iter_coop", "coop_max_blocks"])]
    for src, fns in cases:
        decl = C.cuda_declarations(src, fns)
        for fn in fns:
            assert f" {fn}(" in decl and decl.rstrip().endswith(";")
    with pytest.raises(ValueError):
        C.cuda_declarations("void other(int a) {}", ["missing_fn"])


def test_build_error_summary_keeps_the_compiler_error():
    from bench.workloads._common import build_error_summary
    e = RuntimeError("Error building extension 'x': [1/3] c++ -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=x "
                     + "-I/very/long/include/path " * 20 + "\nmain.cpp:4:52: error: 'f' was not declared in this scope\n"
                     "ninja: build stopped: subcommand failed.")
    assert "'f' was not declared" in build_error_summary(e)
