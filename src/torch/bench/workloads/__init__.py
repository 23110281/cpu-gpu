"""Workload registry. Each module exposes `WORKLOAD` (a core.Workload instance).

Port status follows docs/methodology.md / plan v2: modules are added here as they are
implemented to protocol; the dashboard (results/workloads.json) tracks the rest.
"""
from __future__ import annotations

import importlib

_MODULES = ["b1_saxpy", "b3_spmv", "b4_spgemm", "c1_gemm", "c2_montecarlo", "c3_conv", "c4_fft", "c5_bilateral", "c6_sort", "c7_search", "c8_aes", "c9_sha256", "s1_histogram", "s2_solver"]


def registry() -> dict:
    out = {}
    for m in _MODULES:
        w = importlib.import_module(f"{__name__}.{m}").WORKLOAD
        out[w.id] = w
    return out


def get(key: str):
    reg = registry()
    if key in reg:
        return reg[key]
    for w in reg.values():
        if w.name == key:
            return w
    raise KeyError(f"unknown workload {key!r}; have {sorted(reg)}")
