"""C6 — Key-value sort   (compute-bound via SIMD inefficiency + cache-limited digits; ISCA'10 kernel 10)

Equal work: FULL global stable sort of N int32 keys, producing sorted keys and the int64
source index of every key (the "value" = original position), into preallocated outputs.
Libraries (config key `lib`): GPU torch.sort(stable=True) -> CUB DeviceRadixSort.
CPU: torch (ATen sort), gnu_parallel (libstdc++ parallel mode), tbb (oneTBB parallel_sort)
-- see _cpusort.py; ATen's CPU sort scales only 5x on 24 cores, so the fastest is kept.
Keys: uniform over int32, or Zipf-skewed (many duplicates).
Validation (exact, O(N)): keys non-decreasing; idx is a permutation (sum and squared-sum
checksums in int64/float64 + range check); keys == input[idx]; stability (equal keys keep
ascending idx).
Work model: metric Mpairs/s; compulsory bytes = keys in + keys out + idx out.
"""
from __future__ import annotations

import torch

from ..core import Ctx, Unsupported, Workload
from ..validate import exact
from . import _common as C
from . import _cpusort

SIZES = [1 << 20, 1 << 24, 1 << 27]   # 2^27 pairs = 2 GB working set (>= 4x LLC)


class Sort(Workload):
    id, name, bound, field_ = "C6", "sort", "C", "Database"

    def configs(self, device_class=None):
        libs = ("torch", "gnu_parallel", "tbb") if device_class in (None, "cpu") else ("torch",)
        return [{"precision": "int32", "variant": "base", "n": n, "dist": d, "lib": lib}
                for lib in libs for d in ("uniform", "zipf") for n in SIZES]

    def working_set_bytes(self, p):
        return (4 + 4 + 8) * p["n"]

    def work(self, p):
        return {"flops": 0.0, "bytes": (4.0 + 4.0 + 8.0) * p["n"], "metric_name": "Mpairs/s",
                "metric_units_per_call": p["n"] / 1e6}

    def library(self, ctx):
        return C.library_tag(ctx, "sort/" + ("aten" if ctx.device_class == "cpu" else "CUB"))

    def _library_for(self, p, ctx):
        lib = p.get("lib", "torch")
        return self.library(ctx) if lib == "torch" else _cpusort.version(lib)

    def setup(self, p, ctx: Ctx):
        n = p["n"]
        g = ctx.generator(1)
        if p["dist"] == "uniform":
            keys = torch.randint(-2 ** 31, 2 ** 31 - 1, (n,), generator=g, dtype=torch.int32, device=ctx.device)
        else:   # heavy duplicates: Zipf-like via floor(1/u^1.5), clipped
            u = torch.rand(n, generator=g, device=ctx.device, dtype=torch.float64).clamp_(min=1e-12)
            keys = torch.floor(u.pow_(-1.5)).clamp_(max=2 ** 31 - 2).to(torch.int32)
        out_k = torch.empty_like(keys)
        out_i = torch.empty(n, dtype=torch.int64, device=ctx.device)
        s = {"keys": keys, "k": out_k, "i": out_i, "lib": p.get("lib", "torch")}
        if s["lib"] != "torch":
            if ctx.device_class != "cpu":
                raise Unsupported(f"{s['lib']} is a CPU library")
            e = _cpusort.ext()
            if s["lib"] == "tbb" and not e.has_tbb():
                raise Unsupported("oneTBB headers not installed")
            scratch = torch.empty(n, dtype=torch.int64)
            scratch.fill_(0)
            if s["lib"] == "gnu_parallel":
                s["fn"] = lambda: e.sort_gnu(keys, out_k, out_i, scratch)
            else:
                cpus = torch.tensor(ctx.cpus or list(range(ctx.threads)), dtype=torch.int32)
                s["fn"] = lambda: e.sort_tbb(keys, out_k, out_i, scratch, ctx.threads, cpus)
        return s

    def call(self, s):
        if s["lib"] == "torch":
            torch.sort(s["keys"], stable=True, out=(s["k"], s["i"]))
        else:
            s["fn"]()

    def validate(self, s, ctx):
        self.call(s)
        k, i, src = s["k"], s["i"], s["keys"]
        n = k.numel()
        ok_sorted = bool((k[1:] >= k[:-1]).all())
        ok_range = bool((i.min() >= 0) & (i.max() < n))
        # permutation checksums, exact: sum in int64 (< 2^63 for n <= 2^31); sum of squares in
        # int64 arithmetic modulo 2^64 (each i^2 < 2^62; the sum wraps deterministically),
        # compared with the closed form reduced mod 2^64 in Python big integers.
        def wrap(v):
            v %= 1 << 64
            return v - (1 << 64) if v >= 1 << 63 else v
        ok_perm = int(i.sum()) == n * (n - 1) // 2 and \
            int((i * i).sum()) == wrap((n - 1) * n * (2 * n - 1) // 6)
        ok_gather = bool((src[i] == k).all())
        eq = k[1:] == k[:-1]
        ok_stable = bool((i[1:][eq] > i[:-1][eq]).all())
        bad = [ok_sorted, ok_range, ok_perm, ok_gather, ok_stable].count(False)
        return exact(bad == 0, bad)

    def transfers(self, s):
        return [s["keys"]], [s["k"], s["i"]]


WORKLOAD = Sort()
