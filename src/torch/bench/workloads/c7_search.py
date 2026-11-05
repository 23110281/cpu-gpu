"""C7 — Index search: for each query q, position of q in a sorted key array (ISCA'10 kernel 12)

Equal work: Q int32 queries against K sorted unique int32 keys -> int32 positions
(`torch.searchsorted(..., out_int32=True, out=)`), identical inputs on both devices.
Library: ATen searchsorted (parallel binary search on CPU, CUDA kernel on GPU).
Bound: compute (latency-bound binary search) while the key array fits in the LLC, turning
bandwidth/latency-bound beyond it -> the size grid spans 64 KB .. 2 GB of keys so the
C->B crossover is visible against ws_over_llc. Queries: uniform random or sorted.
Validation: exact match against numpy.searchsorted on the CPU.
Work model: Q*ceil(log2 K) probes; compulsory bytes = queries + results (+ keys once).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..core import Ctx, Workload
from ..validate import exact

KEYS = [1 << 14, 1 << 18, 1 << 22, 1 << 26, 1 << 29]   # int32 keys: 64 KB .. 2 GB
QUERIES = 1 << 24


class Search(Workload):
    id, name, bound, field_ = "C7", "search", "C", "Database"

    def configs(self, device_class=None):
        return [{"precision": "int32", "variant": "base", "keys": k, "queries": QUERIES, "order": o}
                for o in ("random", "sorted") for k in KEYS]

    def working_set_bytes(self, p):
        return 4 * p["keys"] + 8 * p["queries"]

    def work(self, p):
        probes = p["queries"] * math.ceil(math.log2(p["keys"]))
        return {"flops": 0.0, "bytes": 4.0 * p["keys"] + 8.0 * p["queries"],
                "metric_name": "Mqueries/s", "metric_units_per_call": p["queries"] / 1e6,
                "probes": probes}

    def library(self, ctx):
        from ._common import library_tag
        return library_tag(ctx, "searchsorted")

    def setup(self, p, ctx: Ctx):
        g = ctx.generator(1)
        k, q = p["keys"], p["queries"]
        # unique sorted keys with random gaps in [1, max_gap]; max_gap keeps the largest key
        # inside int32 (with 8 for every size, 2^29 keys overflowed int32 -> negative range)
        max_gap = max(1, min(8, (2 ** 31 - 2) // k))
        gaps = torch.randint(1, max_gap + 1, (k,), generator=g, dtype=torch.int64, device=ctx.device)
        keys = torch.cumsum(gaps, 0).to(torch.int32)
        hi = int(keys[-1]) + 1
        qs = torch.randint(0, hi, (q,), generator=ctx.generator(2), dtype=torch.int32, device=ctx.device)
        if p["order"] == "sorted":
            qs = torch.sort(qs).values
        out = torch.empty(q, dtype=torch.int32, device=ctx.device)
        return {"keys": keys, "q": qs, "out": out}

    def call(self, s):
        torch.searchsorted(s["keys"], s["q"], out_int32=True, out=s["out"])

    def validate(self, s, ctx):
        self.call(s)
        ref = np.searchsorted(s["keys"].cpu().numpy(), s["q"].cpu().numpy(), side="left")
        got = s["out"].cpu().numpy()
        mism = int(np.count_nonzero(ref != got))
        return exact(mism == 0, mism)

    def transfers(self, s):
        return [s["keys"], s["q"]], [s["out"]]


WORKLOAD = Search()
