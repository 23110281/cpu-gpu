"""S1 — Histogram of N samples into B bins   (synchronization-bound; ISCA'10 kernel 13)

Equal work: exact integer counts per bin for the same N samples on both devices.
Libraries (config key `lib`):
  histc    : torch.histc on float samples k+0.5 (CPU: per-thread private histograms +
             reduction, i.e. the paper's privatisation; CUDA: shared-memory/global atomics).
             histc counts in the float output dtype (inexact above 2^24), so it is applied
             to exact chunks of 2^24 samples whose counts are summed in int64 -- exact at
             every N; the per-chunk merge is part of the measured work (B adds per chunk).
             Measured on EPYC 7402P: histc scales 17x on 24 cores, bincount only 1.5x.
  bincount : torch.bincount on int samples (CUDA: atomics; CPU ATen kernel)
Inputs: uniform over bins, or Zipf(s=1.1) over bins (skew -> atomic contention on GPU,
load imbalance on CPU). Bins 256 / 4096 / 65536 (privatised copies stop fitting in L1/L2
or GPU shared memory as B grows).
Validation: exact counts vs numpy.bincount.  Metric: Msamples/s.
Sync ablation (methodology / plan Part C) is a separate variant added with the GPU kernels.
"""
from __future__ import annotations

import numpy as np
import torch

from ..core import Ctx, Workload
from ..validate import exact

SAMPLES = [1 << 24, 1 << 27, 1 << 30]
HISTC_CHUNK = 1 << 24            # every per-chunk float32 count is exact (<= 2^24)
BINS = [256, 4096, 65536]


class Histogram(Workload):
    id, name, bound, field_ = "S1", "histogram", "S", "Image / analytics"

    def configs(self, device_class=None):
        return [{"precision": "fp32" if lib == "histc" else "int32", "variant": "base",
                 "n": n, "bins": bn, "dist": d, "lib": lib}
                for lib in ("histc", "bincount") for d in ("uniform", "zipf") for bn in BINS for n in SAMPLES]

    def working_set_bytes(self, p):
        return 4 * p["n"] + 8 * p["bins"]

    def work(self, p):
        return {"flops": 0.0, "bytes": 4.0 * p["n"], "metric_name": "Msamples/s",
                "metric_units_per_call": p["n"] / 1e6}

    def library(self, ctx):
        from ._common import library_tag
        return library_tag(ctx, "histogram")

    def _library_for(self, p, ctx):
        from ._common import library_tag
        return library_tag(ctx, p["lib"])

    def setup(self, p, ctx: Ctx):
        n, b = p["n"], p["bins"]
        if p["dist"] == "uniform":
            idx = torch.randint(0, b, (n,), generator=ctx.generator(1), dtype=torch.int32, device=ctx.device)
        else:   # Zipf(1.1) over bins via inverse CDF (setup only)
            w = 1.0 / torch.arange(1, b + 1, dtype=torch.float64, device=ctx.device) ** 1.1
            cdf = torch.cumsum(w / w.sum(), 0).to(torch.float32)
            cdf[-1] = 1.0
            u = torch.rand(n, generator=ctx.generator(1), device=ctx.device)
            idx = torch.searchsorted(cdf, u, out_int32=True).clamp_(max=b - 1)
        if p["lib"] == "histc":
            x = idx.to(torch.float32).add_(0.5)
            out = torch.zeros(b, dtype=torch.int64, device=ctx.device)
            part = torch.empty(b, dtype=torch.float32, device=ctx.device)
            chunks = [x[i:i + HISTC_CHUNK] for i in range(0, n, HISTC_CHUNK)]

            def fn():
                out.zero_()
                for ch in chunks:
                    torch.histc(ch, bins=b, min=0.0, max=float(b), out=part)
                    out.add_(part.to(torch.int64))
        else:
            x = idx
            out = None
            fn = lambda: torch.bincount(x, minlength=b)  # noqa: E731  (no out= in ATen)
        return {"x": x, "idx": idx, "fn": fn, "out": out, "bins": b}

    def call(self, s):
        r = s["fn"]()
        if s["out"] is None:
            s["last"] = r

    def validate(self, s, ctx):
        self.call(s)
        got = (s["out"] if s["out"] is not None else s["last"]).cpu().numpy().astype(np.int64)
        ref = np.bincount(s["idx"].cpu().numpy(), minlength=s["bins"])
        mism = int(np.count_nonzero(got != ref))
        return exact(mism == 0 and got.sum() == s["idx"].numel(), mism)

    def transfers(self, s):
        return [s["x"]], [s["out"] if s["out"] is not None else s.get("last")]


WORKLOAD = Histogram()
