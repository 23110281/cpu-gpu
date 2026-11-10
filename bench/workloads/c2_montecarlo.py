"""C2 — Monte Carlo European option pricing, fp64   (compute-bound, transcendental; ISCA'10 kernel 2)

Equal work: price of a European call under geometric Brownian motion from N paths, one
exact terminal step per path (S_T = S0 exp((r - s^2/2)T + s sqrt(T) Z)), discounted mean
payoff -- the paper's kernel. Same Philox4x32-10 stream (counter = path-pair index) on both
devices, so both estimate the price from the SAME normals (see _mc_ext.py).
Implementation: native kernels (C++/OpenMP/SIMD on CPU, CUDA on GPU), same algorithm.
Validation (statistical, methodology §8): |price - Black-Scholes| <= 4 standard errors;
the RNG itself is checked against Random123 known-answer vectors (tests/torch).
Metric: billion paths/s ("billion paths/s" as in ISCA'10 Table 3).
"""
from __future__ import annotations

import math

import torch

from ..core import Ctx, Unsupported, Workload
from ..validate import statistical
from . import _mc_ext

PATHS = [1 << 20, 1 << 24, 1 << 28, 1 << 30]
S0, K, R, SIGMA, T = 100.0, 105.0, 0.05, 0.2, 1.0


def black_scholes_call(s0, k, r, sigma, t):
    d1 = (math.log(s0 / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    cdf = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))  # noqa: E731
    return s0 * cdf(d1) - k * math.exp(-r * t) * cdf(d2)


class MonteCarlo(Workload):
    id, name, bound, field_ = "C2", "montecarlo", "C", "Finance"

    def configs(self, device_class=None):
        return [{"precision": "fp64", "variant": "base", "paths": n} for n in PATHS]

    def working_set_bytes(self, p):
        return 0

    def work(self, p):
        return {"flops": 0.0, "bytes": 0.0, "metric_name": "Gpaths/s",
                "metric_units_per_call": p["paths"] / 1e9}

    def library(self, ctx):
        return ("native C++/OpenMP+libmvec Philox4x32-10" if ctx.device_class == "cpu"
                else f"native CUDA Philox4x32-10 (cuda{torch.version.cuda})")

    def setup(self, p, ctx: Ctx):
        n_pairs = p["paths"] // 2
        seed = ctx.seed
        if ctx.device_class == "cpu":
            ext = _mc_ext.cpu()
            fn = lambda: ext.mc_european_cpu(n_pairs, seed, S0, K, R, SIGMA, T)  # noqa: E731
            return {"fn": fn, "n": p["paths"], "gpu": False}
        ext = _mc_ext.cuda()
        acc = torch.zeros(2, dtype=torch.float64, device=ctx.device)
        sms = torch.cuda.get_device_properties(ctx.gpu_index).multi_processor_count
        blocks = sms * 8
        fn = lambda: ext.mc_european_cuda(n_pairs, seed, S0, K, R, SIGMA, T, acc, blocks)  # noqa: E731
        return {"fn": fn, "n": p["paths"], "gpu": True, "acc": acc}

    def call(self, s):
        s["last"] = s["fn"]()

    def validate(self, s, ctx):
        self.call(s)
        if s["gpu"]:
            torch.cuda.synchronize()
            total, total2 = (float(v) for v in s["acc"].cpu())
        else:
            total, total2 = s["last"]
        n = s["n"]
        disc = math.exp(-R * T)
        mean = total / n
        var = max(total2 / n - mean * mean, 0.0)
        price, se = disc * mean, disc * math.sqrt(var / n)
        return statistical(price, black_scholes_call(S0, K, R, SIGMA, T), se)


WORKLOAD = MonteCarlo()
