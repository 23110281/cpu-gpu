"""C4 — FFT, complex single precision, 1-D and 2-D   (compute/BW-bound by size; ISCA'10 kernel 4)

Equal work: forward complex-to-complex transform of the same seeded complex64 signal,
written into a preallocated output (`out=`).
Library: torch.fft -> MKL DFTI (x86 CPU) / pocketfft (other CPUs) and cuFFT (GPU).
Plan creation: cuFFT plans are cached by torch (plan cache warmed during warmup); MKL
descriptors are created per call inside ATen and cannot be hoisted — this planning cost is
therefore inside CPU samples (documented deviation, methodology table). pyFFTW
(FFTW_MEASURE, plans reused) is the planned alternative CPU library.
Validation: normwise vs numpy.fft in complex128; tol = c*u*log2(N).
Work model: 5 N log2 N FLOPs (standard FFT convention); compulsory bytes = in + out.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..core import Ctx, Workload
from ..validate import TOL_C, normwise, unit_roundoff
from . import _common as C

SIZES_1D = [1 << 10, 1 << 14, 1 << 18, 1 << 22, 1 << 26]
SIZES_2D = [1024, 2048, 4096, 8192]


class FFT(Workload):
    id, name, bound, field_ = "C4", "fft", "C", "Signal"

    def configs(self, device_class=None):
        return ([{"precision": "c64", "variant": "base", "dim": 1, "n": n} for n in SIZES_1D] +
                [{"precision": "c64", "variant": "base", "dim": 2, "n": n} for n in SIZES_2D])

    def _points(self, p):
        return p["n"] if p["dim"] == 1 else p["n"] ** 2

    def working_set_bytes(self, p):
        return 2 * 8 * self._points(p)

    def work(self, p):
        N = self._points(p)
        f = 5.0 * N * math.log2(N)
        return {"flops": f, "bytes": 2.0 * 8 * N, "metric_name": "GFLOP/s", "metric_units_per_call": f / 1e9}

    def library(self, ctx):
        return C.library_tag(ctx, "fft/" + (C.cpu_backend("fft") if ctx.device_class == "cpu" else "cuFFT"))

    def setup(self, p, ctx: Ctx):
        shape = (p["n"],) if p["dim"] == 1 else (p["n"], p["n"])
        x = torch.empty(shape, dtype=torch.complex64, device=ctx.device)
        xr = torch.view_as_real(x)
        xr.fill_(0).normal_(generator=ctx.generator(1))
        y = torch.empty_like(x)
        fn = (lambda: torch.fft.fft(x, out=y)) if p["dim"] == 1 else (lambda: torch.fft.fft2(x, out=y))
        return {"x": x, "y": y, "fn": fn, "dim": p["dim"], "N": self._points(p)}

    def call(self, s):
        s["fn"]()

    def validate(self, s, ctx):
        self.call(s)
        xn = s["x"].cpu().numpy().astype(np.complex128)
        ref = np.fft.fft(xn) if s["dim"] == 1 else np.fft.fft2(xn)
        refr = torch.view_as_real(torch.from_numpy(ref))
        got = torch.view_as_real(s["y"].cpu())
        tol = TOL_C * unit_roundoff(torch.float32) * math.log2(s["N"])
        return normwise(got, refr, tol)

    def transfers(self, s):
        return [s["x"]], [s["y"]]


WORKLOAD = FFT()
