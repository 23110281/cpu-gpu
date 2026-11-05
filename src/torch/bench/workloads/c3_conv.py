"""C3 — 2-D image convolution (filtering)   (compute-bound; BW-bound for small filters; ISCA'10 kernel 3)

Equal work: "same"-size output (zero border) of ONE k x k filter applied to every channel
of an H x W image with C in {1 (grayscale), 3 (RGB)} -- the image-filtering definition
(cross-correlation, as both libraries compute); identical seeded image and filter.
Libraries (config key `lib`):
  torch  : F.conv2d, groups=C, filter replicated per channel -> oneDNN (CPU) / cuDNN (GPU);
           cuDNN autotuning (cudnn.benchmark) happens during warmup, never in samples.
  opencv : cv2.filter2D on the HxWxC image (CPU only). Measured on EPYC 7402P, 16K^2, k=5:
           OpenCV 444 ms vs oneDNN 1521 ms -> NN-oriented conv kernels are not the fastest
           image-filter library, so both are timed and the faster is kept.
Validation: normwise vs an fp64 convolution of the same inputs; k*k-term reductions.
Work model: 2*k*k*C*H*W FLOPs; compulsory bytes = input + output (+ filter).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core import Ctx, Unsupported, Workload
from ..validate import normwise, tol_reduction
from . import _common as C

IMAGES = [1024, 4096, 8192]        # 8192^2 = 64x LLC on EPYC; 16384 added hours without new information
KERNELS = [3, 5, 7]
CHANNELS = [1, 3]


def _cv2():
    try:
        import cv2
    except ImportError as e:
        raise Unsupported("opencv-python not installed") from e
    return cv2


class Conv2d(Workload):
    id, name, bound, field_ = "C3", "conv2d", "C", "Image"

    def configs(self, device_class=None):
        libs = ("torch", "opencv") if device_class in (None, "cpu") else ("torch",)
        return [{"precision": "fp32", "variant": "base", "hw": hw, "k": k, "c": c, "lib": lib}
                for lib in libs for c in CHANNELS for k in KERNELS for hw in IMAGES]

    def working_set_bytes(self, p):
        return 2 * p["c"] * p["hw"] ** 2 * 4

    def work(self, p):
        f = 2.0 * p["k"] ** 2 * p["c"] * p["hw"] ** 2
        return {"flops": f, "bytes": 2.0 * p["c"] * p["hw"] ** 2 * 4,
                "metric_name": "Mpixels/s", "metric_units_per_call": p["hw"] ** 2 / 1e6}

    def library(self, ctx):
        return C.library_tag(ctx, "conv2d/" + ("oneDNN" if ctx.device_class == "cpu" else "cuDNN"))

    def _library_for(self, p, ctx):
        if p.get("lib", "torch") == "opencv":
            cv2 = _cv2()
            return f"opencv{cv2.__version__}/filter2D/threads={cv2.getNumThreads()}"
        return self.library(ctx)

    def setup(self, p, ctx: Ctx):
        lib = p.get("lib", "torch")
        if lib == "opencv" and ctx.device_class != "cpu":
            raise Unsupported("opencv variant is CPU-only")
        if ctx.device_class == "gpu":
            torch.backends.cudnn.benchmark = True
        c, hw, k = p["c"], p["hw"], p["k"]
        x = torch.empty(1, c, hw, hw, device=ctx.device)
        x.fill_(0).uniform_(0.0, 1.0, generator=ctx.generator(1))
        filt = torch.randn(k, k, generator=ctx.generator(2), device=ctx.device) / k
        s = {"x": x, "filt": filt, "pad": k // 2, "c": c, "k": k, "lib": lib}
        if lib == "torch":
            s["w"] = filt.expand(c, 1, k, k).contiguous()
            s["y"] = torch.empty_like(x)
        else:
            cv2 = _cv2()
            cv2.setNumThreads(ctx.threads)
            # OpenCV's native layouts: HxW for one channel, HxWxC otherwise
            img = x[0, 0].contiguous() if c == 1 else x[0].permute(1, 2, 0).contiguous()
            s["img"], s["dst"] = img.numpy(), torch.empty_like(img).numpy()
            s["kern"] = filt.numpy()
            s["cv2"] = cv2
        return s

    def call(self, s):
        if s["lib"] == "torch":
            # F.conv2d has no out= variant: each call allocates its output. On GPU the caching
            # allocator reuses the block; on CPU the harness runs under jemalloc with decay
            # disabled (sweep.py, methodology §4) so the block is reused without page faults.
            s["y"] = F.conv2d(s["x"], s["w"], padding=s["pad"], groups=s["c"])
        else:
            s["cv2"].filter2D(s["img"], -1, s["kern"], dst=s["dst"], borderType=s["cv2"].BORDER_CONSTANT)

    def _output_nchw(self, s):
        if s["lib"] == "torch":
            return s["y"]
        d = torch.from_numpy(s["dst"])
        return (d.unsqueeze(-1) if d.dim() == 2 else d).permute(2, 0, 1).unsqueeze(0)

    def validate(self, s, ctx):
        self.call(s)
        w64 = s["filt"].double().expand(s["c"], 1, s["k"], s["k"])
        ref = F.conv2d(s["x"].double(), w64, padding=s["pad"], groups=s["c"])
        return normwise(self._output_nchw(s), ref, tol_reduction(torch.float32, s["k"] ** 2))

    def transfers(self, s):
        return ([s["x"]], [s["y"]]) if s["lib"] == "torch" else ([], [])


WORKLOAD = Conv2d()
