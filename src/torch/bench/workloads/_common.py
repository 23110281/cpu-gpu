"""Helpers shared by workload modules."""
from __future__ import annotations

import torch

from ..core import Ctx, Unsupported

DTYPE = {"fp64": torch.float64, "fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
ELEM = {"fp64": 8, "fp32": 4, "bf16": 2, "fp16": 2}


def first_touch(n_or_shape, dtype, ctx: Ctx) -> torch.Tensor:
    """Allocate uninitialised and write once with a parallel ATen op, so that on CPU each page
    is first touched by the (pinned) thread that will later use it (methodology §3)."""
    t = torch.empty(n_or_shape, dtype=dtype, device=ctx.device)
    t.fill_(0)
    return t


def require_native(p: dict, ctx: Ctx) -> None:
    if p.get("precision") == "bf16" and not ctx.native_bf16:
        raise Unsupported("bf16 not native on this device")


def library_tag(ctx: Ctx, kernel: str) -> str:
    from .. import devices
    if ctx.device_class == "cpu":
        b = devices.torch_backend_info()
        return f"torch{b['torch']}/{kernel}/blas={b['blas']}"
    return f"torch{torch.__version__}/{kernel}/cuda{torch.version.cuda}"
