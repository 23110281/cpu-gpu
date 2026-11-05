"""C9 — SHA-256 over many independent messages (multi-buffer)   (compute-bound, integer)

Equal work: SHA-256 digest (FIPS 180-4, full padding) of each of M messages of L bytes,
same messages on both devices; all M digests materialised. Total 1 GiB per call; L in
{64 B, 1 KiB, 64 KiB}. SHA-256 is serial within a message, so parallelism is across
messages only (unlike AES-CTR, parallel within one stream).
CPU: OpenSSL libcrypto low-level SHA-256 per message, messages over OpenMP threads (SHA-NI or ARMv8
     SHA2 selected by OpenSSL). Variant `sw`: hardware SHA disabled (OPENSSL_ia32cap/armcap).
GPU: custom kernel, one message per thread (_crypto_ext.py).
Validation (exact): FIPS 180-4 vector ("abc"), then 4096 sampled messages (+ first/last)
against python hashlib.  Metric: GB/s of message data (also Mhash/s from M / time).
"""
from __future__ import annotations

import hashlib

import torch

from ..core import Ctx, Unsupported, Workload
from ..validate import exact
from . import _crypto_ext as X
from .c8_aes import NO_HW_CRYPTO_ENV

TOTAL = 1 << 30
LENGTHS = [64, 1024, 65536]


class SHA256(Workload):
    id, name, bound, field_ = "C9", "sha256", "C", "Security"

    def configs(self, device_class=None):
        out = [{"precision": "uint8", "variant": "base", "msg_len": L, "total": TOTAL} for L in LENGTHS]
        if device_class in (None, "cpu"):
            out += [{"precision": "uint8", "variant": "sw", "msg_len": L, "total": TOTAL, "_env": NO_HW_CRYPTO_ENV}
                    for L in LENGTHS]
        return out

    def working_set_bytes(self, p):
        return p["total"] + 32 * (p["total"] // p["msg_len"])

    def work(self, p):
        return {"flops": 0.0, "bytes": float(p["total"]), "metric_name": "GB/s",
                "metric_units_per_call": p["total"] / 1e9, "hashes": p["total"] // p["msg_len"]}

    def library(self, ctx):
        if ctx.device_class == "cpu":
            return f"{X.cpu().openssl_version()}/SHA256+OpenMP"
        return "custom CUDA SHA-256 (one message per thread)"

    def setup(self, p, ctx: Ctx):
        if p["variant"] == "sw" and ctx.device_class != "cpu":
            raise Unsupported("software-SHA variant is CPU-only")
        L, M = p["msg_len"], p["total"] // p["msg_len"]
        msgs = torch.empty(M * L, dtype=torch.uint8, device=ctx.device)
        msgs.fill_(0)
        msgs.copy_(torch.randint(0, 256, (M * L,), generator=ctx.generator(1), dtype=torch.uint8, device=ctx.device))
        dig = torch.empty(M * 32, dtype=torch.uint8, device=ctx.device)
        if ctx.device_class == "cpu":
            ext = X.cpu()
            fn = lambda m, d, l=L: ext.sha256_multi_cpu(m, l, d)  # noqa: E731
        else:
            ext = X.cuda()
            grid = torch.cuda.get_device_properties(ctx.gpu_index).multi_processor_count * 16
            fn = lambda m, d, l=L: ext.sha256_multi_cuda(m, l, d, grid)  # noqa: E731
        return {"msgs": msgs, "dig": dig, "fn": fn, "L": L, "M": M, "device": ctx.device}

    def call(self, s):
        s["fn"](s["msgs"], s["dig"])

    def validate(self, s, ctx):
        dev = s["device"]
        abc = torch.frombuffer(bytearray(b"abc"), dtype=torch.uint8).to(dev)
        d1 = torch.empty(32, dtype=torch.uint8, device=dev)
        s["fn"](abc, d1, 3)
        fips_ok = bytes(d1.cpu().numpy()).hex() == \
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        self.call(s)
        L, M = s["L"], s["M"]
        g = torch.Generator().manual_seed(5)
        idx = sorted(set(torch.randint(0, M, (4096,), generator=g).tolist()) | {0, M - 1})
        msgs, dig = s["msgs"].cpu().numpy(), s["dig"].cpu().numpy()
        bad = sum(hashlib.sha256(msgs[i * L:(i + 1) * L].tobytes()).digest() != dig[i * 32:(i + 1) * 32].tobytes()
                  for i in idx)
        return exact(fips_ok and bad == 0, bad + (0 if fips_ok else 1))

    def transfers(self, s):
        return [s["msgs"]], [s["dig"]]


WORKLOAD = SHA256()
