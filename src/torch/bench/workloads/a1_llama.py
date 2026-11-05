"""A1 — Llama-3 inference: full-model forward, prefill and decode   (application; prefill ~C, decode ~B)

Model: exact Llama-3 architectures (RMSNorm, GQA attention with RoPE theta=500000, SwiGLU
MLP, untied LM head): llama3.2-1b, llama3.2-3b, llama3-8b. Weights are RANDOM (N(0, 0.02),
norms = 1): the official checkpoints are gated (Hugging Face licence + token), and for
dense fp32/bf16 kernels the run time does not depend on weight values -- documented
deviation. llama.cpp/vLLM end-to-end rows need real (or GGUF-converted) weights: future work.
Equal work: identical shapes, seeds, precision and token inputs on both devices.
  prefill : B=1 sequence of S tokens, causal SDPA, logits for the last token
  decode  : B sequences x 1 new token attending to a KV cache of 2048 past tokens (the
            cache is pre-filled in setup); logits for all B
Library: torch eager -- matmuls via oneDNN/MKL (CPU) / cuBLAS (GPU),
F.scaled_dot_product_attention (flash/mem-efficient on GPU, flash CPU kernel), GQA via
enable_gqa. Precision: fp32 everywhere; bf16 (variant ff) only where native.
Validation: a 2-layer instance of the same configuration (same code path per layer) is run
in the benchmark precision and in fp64 on the CPU; normwise error of the logits
(tol: fp32 1e-4, bf16 5e-2). Timing always uses the full model.
Metric: tokens/s (prefill: S per call; decode: B per call).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..core import Ctx, Workload
from ..validate import normwise
from . import _common as C

MODELS = {   # hidden, layers, heads, kv_heads, intermediate, vocab, tied embedding/LM head (as released)
    "llama3.2-1b": dict(d=2048, L=16, h=32, kv=8, ff=8192, V=128256, tied=True),    # 1.236 B params
    "llama3.2-3b": dict(d=3072, L=28, h=24, kv=8, ff=8192, V=128256, tied=True),    # 3.213 B
    "llama3-8b": dict(d=4096, L=32, h=32, kv=8, ff=14336, V=128256, tied=False),    # 8.030 B
}
KV_LEN = 2048
ROPE_THETA = 500000.0
EPS = 1e-5


def n_params(m):
    hd = m["d"] // m["h"]
    attn = m["d"] * (m["h"] * hd) + 2 * m["d"] * (m["kv"] * hd) + (m["h"] * hd) * m["d"]
    mlp = 3 * m["d"] * m["ff"]
    return m["L"] * (attn + mlp + 2 * m["d"]) + (1 if m["tied"] else 2) * m["V"] * m["d"] + m["d"]


class Model:
    def __init__(self, m, layers, dtype, device, gen):
        d, hd = m["d"], m["d"] // m["h"]
        self.m, self.hd, self.L = m, hd, layers
        r = lambda *s: (torch.randn(*s, generator=gen, device=device, dtype=torch.float32) * 0.02).to(dtype)  # noqa: E731
        self.emb = r(m["V"], d)
        self.lm = self.emb if m["tied"] else r(m["V"], d)
        self.norm = torch.ones(d, dtype=dtype, device=device)
        self.layers = [dict(wq=r(m["h"] * hd, d), wk=r(m["kv"] * hd, d), wv=r(m["kv"] * hd, d), wo=r(d, m["h"] * hd),
                            wg=r(m["ff"], d), wu=r(m["ff"], d), wd=r(d, m["ff"]),
                            n1=torch.ones(d, dtype=dtype, device=device), n2=torch.ones(d, dtype=dtype, device=device))
                       for _ in range(layers)]

    def to(self, dtype, device):
        new = object.__new__(Model)
        new.m, new.hd, new.L = self.m, self.hd, self.L
        cv = lambda t: t.to(device=device, dtype=dtype)  # noqa: E731
        new.emb, new.norm = cv(self.emb), cv(self.norm)
        new.lm = new.emb if self.m["tied"] else cv(self.lm)
        new.layers = [{k: cv(v) for k, v in l.items()} for l in self.layers]
        return new


def rmsnorm(x, w):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True).to(x.dtype) + EPS) * w


def rope_tables(positions, hd, dtype, device):
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, hd, 2, dtype=torch.float64, device=device) / hd))
    ang = positions.to(torch.float64)[:, None] * inv[None, :]
    return torch.cos(ang).to(dtype), torch.sin(ang).to(dtype)


def apply_rope(x, cos, sin):
    """x: [B, H, S, hd]; rotate-half convention (as HF Llama)."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    c = torch.cat([cos, cos], -1)[None, None]
    s = torch.cat([sin, sin], -1)[None, None]
    return x * c + torch.cat([-x2, x1], -1) * s


def forward(model: Model, tokens, kv_cache=None, pos0=0):
    """tokens [B, S]. kv_cache: list of (k, v) [B, kv, P, hd] (decode) or None (prefill).
    Returns logits of the last position [B, V]."""
    m, hd = model.m, model.hd
    B, S = tokens.shape
    x = model.emb[tokens]
    positions = torch.arange(pos0, pos0 + S, device=tokens.device)
    cos, sin = rope_tables(positions, hd, x.dtype, x.device)
    for li, l in enumerate(model.layers):
        h = rmsnorm(x, l["n1"])
        q = (h @ l["wq"].T).view(B, S, m["h"], hd).transpose(1, 2)
        k = (h @ l["wk"].T).view(B, S, m["kv"], hd).transpose(1, 2)
        v = (h @ l["wv"].T).view(B, S, m["kv"], hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if kv_cache is not None:
            kc, vc = kv_cache[li]
            k = torch.cat([kc, k], 2)
            v = torch.cat([vc, v], 2)
            a = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)        # new token sees all past
        else:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        x = x + a.transpose(1, 2).reshape(B, S, m["h"] * hd) @ l["wo"].T
        h = rmsnorm(x, l["n2"])
        x = x + (F.silu(h @ l["wg"].T) * (h @ l["wu"].T)) @ l["wd"].T
    x = rmsnorm(x[:, -1], model.norm)
    return x @ model.lm.T


class Llama(Workload):
    id, name, bound, field_ = "A1", "llama3", "A", "LLM / AI"

    def configs(self, device_class=None):
        out = []
        for prec, var in (("fp32", "base"), ("bf16", "ff")):
            for mdl in MODELS:
                for S in (512, 2048):
                    out.append({"precision": prec, "variant": var, "model": mdl, "phase": "prefill", "batch": 1, "seq": S})
                for B in (1, 8, 32):
                    out.append({"precision": prec, "variant": var, "model": mdl, "phase": "decode", "batch": B, "seq": 1})
        return out

    def _kv_bytes(self, p):
        m = MODELS[p["model"]]
        if p["phase"] != "decode":
            return 0
        return 2 * m["L"] * p["batch"] * m["kv"] * (m["d"] // m["h"]) * KV_LEN * C.ELEM[p["precision"]]

    def working_set_bytes(self, p):
        return n_params(MODELS[p["model"]]) * C.ELEM[p["precision"]] + self._kv_bytes(p)

    def work(self, p):
        m = MODELS[p["model"]]
        tok = p["seq"] * p["batch"]
        # matmul FLOPs: every weight except the embedding gather, but the LM head is a matmul
        # even when tied; weights streamed once per call (decode is bound by these bytes)
        mat = n_params(m) - (0 if m["tied"] else m["V"] * m["d"])
        flops = 2.0 * mat * tok
        byt = mat * C.ELEM[p["precision"]] + self._kv_bytes(p)
        return {"flops": flops, "bytes": float(byt), "metric_name": "tokens/s", "metric_units_per_call": float(tok)}

    def library(self, ctx):
        return C.library_tag(ctx, "llama-eager/sdpa+" + ("oneDNN" if ctx.device_class == "cpu" else "cuBLAS"))

    def _inputs(self, p, m, device, dtype, gen):
        B, S = p["batch"], p["seq"]
        tokens = torch.randint(0, m["V"], (B, S), generator=gen, device=device)
        cache = None
        if p["phase"] == "decode":
            hd = m["d"] // m["h"]
            cache = [(torch.randn(B, m["kv"], KV_LEN, hd, generator=gen, device=device).to(dtype),
                      torch.randn(B, m["kv"], KV_LEN, hd, generator=gen, device=device).to(dtype))
                     for _ in range(m["L"])]
        return tokens, cache

    def setup(self, p, ctx: Ctx):
        C.require_native(p, ctx)
        m, dt = MODELS[p["model"]], C.DTYPE[p["precision"]]
        model = Model(m, m["L"], dt, ctx.device, ctx.generator(1))
        tokens, cache = self._inputs(p, m, ctx.device, dt, ctx.generator(2))
        pos0 = KV_LEN if p["phase"] == "decode" else 0
        torch.set_grad_enabled(False)
        return {"model": model, "tokens": tokens, "cache": cache, "pos0": pos0, "p": p, "dt": dt}

    def call(self, s):
        s["out"] = forward(s["model"], s["tokens"], s["cache"], s["pos0"])

    def validate(self, s, ctx):
        p, m = s["p"], MODELS[s["p"]["model"]]
        small = Model(m, 2, s["dt"], ctx.device, ctx.generator(1))
        tokens = s["tokens"]
        cache = None if s["cache"] is None else s["cache"][:2]
        got = forward(small, tokens, cache, s["pos0"]).float().cpu()
        ref = forward(small.to(torch.float64, "cpu"), tokens.cpu(),
                      None if cache is None else [(k.double().cpu(), v.double().cpu()) for k, v in cache],
                      s["pos0"])
        self.call(s)                     # full model once (exercises every layer before timing)
        return normwise(got, ref, 1e-4 if s["dt"] == torch.float32 else 5e-2)

    def transfers(self, s):
        return [s["tokens"]], [s["out"]] if "out" in s else []


WORKLOAD = Llama()
