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
Library: torch eager -- matmuls via torch's CPU BLAS (MKL on x86 wheels: MKL_VERBOSE on Xeon Gold
5512U shows SGEMM for every linear, including batch-1 decode as an N=1 SGEMM, never SGEMV) /
cuBLAS (GPU). Attention:
  prefill: F.scaled_dot_product_attention (causal, GQA via enable_gqa)
  decode : `decode_attention` -- GQA as a reshape (no K/V copies) + two batched matmuls
           (q.K^T, softmax, .V) that stream the KV cache once. SDPA's fused kernels tile over
           queries (64-row tiles on the V100's mem-efficient kernel), so with ONE query token
           they wasted most of the work: on V100 SDPA was 47% (B=1) / 64% (B=8) of the decode
           step. The new token's K/V are written in place into a preallocated cache slot (as an
           inference engine does) instead of torch.cat copying the whole cache every step.
  The old cat + SDPA path is kept as the independent fp64 validation reference.
GPU decode is also timed replayed from a CUDA graph (`lib=cudagraph`, as vLLM/TensorRT-LLM
serve decode): an eager step issues ~500 small kernels from Python, and on the V100 node
(Xeon E5-2667 v4) issuing them took 16 ms vs 8.5 ms of GPU work, i.e. eager batch-1 decode
measured the host's launch rate. Eager rows are kept; the analysis keeps the faster.
Precision: fp32 everywhere; bf16 (variant ff) only where native.
CPU fp32 linear layers (config key `lib`, as in C1): torch (MKL on x86 wheels), OpenBLAS,
BLIS -- the fastest is kept. Batch-1 decode is pure GEMV over the weights; on EPYC 7402P
torch/MKL streamed only ~20 GB/s (one core's worth of the 125 GB/s socket), while the same
call on Intel runs at the full read rate, so the alternative CBLAS libraries are timed too.
Validation: a 2-layer instance of the same configuration (same code path per layer) is run
in the benchmark precision and in fp64 on the CPU; normwise error of the logits
(tol: fp32 1e-4, bf16 5e-2). Timing always uses the full model.
Metric: tokens/s (prefill: S per call; decode: B per call).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..core import Ctx, Unsupported, Workload
from ..validate import normwise
from . import _blas
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


def torch_linear(x, w):
    return x @ w.T


def blas_linear(lib: str, prec: str):
    """x [..., K] @ w[N, K].T through a CBLAS library (CPU, contiguous tensors)."""
    f = _blas.linear_fn(lib, prec)

    def mm(x, w):
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        y = torch.empty(x2.shape[0], w.shape[0], dtype=x.dtype)
        f(x2, w, y)
        return y.view(*x.shape[:-1], w.shape[0])
    return mm


def decode_attention(q, kc, vc):
    """Attention of S (=1) new query tokens over a KV cache, as two batched matmuls.
    q [B, h, S, hd]; kc, vc [B, kv, P, hd] (may be strided views of a larger buffer).
    GQA: query head i uses kv head i // (h/kv), so the h/kv heads of each group are stacked
    as extra rows of one matmul -- every K/V element is read once and never copied.
    The new token attends to every cached position, itself included (no mask needed)."""
    B, h, S, hd = q.shape
    kv = kc.shape[1]
    g = h // kv
    qg = q.reshape(B, kv, g * S, hd)
    scores = torch.matmul(qg, kc.transpose(-1, -2)) * (1.0 / math.sqrt(hd))      # [B, kv, g*S, P]
    acc = torch.float32 if q.dtype in (torch.bfloat16, torch.float16) else q.dtype      # bf16: fp32 softmax
    probs = torch.softmax(scores, dim=-1, dtype=acc).to(q.dtype)
    return torch.matmul(probs, vc).reshape(B, h, S, hd)


def forward(model: Model, tokens, kv_cache=None, pos0=0, mm=torch_linear, reference=False):
    """tokens [B, S]. kv_cache (decode): list of (k, v) buffers [B, kv, >= pos0+S, hd] whose
    first pos0 positions hold the past; the new K/V are written in place at pos0..pos0+S.
    None = prefill. mm(x, w) = x @ w.T for every linear layer. reference=True uses the
    independent cat + SDPA path (validation). Returns logits of the last position [B, V]."""
    m, hd = model.m, model.hd
    B, S = tokens.shape
    x = model.emb[tokens]
    positions = torch.arange(pos0, pos0 + S, device=tokens.device)
    cos, sin = rope_tables(positions, hd, x.dtype, x.device)
    for li, l in enumerate(model.layers):
        h = rmsnorm(x, l["n1"])
        q = mm(h, l["wq"]).view(B, S, m["h"], hd).transpose(1, 2)
        k = mm(h, l["wk"]).view(B, S, m["kv"], hd).transpose(1, 2)
        v = mm(h, l["wv"]).view(B, S, m["kv"], hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if kv_cache is not None and reference:
            kc, vc = kv_cache[li]
            k = torch.cat([kc[:, :, :pos0], k], 2)
            v = torch.cat([vc[:, :, :pos0], v], 2)
            a = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)        # new token sees all past
        elif kv_cache is not None:
            kc, vc = kv_cache[li]
            kc[:, :, pos0:pos0 + S] = k
            vc[:, :, pos0:pos0 + S] = v
            a = decode_attention(q, kc[:, :, :pos0 + S], vc[:, :, :pos0 + S])
        else:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        x = x + mm(a.transpose(1, 2).reshape(B, S, m["h"] * hd), l["wo"])
        h = rmsnorm(x, l["n2"])
        x = x + mm(F.silu(mm(h, l["wg"])) * mm(h, l["wu"]), l["wd"])
    x = rmsnorm(x[:, -1], model.norm)
    return mm(x, model.lm)


class Llama(Workload):
    id, name, bound, field_ = "A1", "llama3", "A", "LLM / AI"

    def configs(self, device_class=None):
        cpu_libs = _blas.CPU_LIBS if device_class in (None, "cpu") else ("torch",)
        graph = ("cudagraph",) if device_class in (None, "gpu") else ()     # GPU decode only
        out = []
        for prec, var, libs in (("fp32", "base", cpu_libs), ("bf16", "ff", ("torch",))):
            for mdl in MODELS:
                for lib in libs:
                    for S in (512, 2048):
                        out.append({"precision": prec, "variant": var, "model": mdl, "phase": "prefill",
                                    "batch": 1, "seq": S, "lib": lib})
                for lib in libs + graph:
                    for B in (1, 8, 32):
                        out.append({"precision": prec, "variant": var, "model": mdl, "phase": "decode",
                                    "batch": B, "seq": 1, "lib": lib})
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
        return C.library_tag(ctx, "llama-eager/sdpa-prefill,bmm-decode+" + ("torch-blas" if ctx.device_class == "cpu" else "cuBLAS"))

    def _library_for(self, p, ctx):
        lib = p.get("lib", "torch")
        if lib == "torch":
            return self.library(ctx)
        if lib == "cudagraph":
            return self.library(ctx) + "+cudagraph"
        if ctx.device_class != "cpu":
            raise Unsupported(f"{lib} is a CPU library")
        return f"llama-eager/sdpa-prefill,bmm-decode+linear={lib}/{_blas.version(lib)}"

    def _inputs(self, p, m, device, dtype, gen):
        B, S = p["batch"], p["seq"]
        tokens = torch.randint(0, m["V"], (B, S), generator=gen, device=device)
        cache = None
        if p["phase"] == "decode":
            hd = m["d"] // m["h"]
            # preallocated: KV_LEN past positions (random) + S slots the step writes in place
            cache = []
            for _ in range(m["L"]):
                kv_pair = []
                for _ in range(2):
                    t = torch.zeros(B, m["kv"], KV_LEN + S, hd, device=device, dtype=dtype)
                    t[:, :, :KV_LEN] = torch.randn(B, m["kv"], KV_LEN, hd, generator=gen, device=device).to(dtype)
                    kv_pair.append(t)
                cache.append(tuple(kv_pair))
        return tokens, cache

    def setup(self, p, ctx: Ctx):
        C.require_native(p, ctx)
        lib = p.get("lib", "torch")
        if lib == "cudagraph" and (ctx.device_class == "cpu" or p["phase"] != "decode"):
            raise Unsupported("cudagraph: GPU decode only")
        if lib in ("torch", "cudagraph"):
            mm = torch_linear
        elif ctx.device_class != "cpu":
            raise Unsupported(f"{lib} is a CPU library")
        else:
            mm = blas_linear(lib, p["precision"])
        m, dt = MODELS[p["model"]], C.DTYPE[p["precision"]]
        model = Model(m, m["L"], dt, ctx.device, ctx.generator(1))
        tokens, cache = self._inputs(p, m, ctx.device, dt, ctx.generator(2))
        pos0 = KV_LEN if p["phase"] == "decode" else 0
        torch.set_grad_enabled(False)
        s = {"model": model, "tokens": tokens, "cache": cache, "pos0": pos0, "p": p, "dt": dt, "mm": mm}
        if lib == "cudagraph":
            self._capture(s)
        return s

    @staticmethod
    def _capture(s):
        """Capture one decode step (static tokens / cache / output addresses) in a CUDA graph."""
        step = lambda: forward(s["model"], s["tokens"], s["cache"], s["pos0"], s["mm"])  # noqa: E731
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):                     # warm up cuBLAS handles/workspaces off-graph
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        s["graph"] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(s["graph"]):
            s["graph_out"] = step()
        torch.cuda.synchronize()

    def call(self, s):
        if "graph" in s:
            s["graph"].replay()
            s["out"] = s["graph_out"]
        else:
            s["out"] = forward(s["model"], s["tokens"], s["cache"], s["pos0"], s["mm"])

    def validate(self, s, ctx):
        p, m = s["p"], MODELS[s["p"]["model"]]
        small = Model(m, 2, s["dt"], ctx.device, ctx.generator(1))
        tokens = s["tokens"]
        cache = None if s["cache"] is None else s["cache"][:2]
        got = forward(small, tokens, cache, s["pos0"], s["mm"]).float().cpu()
        ref = forward(small.to(torch.float64, "cpu"), tokens.cpu(),
                      None if cache is None else [(k.double().cpu(), v.double().cpu()) for k, v in cache],
                      s["pos0"], reference=True)
        tol = 1e-4 if s["dt"] == torch.float32 else 5e-2
        self.call(s)                     # full model once (exercises every layer before timing)
        check = normwise(got, ref, tol)
        if "graph" in s and check.ok:    # the replayed graph must reproduce the eager full model
            eager = forward(s["model"], s["tokens"], s["cache"], s["pos0"], s["mm"])
            g = normwise(s["out"], eager, tol)
            check = g if g.err > check.err else check
        return check

    def transfers(self, s):
        return [s["tokens"]], [s["out"]] if "out" in s else []


WORKLOAD = Llama()
