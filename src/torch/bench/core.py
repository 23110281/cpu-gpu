"""Timing engine and the single-configuration runner (docs/methodology.md §4, §5).

`run_one` is the only code path that produces a result row:
setup (untimed) -> validate at full size -> adaptive warmup -> sampled measurement under
telemetry -> optional GPU transfer timing -> statistics -> one CSV row + raw samples.
"""
from __future__ import annotations

import json
import math
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import torch

from . import schema, stats
from .telemetry import Sampler
from .validate import Check


# ── timing policy ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TimingPolicy:
    min_warmup_calls: int = 5
    min_warmup_s: float = 0.5
    max_warmup_s: float = 10.0
    stable_rel: float = 0.02
    min_samples: int = 30
    min_time_s: float = 1.0
    max_samples: int = 1000
    max_time_s: float = 30.0          # after this, stop once min_samples_long are collected
    min_samples_long: int = 10        # floor for long kernels (each call >= ~1 s)
    hard_cap_s: float = 900.0         # absolute safety stop
    long_call_s: float = 1.0          # warmup needs only 2 calls when a call is this long
    tiny_call_s: float = 50e-6
    target_sample_s: float = 1e-3


PUBLICATION = TimingPolicy()
# Smoke runs check that code paths work; their rows are flagged and never analysed.
SMOKE = TimingPolicy(min_warmup_calls=2, min_warmup_s=0.0, max_warmup_s=1.0, min_samples=5,
                     min_time_s=0.05, max_samples=50, max_time_s=2.0, min_samples_long=2, hard_cap_s=60.0)


class _Timer:
    """Times `k` back-to-back calls of fn. GPU: CUDA events (device time); CPU: perf_counter_ns."""

    def __init__(self, device_class: str):
        self.gpu = device_class == "gpu"
        if self.gpu:
            self.e0 = torch.cuda.Event(enable_timing=True)
            self.e1 = torch.cuda.Event(enable_timing=True)

    def __call__(self, fn, k: int) -> float:
        if self.gpu:
            self.e0.record()
            for _ in range(k):
                fn()
            self.e1.record()
            self.e1.synchronize()
            return self.e0.elapsed_time(self.e1) / k
        t0 = time.perf_counter_ns()
        for _ in range(k):
            fn()
        return (time.perf_counter_ns() - t0) / 1e6 / k


def measure(fn, device_class: str, policy: TimingPolicy = PUBLICATION, on_measure_start=None):
    """Returns (samples_ms_per_call, calls_per_sample, measured_calls, warmup_calls)."""
    timer = _Timer(device_class)
    # adaptive warmup
    warm, t_start = [], time.perf_counter()
    while True:
        warm.append(timer(fn, 1))
        elapsed = time.perf_counter() - t_start
        if len(warm) >= 2 and min(warm[-2:]) / 1e3 >= policy.long_call_s:
            break     # long kernels: first call absorbs lazy init; one more confirms steady state
        if len(warm) >= policy.min_warmup_calls and elapsed >= policy.min_warmup_s:
            if len(warm) >= 10:
                prev, last = median(warm[-10:-5]), median(warm[-5:])
                if prev > 0 and abs(last - prev) / prev < policy.stable_rel:
                    break
            elif policy is SMOKE:
                break
        if elapsed >= policy.max_warmup_s:
            break
    per_call_s = median(warm[-5:]) / 1e3
    k = 1
    if per_call_s < policy.tiny_call_s:
        k = max(1, math.ceil(policy.target_sample_s / max(per_call_s, 1e-9)))
    if on_measure_start:
        on_measure_start()
    samples, t_start = [], time.perf_counter()
    while True:
        samples.append(timer(fn, k))
        elapsed = time.perf_counter() - t_start
        n = len(samples)
        if (n >= policy.min_samples and elapsed >= policy.min_time_s) or n >= policy.max_samples \
                or (elapsed >= policy.max_time_s and n >= policy.min_samples_long) \
                or elapsed >= policy.hard_cap_s:
            break
    return samples, k, len(samples) * k, len(warm)


# ── workload interface ───────────────────────────────────────────────────────
@dataclass
class Ctx:
    device_class: str                  # "cpu" | "gpu"
    device: torch.device
    device_id: str
    threads: int
    placement: str
    gpu_index: int
    seed: int
    native_bf16: bool
    llc_bytes: int
    smoke: bool = False
    cpus: list = field(default_factory=list)
    sockets: set | None = None

    def generator(self, offset: int = 0) -> torch.Generator:
        return torch.Generator(device=self.device).manual_seed(self.seed + offset)


class Unsupported(Exception):
    """Raised by a workload when a configuration cannot run natively on this device."""


class Workload:
    id: str = "?"
    name: str = "?"
    bound: str = "?"
    field_: str = ""

    def configs(self, device_class: str | None = None) -> list[dict]:
        """Absolute config grid. device_class (cpu|gpu) may add device-specific *library*
        variants (e.g. several CPU BLAS libraries); sizes never depend on the device (§6)."""
        raise NotImplementedError

    def smoke_configs(self, device_class: str | None = None) -> list[dict]:
        seen, out = set(), []
        for c in self.configs(device_class):
            key = (c.get("precision"), c.get("variant"), c.get("lib"))
            if key not in seen:
                seen.add(key)
                out.append(c)
        return out

    def working_set_bytes(self, p: dict) -> int:
        raise NotImplementedError

    def library(self, ctx: Ctx) -> str:
        raise NotImplementedError

    def setup(self, p: dict, ctx: Ctx):
        raise NotImplementedError

    def call(self, state) -> None:
        raise NotImplementedError

    def validate(self, state, ctx: Ctx) -> Check:
        raise NotImplementedError

    def work(self, p: dict) -> dict:
        """{'flops': float, 'bytes': float, 'metric_name': str, 'metric_units_per_call': float}"""
        raise NotImplementedError

    def transfers(self, state) -> tuple[list, list]:
        """(device inputs, device outputs) for H2D/D2H timing on GPU. Empty = not applicable."""
        return [], []


# ── calibration ──────────────────────────────────────────────────────────────
CALIB_DIR = Path(__file__).resolve().parents[3] / "results" / "calibration"


def _calibration(device_id: str) -> dict:
    p = CALIB_DIR / f"{device_id}.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def _pct(bound: str, precision: str, gflops: float, gbps: float, table: dict) -> float:
    if not table:
        return math.nan
    if bound == "B" and table.get("bandwidth_gbps"):
        return 100.0 * gbps / table["bandwidth_gbps"]
    peak = (table.get("peak_gflops") or {}).get(precision)
    return 100.0 * gflops / peak if peak else math.nan


def _is_oom(e: BaseException) -> bool:
    if isinstance(e, (torch.OutOfMemoryError, MemoryError)):
        return True
    s = str(e).lower()
    return "out of memory" in s or "not enough memory" in s or "can't allocate memory" in s


def _transfer_ms(tensors, to_device: bool, reps: int = 5) -> float:
    """Median pinned-memory copy time (ms) for a list of tensors."""
    if not tensors:
        return math.nan
    if to_device:
        src = [t.detach().to("cpu").pin_memory() for t in tensors]
        dst = [torch.empty_like(t) for t in tensors]
    else:
        src = list(tensors)
        dst = [torch.empty(t.shape, dtype=t.dtype, pin_memory=True) for t in tensors]
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(reps):
        e0.record()
        for s, d in zip(src, dst):
            d.copy_(s, non_blocking=True)
        e1.record()
        e1.synchronize()
        times.append(e0.elapsed_time(e1))
    del src, dst
    return median(times)


def run_one(w: Workload, p: dict, ctx: Ctx, *, trial: int, run_id: str, git_sha: str,
            flags: list[str], out_root: Path, policy: TimingPolicy) -> dict:
    paths = schema.output_paths(out_root, ctx.device_class, ctx.device_id, w.id, w.name)
    chash = schema.config_hash(w.id, p, ctx.placement, ctx.device_id)
    ws = w.working_set_bytes(p)
    row = {
        "schema": schema.SCHEMA_VERSION, "run_id": run_id, "git_sha": git_sha, "trial": trial,
        "workload_id": w.id, "workload": w.name, "variant": p.get("variant", "base"), "bound": w.bound,
        "device_class": ctx.device_class, "device": ctx.device_id, "precision": p.get("precision", ""),
        "threads": ctx.threads if ctx.device_class == "cpu" else "", "placement": ctx.placement,
        "gpu_index": ctx.gpu_index if ctx.device_class == "gpu" else "",
        "params_json": p, "config_hash": chash, "working_set_bytes": ws,
        "ws_over_llc": ws / ctx.llc_bytes if ctx.llc_bytes else math.nan,
        "seed": ctx.seed, "status": "error",
    }
    row["regime"] = "dram" if row["ws_over_llc"] >= 4 else "cache"
    wk = w.work(p)
    row.update(flops=wk["flops"], bytes=wk["bytes"], metric_name=wk["metric_name"])
    try:
        row["library"] = w._library_for(p, ctx) if hasattr(w, "_library_for") else w.library(ctx)
        t0 = time.perf_counter()
        state = w.setup(p, ctx)
        if ctx.device_class == "gpu":
            torch.cuda.synchronize()
        row["setup_ms"] = (time.perf_counter() - t0) * 1e3
        # workloads whose exact work is only known after building inputs (e.g. sparse nnz)
        if isinstance(state, dict) and "work" in state:
            wk = state["work"]
            row.update(flops=wk["flops"], bytes=wk["bytes"], metric_name=wk["metric_name"])
            if "working_set_bytes" in wk:
                row["working_set_bytes"] = wk["working_set_bytes"]
                row["ws_over_llc"] = wk["working_set_bytes"] / ctx.llc_bytes if ctx.llc_bytes else math.nan
                row["regime"] = "dram" if row["ws_over_llc"] >= 4 else "cache"

        chk = w.validate(state, ctx)
        row.update(validation=chk.kind, max_err=chk.err, tol=chk.tol)
        if not chk.ok:
            row["status"] = "validation_failed"
            raise _Done

        sampler = Sampler(ctx.device_class, cpus=ctx.cpus, sockets=ctx.sockets, gpu_index=ctx.gpu_index)
        with sampler:
            samples, k, calls, _ = measure(lambda: w.call(state), ctx.device_class, policy)
        tel = sampler.result(calls)
        flags = flags + tel.pop("_flags")
        row.update(tel)

        s = stats.summarize(samples)
        row.update(s, calls_per_sample=k)
        sec = s["median_ms"] / 1e3
        row["gflops"] = wk["flops"] / sec / 1e9 if wk["flops"] else math.nan
        row["gbps"] = wk["bytes"] / sec / 1e9 if wk["bytes"] else math.nan
        row["metric_value"] = wk["metric_units_per_call"] / sec
        cal = _calibration(ctx.device_id)
        row["pct_measured_peak"] = _pct(w.bound, row["precision"], row["gflops"], row["gbps"], cal.get("measured", {}))
        row["pct_datasheet_peak"] = _pct(w.bound, row["precision"], row["gflops"], row["gbps"], cal.get("datasheet", {}))
        if s["cv"] > stats.UNSTABLE_CV:
            flags.append("unstable")

        if ctx.device_class == "gpu":
            ins, outs = w.transfers(state)
            row["h2d_ms"] = _transfer_ms(ins, True)
            row["d2h_ms"] = _transfer_ms(outs, False)
            if not (math.isnan(row["h2d_ms"]) or math.isnan(row["d2h_ms"])):
                row["e2e_ms"] = row["h2d_ms"] + s["median_ms"] + row["d2h_ms"]

        schema.write_samples(paths["samples"] / f"{run_id}_{chash}_t{trial}.csv.gz", samples, k)
        row["status"] = "ok"
    except _Done:
        pass
    except Unsupported as e:
        row["status"] = "unsupported"
        flags.append(f"unsupported:{e}")
    except BaseException as e:  # noqa: BLE001 - every failure must still produce a row
        if isinstance(e, KeyboardInterrupt):
            raise
        row["status"] = "oom" if _is_oom(e) else "error"
        flags.append(f"{type(e).__name__}:{str(e)[:160]}")
        if row["status"] == "error":
            traceback.print_exc()
    if policy is SMOKE:
        flags.append("smoke")
    row["flag"] = ";".join(dict.fromkeys(f.replace(";", ",") for f in flags))
    schema.append_row(paths["rows"], row)
    return row


class _Done(Exception):
    pass
