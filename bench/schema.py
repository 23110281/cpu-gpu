"""The single long-format result schema (docs/methodology.md §13).

One CSV row per (configuration, trial). Raw per-sample timings are written next to
the rows so every statistic can be recomputed from data.
"""
from __future__ import annotations

import csv
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

SCHEMA_VERSION = 2

COLUMNS = [
    # identity / provenance
    "schema", "run_id", "git_sha", "trial", "workload_id", "workload", "variant", "bound",
    "device_class", "device", "library", "precision", "threads", "placement", "gpu_index",
    # problem
    "params_json", "config_hash", "working_set_bytes", "ws_over_llc", "regime", "seed",
    # timing (milliseconds per call)
    "n_samples", "calls_per_sample", "median_ms", "iqr_ms", "min_ms", "cv",
    "ci95_lo_ms", "ci95_hi_ms", "setup_ms", "h2d_ms", "d2h_ms", "e2e_ms",
    # throughput
    "flops", "bytes", "metric_name", "metric_value", "gflops", "gbps",
    "pct_measured_peak", "pct_datasheet_peak",
    # power / clocks
    "power_w", "energy_j_per_op", "power_valid", "freq_mhz", "temp_c", "throttle",
    # outcome
    "validation", "max_err", "tol", "status", "flag",
]

STATUSES = ("ok", "validation_failed", "oom", "unsupported", "error")


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return repr(v)
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True, separators=(",", ":"))
    return str(v)


def append_row(path: Path, row: dict) -> None:
    """Append one row under an exclusive lock; writes the header on first use."""
    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise KeyError(f"columns not in schema: {sorted(unknown)}")
    if row.get("status") not in STATUSES:
        raise ValueError(f"invalid status {row.get('status')!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            new = f.tell() == 0
            w = csv.writer(f)
            if new:
                w.writerow(COLUMNS)
            w.writerow([_fmt(row.get(c, "")) for c in COLUMNS])
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_samples(path: Path, samples_ms, calls_per_sample: int) -> None:
    """Raw per-call times (ms), one per line, gzip-compressed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        f.write(f"# calls_per_sample={calls_per_sample}\nms_per_call\n")
        for s in samples_ms:
            f.write(f"{s!r}\n")


def read_samples(path: Path) -> list[float]:
    with gzip.open(path, "rt") as f:
        return [float(l) for l in f if l and l[0] not in "#m"]


def output_paths(root: Path, device_class: str, device: str, workload_id: str, workload: str):
    base = root / device_class / device
    return {
        "rows": base / f"{workload_id}_{workload}.csv",
        "samples": base / "samples",
        "env": base / "env",
    }


def default_root() -> Path:
    return Path(os.environ.get("BENCH_OUT", Path(__file__).resolve().parents[1] / "results" / "data"))


def config_hash(workload_id: str, params: dict, placement: str, device_id: str) -> str:
    """Stable id of one configuration (independent of trial and run)."""
    blob = json.dumps({"w": workload_id, "p": params, "pl": placement, "d": device_id}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]
