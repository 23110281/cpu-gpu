"""Collection driver (docs/methodology.md §3, §5, §13).

One subprocess per (configuration, placement, trial): PyTorch thread settings are
once-per-process, and isolation prevents allocator / cache state leaking between configs.
Within a trial the job order is shuffled with a recorded seed. A campaign can be resumed:
jobs whose (config_hash, trial) already have a terminal row in this campaign are skipped.

Device configs: configs/devices/<DEVICE>.json, e.g.
  {"device_id": "INTEL_C6620", "device_class": "cpu",
   "placements": ["socket0", "socket0-smt"],
   "scaling": {"workloads": ["B1", "C1"], "base": "socket0", "threads": [1, 2, 4, 8, 16]},
   "max_mem_gb": 200}
  {"device_id": "NVIDIA_A100", "device_class": "gpu", "gpu_index": 0, "max_mem_gb": 40}
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import random
import socket
import subprocess
import sys
from pathlib import Path

from . import devices, schema

REPO = Path(__file__).resolve().parents[3]
TERMINAL = {"ok", "validation_failed", "oom", "unsupported"}
MEM_HEADROOM = 2.5   # working set x this must fit (validation clones, workspace)


def _campaign_id() -> str:
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"c{ts}_{socket.gethostname().split('.')[0]}"


def _done(out_root: Path, dev_cls: str, dev_id: str, w, campaign: str) -> set:
    p = schema.output_paths(out_root, dev_cls, dev_id, w.id, w.name)["rows"]
    if not p.exists():
        return set()
    return {(r["config_hash"], int(r["trial"])) for r in schema.read_rows(p)
            if r["run_id"].startswith(campaign + "-") and r["status"] in TERMINAL}


def _jobs(dcfg: dict, selected, smoke: bool):
    cls = dcfg["device_class"]
    if cls == "cpu":
        # drop placements that resolve to an identical CPU set (e.g. socket0-smt without SMT)
        placements, seen = [], set()
        for pl in dcfg.get("placements", ["socket0"]):
            key = tuple(devices.resolve_placement(pl).cpus)
            if key not in seen:
                seen.add(key)
                placements.append(pl)
    else:
        placements = [f"gpu{dcfg.get('gpu_index', 0)}"]
    jobs = []
    for w in selected:
        cfgs = w.smoke_configs(cls) if smoke else w.configs(cls)
        for c in cfgs:
            for pl in placements:
                jobs.append((w, c, pl))
    sc = dcfg.get("scaling")
    if sc and cls == "cpu" and not smoke:
        for w in selected:
            if w.id not in sc.get("workloads", []):
                continue
            # MEDIAN-size config per (precision, variant, lib): thread scaling at the largest size
            # is prohibitively long at 1 thread (e.g. fp64 GEMM 8192^3) and adds no information
            groups = {}
            for c in w.configs(cls):
                groups.setdefault((c.get("precision"), c.get("variant"), c.get("lib")), []).append(c)
            chosen = [sorted(g, key=w.working_set_bytes)[len(g) // 2] for g in groups.values()]
            base = sc.get("base", "socket0")
            avail = devices.resolve_placement(base).threads
            for c in chosen:
                for t in sorted({t for t in sc["threads"] if t < avail}):  # t == avail is the headline row
                    jobs.append((w, c, f"{base}:{t}"))
    return jobs


def _preflight_oom_row(w, c, dcfg, pl, trial, campaign, out_root):
    from . import env
    sha, dirty = env.git_state()
    flag = f"preflight:ws*{MEM_HEADROOM}>max_mem_gb={dcfg['max_mem_gb']}" + (";dirty_tree" if dirty else "")
    row = {"schema": schema.SCHEMA_VERSION, "run_id": f"{campaign}-preflight", "trial": trial, "git_sha": sha,
           "workload_id": w.id, "workload": w.name, "variant": c.get("variant", "base"), "bound": w.bound,
           "device_class": dcfg["device_class"], "device": dcfg["device_id"], "precision": c.get("precision", ""),
           "placement": pl, "params_json": c, "working_set_bytes": w.working_set_bytes(c),
           "config_hash": schema.config_hash(w.id, c, pl, dcfg["device_id"]),
           "status": "oom", "flag": flag}
    p = schema.output_paths(out_root, dcfg["device_class"], dcfg["device_id"], w.id, w.name)["rows"]
    schema.append_row(p, row)


def run_sweep(dcfg: dict, workload_keys, trials: int, out_root: Path, campaign: str,
              smoke: bool = False, dry_run: bool = False, order_seed: int = 0) -> int:
    from .workloads import get, registry
    selected = [get(k) for k in workload_keys] if workload_keys else list(registry().values())
    jobs = _jobs(dcfg, selected, smoke)
    cls, dev_id = dcfg["device_class"], dcfg["device_id"]
    max_bytes = dcfg.get("max_mem_gb", float("inf")) * (1 << 30)
    print(f"[sweep] campaign={campaign} device={dev_id} jobs/trial={len(jobs)} trials={trials} "
          f"order_seed={order_seed}", flush=True)
    failures = 0
    for trial in range(1, trials + 1):
        order = list(jobs)
        random.Random(order_seed * 1000 + trial).shuffle(order)
        done = {w.id: _done(out_root, cls, dev_id, w, campaign) for w in selected}
        for i, (w, c, pl) in enumerate(order, 1):
            chash = schema.config_hash(w.id, c, pl, dev_id)
            if (chash, trial) in done[w.id]:
                continue
            tag = f"[t{trial} {i}/{len(order)}] {w.id} {json.dumps(c, sort_keys=True)} @{pl}"
            if w.working_set_bytes(c) * MEM_HEADROOM > max_bytes:
                print(f"{tag}  -> oom (preflight)", flush=True)
                if not dry_run:
                    _preflight_oom_row(w, c, dcfg, pl, trial, campaign, out_root)
                continue
            cmd, env = _command(w, c, pl, dcfg, trial, campaign, out_root, smoke)
            if dry_run:
                print(f"{tag}\n    {' '.join(cmd)}")
                continue
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            last = (r.stdout.strip().splitlines() or ["{}"])[-1]
            print(f"{tag}  rc={r.returncode} {last}", flush=True)
            if r.returncode != 0:
                failures += 1
                sys.stderr.write(r.stderr[-4000:])
    return 1 if failures else 0


def _command(w, c, pl, dcfg, trial, campaign, out_root, smoke):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src" / "torch") + os.pathsep + env.get("PYTHONPATH", "")
    prefix = []
    args = ["--workload", w.id, "--params", json.dumps(c, sort_keys=True), "--trial", str(trial),
            "--device-id", dcfg["device_id"], "--campaign", campaign, "--out", str(out_root)]
    if dcfg["device_class"] == "cpu":
        p = devices.resolve_placement(pl)
        env.update(devices.placement_env(p))
        env.update(devices.cpu_allocator_env())
        prefix = devices.numactl_prefix(p)
        if prefix:
            env["BENCH_MEMBIND"] = ",".join(map(str, p.nodes))
        args += ["--device", "cpu", "--placement", pl]
    else:
        args += ["--device", "cuda", "--gpu", str(dcfg.get("gpu_index", 0))]
    if smoke:
        args.append("--smoke")
    env.update(c.get("_env", {}))     # e.g. OpenSSL capability masks: must precede library init
    return prefix + [sys.executable, "-m", "bench", "run"] + args, env


def detected_mem_gb(cfg: dict) -> float:
    """Usable memory for one job when the device config does not pin it: CPU = MemTotal
    divided across sockets (socket placements bind memory to one node), x0.8; GPU = total
    device memory from nvidia-smi, x0.9."""
    try:
        if cfg["device_class"] == "cpu":
            kb = int(next(l for l in Path("/proc/meminfo").read_text().splitlines()
                          if l.startswith("MemTotal")).split()[1])
            return kb / (1 << 20) / max(1, len(devices.sockets())) * 0.8
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits",
                              "-i", str(cfg.get("gpu_index", 0))], capture_output=True, text=True).stdout
        return float(out.strip().splitlines()[0]) / 1024 * 0.9
    except Exception:
        return float("inf")


def _load_device_config(path: str) -> dict:
    cfg = json.loads(Path(path).read_text())
    for k in ("device_id", "device_class"):
        if k not in cfg:
            raise SystemExit(f"{path}: missing '{k}'")
    cfg.setdefault("max_mem_gb", detected_mem_gb(cfg))
    return cfg


def _with_mem(cfg: dict) -> dict:
    cfg.setdefault("max_mem_gb", detected_mem_gb(cfg))
    return cfg


def add_parsers(sub) -> None:
    s = sub.add_parser("sweep", help="collect a device's full matrix (subprocess per config x trial)")
    s.add_argument("--device-config", required=True, help="configs/devices/<DEVICE>.json")
    s.add_argument("--workloads", default="", help="comma-separated ids (default: all registered)")
    s.add_argument("--trials", type=int, default=3)
    s.add_argument("--campaign", help="resume an existing campaign id")
    s.add_argument("--order-seed", type=int, default=0)
    s.add_argument("--out", help="output root (default results/data/v2)")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=lambda a: run_sweep(
        _load_device_config(a.device_config), [k for k in a.workloads.split(",") if k], a.trials,
        Path(a.out) if a.out else schema.default_root(), a.campaign or _campaign_id(),
        dry_run=a.dry_run, order_seed=a.order_seed))

    m = sub.add_parser("smoke", help="every workload, smallest configs, short timing; output to results/data/smoke")
    m.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    m.add_argument("--gpu", type=int, default=0)
    m.add_argument("--placement", default="node", help="CPU placement for the smoke run")
    m.add_argument("--workloads", default="")
    m.add_argument("--out", default=str(REPO / "results" / "data" / "smoke"))
    m.set_defaults(fn=lambda a: run_sweep(
        _with_mem({"device_id": f"smoke_{socket.gethostname().split('.')[0]}",
                   "device_class": "cpu" if a.device == "cpu" else "gpu",
                   "placements": [a.placement], "gpu_index": a.gpu}),
        [k for k in a.workloads.split(",") if k], 1, Path(a.out), _campaign_id(), smoke=True))
