"""CLI:  python -m bench {run,sweep,smoke,list,describe}    (run from the repo root: PYTHONPATH=.)

`run` executes ONE configuration in this process. CPU thread placement is applied before
torch is imported (OpenMP reads its environment once). Use `sweep` for real collections:
it launches one `run` subprocess per (configuration, trial) under numactl + OMP pinning.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_run(a) -> int:
    import os
    early = json.loads(a.params) if a.params else {}
    for k, v in early.get("_env", {}).items():     # before torch / libcrypto initialise
        if os.environ.get(k) != v:
            os.environ[k] = v
    from . import devices, env, schema
    flags: list[str] = [f"env:{k}" for k in early.get("_env", {})]
    placement = None
    if a.device == "cpu":
        placement = devices.resolve_placement(a.placement)
        flags += devices.apply_cpu_placement(placement)
        if devices.jemalloc_active():
            flags.append("alloc=jemalloc")
    # ---- torch may be imported only from here on ----
    import torch
    from . import core
    from .workloads import get

    if a.device == "cpu":
        devices.configure_torch_cpu(placement.threads)
        dev = torch.device("cpu")
        llc = devices.llc_bytes_for(placement)
        device_class = "cpu"
    else:
        if not torch.cuda.is_available():
            print("CUDA not available", file=sys.stderr)
            return 2
        devices.configure_torch_cuda(a.gpu)
        dev = torch.device(f"cuda:{a.gpu}")
        llc = devices.gpu_llc_bytes(a.gpu)
        device_class = "gpu"

    device_id = a.device_id or devices.default_device_id(device_class, a.gpu)
    w = get(a.workload)
    params = json.loads(a.params) if a.params else w.configs(device_class)[0]
    # run ids are prefixed by the campaign so a sweep can be resumed (sweep._done)
    run_id = f"{a.campaign}-{env.new_run_id()}" if a.campaign else env.new_run_id()
    out_root = Path(a.out) if a.out else schema.default_root()
    paths = schema.output_paths(out_root, device_class, device_id, w.id, w.name)

    rec = env.collect(run_id, device_class, sys.argv, extra={
        "campaign": a.campaign, "device_id": device_id, "placement": a.placement if placement else "",
        "cpus": placement.cpus if placement else [], "membind": a.device == "cpu" and "no_membind" not in flags})
    if a.campaign and (paths["env"] / f"{a.campaign}.json").exists():
        rec = env.light(rec)
    elif a.campaign:
        env.write(paths["env"] / f"{a.campaign}.json", rec)
        rec = env.light(rec)
    env.write(paths["env"] / f"{run_id}.json", rec)
    if rec["git_dirty"]:
        flags.append("dirty_tree")

    ctx = core.Ctx(device_class=device_class, device=dev, device_id=device_id,
                   threads=placement.threads if placement else 0,
                   placement=a.placement if placement else f"gpu{a.gpu}", gpu_index=a.gpu,
                   seed=a.seed, native_bf16=devices.native_bf16(device_class, a.gpu), llc_bytes=llc,
                   smoke=a.smoke, cpus=placement.cpus if placement else [],
                   sockets=set(placement.sockets) if placement else None)
    policy = core.SMOKE if a.smoke else core.PUBLICATION
    row = core.run_one(w, params, ctx, trial=a.trial, run_id=run_id, git_sha=rec["git_sha"],
                       flags=flags, out_root=out_root, policy=policy)
    summary = {k: row.get(k) for k in ("workload_id", "precision", "params_json", "status",
                                         "median_ms", "cv", "metric_value", "metric_name",
                                         "max_err", "tol", "flag")}
    print(json.dumps(summary, default=str))
    return 0 if row["status"] in ("ok", "unsupported", "oom") else 1


def _cmd_list(a) -> int:
    from .workloads import registry
    for wid, w in registry().items():
        print(f"{wid:3} {w.name:10} bound={w.bound}  {len(w.configs())} configs  ({w.field_})")
    return 0


def _describe_cpu() -> dict:
    from . import devices
    topo = devices.topology()
    socks = devices.sockets()
    s0 = [t for t in topo if t.socket == socks[0]]
    cores = len({t.core for t in s0})
    threads = len(s0)
    nodes = sorted({t.node for t in s0})
    print(f"CPU      : {devices.cpu_model()}")
    print(f"sockets  : {len(socks)}  (cores/socket {cores}, hardware threads/socket {threads}, "
          f"NUMA nodes/socket {len(nodes)})")
    print(f"LLC      : {devices.llc_bytes_per_socket() / 2**20:.0f} MiB per socket   native bf16: {devices.native_bf16('cpu')}")
    scaling = sorted({t for t in (1, 2, 4, 8, 16, 32, 64, 128) if t < cores})
    return {"device_class": "cpu",
            "label": f"{devices.cpu_model()} ({cores}C/{threads}T per socket, {len(socks)} socket(s)) -- <CloudLab node type>",
            "placements": ["socket0"] + (["socket0-smt"] if threads > cores else []),
            "scaling": {"workloads": ["B1", "C1", "S1", "S2"], "base": "socket0", "threads": scaling},
            "_note": "Fill 'datasheet' (bandwidth_gbps, peak_gflops fp32/fp64, derivation) from the vendor spec; "
                     "see configs/devices/AMD_R6615.json for the format."}


def _describe_gpu() -> dict | None:
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    rows = [r.split(", ") for r in out.splitlines() if r]
    for idx, name, mem, cc, drv in rows:
        print(f"GPU {idx}    : {name}  {mem}  compute capability {cc}  driver {drv}"
              f"{'' if float(cc) >= 8.0 else '  (no native bf16: bf16 rows will be unsupported)'}")
    return {"device_class": "gpu", "label": f"{rows[0][1]} -- <CloudLab node type>", "gpu_index": 0,
            "_note": "Fill 'datasheet' (bandwidth_gbps, peak_gflops fp32/fp64, derivation) from the vendor spec; "
                     "see configs/devices/NVIDIA_V100.json for the format."}


def _cmd_describe(a) -> int:
    """What the harness sees on this node, and a suggested configs/devices/<ID>.json."""
    import re
    from . import devices
    gpu = None if a.cpu else _describe_gpu()
    cfg = gpu if gpu else _describe_cpu()
    dev_id = a.id or re.sub(r"[^A-Za-z0-9]+", "_", cfg["label"].split(" -- ")[0].split(" (")[0]).strip("_").upper()
    cfg = {"device_id": dev_id, **cfg}
    known = sorted(p.stem for p in (Path(__file__).resolve().parents[1] / "configs" / "devices").glob("*.json"))
    print(f"\nexisting device configs: {', '.join(known)}")
    print(f"suggested configs/devices/{dev_id}.json (pick a short id such as INTEL_C6620 with --id):")
    print(json.dumps(cfg, indent=2))
    if a.write:
        dst = Path(__file__).resolve().parents[1] / "configs" / "devices" / f"{dev_id}.json"
        if dst.exists():
            print(f"not written: {dst} already exists")
            return 1
        dst.write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"written: {dst}  (now add its datasheet numbers)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one configuration (one trial) in this process")
    r.add_argument("--workload", required=True, help="workload id (B1) or name (saxpy)")
    r.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    r.add_argument("--params", help="JSON config dict (default: first config of the workload)")
    r.add_argument("--placement", default="socket0", help="CPU: socketS[-smt][:T] | node[-smt][:T]")
    r.add_argument("--gpu", type=int, default=0)
    r.add_argument("--trial", type=int, default=1)
    r.add_argument("--seed", type=int, default=1234)
    r.add_argument("--device-id", help="device directory name (e.g. INTEL_C6620, NVIDIA_A100)")
    r.add_argument("--campaign", default="", help="campaign id shared by all runs of one sweep")
    r.add_argument("--out", help="output root (default results/data or $BENCH_OUT)")
    r.add_argument("--smoke", action="store_true", help="short timing policy; rows flagged 'smoke'")
    r.set_defaults(fn=_cmd_run)

    from . import sweep
    sweep.add_parsers(sub)

    l = sub.add_parser("list", help="list registered workloads")
    l.set_defaults(fn=_cmd_list)

    d = sub.add_parser("describe", help="show the detected CPU/GPU and a suggested device config")
    d.add_argument("--id", help="device id to use, e.g. INTEL_C6620 (default: derived from the model name)")
    d.add_argument("--cpu", action="store_true", help="describe the CPU even on a GPU node")
    d.add_argument("--write", action="store_true", help="save the suggestion to configs/devices/<id>.json")
    d.set_defaults(fn=_cmd_describe)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
