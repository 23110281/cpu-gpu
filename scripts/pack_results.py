#!/usr/bin/env python3
"""Pack the per-job files of each device folder into two files, so git sees 2 files instead of thousands.

On a node the harness writes one file per job (safe while jobs run):
    <dir>/samples/<run>_<hash>_t<trial>.csv.gz   raw per-call timings (read by analysis/analyze.py)
    <dir>/env/<run>.json                          machine + software record of that job (provenance)
This merges them, losslessly, into
    <dir>/samples.tar       the .csv.gz files unchanged (tar, no recompression)
    <dir>/machine.json      one record per device: software versions and machine state (constant across
                            jobs), CPU/thread settings per placement, campaigns, git commits, dirty-job count
and deletes the loose files (samples only after every one has been read back from the tar).
Per-job fields not kept (run_id, time, git_sha, placement) are already in every CSV row.
Re-running it merges newly copied files into existing packs (resumed campaigns, a second copy-back).

Usage (laptop, after copying results back):  python3 scripts/pack_results.py   [dirs...]
Default dirs: results/data/{cpu,gpu}/*  and  results/calibration/_runs/{cpu,gpu}/*
"""
import gzip
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ["results/data/cpu/*", "results/data/gpu/*", "results/calibration/_runs/cpu/*", "results/calibration/_runs/gpu/*"]


def _tarinfo(name: str, size: int) -> tarfile.TarInfo:
    ti = tarfile.TarInfo(name)
    ti.size, ti.mtime, ti.mode = size, 0, 0o644          # deterministic: same files -> same bytes
    return ti


def pack_samples(d: Path) -> int:
    loose = sorted((d / "samples").glob("*.csv.gz")) if (d / "samples").is_dir() else []
    if not loose:
        return 0
    tar = d / "samples.tar"
    have = {}
    if tar.exists():
        with tarfile.open(tar) as t:
            have = {m.name: t.extractfile(m).read() for m in t.getmembers()}
    for f in loose:
        have[f.name] = f.read_bytes()
    tmp = tar.with_suffix(".tar.tmp")
    with tarfile.open(tmp, "w", format=tarfile.USTAR_FORMAT) as t:
        for name in sorted(have):
            t.addfile(_tarinfo(name, len(have[name])), io.BytesIO(have[name]))
    with tarfile.open(tmp) as t:                                   # verify before deleting anything
        back = {m.name: t.extractfile(m).read() for m in t.getmembers()}
    assert all(back.get(f.name) == f.read_bytes() for f in loose), f"{tmp}: verification failed"
    tmp.replace(tar)
    for f in loose:
        f.unlink()
    (d / "samples").rmdir()
    return len(loose)


# per-job fields that are already in the CSV rows (run_id, git_sha, placement) or rebuildable from them
PER_JOB = {"_file", "argv", "run_id", "utc", "git_sha", "campaign", "placement", "cpus", "env"}


def _norm(k: str, v):
    """Drop live values that differ between jobs without meaning anything (free memory)."""
    if k == "numactl_hardware" and isinstance(v, str):
        return "\n".join(l for l in v.splitlines() if " free:" not in l)
    if k == "torch_backend" and isinstance(v, dict):
        return {a: b for a, b in v.items() if a != "parallel_info"}       # thread counts: per placement
    return v


def _summarise(recs: list[dict]) -> dict:
    """One machine record: fields that are constant across jobs, per-placement CPU/thread settings,
    and (should never happen) any other field that changed during the campaign, with every value seen.
    Fields recorded only on some jobs (full lscpu / nvidia-smi dumps) count as constant if they agree."""
    out, changed = {}, {}
    for k in sorted({k for r in recs for k in r} - PER_JOB):
        vals = {json.dumps(_norm(k, r.get(k)), sort_keys=True) for r in recs if r.get(k) is not None}
        if k == "torch_backend":            # a few jobs record the full build config, the rest a short form
            full = [v for v in vals if '"config"' in v]
            short = {json.dumps({a: b for a, b in json.loads(v).items() if a != "config"}, sort_keys=True) for v in vals}
            vals = set(full) if len(short) == 1 and len(set(full)) == 1 else vals
        if len(vals) == 1:
            out[k] = json.loads(next(iter(vals)))
        elif not vals:
            out[k] = None
        else:
            changed[k] = [json.loads(v) for v in sorted(vals)]
    out["placements"] = {r.get("placement") or "default": {"cpus": r.get("cpus"), "env": r.get("env")}
                         for r in sorted(recs, key=lambda r: r["_file"])}
    for r in recs:                         # torch's thread report, per placement
        pi = (r.get("torch_backend") or {}).get("parallel_info")
        if pi and (r.get("placement") or "default") in out["placements"]:
            out["placements"][r.get("placement") or "default"]["torch_parallel_info"] = pi
    out["campaigns"] = sorted({r.get("campaign") for r in recs if r.get("campaign")})
    out["git_shas"] = sorted({r.get("git_sha") for r in recs if r.get("git_sha")})
    out["git_dirty_jobs"] = sum(bool(r.get("git_dirty")) for r in recs)
    out["jobs"] = len(recs)
    if changed:
        out["changed_during_campaign"] = changed
    return out


def pack_env(d: Path) -> int:
    """env/*.json (one per job, ~99% identical) -> machine.json (one per device, a few KB)."""
    loose = sorted((d / "env").glob("*.json")) if (d / "env").is_dir() else []
    if not loose:
        return 0
    out = d / "machine.json"
    recs = [{"_file": f.name, **json.loads(f.read_text())} for f in loose]
    prev = json.loads(out.read_text()) if out.exists() else None
    summary = _summarise(recs)
    if prev:                                   # second copy-back: merge with what was packed before
        summary = _merge(prev, summary)
    out.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    if summary.get("changed_during_campaign"):
        print(f"WARNING {d}: fields changed during the campaign: {sorted(summary['changed_during_campaign'])}")
    for f in loose:
        f.unlink()
    (d / "env").rmdir()
    return len(loose)


def _merge(a: dict, b: dict) -> dict:
    """Combine two machine records (first and later copy-backs of the same device)."""
    out = {"placements": {**a.get("placements", {}), **b.get("placements", {})},
           "campaigns": sorted(set(a.get("campaigns", [])) | set(b.get("campaigns", []))),
           "git_shas": sorted(set(a.get("git_shas", [])) | set(b.get("git_shas", []))),
           "git_dirty_jobs": a.get("git_dirty_jobs", 0) + b.get("git_dirty_jobs", 0),
           "jobs": a.get("jobs", 0) + b.get("jobs", 0)}
    def seen(m):                                       # field -> every value recorded for it
        v = {k: [x] for k, x in m.items() if k not in out and k != "changed_during_campaign"}
        v.update(m.get("changed_during_campaign", {}))
        return v
    va, vb = seen(a), seen(b)
    changed = {}
    for k in sorted(set(va) | set(vb)):
        vals = sorted({json.dumps(x, sort_keys=True) for x in va.get(k, []) + vb.get(k, [])})
        if len(vals) == 1:
            out[k] = json.loads(vals[0])
        else:
            changed[k] = [json.loads(x) for x in vals]
    if changed:
        out["changed_during_campaign"] = changed
    return out


def main(argv):
    dirs = [Path(a) for a in argv] or sorted(p for g in DEFAULT for p in ROOT.glob(g) if p.is_dir())
    for d in dirs:
        s, e = pack_samples(d), pack_env(d)
        if s or e:
            print(f"{d.relative_to(ROOT) if d.is_absolute() else d}: packed {s} sample files, {e} env files")


if __name__ == "__main__":
    main(sys.argv[1:])
