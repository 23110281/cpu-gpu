#!/usr/bin/env python3
"""v2 analysis (plan Part E): headline numbers, library comparison, bound verification,
device ratios with bootstrap CIs, and per-workload plots -- from raw v2 data only.

Inputs : results/data/v2/<cpu|gpu>/<DEVICE>/<ID>_<name>.csv (+ samples/*.csv.gz)
         results/calibration/<DEVICE>.json
Outputs: results/analysis/<DEVICE>_headline.csv   best (library, placement) per problem
         results/analysis/<DEVICE>_libraries.csv  every library per problem (fairness evidence)
         results/analysis/<DEVICE>_bounds.csv     bound-label verification
         results/analysis/ratios.csv              GPU/CPU (or any pair) ratios, 95% CI, geomean
         results/plots/v2/<DEVICE>/<ID>.png
Rules (docs/methodology.md): only status=ok rows, never smoke; a problem = workload + params
without lib/_env; headline = fastest library & placement (median over trials); CIs come from
bootstrapping the pooled per-call samples of all trials of that (library, placement).
"""
from __future__ import annotations

import glob
import gzip
import warnings
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("ANALYSIS_DATA", ROOT / "results" / "data" / "v2"))
CAL = Path(os.environ.get("ANALYSIS_CAL", ROOT / "results" / "calibration"))
OUT = Path(os.environ.get("ANALYSIS_OUT", ROOT / "results" / "analysis"))
PLOTS = Path(os.environ.get("ANALYSIS_PLOTS", ROOT / "results" / "plots" / "v2"))
sys.path.insert(0, str(ROOT / "src" / "torch"))
from bench import stats  # noqa: E402

DROP_KEYS = ("lib", "_env")
warnings.filterwarnings("ignore", category=RuntimeWarning)    # medians of all-NaN power columns
warnings.filterwarnings("ignore", category=FutureWarning)


def problem_key(params_json: str) -> str:
    p = json.loads(params_json)
    return json.dumps({k: v for k, v in sorted(p.items()) if k not in DROP_KEYS}, sort_keys=True)


def lib_label(params_json: str, library) -> str:
    """The config's `lib` key; workloads with a single implementation have none, so the label
    comes from the row's `library` column ("torch2.x/...", "native C++/...", "OpenSSL ...")."""
    lib = json.loads(params_json).get("lib")
    if lib:
        return lib
    m = re.match(r"[A-Za-z]+", str(library or ""))
    return m.group(0).lower() if m else "unknown"


def load() -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(DATA / "*" / "*" / "*.csv"))):
        d = pd.read_csv(f)
        d["_file_dir"] = str(Path(f).parent)
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["status"] == "ok") & ~df["flag"].fillna("").str.contains("smoke")].copy()
    df["lib"] = [lib_label(p, l) for p, l in zip(df["params_json"], df["library"])]
    df["problem"] = df["params_json"].map(problem_key)
    df["scaling"] = df["placement"].fillna("").str.contains(":")
    return df


def samples_for(rows: pd.DataFrame) -> np.ndarray:
    out = []
    for _, r in rows.iterrows():
        f = Path(r["_file_dir"]) / "samples" / f"{r['run_id']}_{r['config_hash']}_t{int(r['trial'])}.csv.gz"
        if f.exists():
            with gzip.open(f, "rt") as fh:
                out += [float(l) for l in fh if l and l[0] not in "#m"]
    return np.asarray(out, dtype=float)


def summarise_combo(rows: pd.DataFrame) -> dict:
    """One (problem, lib, placement) across trials."""
    units = float((rows["metric_value"] * rows["median_ms"] / 1e3).median())   # units per call
    s = samples_for(rows)
    med_ms = float(np.median(s)) if s.size else float(rows["median_ms"].median())
    lo, hi = stats.bootstrap_ci_median(s) if s.size > 1 else (med_ms, med_ms)
    return {"metric": units / (med_ms / 1e3), "metric_ci_lo": units / (hi / 1e3), "metric_ci_hi": units / (lo / 1e3),
            "median_ms": med_ms, "trials": int(rows["trial"].nunique()), "n_samples": int(s.size),
            "cv_max": float(rows["cv"].max()), "unstable": bool(rows["flag"].fillna("").str.contains("unstable").any()),
            "gflops": float(rows["gflops"].median()), "gbps": float(rows["gbps"].median()),
            "power_w": float(rows["power_w"].median()), "energy_j_per_op": float(rows["energy_j_per_op"].median()),
            "freq_mhz": float(rows["freq_mhz"].median()), "library_detail": rows["library"].iloc[0],
            "ws_over_llc": float(rows["ws_over_llc"].median()), "regime": rows["regime"].iloc[0],
            "metric_name": rows["metric_name"].iloc[0], "units_per_call": units, "_samples": s}


def per_device(df: pd.DataFrame, dev: str):
    d = df[(df["device"] == dev) & ~df["scaling"]]
    combos, heads = [], []
    for (wid, prob), g in d.groupby(["workload_id", "problem"]):
        best = None
        for (lib, pl), gg in g.groupby(["lib", "placement"]):
            s = summarise_combo(gg)
            rec = {"workload_id": wid, "workload": gg["workload"].iloc[0], "bound": gg["bound"].iloc[0],
                   "problem": prob, "precision": gg["precision"].iloc[0], "variant": gg["variant"].iloc[0],
                   "lib": lib, "placement": pl, **s}
            combos.append(rec)
            if best is None or rec["metric"] > best["metric"]:
                best = rec
        heads.append(best)
    strip = lambda recs: pd.DataFrame([{k: v for k, v in r.items() if k != "_samples"} for r in recs])  # noqa: E731
    return strip(heads), strip(combos), heads


def calibration(dev: str) -> dict:
    try:
        return json.loads((CAL / f"{dev}.json").read_text())
    except (OSError, ValueError):
        return {}


def bounds_table(head: pd.DataFrame, df: pd.DataFrame, dev: str) -> pd.DataFrame:
    cal = calibration(dev).get("measured", {})
    bw = cal.get("bandwidth_gbps")
    peak = dict(cal.get("peak_gflops", {}) or {})
    gem = head[head["workload_id"] == "C1"]
    for prec, g in gem.groupby("precision"):          # sweep's own best GEMM is the compute ceiling
        peak[prec] = max(peak.get(prec, 0.0), float(g["gflops"].max()))
    rows = []
    for _, r in head.iterrows():
        rec = {"workload_id": r["workload_id"], "bound": r["bound"], "problem": r["problem"], "lib": r["lib"],
               "placement": r["placement"], "regime": r["regime"], "ws_over_llc": r["ws_over_llc"]}
        if r["bound"] == "B" and bw and r["regime"] == "dram" and r["gbps"] == r["gbps"]:
            rec["pct_of_measured_bw"] = 100 * r["gbps"] / bw
            rec["label_ok"] = rec["pct_of_measured_bw"] >= 60
        elif r["bound"] == "C" and r["gflops"] and r["gflops"] == r["gflops"] and peak.get(r["precision"]):
            rec["pct_of_measured_peak"] = 100 * r["gflops"] / peak[r["precision"]]
        rows.append(rec)
    out = pd.DataFrame(rows)
    out = pd.concat([out, scaling_table(df, dev)])
    # sync-bound evidence: S2 batch-size sensitivity (scaling efficiency is in scaling_table)
    s2 = head[head["workload_id"] == "S2"]
    if len(s2) > 1:
        b = s2.assign(batch=s2["problem"].map(lambda p: json.loads(p)["batch"])).sort_values("batch")
        out = pd.concat([out, pd.DataFrame([{"workload_id": "S2", "bound": "S",
                                              "problem": "iterations/s, smallest vs largest batch",
                                              "sync_slowdown": float(b["metric"].iloc[-1] / b["metric"].iloc[0])}])])
    return out


def scaling_table(df: pd.DataFrame, dev: str) -> pd.DataFrame:
    """Thread-scaling efficiency per (workload, problem, lib): the scaling rows (`<base>:<T>`)
    plus the full-placement row of the same problem and library as the top point. Each series
    is one configuration -- never a median across libraries or precisions."""
    d = df[df["device"] == dev]
    sc = d[d["scaling"]]
    rows = []
    for (wid, prob, lib), g in sc.groupby(["workload_id", "problem", "lib"]):
        base = g["placement"].iloc[0].split(":")[0]
        full = d[(d["workload_id"] == wid) & (d["problem"] == prob) & (d["lib"] == lib) & (d["placement"] == base)]
        g = pd.concat([g, full]).assign(t=lambda x: x["threads"].astype(int))
        s = g.groupby("t")["metric_value"].median().sort_index()
        if len(s) < 2:
            continue
        eff = (s / s.iloc[0]) / (s.index / s.index[0])
        rows.append({"workload_id": wid, "bound": "scaling", "problem": prob, "lib": lib, "placement": base,
                     "threads": " ".join(map(str, s.index)),
                     "speedup_at_max": float(s.iloc[-1] / s.iloc[0]),
                     "scaling_efficiency_at_max": float(eff.iloc[-1])})
    return pd.DataFrame(rows)


def ratios(heads_by_dev: dict) -> pd.DataFrame:
    devs = sorted(heads_by_dev)
    rows = []
    for i, a in enumerate(devs):
        for b in devs[i + 1:]:
            A = {(r["workload_id"], r["problem"]): r for r in heads_by_dev[a]}
            B = {(r["workload_id"], r["problem"]): r for r in heads_by_dev[b]}
            for key in sorted(set(A) & set(B)):
                ra, rb = A[key], B[key]
                if ra["_samples"].size < 2 or rb["_samples"].size < 2:
                    continue
                # speed-up of b over a = time_a / time_b, normalised per unit of work
                ta = ra["_samples"] / ra["units_per_call"]
                tb = rb["_samples"] / rb["units_per_call"]
                r, lo, hi = stats.ratio_ci(ta, tb)
                rows.append({"device_a": a, "device_b": b, "workload_id": key[0], "problem": key[1],
                             "speedup_b_over_a": r, "ci95_lo": lo, "ci95_hi": hi,
                             "lib_a": ra["lib"], "lib_b": rb["lib"]})
    out = pd.DataFrame(rows)
    if len(out):
        g = out.groupby(["device_a", "device_b"])["speedup_b_over_a"].apply(lambda s: float(np.exp(np.log(s).mean())))
        for (a, b), v in g.items():
            out = pd.concat([out, pd.DataFrame([{"device_a": a, "device_b": b, "workload_id": "GEOMEAN",
                                                  "speedup_b_over_a": v}])])
    return out


def workload_names() -> dict:
    try:
        return {w["id"]: w["name"] for w in json.loads((ROOT / "results" / "workloads.json").read_text())["workloads"]}
    except (OSError, ValueError, KeyError):
        return {}


def plots(combos: pd.DataFrame, bounds: pd.DataFrame, dev: str):
    """One figure per workload (results/plots/v2/<DEV>/<ID>.png): every library at its best
    placement, median with 95% CI, plus the measured ceiling of each problem where the bound
    check has one. Style and colours: results/plotting/v2/figures.py."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import figures
    d = PLOTS / dev
    d.mkdir(parents=True, exist_ok=True)
    names = workload_names()
    b = bounds[bounds["bound"].isin(["B", "C"])] if len(bounds) else bounds
    meas = calibration(dev).get("measured", {})
    for wid, g in combos.groupby("workload_id"):
        title = f"{wid} {names.get(wid, g['workload'].iloc[0])} — {dev.replace('_', ' ')}"
        extra = a1_ceilings(g, meas) if wid == "A1" else None
        figures.workload_figure(g, b[b["workload_id"] == wid] if len(b) else None, d / f"{wid}.png",
                                title, g["metric_name"].iloc[0], extra)


def a1_ceilings(g: pd.DataFrame, meas: dict) -> dict:
    """A1 has bound 'A' (no automatic bound check): decode is limited by streaming the weights
    + KV cache (measured bandwidth), prefill by the GEMM peak of its precision. Ceiling in
    tokens/s = best metric x (ceiling / achieved)."""
    bw, peak = meas.get("bandwidth_gbps"), meas.get("peak_gflops") or {}
    out = {}
    for prob, gg in g.groupby("problem"):
        r = gg.sort_values("metric").iloc[-1]
        phase = json.loads(prob).get("phase")
        if phase == "decode" and bw and r["gbps"] > 0:
            out[prob] = r["metric"] * bw / r["gbps"]
        elif phase == "prefill" and peak.get(r["precision"]) and r["gflops"] > 0:
            out[prob] = r["metric"] * peak[r["precision"]] / r["gflops"]
    return out


def main():
    df = load()
    if df.empty:
        print("no v2 data under", DATA)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    heads_by_dev = {}
    for dev in sorted(df["device"].unique()):
        head, combos, heads = per_device(df, dev)
        heads_by_dev[dev] = heads
        head.to_csv(OUT / f"{dev}_headline.csv", index=False)
        combos.to_csv(OUT / f"{dev}_libraries.csv", index=False)
        bt = bounds_table(head, df, dev)
        bt.to_csv(OUT / f"{dev}_bounds.csv", index=False)
        plots(combos, bt, dev)
        print(f"{dev}: {len(head)} problems, {len(combos)} (problem, library, placement) combinations")
    r = ratios(heads_by_dev)
    if len(r):
        r.to_csv(OUT / "ratios.csv", index=False)
        print(f"ratios: {len(r)} rows over {r[['device_a', 'device_b']].drop_duplicates().shape[0]} device pairs")


if __name__ == "__main__":
    main()
