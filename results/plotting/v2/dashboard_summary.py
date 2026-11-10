#!/usr/bin/env python3
"""Per-device dashboard summary: headline statements with numbers, a per-workload table, the
overview figure, and curated findings -- all computed from the v2 analysis outputs, so the text
cannot drift from the data.

Inputs : results/analysis/<DEV>_{headline,libraries,bounds}.csv, library_summary.csv  (analyze.py,
         library_report.py), results/calibration/<DEV>.json, configs/devices/<DEV>.json,
         results/data/v2/<cpu|gpu>/<DEV>/*.csv (row counts), results/dashboard_notes.json (curated)
Outputs: results/plots/v2/<DEV>/summary.json, results/plots/v2/<DEV>/overview.png
Run after analyze.py and library_report.py (make plots / make analysis do).
"""
from __future__ import annotations

import fnmatch
import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import figures  # noqa: E402

DATA = Path(os.environ.get("ANALYSIS_DATA", ROOT / "results" / "data" / "v2"))
CAL = Path(os.environ.get("ANALYSIS_CAL", ROOT / "results" / "calibration"))
OUT = Path(os.environ.get("ANALYSIS_OUT", ROOT / "results" / "analysis"))
PLOTS = Path(os.environ.get("ANALYSIS_PLOTS", ROOT / "results" / "plots" / "v2"))
NOTES = Path(os.environ.get("DASHBOARD_NOTES", ROOT / "results" / "dashboard_notes.json"))
CONFIGS = ROOT / "configs" / "devices"

# overview: one bar per (workload, facet), at the largest working set of that facet
FACETS = {"B1": "precision", "B2": "precision", "B3": "matrix", "B4": "matrix", "C1": "precision",
          "C3": "k", "A1": "phase_batch"}


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _read(name: str) -> pd.DataFrame:
    try:
        return pd.read_csv(OUT / name)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def fmt_metric(v: float, unit: str) -> str:
    if unit == "GFLOP/s" and v >= 1000:
        return f"{v / 1000:.1f} TFLOP/s"
    if v >= 10000:
        return f"{v / 1000:,.1f}K {unit}"
    return f"{v:,.0f} {unit}" if v >= 100 else f"{v:.3g} {unit}"


def problem_text(wid: str, prob: str) -> str:
    p = json.loads(prob)
    if wid == "A1":
        return (f"{p['model']} prefill S={p['seq']}" if p["phase"] == "prefill"
                else f"{p['model']} decode batch {p['batch']}") + f", {p['precision']}"
    keys = [k for k in sorted(p) if k not in figures.HIDDEN_KEYS]
    return ", ".join([f"{k}={figures.fmt_num(p[k])}" for k in keys] + [p.get("precision", "")]).strip(", ")


def row_counts(dev: str) -> dict:
    files = glob.glob(str(DATA / "*" / dev / "*.csv"))
    if not files:
        return {}
    d = pd.concat([pd.read_csv(f, usecols=lambda c: c in ("status", "flag", "git_sha", "trial", "run_id"))
                   for f in files], ignore_index=True)
    flag = d["flag"].fillna("")
    ok = d[(d["status"] == "ok") & ~flag.str.contains("smoke")]
    return {"ok": int(len(ok)), "unsupported": int((d["status"] == "unsupported").sum()),
            "oom": int((d["status"] == "oom").sum()), "error": int((d["status"] == "error").sum()),
            "validation_failed": int((d["status"] == "validation_failed").sum()),
            "unstable": int(flag[d["status"] == "ok"].str.contains("unstable").sum()),
            "trials": int(ok["trial"].nunique()) if len(ok) else 0,
            "git_shas": sorted({str(s)[:7] for s in ok["git_sha"].dropna()}),
            "unsupported_reasons": sorted({f for f in flag[d["status"] == "unsupported"] if f})[:4]}


def ceiling_pct(r, bounds: pd.DataFrame, meas: dict):
    """(% of measured ceiling, kind) for one headline row, or (nan, '')."""
    if r["workload_id"] == "A1":
        phase = json.loads(r["problem"]).get("phase")
        if phase == "decode" and meas.get("bandwidth_gbps") and r["gbps"] > 0:
            return 100 * r["gbps"] / meas["bandwidth_gbps"], "bandwidth"
        peak = (meas.get("peak_gflops") or {}).get(r["precision"])
        if phase == "prefill" and peak and r["gflops"] > 0:
            return 100 * r["gflops"] / peak, "GEMM peak"
        return float("nan"), ""
    b = bounds[(bounds["workload_id"] == r["workload_id"]) & (bounds["problem"] == r["problem"])] if len(bounds) else bounds
    if len(b):
        bw = b["pct_of_measured_bw"].dropna() if "pct_of_measured_bw" in b else []
        pk = b["pct_of_measured_peak"].dropna() if "pct_of_measured_peak" in b else []
        if len(bw):
            return float(bw.iloc[0]), "bandwidth"
        if len(pk):
            return float(pk.iloc[0]), "GEMM peak"
    return float("nan"), ""


def facet_of(wid: str, prob: str) -> str | None:
    key = FACETS.get(wid)
    if key is None:
        return None
    p = json.loads(prob)
    if key == "phase_batch":
        return f"prefill" if p["phase"] == "prefill" else f"decode B={p['batch']}"
    return f"{key}={figures.fmt_num(p[key])}" if key not in ("precision", "matrix") else str(p[key])


def build(dev: str, names: dict, order: list[str]) -> dict:
    head = _read(f"{dev}_headline.csv")
    if head.empty:
        return {}
    libs = _read(f"{dev}_libraries.csv")
    bounds = _read(f"{dev}_bounds.csv")
    lsum = _read("library_summary.csv")
    lsum = lsum[lsum["device"] == dev] if len(lsum) else lsum
    cal = _json(CAL / f"{dev}.json")
    meas = cal.get("measured", {})
    cfg = _json(CONFIGS / f"{dev}.json")
    ds = cfg.get("datasheet", {}) or cal.get("datasheet", {})
    counts = row_counts(dev)
    head = head.assign(pct=float("nan"), kind="")
    for i, r in head.iterrows():
        head.loc[i, ["pct", "kind"]] = ceiling_pct(r, bounds, meas)

    # ── ceilings tiles ────────────────────────────────────────────────────────
    tiles = []
    bw = meas.get("bandwidth_gbps")
    if bw:
        t = {"k": "Memory bandwidth (measured)", "v": f"{bw:,.0f} GB/s",
             "s": meas.get("bandwidth_convention", "")}
        if ds.get("bandwidth_gbps"):
            t["s"] = f"{100 * bw / ds['bandwidth_gbps']:.0f}% of the {ds['bandwidth_gbps']:,.0f} GB/s datasheet"
        tiles.append(t)
    for prec in ("fp32", "fp64"):
        v = (meas.get("peak_gflops") or {}).get(prec)
        if v:
            dv = (ds.get("peak_gflops") or {}).get(prec)
            tiles.append({"k": f"GEMM peak {prec} (measured)", "v": fmt_metric(v, "GFLOP/s"),
                          "s": f"{100 * v / dv:.0f}% of the {dv / 1000:.1f} TFLOP/s datasheet" if dv else ""})
    if counts:
        tiles.append({"k": "Validated results", "v": f"{counts['ok']}",
                      "s": f"{head['problem'].nunique()} problems × {counts['trials']} trials · "
                           f"{counts['error'] + counts['validation_failed']} errors · {counts['unstable']} unstable"})

    # ── overview entries + statements ─────────────────────────────────────────
    entries = []
    for wid in order:
        g = head[(head["workload_id"] == wid) & head["pct"].notna()]
        if g.empty:
            continue
        g = g.assign(facet=g["problem"].map(lambda p: facet_of(wid, p)))
        for facet, gg in g.groupby("facet", sort=False):
            r = gg.sort_values(["ws_over_llc", "metric"]).iloc[-1]
            entries.append({"wid": wid, "label": f"{wid} {names.get(wid, wid)} · {facet}", "pct": float(r["pct"]),
                            "kind": r["kind"], "problem": problem_text(wid, r["problem"])})
    entries.sort(key=lambda e: (order.index(e["wid"]), e["label"]))
    statements = []
    if tiles and bw:
        s = f"Measured ceilings: memory bandwidth {bw:,.0f} GB/s"
        if ds.get("bandwidth_gbps"):
            s += f" ({100 * bw / ds['bandwidth_gbps']:.0f}% of the datasheet {ds['bandwidth_gbps']:,.0f} GB/s)"
        pk = meas.get("peak_gflops") or {}
        if pk:
            s += "; GEMM " + ", ".join(f"{fmt_metric(v, 'GFLOP/s')} {p}" for p, v in sorted(pk.items(), reverse=True))
        statements.append(s + ". Every percentage below is relative to these measured ceilings.")

    def span(kind, wids):
        parts = []
        for wid in wids:
            v = [e["pct"] for e in entries if e["wid"] == wid and e["kind"] == kind]
            if v:
                lo, hi = min(v), max(v)
                parts.append(f"{wid} {names.get(wid, wid)} {hi:.0f}%" if round(lo) == round(hi)
                             else f"{wid} {names.get(wid, wid)} {lo:.0f}–{hi:.0f}%")
        return parts
    b_parts = span("bandwidth", [w for w in order if w.startswith("B")])
    if b_parts:
        statements.append("Bandwidth-bound workloads at their largest size reach " + "; ".join(b_parts)
                          + " of measured bandwidth.")
    c_parts = span("GEMM peak", [w for w in order if w.startswith("C")])
    if c_parts:
        statements.append("Compute-bound workloads with a FLOP model reach " + "; ".join(c_parts)
                          + " of the measured GEMM peak.")
    a = [e for e in entries if e["wid"] == "A1"]
    if a:
        statements.append(f"{names.get('A1', 'A1')} ({a[0]['problem'].split(' ')[0]}): " + "; ".join(
            f"{e['label'].split(' · ')[-1]} at {e['pct']:.0f}% of measured {e['kind']}" for e in a) + ".")
    lib_parts = []
    for (wid, prec), g in (lsum.groupby(["workload_id", "precision"]) if len(lsum) else []):
        g = g.sort_values(["wins", "median_speed_vs_winner"], ascending=False)
        w, rest = g.iloc[0], g.iloc[1:]
        if w["wins"] == 0 or rest.empty:
            continue
        others = ", ".join(f"{figures.LIB_NAMES.get(r['library'], r['library'])} at {r['median_speed_vs_winner']:.2f}×"
                           for _, r in rest.iterrows())
        lib_parts.append(f"{wid}: {figures.LIB_NAMES.get(w['library'], w['library'])} fastest on "
                         f"{w['wins']}/{w['problems']} sizes ({others} median)")
    if lib_parts:
        statements.append("Fastest implementation where several were timed — " + "; ".join(lib_parts) + ".")
    c1 = head[head["workload_id"] == "C1"]
    if len(c1) and c1["freq_mhz"].notna().any():
        big = c1.sort_values("ws_over_llc").iloc[-1]
        statements.append(f"Clocks were not locked: the median clock during the largest GEMM was "
                          f"{big['freq_mhz']:,.0f} MHz (recorded in every row as freq_mhz).")
    if counts:
        skip = []
        if counts["unsupported"]:
            skip.append(f"{counts['unsupported']} unsupported"
                        + (f" ({'; '.join(counts['unsupported_reasons'])})" if counts["unsupported_reasons"] else ""))
        if counts["oom"]:
            skip.append(f"{counts['oom']} skipped because the working set exceeds device memory")
        statements.append(f"Data: {counts['ok']} validated rows from {counts['trials']} process-isolated trials "
                          f"(commits {', '.join(counts['git_shas'])}); "
                          + "; ".join(skip + [f"{counts['error'] + counts['validation_failed']} errors",
                                              f"{counts['unstable']} rows flagged unstable (CV > 5%)"]) + ".")

    # ── per-workload table ────────────────────────────────────────────────────
    table = []
    for wid in order:
        g = head[head["workload_id"] == wid]
        if g.empty:
            continue
        r = g.sort_values(["ws_over_llc", "metric"]).iloc[-1]
        n_libs = libs[libs["workload_id"] == wid]["lib"].nunique() if len(libs) else 1
        table.append({"id": wid, "name": names.get(wid, wid), "bound": r["bound"],
                      "result": fmt_metric(float(r["metric"]), r["metric_name"]),
                      "problem": problem_text(wid, r["problem"]),
                      "pct": None if r["pct"] != r["pct"] else round(float(r["pct"]), 1), "kind": r["kind"],
                      "impl": figures.LIB_NAMES.get(r["lib"], r["lib"]),
                      "n_impl": int(n_libs), "problems": int(g["problem"].nunique())})

    # ── curated findings ──────────────────────────────────────────────────────
    findings = [n for n in _json(NOTES).get("notes", []) if fnmatch.fnmatch(dev, n.get("device", "*"))]

    d = PLOTS / dev
    d.mkdir(parents=True, exist_ok=True)
    figs = []
    if entries:
        figures.overview_figure(entries, d / "overview.png",
                                f"% of measured ceiling — {dev.replace('_', ' ')}")
        figs.append({"file": "overview.png", "title": "Overview: % of measured ceiling", "wide": True,
                     "caption": "One bar per workload variant at its largest problem size."
                                + (" Not shown (no simple hardware ceiling): "
                                   + ", ".join(x["id"] for x in table if x["pct"] is None) + "."
                                   if any(x["pct"] is None for x in table) else "")})
    for t in table:
        if (d / f"{t['id']}.png").exists():
            cap = f"{t['problems']} problem sizes; largest: {t['result']} ({t['problem']})"
            if t["pct"] is not None:
                cap += f", {t['pct']:.0f}% of measured {t['kind']}"
            figs.append({"file": f"{t['id']}.png", "title": f"{t['id']} {t['name']}", "caption": cap + ".",
                         "wide": t["problems"] > 10})
    return {"device": dev, "label": cfg.get("label", dev), "device_class": cfg.get("device_class", ""),
            "tiles": tiles, "statements": statements, "table": table, "findings": findings, "figures": figs,
            "counts": counts}


def main():
    manifest = _json(ROOT / "results" / "workloads.json")
    names = {w["id"]: w["name"] for w in manifest.get("workloads", [])}
    order = [w["id"] for w in manifest.get("workloads", [])]
    devs = sorted(Path(f).name[: -len("_headline.csv")] for f in glob.glob(str(OUT / "*_headline.csv")))
    for dev in devs:
        s = build(dev, names, order)
        if s:
            (PLOTS / dev / "summary.json").write_text(json.dumps(s, indent=1, ensure_ascii=False) + "\n")
            print(f"{dev}: {len(s['statements'])} statements, {len(s['table'])} workloads, "
                  f"{len(s['findings'])} curated findings -> {PLOTS / dev}/summary.json")


if __name__ == "__main__":
    main()
