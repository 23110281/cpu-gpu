#!/usr/bin/env python3
"""Per-machine dashboard pages: machine limits, an overview chart, and one page per workload
(header box: theoretical peak, measured limit, achieved range, why; then the plots).

Final-script machines : results/analysis/<DEV>_libraries.csv (analyze.py), raw rows for thread scaling,
                        results/calibration/<DEV>.json (measured limits), configs/devices/<DEV>.json (datasheet)
Earlier-script machines: analysis/legacy.py (configs/legacy.json decides what is usable)
Text                  : configs/workloads.json (plain names), configs/explanations.json (why lines)
Outputs               : results/site/devices/<DEV>/summary.json + PNGs
"""
from __future__ import annotations

import fnmatch
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import figures as F  # noqa: E402
import legacy  # noqa: E402
import plots as P  # noqa: E402

CAL = Path(os.environ.get("ANALYSIS_CAL", ROOT / "results" / "calibration"))
OUT = Path(os.environ.get("ANALYSIS_OUT", ROOT / "results" / "analysis"))
SITE = Path(os.environ.get("ANALYSIS_PLOTS", ROOT / "results" / "site" / "devices"))
DATA = Path(os.environ.get("ANALYSIS_DATA", ROOT / "results" / "data"))
CONFIGS = ROOT / "configs"
ORDER = ["B1", "B2", "B3", "B4", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "S1", "S2", "A1", "GEMV"]
MEMORY_BOUND = {"B1", "B2", "B3", "B4", "GEMV"}
CACHE_FILTER = {"B1", "B2", "B3", "GEMV"}          # sizes that fit in the cache are reported separately
COMPUTE_WITH_FLOPS = {"C1", "C3", "C4"}


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def fmt(v: float, unit: str) -> str:
    if v != v:
        return "—"
    if unit == "GFLOP/s" and v >= 1000:
        return f"{v / 1000:,.1f} TFLOP/s"
    if unit == "GB/s" and v >= 1000:
        return f"{v / 1000:,.2f} TB/s"
    return f"{v:,.0f} {unit}" if v >= 100 else f"{v:.3g} {unit}"


def plain_names() -> dict:
    w = _json(CONFIGS / "workloads.json").get("workloads", [])
    names = {x["id"]: {"name": x.get("plain", x["name"]), "what": x.get("what", "")} for x in w}
    names["GEMV"] = {"name": "Matrix × vector (GEMV)",
                     "what": "Multiplies a dense matrix by a vector (an earlier benchmark, not one of the 16 final workloads)."}
    return names


def why_lines(dev: str, wid: str) -> list[str]:
    ex = _json(CONFIGS / "explanations.json")
    out = [ex.get("workloads", {}).get(wid, "")]
    for n in ex.get("device_notes", []):
        if fnmatch.fnmatch(dev, n["device"]) and fnmatch.fnmatch(wid, n["workload"]):
            out.append(n["text"])
    return [x for x in out if x]


def _short(text: str) -> str:
    return text.split(" (")[0].strip()


def theory_lines(cfg: dict) -> dict:
    ds = cfg.get("datasheet", {}) or {}
    der = ds.get("derivation", {}) or {}
    pk = ds.get("peak_gflops") or {}
    mem = f"Memory: {_short(der['bandwidth'])}" if der.get("bandwidth") else (
        f"Memory: {ds['bandwidth_gbps']:,.0f} GB/s (datasheet)" if ds.get("bandwidth_gbps") else "")
    comp = ""
    if der.get("compute"):
        comp = f"Compute: {_short(der['compute'])}"
    elif pk:
        comp = "Compute: " + " · ".join(f"{fmt(v, 'GFLOP/s')} {k}" for k, v in pk.items()) + " (datasheet)"
    if comp and not comp.rstrip().endswith(")") and "=" not in comp and pk:
        comp += " = " + " · ".join(f"{fmt(v, 'GFLOP/s')} {k}" for k, v in pk.items())
    return {"memory": mem, "compute": comp, "bw": ds.get("bandwidth_gbps"), "peak": pk}


def size_text(row: dict, wid: str) -> str:
    cfg = P.CFG.get(wid, {})
    x = cfg.get("x")
    parts = []
    if wid == "A1":
        return f"{row['model']}, " + (f"generating with batch {row['batch']}" if row.get("phase") == "decode"
                                      else f"a {row['seq']}-token prompt")
    for k in [x] + [k for k in cfg.get("series", []) if k != x]:
        if k and k in row and row[k] is not None and row[k] == row[k]:
            parts.append((P.XLABEL.get(wid, "size").split(" (")[0] + " " + P._pretty(k, row[k])) if k == x
                         else P._pretty(k, row[k]))
    return ", ".join(parts)


# ── final-script machines ──────────────────────────────────────────────────
def final_device(dev: str, cls: str, raw: pd.DataFrame, names: dict, core: bool) -> dict:
    libs = pd.read_csv(OUT / f"{dev}_libraries.csv")
    cfg = _json(CONFIGS / "devices" / f"{dev}.json")
    meas = _json(CAL / f"{dev}.json").get("measured", {})
    th = theory_lines(cfg)
    m = {"bw": meas.get("bandwidth_gbps"), "peak": meas.get("peak_gflops") or {}}
    # every percentage on the dashboard is relative to the THEORETICAL peak; the measured limit is a reference line
    lim = {"bw": th["bw"], "peak": th["peak"], "mbw": m["bw"], "mpeak": m["peak"]}
    out = SITE / dev
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    limits = [x for x in (th["memory"], th["compute"]) if x]
    mline = []
    if m["bw"]:
        pct = f" ({100 * m['bw'] / th['bw']:.0f}% of theory)" if th["bw"] else ""
        conv = _json(CAL / f"{dev}.json").get("measured", {}).get("bandwidth_convention", "")
        how = ("pure-read memory test" if "Dot" in conv else
               "memory copy test, counting the extra read before each write" if "write-allocate" in conv
               else "memory streaming test")
        mline.append(f"memory {fmt(m['bw'], 'GB/s')}{pct}, {how}")
    if m["peak"]:
        parts = []
        for pr in ("fp32", "fp64"):
            if pr in m["peak"]:
                t = th["peak"].get(pr)
                parts.append(f"{fmt(m['peak'][pr], 'GFLOP/s')} {pr}" + (f" ({100 * m['peak'][pr] / t:.0f}% of theory)" if t else ""))
        mline.append("compute " + " · ".join(parts) + ", best matrix multiply")
    if mline:
        limits.append("Measured: " + "; ".join(mline))
    pages, entries = [], []
    for wid in ORDER:
        d = P.prepare(libs, wid)
        if d.empty:
            continue
        sc = pd.DataFrame()
        if cls == "cpu" and len(raw):
            r = raw[(raw["device"] == dev) & (raw["workload_id"] == wid)]
            sp = set(r.loc[r["scaling"], "problem"])
            if sp:
                sc = r[r["problem"].isin(sp) & (r["scaling"] | (r["placement"] == "socket0"))]
        figs = P.workload_plots(dev, wid, d, sc, lim, out, cls)
        fast = P.fastest(d)
        main = fast[(fast["variant"] == "base") | (wid in ("C8", "C9"))] if wid != "C1" else fast
        if wid == "A1":
            main = main[main["precision"] == "fp32"]
        unit = d["metric_name"].iloc[0]
        rows = main.to_dict("records")
        for r in rows:
            r["pct"] = P.pct_of_limit(r, wid, lim)
        header = header_box(wid, rows, unit, th, lim, "theoretical")
        pages.append({"id": wid, **names.get(wid, {"name": wid, "what": ""}), "key": key_result(rows, wid, unit),
                      "header": {**header, "why": why_lines(dev, wid)}, "figures": figs,
                      "note": "Final scripts: every result checked against a reference; median of 3 separate runs."})
        entries += overview_entries(wid, rows, unit, lim, names)
    ov = P.overview(entries, out / "overview.png", "theoretical")
    no_lim = [names[w]["name"] for w in [p["id"] for p in pages] if w not in {e["wid"] for e in entries}]
    return {"device": dev, "cls": cls, "category": "final", "core": core, **display(cfg, dev),
            "limits": limits, "overview": ov, "no_limit": no_lim, "workloads": pages}


def header_box(wid: str, rows: list[dict], unit: str, th: dict, lim: dict, word: str) -> dict:
    peak, measured = [], []
    no_limit = _json(CONFIGS / "explanations.json").get("no_limit", {})
    if wid in MEMORY_BOUND or wid == "A1":
        if th["memory"]:
            peak.append(th["memory"])
        if lim.get("mbw"):
            measured.append(f"Measured memory limit: {fmt(lim['mbw'], 'GB/s')} ({100 * lim['mbw'] / th['bw']:.0f}% of theory)" if th["bw"] else
                            f"Measured memory limit: {fmt(lim['mbw'], 'GB/s')}")
    if wid in COMPUTE_WITH_FLOPS or wid == "A1":
        if th["compute"]:
            peak.append(th["compute"])
        if lim.get("mpeak"):
            measured.append("Measured compute limit: " + " · ".join(
                f"{fmt(v, 'GFLOP/s')} {k}" + (f" ({100 * v / th['peak'][k]:.0f}% of theory)" if th["peak"].get(k) else "")
                for k, v in lim["mpeak"].items() if k in ("fp32", "fp64")))
    if not peak:
        peak.append(f"No single hardware limit applies: speed is set by {no_limit.get(wid, 'the algorithm')}, "
                    f"so results are reported in {unit}.")
    big = [r for r in rows if _is_dram(r)] if wid in CACHE_FILTER else rows
    rows = big or rows
    vals = [r["metric"] for r in rows]
    pcts = [r["pct"] for r in rows if r.get("pct") == r.get("pct") and r.get("pct") is not None]
    best = max(rows, key=lambda r: r["metric"])
    if wid == "A1":
        parts = []
        for ph, what, lw in (("decode", "Token generation", "memory"), ("prefill", "Prompt processing", "compute")):
            rr = [r for r in rows if r.get("phase") == ph]
            if rr:
                b = max(rr, key=lambda r: r["metric"])
                pp = [r["pct"] for r in rr if r["pct"] == r["pct"]]
                parts.append(f"{what}: up to {fmt(b['metric'], unit)}" +
                             (f" ({max(pp):.0f}% of the {word} {lw} limit at best)" if pp else ""))
        achieved = "; ".join(parts) + "."
    else:
        extra = ""
        special = {"bf16": "16-bit (bf16)", "fp64 (tensor cores)": "fp64 on tensor cores"}
        if wid == "C1" and any(r.get("precision") in special for r in rows):
            for pr, nm in special.items():
                xs = [r["metric"] for r in rows if r.get("precision") == pr]
                if xs:
                    extra += f" {nm}: up to {fmt(max(xs), unit)}."
            rows = [r for r in rows if r.get("precision") not in special]
            vals = [r["metric"] for r in rows]
            pcts = [r["pct"] for r in rows if r.get("pct") == r.get("pct") and r.get("pct") is not None]
            best = max(rows, key=lambda r: r["metric"])
        achieved = (f"At sizes larger than the cache: {fmt(min(vals), unit)} to {fmt(max(vals), unit)}" if wid in CACHE_FILTER and big
                    else f"Reached {fmt(min(vals), unit)} to {fmt(max(vals), unit)} across sizes")
        if pcts:
            rng = f"{min(pcts):.0f}%" if round(min(pcts)) == round(max(pcts)) else f"{min(pcts):.0f}–{max(pcts):.0f}%"
            achieved += f" ({rng} of the {word} limit)"
        achieved += f"; best at {size_text(best, wid)}." + (extra if wid == "C1" else "")
    return {"peak": peak, "measured": measured, "achieved": achieved}


def _is_dram(r: dict) -> bool:
    return bool(r["dram"]) if "dram" in r else r.get("regime", "dram") == "dram"


def key_result(rows: list[dict], wid: str, unit: str) -> str:
    """Headline number for a workload card: the largest size (memory-limited: largest size beyond the cache)."""
    rr = [r for r in rows if _is_dram(r)] if wid in CACHE_FILTER else rows
    if wid == "C1":
        rr = [r for r in rr if r.get("precision") == "fp32"] or rr
    rr = rr or rows
    if wid == "A1":
        rr = [r for r in rows if r.get("phase") == "decode"] or rows
        top = max(MODEL_ORDER_KEY(r) for r in rr)
        return fmt(max(r["metric"] for r in rr if MODEL_ORDER_KEY(r) == top), unit) + " (largest model, generation)"
    x = P.CFG.get(wid, {}).get("x") or "size"
    top = max(_num(r.get(x)) for r in rr)
    return fmt(max(r["metric"] for r in rr if _num(r.get(x)) == top), unit)


def MODEL_ORDER_KEY(r):
    return P.MODEL_ORDER.get(r.get("model"), -1)


def overview_entries(wid: str, rows: list[dict], unit: str, lim: dict, names: dict) -> list[dict]:
    """One bar per workload (per number format for B1/B2/C1, per phase for A1) at its largest size."""
    out = []
    rows = [r for r in rows if r.get("pct") == r.get("pct") and r.get("pct") is not None]
    if not rows:
        return out
    x = P.CFG.get(wid, {}).get("x")
    facet = {"B1": "precision", "B2": "precision", "C1": "precision", "A1": "phase", "B3": "matrix"}.get(wid)
    groups = {}
    for r in rows:
        groups.setdefault(r.get(facet) if facet else "", []).append(r)
    for f, rr in groups.items():
        if wid == "A1":
            r = max(rr, key=lambda r: r["pct"])
        else:
            r = max(rr, key=lambda r: (_num(r.get(x)), r["metric"]))
        kind = "memory" if (wid in MEMORY_BOUND or (wid == "A1" and f == "decode")) else "compute"
        label = names.get(wid, {}).get("name", wid) + (f" ({P._pretty(facet, f) if facet != 'phase' else ('generation' if f == 'decode' else 'prompt')})" if facet else "")
        limit_val = lim.get("bw") if kind == "memory" else (lim.get("peak") or {}).get("fp32" if r.get("precision") in ("c64", None) else r.get("precision"))
        got = r.get("gbps") if kind == "memory" else r.get("gflops")
        u = "GB/s" if kind == "memory" else "GFLOP/s"
        out.append({"wid": wid, "label": label, "pct": float(r["pct"]), "kind": kind,
                    "text": f"{fmt(got, u)} of {fmt(limit_val, u)}" if limit_val and got else ""})
    return out


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def display(cfg: dict, dev: str) -> dict:
    d = cfg.get("display", {})
    return {"name": d.get("name", dev.replace("_", " ")), "spec": d.get("spec", "")}


# ── earlier-script machines ────────────────────────────────────────────────
def legacy_device(dev: str, cls: str, names: dict, core: bool) -> dict | None:
    rows = legacy.load()
    rows = rows[rows["device"] == dev] if len(rows) else rows
    if rows.empty:
        return None
    main = legacy.main_rows(rows)
    cfg = _json(CONFIGS / "devices" / f"{dev}.json")
    th = theory_lines(cfg)
    lim = {"bw": th["bw"], "peak": th["peak"]}
    caveats = legacy.config().get("caveats", {})
    out = SITE / dev
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    limits = [x for x in (th["memory"], th["compute"]) if x]
    # measured limits from the machine's own correct earlier data
    mparts = []
    mem_src = main[main["dram"] & main["bench"].isin(["saxpy"])]
    if mem_src.empty:
        mem_src = main[main["dram"] & main["bench"].isin(["gemv"])]
    if not mem_src.empty:
        b = mem_src["metric"].max()
        bench = mem_src.loc[mem_src["metric"].idxmax(), "bench"].upper()
        mparts.append(f"memory {fmt(b, 'GB/s')}" + (f" ({100 * b / th['bw']:.0f}% of theory)" if th["bw"] else "") +
                      f", best {bench} larger than the cache")
    g = main[main["bench"] == "gemm"]
    cp = []
    for pr in ("fp32", "fp64"):
        gg = g[g["precision"] == pr]
        if not gg.empty:
            v = gg["metric"].max()
            t = th["peak"].get(pr)
            cp.append(f"{fmt(v, 'GFLOP/s')} {pr}" + (f" ({100 * v / t:.0f}% of theory)" if t else ""))
    if cp:
        mparts.append("compute " + " · ".join(cp) + ", best matrix multiply")
    if mparts:
        limits.append("Measured by the earlier scripts: " + "; ".join(mparts))
    pages, entries = [], []
    for bench in ("saxpy", "spmv", "spgemm", "gemm", "gemv"):
        rb, mb = rows[rows["bench"] == bench], main[main["bench"] == bench]
        if mb.empty:
            continue
        wid = legacy.WORKLOAD[bench]
        figs = P.legacy_plots(dev, bench, rb, mb, lim, out, cls)
        unit = mb["unit"].iloc[0]
        recs = []
        for r in mb.to_dict("records"):
            if unit == "GB/s" and th["bw"] and r["dram"] and bench != "spgemm":
                r["pct"] = 100 * r["metric"] / th["bw"]
            elif unit == "GFLOP/s" and th["peak"]:
                key = "fp64t" if r["precision"].startswith("fp64 (") else r["precision"]
                r["pct"] = 100 * r["metric"] / th["peak"][key] if th["peak"].get(key) else float("nan")
            else:
                r["pct"] = float("nan")
            recs.append(r)
        header = header_box(wid if wid != "GEMV" else "GEMV", recs, unit, th, {}, "theoretical")
        if bench == "spgemm":
            header["peak"] = [f"No single hardware limit applies: building a sparse result needs scattered memory accesses, so results are reported in {unit}."]
        rr = [r for r in recs if r["dram"]] if bench in ("saxpy", "gemv", "spmv") else recs
        if bench == "gemm":
            rr = [r for r in rr if r["precision"] in ("fp32", "fp64")] or rr
        rr = rr or recs
        best = max(rr, key=lambda r: r["metric"])
        tail = header["achieved"].split("; best at")[1].split(".", 1)[1] if "; best at" in header["achieved"] else ""
        header["achieved"] = header["achieved"].split("; best at")[0] + \
            f"; best at size {best['size_label']}" + (f", {best['precision']}" if mb['precision'].nunique() > 1 else "") + "." + tail
        if bench == "gemm":
            rr = [r for r in rr if r["precision"] == "fp32"] or rr
        top = max(r["size"] for r in rr)
        pages.append({"id": wid, **names.get(wid, {"name": wid, "what": ""}),
                      "key": fmt(max(r["metric"] for r in rr if r["size"] == top), unit),
                      "header": {**header, "why": why_lines(dev, wid)}, "figures": figs, "note": caveats.get(bench, "")})
        for r in recs:
            r["phase"] = None
        if bench in ("gemm", "saxpy", "gemv"):
            for pr in ("fp32", "fp64"):
                rr = [r for r in recs if r["precision"] == pr and r["pct"] == r["pct"]]
                if rr:
                    r = max(rr, key=lambda r: (r["size"], r["metric"]))
                    kind = "compute" if bench == "gemm" else "memory"
                    lv = th["bw"] if kind == "memory" else th["peak"].get(pr)
                    entries.append({"wid": wid, "label": f"{names[wid]['name']} ({pr})", "pct": r["pct"], "kind": kind,
                                    "text": f"{fmt(r['metric'], unit)} of {fmt(lv, unit)}"})
    pages.sort(key=lambda p: ORDER.index(p["id"]))
    ov = P.overview(entries, out / "overview.png", "theoretical")
    no_lim = [p["name"] for p in pages if p["id"] not in {e["wid"] for e in entries}]
    return {"device": dev, "cls": cls, "category": "earlier", "core": core, **display(cfg, dev),
            "limits": limits, "overview": ov, "no_limit": no_lim, "workloads": pages}


def main():
    import analyze
    names = plain_names()
    devs = _json(CONFIGS / "workloads.json").get("devices", {})
    core = {d["id"] for side in devs.values() for d in side if d.get("core")}
    raw = analyze.load()
    done = set()
    for cls in ("cpu", "gpu"):
        for p in sorted((DATA / cls).glob("*")):
            dev = p.name
            if not p.is_dir() or dev in ("smoke",) or not (OUT / f"{dev}_libraries.csv").exists():
                continue
            s = final_device(dev, cls, raw, names, dev in core)
            (SITE / dev / "summary.json").write_text(json.dumps(s, indent=1, default=float) + "\n")
            done.add(dev)
            print(f"{dev}: final scripts, {len(s['workloads'])} workloads, {sum(len(w['figures']) for w in s['workloads'])} plots")
    for dev, d in legacy.config()["devices"].items():
        if dev in done:
            continue
        s = legacy_device(dev, d["class"], names, dev in core)
        if s:
            (SITE / dev / "summary.json").write_text(json.dumps(s, indent=1, default=float) + "\n")
            print(f"{dev}: earlier scripts, {len(s['workloads'])} workloads, {sum(len(w['figures']) for w in s['workloads'])} plots")


if __name__ == "__main__":
    main()
