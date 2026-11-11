#!/usr/bin/env python3
"""Data for the interactive CPU vs GPU page: results/analysis/compare.json.

For every machine with correct data (final scripts: <DEV>_headline.csv; earlier scripts: analysis/legacy.py)
and every workload, a list of points {key, label, prec, v, lo, hi, sz, dram}:
  key  -- matching key: two machines are compared at a size only when the keys are equal
          (C1 / B1 use "n|precision" so earlier-script GEMM and SAXPY match ours; others use the problem itself)
  v    -- fastest result at that size (unit per workload); lo/hi = 95% range (final scripts only)
  sz   -- size order, to pick "the largest size both machines ran"
Final-vs-final ratio ranges come from results/analysis/ratios.csv (bootstrap over all samples).
The browser computes the other ratios (GPU result / CPU result).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import legacy  # noqa: E402
import plots as P  # noqa: E402

OUT = ROOT / "results" / "analysis"
SITE = ROOT / "results" / "site" / "devices"
MODEL_SIZE = {"llama3.2-1b": 1, "llama3.2-3b": 3, "llama3-8b": 8}


def _key(wid: str, p: dict) -> str:
    if wid in ("C1", "B1", "B2"):
        size = p.get("n")
        return f"n={size}|{p.get('precision')}"
    q = {k: v for k, v in p.items() if k not in ("variant", "_env") and not (wid == "S1" and k == "precision")}
    return json.dumps(q, sort_keys=True)


def _sz(wid: str, p: dict) -> float:
    if wid == "A1":
        return MODEL_SIZE.get(p.get("model"), 0) * 1e6 + (p.get("batch") or 0) * 1e3 + (p.get("seq") or 0)
    if wid == "C4":
        return float(p.get("n", 1)) ** int(p.get("dim", 1))
    x = P.CFG.get(wid, {}).get("x")
    v = 1.0
    for k in ([x] if x else []) + [k for k in ("c", "k", "bins", "r") if k in p]:
        try:
            v *= float(p.get(k) or 1)
        except (TypeError, ValueError):
            pass
    if wid in ("B3", "B4"):
        v = float(p.get("size", 1))
    return v


def _label(wid: str, p: dict) -> str:
    if wid == "A1":
        return f"{p['model']}, " + (f"generation batch {p['batch']}" if p.get("phase") == "decode" else f"{p['seq']}-token prompt")
    keys = [P.CFG.get(wid, {}).get("x")] + [k for k in P.CFG.get(wid, {}).get("series", []) if k != "_phase"]
    parts = [P._pretty(k, p[k]) for k in keys if k and k in p and k != "_rank"]
    if wid in ("B3", "B4"):
        parts = [P.PLAIN.get(p.get("matrix"), p.get("matrix")), f"size {P.F.fmt_num(int(p['size']))}"]
    if "precision" in p and wid in ("C1", "B1", "B2") and "precision" not in keys:
        parts.append(p["precision"])
    return ", ".join(dict.fromkeys(parts))


def final_points(dev: str) -> dict:
    h = pd.read_csv(OUT / f"{dev}_headline.csv")
    out = {}
    for r in h.to_dict("records"):
        p = json.loads(r["problem"])
        wid = r["workload_id"]
        if p.get("variant") not in ("base", None) and not (p.get("variant") == "ff" and p.get("precision") == "bf16"):
            continue
        out.setdefault(wid, {"unit": r["metric_name"], "pts": {}})
        k = _key(wid, p)
        cur = out[wid]["pts"].get(k)
        if cur is None or r["metric"] > cur["v"]:
            out[wid]["pts"][k] = {"key": k, "label": _label(wid, p), "prec": p.get("precision", ""), "v": r["metric"],
                                  "lo": r["metric_ci_lo"], "hi": r["metric_ci_hi"], "sz": _sz(wid, p),
                                  "dram": r.get("regime", "dram") == "dram", "problem": r["problem"]}
    return {w: {"unit": d["unit"], "pts": list(d["pts"].values())} for w, d in out.items()}


def legacy_points(dev: str) -> dict:
    df = legacy.load()
    m = legacy.main_rows(df[df["device"] == dev]) if len(df) else df
    out = {}
    for r in m.to_dict("records"):
        wid = r["workload_id"]
        if wid == "GEMV":
            continue
        prec = r["precision"]
        if wid in ("C1", "B1"):
            k = f"n={r['size']}|{prec}"
            label = f"n = {P.F.fmt_num(int(r['size']))}, {prec}"
        else:
            k = f"legacy|{r['size']}|{r['density']}|{prec}"
            label = f"size {r['size_label']}, {prec}"
        out.setdefault(wid, {"unit": r["unit"], "pts": []})
        out[wid]["pts"].append({"key": k, "label": label, "prec": prec, "v": r["metric"], "lo": None, "hi": None,
                                "sz": float(r["size"]) * (1 + float(r["density"] if r["density"] == r["density"] else 0)),
                                "dram": bool(r["dram"]) if wid != "C1" else True})
    return out


def _clean(o):
    if isinstance(o, float):
        return None if not math.isfinite(o) else round(o, 6)
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def main():
    machines, data = [], {}
    for f in sorted(SITE.glob("*/summary.json")):
        s = json.loads(f.read_text())
        dev = s["device"]
        machines.append({"id": dev, "name": s["name"], "spec": s.get("spec", ""), "cls": s["cls"],
                         "category": s["category"], "core": s.get("core", False)})
        data[dev] = final_points(dev) if s["category"] == "final" else legacy_points(dev)
    ratios = {}
    try:
        r = pd.read_csv(OUT / "ratios.csv")
        for x in r.to_dict("records"):
            if x["workload_id"] == "GEOMEAN":
                continue
            p = json.loads(x["problem"])
            ratios[f"{x['device_a']}|{x['device_b']}|{x['workload_id']}|{_key(x['workload_id'], p)}"] = \
                [x["speedup_b_over_a"], x["ci95_lo"], x["ci95_hi"]]
    except (OSError, ValueError):
        pass
    names = {w["id"]: w.get("plain", w["name"]) for w in json.loads((ROOT / "configs" / "workloads.json").read_text())["workloads"]}
    out = {"machines": machines, "data": data, "ratios": ratios, "workload_names": names,
           "order": ["B1", "B2", "B3", "B4", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "S1", "S2", "A1"]}
    (OUT / "compare.json").write_text(json.dumps(_clean(out), separators=(",", ":")) + "\n")
    print(f"compare: {len(machines)} machines, {len(ratios)} final-vs-final ratios -> {OUT / 'compare.json'}")


if __name__ == "__main__":
    main()
