"""Dashboard plot catalogue (one PNG per plot), black and yellow theme from figures.py.

Final-script machines (our harness): per workload, at most 10 plots, at most 2 heatmaps, in this order
  heatmap #1, heatmap #2, P1 speed vs size, P2 share of limit, P4 software comparison,
  P5 thread scaling, P8 precision, P7 energy efficiency, P6 power, P9 hyper-threading.
Earlier-script machines (legacy/): the same names where the data allows, plus a time breakdown (GPUs) and
speed vs sparsity (sparse benches).
Every plot returns {"file", "title", "caption"}; captions are one plain sentence on how to read it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import figures as F

MAX_PLOTS = 10

PLAIN = {
    "poisson3d": "3-D grid matrix", "powerlaw": "power-law graph matrix", "uniform": "uniform data",
    "zipf": "skewed data", "random": "random queries", "sorted": "sorted queries",
    "base": "hardware crypto on", "sw": "hardware crypto off",
}
XLABEL = {
    "B1": "vector length (elements)", "B2": "grid size (cells per side)", "B3": "matrix size", "B4": "matrix size",
    "C1": "matrix size n (n × n)", "C2": "number of simulated paths", "C3": "image size (pixels per side)",
    "C4": "FFT size (total points)", "C5": "image size (pixels per side)", "C6": "number of key-value pairs",
    "C7": "number of sorted keys", "C8": "data size (bytes)", "C9": "message length (bytes)",
    "S1": "number of samples", "S2": "constraints per batch", "A1": "model",
}
CFG = {
    "B1": {"x": "n", "series": ["precision"], "heat": [("precision", "n", {})]},
    "B2": {"x": "n", "series": ["precision"], "heat": [("precision", "n", {})]},
    "B3": {"x": "_rank", "series": ["matrix"], "heat": [("matrix", "_rank", {})]},
    "B4": {"x": "_rank", "series": ["matrix"], "heat": [("matrix", "_rank", {})]},
    "C1": {"x": "n", "series": ["precision"], "heat": [("lib", "n", {"precision": "fp32"}), ("lib", "n", {"precision": "fp64"})]},
    "C2": {"x": "paths", "series": [], "heat": []},
    "C3": {"x": "hw", "series": ["c", "k"], "heat": [("k", "hw", {"c": 1}), ("k", "hw", {"c": 3})]},
    "C4": {"x": "_points", "series": ["dim"], "heat": []},
    "C5": {"x": "hw", "series": ["r"], "heat": [("r", "hw", {})]},
    "C6": {"x": "n", "series": ["dist"], "heat": [("dist", "n", {})]},
    "C7": {"x": "keys", "series": ["order"], "heat": [("order", "keys", {})]},
    "C8": {"x": "bytes", "series": ["variant"], "heat": []},
    "C9": {"x": "msg_len", "series": ["variant"], "heat": [("variant", "msg_len", {})]},
    "S1": {"x": "n", "series": ["dist", "bins"], "heat": [("bins", "n", {"dist": "uniform"}), ("bins", "n", {"dist": "zipf"})]},
    "S2": {"x": "batch", "series": [], "heat": []},
    "A1": {"x": "model", "series": ["_phase"], "heat": [("model", "batch", {"phase": "decode"}), ("model", "seq", {"phase": "prefill"})]},
}
SIZE_NAMES = ["small", "medium", "large", "larger", "largest"]
MODEL_ORDER = {"llama3.2-1b": 0, "llama3.2-3b": 1, "llama3-8b": 2}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    F.style(plt)
    return plt


def per_joule(metric_name: str) -> str:
    return metric_name[:-2] + " per joule" if metric_name.endswith("/s") else metric_name + " per joule"


def _val(v) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    a = abs(v)
    return f"{v:,.0f}" if a >= 100 else f"{v:.3g}"


def _pretty(key: str, v) -> str:
    if key == "precision":
        return str(v)
    if key == "c":
        return f"{v} channel" + ("s" if int(v) > 1 else "")
    if key == "k":
        return f"{v}×{v} filter"
    if key == "r":
        return f"radius {v}"
    if key == "bins":
        return f"{F.fmt_num(int(v))} bins"
    if key == "dim":
        return f"{v}-D"
    if key == "lib":
        return F.LIB_NAMES.get(v, v)
    if key == "_points":
        return F.fmt_num(int(v))
    if key in ("_rank",):
        return SIZE_NAMES[int(v)] if int(v) < len(SIZE_NAMES) else str(v)
    if key in ("batch",):
        return f"batch {v}"
    if key in ("seq",):
        return f"{v} tokens"
    if isinstance(v, (int, np.integer)) or (isinstance(v, float) and float(v).is_integer()):
        return F.fmt_num(int(v))
    return PLAIN.get(str(v), str(v))


def _label(keys, row) -> str:
    return ", ".join(_pretty(k, row[k]) for k in keys) or "fastest software"


# ── data preparation (final-script machines) ───────────────────────────────
def prepare(lib_df: pd.DataFrame, wid: str) -> pd.DataFrame:
    d = lib_df[lib_df["workload_id"] == wid].copy()
    if d.empty:
        return d
    params = [json.loads(p) for p in d["problem"]]
    keys = sorted({k for p in params for k in p})
    for k in keys:
        d[k] = [p.get(k) for p in params]
    if wid in ("B3", "B4"):
        size_col = "size"
        d["_rank"] = d.groupby("matrix")[size_col].rank(method="dense").astype(int) - 1
    if wid == "C4":
        d["_points"] = [int(n) if int(dm) == 1 else int(n) ** 2 for n, dm in zip(d["n"], d["dim"])]
    if wid == "A1":
        d["_phase"] = [f"token generation, batch {b}" if ph == "decode" else f"prompt of {s} tokens"
                       for ph, b, s in zip(d["phase"], d["batch"], d["seq"])]
    # problems compared across input types (S1: histc fp32 vs bincount int32 count the same samples)
    d["cmp"] = [json.dumps({k: v for k, v in json.loads(p).items() if not (wid == "S1" and k == "precision")},
                           sort_keys=True) for p in d["problem"]]
    return d


def fastest(d: pd.DataFrame) -> pd.DataFrame:
    return d.loc[d.groupby("cmp")["metric"].idxmax()]


def best_by(d: pd.DataFrame, col: str) -> pd.DataFrame:
    return d.loc[d.groupby(["cmp", col])["metric"].idxmax()]


def pct_of_limit(r, wid: str, lim: dict) -> float:
    bw, peak = lim.get("bw"), lim.get("peak") or {}
    prec = "fp32" if r.get("precision") == "c64" else r.get("precision")
    if wid == "A1":
        if r.get("phase") == "decode" and bw and r.get("gbps", 0) > 0:
            return 100 * r["gbps"] / bw
        if r.get("phase") == "prefill" and peak.get(prec) and r.get("gflops", 0) > 0:
            return 100 * r["gflops"] / peak[prec]
        return float("nan")
    if r.get("bound") == "B" and bw and r.get("gbps", 0) > 0:
        return 100 * r["gbps"] / bw if r.get("regime", "dram") == "dram" else float("nan")
    if r.get("bound") == "C" and peak.get(prec) and r.get("gflops", 0) > 0:
        return 100 * r["gflops"] / peak[prec]
    return float("nan")


# ── generic drawing helpers ────────────────────────────────────────────────
def _x_order(d: pd.DataFrame, x: str):
    vals = d[x].dropna().unique().tolist()
    if x == "model":
        return sorted(vals, key=lambda v: MODEL_ORDER.get(v, 9))
    try:
        return sorted(vals, key=float)
    except (TypeError, ValueError):
        return sorted(vals, key=str)


def _logy(ax, values):
    v = [x for x in values if x and x > 0 and math.isfinite(x)]
    if v and max(v) / min(v) > 20:
        ax.set_yscale("log")


def _save(fig, path: Path, title: str, caption: str) -> dict:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return {"file": path.name, "title": title, "caption": caption}


def _legend(ax, n):
    if n >= 2:
        ax.legend(loc="best", ncol=1 if n <= 4 else 2)


def _cat_axis(ax, labels, xlabel):
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0 if len(labels) <= 6 else 45, ha="center" if len(labels) <= 6 else "right")
    ax.set_xlabel(xlabel)


def heatmap(pivot: pd.DataFrame, path: Path, title: str, unit: str, row_key: str, col_key: str,
            xlabel: str, caption: str) -> dict | None:
    if pivot.empty or pivot.shape[0] < 1 or pivot.shape[1] < 2:
        return None
    plt = _plt()
    from matplotlib.colors import LogNorm, Normalize
    v = pivot.to_numpy(dtype=float)
    fin = v[np.isfinite(v) & (v > 0)]
    if fin.size == 0:
        return None
    norm = LogNorm(fin.min(), fin.max()) if fin.max() / fin.min() > 20 else Normalize(fin.min() * 0.9, fin.max())
    fig, ax = plt.subplots(figsize=(6.4, 0.9 + 0.55 * max(2, pivot.shape[0]) + 1.2))
    im = ax.imshow(np.ma.masked_invalid(v), cmap=F.ramp_cmap(), norm=norm, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([_pretty(col_key, c) for c in pivot.columns], rotation=0 if pivot.shape[1] <= 6 else 45)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([_pretty(row_key, r) for r in pivot.index])
    ax.set_xlabel(xlabel)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if np.isfinite(v[i, j]):
                bright = norm(v[i, j]) > 0.55
                ax.text(j, i, _val(v[i, j]), ha="center", va="center", fontsize=8,
                        color="#000000" if bright else F.INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(unit, color=F.INK_MUTED)
    cb.outline.set_visible(False)
    pass  # title is shown by the page
    return _save(fig, path, title, caption)


def lines(series: list[tuple[str, list, list, list | None]], path: Path, title: str, ylabel: str, xlabel: str,
          caption: str, limits: list[tuple[float, str]] = (), xlabels: list[str] | None = None,
          ref100: bool = False, logy_auto: bool = True, connect: bool = True) -> dict | None:
    """series = [(name, x positions, y, (lo, hi) or None)], categorical x if xlabels is given."""
    if not series:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    ys = []
    for i, (name, xs, y, ci) in enumerate(series):
        c = F.SERIES[i % len(F.SERIES)] if i < len(F.SERIES) else F.INK_MUTED
        ax.plot(xs, y, "-o" if connect else "o", color=c, label=name, markersize=6 if connect else 7)
        if ci is not None:
            lo, hi = ci
            ax.errorbar(xs, y, yerr=[np.maximum(0, np.array(y) - np.array(lo)), np.maximum(0, np.array(hi) - np.array(y))],
                        fmt="none", ecolor=c, elinewidth=1.2, capsize=2)
        ys += [v for v in y if v == v]
    for val, name in limits:
        ax.axhline(val, color=F.LIMIT, ls="--", lw=1.2, alpha=0.8)
        ax.annotate(name, xy=(1, val), xycoords=("axes fraction", "data"), ha="right", va="bottom",
                    fontsize=8, color=F.INK_MUTED)
        ys.append(val)
    if ref100:
        ax.axhline(100, color=F.LIMIT, ls="--", lw=1.2, alpha=0.8)
    if logy_auto:
        _logy(ax, ys)
    if xlabels is not None:
        _cat_axis(ax, xlabels, xlabel)
    else:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    pass  # title is shown by the page
    _legend(ax, len(series))
    return _save(fig, path, title, caption)


def bars(groups: list[str], series: list[tuple[str, list]], path: Path, title: str, ylabel: str, xlabel: str,
         caption: str, ref: float | None = None, logy: bool = False, stacked: bool = False) -> dict | None:
    if not series or not groups:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    n = len(series)
    w = 0.8 / (1 if stacked else n)
    bottom = np.zeros(len(groups))
    for i, (name, vals) in enumerate(series):
        vals = np.array([v if v == v else 0 for v in vals], dtype=float)
        x = np.arange(len(groups)) + (0 if stacked else (i - (n - 1) / 2) * w)
        ax.bar(x, vals, width=w * 0.92, bottom=bottom if stacked else None, color=F.SERIES[i % len(F.SERIES)],
               label=name, edgecolor=F.SURFACE, linewidth=1)
        if stacked:
            bottom = bottom + vals
    if ref is not None:
        ax.axhline(ref, color=F.LIMIT, ls="--", lw=1.2, alpha=0.8)
    if logy:
        ax.set_yscale("log")
    _cat_axis(ax, groups, xlabel)
    ax.set_ylabel(ylabel)
    if stacked and n >= 2:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=n)
    else:
        _legend(ax, n)
    return _save(fig, path, title, caption)


# ── final-script workload pages ────────────────────────────────────────────
def workload_plots(dev: str, wid: str, d: pd.DataFrame, scaling: pd.DataFrame, lim: dict, out: Path,
                   cls: str) -> list[dict]:
    """d = prepare(...) rows of one workload; scaling = raw rows of its thread sweep (may be empty)."""
    if d.empty:
        return []
    cfg = CFG.get(wid, {"x": None, "series": [], "heat": []})
    unit = d["metric_name"].iloc[0]
    x = cfg["x"]
    xl = XLABEL.get(wid, "problem size")
    fast = fastest(d)
    if wid == "A1":
        fast_main = fast[(fast["precision"] == "fp32") & (fast["variant"] == "base")]
    elif wid in ("C1", "B1", "B2"):
        fast_main = fast
    else:
        fast_main = fast[fast["variant"].isin(["base", "ff"]) | (wid in ("C8", "C9"))]
    plots: list[dict] = []
    p = lambda name: out / f"{wid}_{name}.png"  # noqa: E731

    # heatmaps (at most 2)
    for hi, (rk, ck, filt) in enumerate(cfg["heat"][:2]):
        src = best_by(d, "lib") if "lib" in (rk, ck) else fast_main
        if wid == "A1":
            src = fast[(fast["precision"] == "fp32") & (fast["variant"] == "base")]
        for k, v in filt.items():
            src = src[src[k] == v]
        if src.empty:
            continue
        if wid not in ("A1",):
            others = [c for c in cfg["series"] if c not in (rk, ck) and c in src and c != "precision" and src[c].nunique() > 1]
            for c in others:                     # keep one slice of any other swept parameter (its largest value)
                src = src[src[c] == _x_order(src, c)[-1]]
        pv = src.pivot_table(index=rk, columns=ck, values="metric", aggfunc="max")
        pv = pv.reindex(index=_x_order(src, rk), columns=_x_order(src, ck))
        tag = ", ".join(_pretty(k, v) for k, v in filt.items() if k not in ("phase",))
        what = {"decode": "token generation", "prefill": "prompt processing"}.get(filt.get("phase"), "")
        title = f"{unit} by {_axis_name(rk)} and {_axis_name(ck)}" + (f" ({tag or what})" if (tag or what) else "")
        r = heatmap(pv, p(f"heat{hi + 1}"), title, unit, rk, ck, _axis_name(ck).capitalize(),
                    "Brighter cells are faster; each cell shows the measured value.")
        if r:
            plots.append(r)

    # P1 speed vs size
    keys = cfg["series"]
    ser, limits = [], []
    xs_all = _x_order(fast_main, x) if x else []
    xlabels = [_pretty(x, v) for v in xs_all]
    groups = fast_main.groupby(keys) if keys else [((), fast_main)]
    for gk, g in groups:
        g = g.sort_values("metric").drop_duplicates(x, keep="last").set_index(x).reindex(xs_all)
        name = _label(keys, dict(zip(keys, gk if isinstance(gk, tuple) else (gk,)))) if keys else "fastest software"
        ser.append((name, list(range(len(xs_all))), g["metric"].tolist(), (g["metric_ci_lo"].tolist(), g["metric_ci_hi"].tolist())))
    ser = ser[:len(F.SERIES)]
    if unit == "GB/s" and lim.get("bw"):
        limits.append((lim["bw"], f"measured memory limit {lim['bw']:,.0f} GB/s"))
    if unit == "GFLOP/s":
        precs = sorted(set(fast_main["precision"]))
        for pr in precs:
            pk = (lim.get("peak") or {}).get("fp32" if pr == "c64" else pr)
            if pk and wid in ("C1",):
                limits.append((pk, f"{pr} compute limit {pk:,.0f}"))
    r = lines(ser, p("speed"), f"Speed vs problem size ({unit})", unit, xl,
              "Each line is one setting; points are the median of three runs with their 95% range; dashed lines are the machine's measured limits.",
              limits, xlabels)
    if r:
        plots.append(r)

    # P2 share of limit
    pr_rows = fast_main.copy()
    pr_rows["pct"] = [pct_of_limit(row, wid, lim) for row in pr_rows.to_dict("records")]
    pr_rows = pr_rows[pr_rows["pct"].notna()]
    if not pr_rows.empty:
        ser = []
        for gk, g in (pr_rows.groupby(keys) if keys else [((), pr_rows)]):
            g = g.sort_values("metric").drop_duplicates(x, keep="last").set_index(x).reindex(xs_all)
            name = _label(keys, dict(zip(keys, gk if isinstance(gk, tuple) else (gk,)))) if keys else "fastest software"
            ser.append((name, list(range(len(xs_all))), g["pct"].tolist(), None))
        kind = "memory" if d["bound"].iloc[0] == "B" else ("both" if wid == "A1" else "compute")
        cap = {"memory": "100% (dashed) is the measured memory limit; sizes that fit in the cache are left out, and sizes only a few times larger than the cache can still pass 100%.",
               "compute": "100% (dashed) is the best speed measured when calibrating this machine; single runs can be slightly faster.",
               "both": "Token generation is compared with the memory limit, prompt processing with the compute limit; 100% (dashed) is the limit."}[kind]
        r = lines(ser[:len(F.SERIES)], p("share"), "Share of the machine's measured limit (%)", "% of measured limit", xl, cap,
                  xlabels=xlabels, ref100=True, logy_auto=False)
        if r:
            plots.append(r)

    # P4 software comparison
    if d["lib"].nunique() > 1:
        bl = best_by(d if wid != "A1" else d[d["variant"] == "base"], "lib")
        best = bl.groupby("cmp")["metric"].transform("max")
        bl = bl.assign(rel=bl["metric"] / best)
        order = fast_main.sort_values([c for c in [x] if c in fast_main] or ["cmp"])["cmp"].tolist()
        order = [c for c in order if c in set(bl["cmp"])]
        lab = [_short_problem(c, wid) for c in order]
        ser = []
        for lib, g in bl.groupby("lib"):
            g = g.set_index("cmp").reindex(order)
            ser.append((F.LIB_NAMES.get(lib, lib), list(range(len(order))), g["rel"].tolist(), None))
        r = lines(ser[:len(F.SERIES)], p("software"), "Software comparison (speed relative to the fastest)",
                  "speed ÷ fastest (1 = fastest)", "problem", "Each colour is one software library; 1.0 means it was the fastest at that size.",
                  xlabels=lab, logy_auto=False, connect=False)
        if r:
            plots.append(r)

    # P5 thread scaling (CPU)
    if cls == "cpu" and not scaling.empty:
        r = thread_scaling(scaling, p("threads"), unit, wid)
        if r:
            plots.append(r)

    # P8 precision
    if wid in ("B1", "B2", "C1", "A1") and fast["precision"].nunique() > 1:
        src = fast if wid != "A1" else fast[fast["phase"].notna()]
        gx = "_phase_model" if wid == "A1" else x
        if wid == "A1":
            src = src.assign(_phase_model=[f"{m.replace('llama', '')}, {ph}" for m, ph in zip(src["model"], src["_phase"])])
        groups_x = _x_order(src, gx) if wid != "A1" else sorted(src[gx].unique())
        ser = []
        for pr, g in src.groupby("precision"):
            g = g.groupby(gx)["metric"].max().reindex(groups_x)
            ser.append((pr, g.tolist()))
        r = bars([_pretty(gx, v) if wid != "A1" else v for v in groups_x], ser, p("precision"),
                 f"Number format comparison ({unit})", unit, xl if wid != "A1" else "model and phase",
                 "Each bar group is one size; bars compare number formats (fp64 = 64-bit, fp32 = 32-bit, bf16 = 16-bit).",
                 logy=True)
        if r:
            plots.append(r)

    # P7 energy, P6 power
    pw = fast_main[fast_main["power_w"] > 0] if "power_w" in fast_main else fast_main.iloc[0:0]
    if not pw.empty:
        ser_e, ser_p = [], []
        for gk, g in (pw.groupby(keys) if keys else [((), pw)]):
            g = g.sort_values("metric").drop_duplicates(x, keep="last").set_index(x).reindex(xs_all)
            name = _label(keys, dict(zip(keys, gk if isinstance(gk, tuple) else (gk,)))) if keys else "fastest software"
            ser_e.append((name, list(range(len(xs_all))), (g["metric"] / g["power_w"]).tolist(), None))
            ser_p.append((name, list(range(len(xs_all))), g["power_w"].tolist(), None))
        r = lines(ser_e[:6], p("energy"), f"Energy efficiency ({per_joule(unit)})", per_joule(unit), xl,
                  "Work done per joule of energy (higher is better), from the chip's own power reading.", xlabels=xlabels)
        if r:
            plots.append(r)
        r = lines(ser_p[:6], p("power"), "Average power while running (watts)", "watts", xl,
                  "Average power drawn by the " + ("processor package" if cls == "cpu" else "GPU board") + " during the measurement.",
                  xlabels=xlabels, logy_auto=False)
        if r:
            plots.append(r)

    # P9 hyper-threading (CPU)
    if cls == "cpu" and d["placement"].nunique() > 1:
        bp = best_by(d, "placement")
        piv = bp.pivot_table(index="cmp", columns="placement", values="metric", aggfunc="max")
        if {"socket0", "socket0-smt"} <= set(piv.columns):
            ratio = (piv["socket0-smt"] / piv["socket0"]).dropna()
            order = [c for c in fast_main.sort_values(x)["cmp"] if c in ratio.index] if x else list(ratio.index)
            r = bars([_short_problem(c, wid) for c in order], [("all hardware threads ÷ one per core", ratio.reindex(order).tolist())],
                     p("smt"), "Hyper-threading effect", "speed ratio (1 = no change)", "problem",
                     "Above 1: using both hardware threads of every core was faster; below 1: one thread per core was faster.",
                     ref=1.0)
            if r:
                plots.append(r)
    return plots[:MAX_PLOTS]


def _axis_name(k: str) -> str:
    return {"precision": "number format", "n": "size", "matrix": "matrix type", "_rank": "size", "lib": "software",
            "k": "filter size", "hw": "image size", "c": "channels", "r": "filter radius", "dist": "data distribution",
            "keys": "number of keys", "order": "query order", "variant": "hardware crypto", "msg_len": "message length",
            "bins": "number of bins", "model": "model", "batch": "batch size", "seq": "prompt length"}.get(k, k)


def _short_problem(cmp: str, wid: str) -> str:
    p = json.loads(cmp)
    keep = [k for k in p if k not in ("variant", "_env", "precision", "index", "total") or (k == "precision" and wid in ("C1", "B1", "B2"))]
    if wid == "A1":
        return f"{p['model'].replace('llama', '')} " + (f"gen b{p['batch']}" if p.get("phase") == "decode" else f"prompt {p['seq']}")
    return " ".join(F.fmt_num(p[k]) if isinstance(p[k], (int, float)) else str(p[k])[:6] for k in keep)


def thread_scaling(s: pd.DataFrame, path: Path, unit: str, wid: str) -> dict | None:
    """s = raw rows of one workload's scaling problem: socket0:<n> placements + the full-socket rows."""
    s = s.copy()
    s["series"] = [f"{F.LIB_NAMES.get(l, l)} · {p}" if wid == "C1" else (p if wid == "B1" else F.LIB_NAMES.get(l, l))
                   for l, p in zip(s["lib"], s["precision"])]
    g = s.groupby(["series", "threads"])["metric_value"].median().reset_index()
    if wid == "C1":                                    # fastest library per precision only
        full = g[g["threads"] == g["threads"].max()]
        best = {}
        for ser_name, v in zip(full["series"], full["metric_value"]):
            pr = ser_name.split(" · ")[-1]
            if v > best.get(pr, (None, -1))[1]:
                best[pr] = (ser_name, v)
        g = g[g["series"].isin([b[0] for b in best.values()])]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    n = 0
    for i, (name, gg) in enumerate(g.groupby("series")):
        gg = gg.sort_values("threads")
        if len(gg) < 3:
            continue
        c = F.SERIES[i % len(F.SERIES)]
        t, y = gg["threads"].to_numpy(), gg["metric_value"].to_numpy()
        ax.plot(t, y, "-o", color=c, label=name)
        ax.plot(t, y[0] * t / t[0], ":", color=c, lw=1.2, alpha=0.7)
        eff = (y[-1] / y[0]) / (t[-1] / t[0])
        ax.annotate(f"{100 * eff:.0f}% efficient", (t[-1], y[-1]), textcoords="offset points", xytext=(-4, 6),
                    ha="right", fontsize=8, color=F.INK)
        n += 1
    if not n:
        import matplotlib.pyplot as plt2
        plt2.close(fig)
        return None
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sorted(g["threads"].unique()))
    ax.set_xticklabels([str(int(v)) for v in sorted(g["threads"].unique())])
    ax.set_xlabel("CPU threads used")
    ax.set_ylabel(unit)
    title = "Thread scaling"
    pass  # title is shown by the page
    _legend(ax, n)
    return _save(fig, path, title, "Solid lines are measured; dotted lines show perfect scaling from one thread. "
                                   "'Efficient' = measured speed-up ÷ thread count at the largest count.")


# ── machine overview chart ─────────────────────────────────────────────────
def overview(entries: list[dict], path: Path, limit_word: str) -> dict | None:
    """entries: [{label, pct, kind ('memory'|'compute'), text}] -> horizontal bars, one per workload."""
    if not entries:
        return None
    plt = _plt()
    entries = sorted(entries, key=lambda e: (e["kind"] != "memory", -e["pct"]))
    fig, ax = plt.subplots(figsize=(9.2, 1.3 + 0.42 * len(entries)))
    y = np.arange(len(entries))[::-1]
    colors = {"memory": F.SERIES[1], "compute": F.SERIES[0]}
    for yi, e in zip(y, entries):
        ax.barh(yi, min(e["pct"], 110), color=colors[e["kind"]], height=0.62, edgecolor=F.SURFACE)
        ax.text(min(e["pct"], 110) + 1.5, yi, f"{e['pct']:.0f}%  ·  {e['text']}", va="center", fontsize=8, color=F.INK)
    ax.axvline(100, color=F.LIMIT, ls="--", lw=1.2, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([e["label"] for e in entries])
    ax.set_xlim(0, 200)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(f"% of this machine's {limit_word} limit (largest size, fastest software)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=colors["memory"], label="limited by memory speed (GB/s)"),
                       Patch(color=colors["compute"], label="limited by compute (GFLOP/s)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.3 / max(1, len(entries))), ncol=2)
    ax.grid(axis="y", visible=False)
    title = f"How close each workload gets to the {limit_word} limit"
    ax.set_xlabel(ax.get_xlabel())
    pass  # title is shown by the page
    return _save(fig, path, title, "Each bar is one workload at its largest size; 100% (dashed) is the hardware limit.")


# ── earlier-script (legacy) pages ──────────────────────────────────────────
LEG_X = {"gemm": "matrix size n (n × n)", "saxpy": "vector length (elements)", "gemv": "matrix size n (n × n)",
         "spmv": "matrix size n (n × n)", "spgemm": "matrix size n (n × n)"}
PREC_ORDER = ["fp64", "fp64 (tensor cores)", "fp32", "bf16"]


def legacy_plots(dev: str, bench: str, rows: pd.DataFrame, main: pd.DataFrame, lim: dict, out: Path,
                 cls: str) -> list[dict]:
    """rows = every usable row of this bench (all GPU / thread counts); main = one value per size."""
    if main.empty:
        return []
    unit = main["unit"].iloc[0]
    wid = main["workload_id"].iloc[0]
    xl = LEG_X[bench]
    p = lambda name: out / f"{wid}_{name}.png"  # noqa: E731
    plots = []
    sparse = bench in ("spmv", "spgemm")
    if sparse:                                   # main view: the most common density
        dens = main["density"].round(6).value_counts().index[0]
        mview = main[main["density"].round(6) == dens]
    else:
        mview = main
    # heatmap: size x number of GPUs / threads
    h = rows[(rows["precision"] == "fp32")]
    if sparse:
        h = h[h["density"].round(6) == dens]
    if h["units"].nunique() > 1:
        pv = h.pivot_table(index="units", columns="size", values="metric", aggfunc="max")
        what = "GPUs" if cls == "gpu" else "CPU threads"
        r = heatmap(pv, p("heat1"), f"{unit} by number of {what} and size (fp32)", unit, "units", "size", xl,
                    f"Brighter cells are faster; rows are the number of {what} used together.")
        if r:
            plots.append(r)
    # heatmap: number format x size (single-GPU / full-socket results), when no GPU-count heatmap exists
    if not plots and mview["precision"].nunique() > 1:
        pv = mview.pivot_table(index="precision", columns="size", values="metric", aggfunc="max")
        pv = pv.reindex(index=[q for q in PREC_ORDER if q in pv.index])
        r = heatmap(pv, p("heat1"), f"{unit} by number format and size", unit, "precision", "size", xl,
                    "Brighter cells are faster; each cell shows the measured value.")
        if r:
            plots.append(r)
    # P1 speed vs size, per number format
    xs = sorted(mview["size"].unique())
    ser = []
    for pr in [q for q in PREC_ORDER if q in set(mview["precision"])]:
        g = mview[mview["precision"] == pr].set_index("size")["metric"].reindex(xs)
        ser.append((pr, list(range(len(xs))), g.tolist(), None))
    limits = []
    if unit == "GB/s" and lim.get("bw"):
        limits.append((lim["bw"], f"memory limit (datasheet) {lim['bw']:,.0f} GB/s"))
    if unit == "GFLOP/s":
        for pr in ("fp32", "fp64"):
            pk = (lim.get("peak") or {}).get(pr)
            if pk and pr in set(mview["precision"]):
                limits.append((pk, f"{pr} compute limit (datasheet) {pk:,.0f}"))
    xlabels = [F.fmt_num(int(v)) for v in xs]
    r = lines(ser, p("speed"), f"Speed vs problem size ({unit})", unit, xl,
              "Each line is one number format (mean of the earlier script's runs); dashed lines are datasheet limits."
              + (" Small sizes can exceed the memory limit because they fit in the on-chip cache." if unit == "GB/s" else ""),
              limits, xlabels)
    if r:
        plots.append(r)
    # P2 share of datasheet limit
    sh = []
    for pr in [q for q in PREC_ORDER if q in set(mview["precision"])]:
        g = mview[mview["precision"] == pr].set_index("size")
        if unit == "GB/s" and lim.get("bw"):
            g = g[g["dram"]] if "dram" in g else g
            v = 100 * g["metric"] / lim["bw"]
        elif unit == "GFLOP/s" and (lim.get("peak") or {}).get(pr.split(" ")[0] if pr != "fp64 (tensor cores)" else "fp64t"):
            v = 100 * g["metric"] / lim["peak"][pr.split(" ")[0] if pr != "fp64 (tensor cores)" else "fp64t"]
        else:
            continue
        sh.append((pr, list(range(len(xs))), v.reindex(xs).tolist(), None))
    if sh and not sparse:
        r = lines(sh, p("share"), "Share of the datasheet limit (%)", "% of datasheet limit", xl,
                  "100% (dashed) is the vendor's datasheet limit; memory-limited sizes that fit in cache are left out.",
                  xlabels=xlabels, ref100=True, logy_auto=False)
        if r:
            plots.append(r)
    # thread scaling (CPU)
    if cls == "cpu" and rows["units"].nunique() > 2:
        big = rows[rows["size"] == rows["size"].max()]
        if sparse:
            big = big[big["density"].round(6) == dens]
        s = big.rename(columns={"metric": "metric_value", "units": "threads"}).assign(lib="earlier script")
        s = s[s["precision"].isin(["fp32", "fp64"])]
        r = thread_scaling(s.assign(lib=s["precision"]), p("threads"), unit, "B1")
        if r:
            plots.append(r)
    # precision bars
    if mview["precision"].nunique() > 1:
        ser = [(pr, mview[mview["precision"] == pr].set_index("size")["metric"].reindex(xs).tolist())
               for pr in [q for q in PREC_ORDER if q in set(mview["precision"])]]
        r = bars(xlabels, ser, p("precision"), f"Number format comparison ({unit})", unit, xl,
                 "Each bar group is one size; bars compare number formats.", logy=True)
        if r:
            plots.append(r)
    # speed vs sparsity
    if sparse and main["density"].nunique() > 1:
        f32 = main[main["precision"] == "fp32"]
        dx = sorted(f32["density"].unique())
        ser = []
        for s_ in sorted(f32["size"].unique())[-4:]:
            g = f32[f32["size"] == s_].set_index("density")["metric"].reindex(dx)
            ser.append((f"n = {F.fmt_num(int(s_))}", list(range(len(dx))), g.tolist(), None))
        r = lines(ser, p("sparsity"), f"Speed vs sparsity ({unit}, fp32)", unit, "fraction of non-zero entries",
                  "Each line is one matrix size; further right = denser matrix.", xlabels=[f"{v:g}" for v in dx])
        if r:
            plots.append(r)
    # time breakdown (GPU)
    if cls == "gpu" and mview[["h2d_ms", "compute_ms", "d2h_ms"]].notna().all(axis=1).any():
        f = mview[mview["precision"] == "fp32"].set_index("size").reindex(xs)
        tot = f[["h2d_ms", "compute_ms", "d2h_ms"]].sum(axis=1)
        if tot.notna().any() and (tot > 0).any():
            ser = [(name, (100 * f[c] / tot).tolist()) for name, c in
                   (("copy to GPU", "h2d_ms"), ("compute", "compute_ms"), ("copy back", "d2h_ms"))]
            r = bars(xlabels, ser, p("time"), "Where the time goes (fp32, % of total)", "% of total time", xl,
                     "Our speed numbers count only the compute part; this shows how much time copying data would add.",
                     stacked=True)
            if r:
                plots.append(r)
    # energy and power (GPU only; earlier CPU power readings were wrong)
    if cls == "gpu" and mview["power_w"].notna().any():
        ser_e, ser_p = [], []
        for pr in [q for q in PREC_ORDER if q in set(mview["precision"])]:
            g = mview[mview["precision"] == pr].set_index("size").reindex(xs)
            ser_e.append((pr, list(range(len(xs))), (g["metric"] / g["power_w"]).tolist(), None))
            ser_p.append((pr, list(range(len(xs))), g["power_w"].tolist(), None))
        r = lines(ser_e, p("energy"), f"Energy efficiency ({per_joule(unit)})", per_joule(unit), xl,
                  "Work done per joule of energy (higher is better).", xlabels=xlabels)
        if r:
            plots.append(r)
        r = lines(ser_p, p("power"), "Average power while running (watts)", "watts", xl,
                  "Average GPU board power during the measurement.", xlabels=xlabels, logy_auto=False)
        if r:
            plots.append(r)
    return plots[:MAX_PLOTS]
