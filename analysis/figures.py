"""Dashboard figures (dark theme matching analysis/dashboard.html).

Colour roles (dataviz reference palette, dark steps, fixed order -- never cycled):
  series 1..4 = blue, orange, aqua, yellow; the measured ceiling is neutral ink, not a series.
Every figure: one y-axis, CI error bars, x = problem size ordered by working set,
legend whenever there are >= 2 series.
"""
from __future__ import annotations

import json
import math

SURFACE = "#050b14"          # .card-img-wrap background in analysis/dashboard.html
INK = "#e2e8f0"
INK_MUTED = "#94a3b8"
GRID = "#1e293b"
CEILING = "#64748b"
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

LIB_NAMES = {
    "torch": "PyTorch", "cudagraph": "PyTorch + CUDA graph", "native": "native kernel",
    "graph": "CUDA graph launch", "coop": "cooperative launch", "custom": "custom CUDA kernel",
    "fast": "native, fast exp (__expf)", "histc": "torch.histc", "bincount": "torch.bincount",
    "openblas": "OpenBLAS", "blis": "BLIS", "opencv": "OpenCV", "openssl": "OpenSSL",
}
HIDDEN_KEYS = ("precision", "variant", "lib", "_env")


def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_MUTED, "axes.titlecolor": INK,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "text.color": INK,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True,
        "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.fontsize": 8,
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    })


def fmt_num(v) -> str:
    """Compact binary-ish size: 1048576 -> 1M, 16384 -> 16K, 65 -> 65."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    v = int(v) if float(v).is_integer() else v
    if isinstance(v, int):
        for unit, s in ((1 << 30, "G"), (1 << 20, "M"), (1 << 10, "K")):
            if v >= unit and v % unit == 0:
                return f"{v // unit}{s}"
    return f"{v:g}"


def size_label(wid: str, params: dict, varying: list[str]) -> str:
    if wid == "A1":
        return (f"prefill S={params['seq']}" if params.get("phase") == "prefill" else f"decode B={params['batch']}") \
            + ("" if "model" not in varying else f"\n{params['model']}")
    return "  ".join(f"{k}={fmt_num(params[k])}" for k in varying) or "single size"


def series_name(lib: str, prec: str, var: str, multi_prec: bool) -> str:
    name = LIB_NAMES.get(lib, lib)
    if multi_prec:
        name += f" · {prec}"
    if var not in ("base", "", None) and lib != "fast":
        name += f" ({var})"
    return name


def workload_figure(g, bounds, path, title: str, metric_name: str, extra_ceiling: dict | None = None) -> dict:
    """g: the device's *_libraries rows for one workload; bounds: its *_bounds rows;
    extra_ceiling: {problem: ceiling in metric units} for workloads the bound check does not
    cover (A1: decode = bandwidth, prefill = GEMM peak). Returns a caption dict."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    g = g.copy()
    g["params"] = g["problem"].map(json.loads)
    keys = sorted({k for p in g["params"] for k in p if k not in HIDDEN_KEYS})
    varying = [k for k in keys if g["params"].map(lambda p: json.dumps(p.get(k))).nunique() > 1]
    wid = g["workload_id"].iloc[0]
    g["size"] = g["params"].map(lambda p: size_label(wid, p, varying))
    order = g.groupby("size")["ws_over_llc"].median().sort_values().index.tolist()
    xi = {s: i for i, s in enumerate(order)}
    multi_prec = g["precision"].nunique() > 1
    fig_w = min(13.0, max(6.5, 0.55 * len(order) + 3.0))
    fig, ax = plt.subplots(figsize=(fig_w, 4.3))

    # measured ceiling per problem (from the analysis' own bound check): metric / (pct / 100)
    ceil_pts = []
    if bounds is not None and len(bounds):
        b = bounds.set_index("problem")
        heads = g.sort_values("metric", ascending=False).drop_duplicates("problem")
        for _, r in heads.iterrows():
            if r["problem"] not in b.index:
                continue
            br = b.loc[[r["problem"]]].iloc[0]
            pct = br.get("pct_of_measured_bw")
            if not (pct == pct and pct):
                pct = br.get("pct_of_measured_peak")
            if pct == pct and pct and pct > 0:
                ceil_pts.append((xi[r["size"]], r["metric"] / (pct / 100.0)))
    if extra_ceiling:
        heads = g.drop_duplicates("problem")
        ceil_pts += [(xi[r["size"]], extra_ceiling[r["problem"]]) for _, r in heads.iterrows()
                     if r["problem"] in extra_ceiling]
    if ceil_pts:
        per_x = {}
        for x, y in ceil_pts:
            per_x.setdefault(x, []).append(y)
        xs = list(per_x)
        ys = [sorted(v)[len(v) // 2] for v in per_x.values()]
        ax.scatter(xs, ys, marker="_", s=260, linewidths=2, color=CEILING, zorder=2,
                   label="measured ceiling (bandwidth or GEMM peak)")

    from matplotlib.lines import Line2D
    handles = []
    if ceil_pts:
        handles.append(Line2D([], [], ls="none", marker="_", ms=14, mew=2, color=CEILING,
                              label="measured ceiling"))
    groups = list(g.groupby(["lib", "precision", "variant"], sort=True))
    n = len(groups)
    width = 0.5
    for i, ((lib, prec, var), gg) in enumerate(groups):
        gg = gg.sort_values("metric", ascending=False).drop_duplicates("size")     # best placement
        off = (i - (n - 1) / 2) * (width / max(n, 1)) if n > 1 else 0.0
        x = [xi[s] + off for s in gg["size"]]
        y = gg["metric"].to_numpy()
        lo = (gg["metric"] - gg["metric_ci_lo"]).clip(lower=0).to_numpy()
        hi = (gg["metric_ci_hi"] - gg["metric"]).clip(lower=0).to_numpy()
        ax.errorbar(x, y, yerr=[lo, hi], fmt="o", ms=6, mew=1.5, mec=SURFACE, elinewidth=1.2, capsize=2,
                    color=SERIES[i % len(SERIES)], zorder=3)
        handles.append(Line2D([], [], ls="none", marker="o", ms=6, color=SERIES[i % len(SERIES)],
                              label=series_name(lib, prec, var, multi_prec)))
    ax.set_yscale("log")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=35 if len(order) > 4 else 0, ha="right" if len(order) > 4 else "center",
                       fontsize=8)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel(metric_name + "  (log scale, higher is better)")
    ax.set_xlabel("problem size (ordered by working set)")
    ax.set_title(title, loc="left", pad=26)
    ax.grid(axis="x", visible=False)
    if len(handles) >= 2:
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=len(handles),
                  handletextpad=0.3, columnspacing=1.4, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {"series": [series_name(l, p, v, multi_prec) for (l, p, v), _ in groups], "sizes": len(order),
            "has_ceiling": bool(ceil_pts)}


def overview_figure(entries: list[dict], path, title: str):
    """entries: [{"label", "pct", "kind"}] -- % of the measured ceiling at the largest size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    entries = list(reversed(entries))
    fig, ax = plt.subplots(figsize=(8.5, 0.36 * len(entries) + 1.3))
    y = range(len(entries))
    vals = [min(e["pct"], 110.0) for e in entries]
    ax.barh(y, vals, height=0.62, color=SERIES[0], zorder=3)
    for yi, e, v in zip(y, entries, vals):
        ax.text(v + 1.2, yi, f"{e['pct']:.0f}%", va="center", ha="left", fontsize=8, color=INK)
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{e['label']}  ·  {e['kind']}" for e in entries], fontsize=8)
    ax.axvline(100, color=CEILING, lw=1.2, ls="--", zorder=2)
    ax.text(100, len(entries) - 0.35, "measured ceiling", ha="right", va="bottom", fontsize=7.5, color=INK_MUTED)
    ax.set_xlim(0, 118)
    ax.set_xlabel("% of the device's measured ceiling, at the largest problem size")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def pct(v) -> float:
    return float(v) if v == v and v is not None and not math.isinf(v) else float("nan")
