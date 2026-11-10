#!/usr/bin/env python3
"""Library comparison report: which library wins each problem on each device, by how much,
and whether the win is statistically real -- built from analyze.py's per-library output.

Inputs : results/analysis/<DEVICE>_libraries.csv   (written by analyze.py; every library x placement)
         results/calibration/<DEVICE>.json          (measured ceilings, for % of peak)
         configs/library_notes.json                 (curated explanations; optional)
Outputs: results/analysis/library_report.csv        one row per (device, problem) with >= 2 libraries
         results/analysis/library_summary.csv       per (device, workload, precision, library): wins, ties, speed vs winner
         results/analysis/variant_effects.csv       base vs each variant (fast-math, hardware-off, ...): not libraries
         results/analysis/library_report.json       the same, structured (dashboard "Library analysis" page)
         results/analysis/library_report.md         readable tables, winners in bold, notes
         README.md (with --readme README.md)        the block between the library-table markers

Rules:
  - A library's entry for a problem is its BEST placement (placement is a separate choice).
  - Winner = highest metric (all metrics are higher-is-better rates).
  - tie = the winner's 95% CI overlaps the runner-up's: not a statistically real win.
  - Notes are only shown as written in library_notes.json; nothing is inferred automatically.
  - COMPARE_IGNORING lets libraries tied to an input type compete (S1: histc needs float input,
    bincount integer input -- same samples, same bins, same exact counts).
  - Variants (`variant` != base, same precision) are not libraries: reported as variant effects.
Only devices with full campaigns appear: the inputs come from results/data/<cpu|gpu>/ (analyze.py),
never from results/data/sessions/.
"""
from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures import LIB_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("ANALYSIS_OUT", ROOT / "results" / "analysis"))
CAL = Path(os.environ.get("ANALYSIS_CAL", ROOT / "results" / "calibration"))
NOTES = Path(os.environ.get("LIBRARY_NOTES", ROOT / "configs" / "library_notes.json"))
MANIFEST = ROOT / "configs" / "workloads.json"
COMPARE_IGNORING = {"S1": ("precision",)}
VARIANT_NAMES = {"ff": "fast-math / fixed-function variant (C5: __expf instead of expf)",
                 "sw": "hardware crypto instructions disabled (AES-NI / SHA-NI off)"}
MARK_START, MARK_END = "<!-- library-table:start -->", "<!-- library-table:end -->"


def compare_key(wid: str, problem: str) -> str:
    drop = COMPARE_IGNORING.get(wid)
    if not drop:
        return problem
    return json.dumps({k: v for k, v in sorted(json.loads(problem).items()) if k not in drop}, sort_keys=True)


def lib_name(lib: str) -> str:
    return LIB_NAMES.get(lib, lib)


def _calibration(dev: str) -> dict:
    try:
        return json.loads((CAL / f"{dev}.json").read_text()).get("measured", {})
    except (OSError, ValueError):
        return {}


def _pct_peak(r, cal: dict) -> float:
    """% of the device's measured ceiling for this row's bound (B: bandwidth, C: GEMM peak)."""
    if r["bound"] == "B" and cal.get("bandwidth_gbps") and r["gbps"] == r["gbps"]:
        return 100.0 * r["gbps"] / cal["bandwidth_gbps"]
    peak = (cal.get("peak_gflops") or {}).get(r["precision"])
    if r["bound"] == "C" and peak and r["gflops"] == r["gflops"] and r["gflops"] > 0:
        return 100.0 * r["gflops"] / peak
    return float("nan")


def _notes() -> list[dict]:
    try:
        return json.loads(NOTES.read_text()).get("notes", [])
    except (OSError, ValueError):
        return []


def notes_for(dev: str, wid: str, lib: str, notes: list[dict]) -> list[dict]:
    return [n for n in notes if fnmatch.fnmatch(dev, n.get("device", "*"))
            and n.get("workload") in (wid, "*") and n.get("library") in (lib, "*")]


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for f in sorted(glob.glob(str(OUT / "*_libraries.csv"))):
        d = pd.read_csv(f)
        d["device"] = Path(f).name[: -len("_libraries.csv")]
        frames.append(d)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["cmp"] = [compare_key(w, p) for w, p in zip(df["workload_id"], df["problem"])]
    rows = []
    for (dev, wid, prob), g in df.groupby(["device", "workload_id", "cmp"]):
        best = g.sort_values("metric", ascending=False).groupby("lib", as_index=False).first()
        if len(best) < 2:
            continue                                   # single-library problem: nothing to compare
        cal = _calibration(dev)
        best = best.sort_values("metric", ascending=False).reset_index(drop=True)
        w, ru = best.iloc[0], best.iloc[1]
        rec = {"device": dev, "workload_id": wid, "workload": w["workload"], "bound": w["bound"],
               "precision": "/".join(sorted(set(best["precision"]))), "variant": w["variant"], "problem": prob,
               "regime": w["regime"], "ws_over_llc": w["ws_over_llc"], "metric_name": w["metric_name"],
               "winner": w["lib"], "winner_metric": w["metric"], "winner_placement": w["placement"],
               "winner_library_detail": w["library_detail"], "winner_pct_measured_peak": _pct_peak(w, cal),
               "runner_up": ru["lib"], "runner_up_metric": ru["metric"],
               "margin_pct": 100.0 * (w["metric"] / ru["metric"] - 1.0) if ru["metric"] > 0 else float("nan"),
               "tie": bool(w["metric_ci_lo"] <= ru["metric_ci_hi"]),
               "n_libraries": len(best)}
        for _, b in best.iterrows():
            rec[f"metric[{b['lib']}]"] = b["metric"]
            rec[f"vs_winner[{b['lib']}]"] = b["metric"] / w["metric"] if w["metric"] > 0 else float("nan")
            rec[f"placement[{b['lib']}]"] = b["placement"]
        rows.append(rec)
    report = pd.DataFrame(rows)
    if report.empty:
        return report, pd.DataFrame()
    summ = []
    for (dev, wid, prec), g in report.groupby(["device", "workload_id", "precision"]):
        libs = sorted({c[len("metric["):-1] for c in g.columns if c.startswith("metric[") and g[c].notna().any()})
        for lib in libs:
            entered = g[f"metric[{lib}]"].notna()
            if not entered.any():
                continue
            won = (g["winner"] == lib) & entered
            vs = g.loc[entered, f"vs_winner[{lib}]"]
            summ.append({"device": dev, "workload_id": wid, "workload": g["workload"].iloc[0], "precision": prec,
                         "library": lib, "problems": int(entered.sum()), "wins": int(won.sum()),
                         "clear_wins": int((won & ~g["tie"]).sum()), "tied_wins": int((won & g["tie"]).sum()),
                         "median_speed_vs_winner": float(vs.median()), "worst_speed_vs_winner": float(vs.min())})
    return report, pd.DataFrame(summ)


def variant_effects() -> pd.DataFrame:
    """Best base implementation vs best implementation of each variant, same problem otherwise
    (same precision): e.g. C5 __expf (ff) vs expf, C8/C9 hardware crypto off (sw) vs on."""
    frames = [pd.read_csv(f).assign(device=Path(f).name[: -len("_libraries.csv")])
              for f in sorted(glob.glob(str(OUT / "*_libraries.csv")))]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["base_key"] = [json.dumps({k: v for k, v in sorted(json.loads(p).items()) if k != "variant"}, sort_keys=True)
                      for p in df["problem"]]
    rows = []
    for (dev, wid, key), g in df.groupby(["device", "workload_id", "base_key"]):
        if "base" not in set(g["variant"]) or g["variant"].nunique() < 2:
            continue
        b = g[g["variant"] == "base"].sort_values("metric").iloc[-1]
        for var, gv in g[g["variant"] != "base"].groupby("variant"):
            v = gv.sort_values("metric").iloc[-1]
            rows.append({"device": dev, "workload_id": wid, "workload": b["workload"], "variant": var, "problem": key,
                         "base_lib": b["lib"], "variant_lib": v["lib"], "metric_name": b["metric_name"],
                         "base_metric": b["metric"], "variant_metric": v["metric"],
                         "variant_over_base": v["metric"] / b["metric"] if b["metric"] > 0 else float("nan")})
    return pd.DataFrame(rows)


def _order() -> list[str]:
    try:
        return [w["id"] for w in json.loads(MANIFEST.read_text())["workloads"]]
    except (OSError, ValueError, KeyError):
        return []


def readme_block(summ: pd.DataFrame, var: pd.DataFrame, notes: list[dict]) -> str:
    """Markdown for README.md: rows = workload (precision), columns = devices with full campaigns."""
    devs = sorted(summ["device"].unique())
    order = _order()
    keys = sorted({(w, p, n) for w, p, n in zip(summ["workload_id"], summ["precision"], summ["workload"])},
                  key=lambda k: (order.index(k[0]) if k[0] in order else 99, k[1]))
    L = [MARK_START,
         "_Generated by `make readme` from the campaign data (`analysis/library_report.py`); do not edit by hand._",
         "Each cell: **fastest implementation**, sizes it won / sizes compared, then the others at their median speed",
         "relative to the winner (1.00× = as fast). ‘tie’ = the 95% confidence intervals overlap.", "",
         "| Workload | " + " | ".join(d.replace("_", " ") for d in devs) + " |",
         "|---|" + "---|" * len(devs)]
    used_notes = []
    for wid, prec, wname in keys:
        cells = []
        for dev in devs:
            g = summ[(summ["device"] == dev) & (summ["workload_id"] == wid) & (summ["precision"] == prec)]
            if g.empty:
                cells.append("—")
                continue
            g = g.sort_values(["wins", "median_speed_vs_winner"], ascending=False)
            w, rest = g.iloc[0], g.iloc[1:]
            cell = f"**{lib_name(w['library'])}** {w['wins']}/{w['problems']}"
            if w["tied_wins"]:
                cell += f" ({w['tied_wins']} tie)"
            if len(rest):
                cell += "; " + ", ".join(f"{lib_name(r['library'])} {r['median_speed_vs_winner']:.2f}×" for _, r in rest.iterrows())
            for n in notes_for(dev, wid, w["library"], notes):
                if n not in used_notes:
                    used_notes.append(n)
                cell += f" [{used_notes.index(n) + 1}]"
            cells.append(cell)
        L.append(f"| {wid} {wname} ({prec}) | " + " | ".join(cells) + " |")
    if len(var):
        L += ["", "Variants (not library choices):"]
        for (dev, wid, v), g in var.groupby(["device", "workload_id", "variant"]):
            r = g["variant_over_base"]
            L.append(f"- {dev.replace('_', ' ')}, {wid} — {VARIANT_NAMES.get(v, v)}: {r.min():.2f}–{r.max():.2f}× "
                     f"the base implementation ({len(g)} sizes).")
    if used_notes:
        L += ["", "Notes:"]
        for i, n in enumerate(used_notes, 1):
            tag = "" if n.get("status") == "verified" else " _(hypothesis)_"
            L.append(f"{i}. {n['note']}{tag}")
    L.append(MARK_END)
    return "\n".join(L)


def write_readme(path: Path, block: str):
    text = path.read_text()
    if MARK_START not in text or MARK_END not in text:
        raise SystemExit(f"{path}: library-table markers not found ({MARK_START} ... {MARK_END})")
    head, rest = text.split(MARK_START, 1)
    _, tail = rest.split(MARK_END, 1)
    new = head + block + tail
    if new != text:
        path.write_text(new)
    print(f"{path}: library table {'updated' if new != text else 'unchanged'}")


def _md(report: pd.DataFrame, summ: pd.DataFrame, notes: list[dict]) -> str:
    L = ["# Library comparison report", "",
         "Generated by `analysis/library_report.py` from `results/analysis/*_libraries.csv`.",
         "Each library is shown at its best placement. **Bold** = winner. `tie` = the winner's 95% CI overlaps",
         "the runner-up's (not a statistically real win). `speed vs winner` = metric / winner's metric.", ""]
    for dev, gd in summ.groupby("device"):
        L += [f"## {dev}", "", "| Workload | Precision | Library | Wins / problems (clear, tied) | Median speed vs winner | Worst |",
              "|---|---|---|---|---|---|"]
        for (wid, prec), g in gd.groupby(["workload_id", "precision"]):
            g = g.sort_values(["wins", "median_speed_vs_winner"], ascending=False)
            for i, (_, r) in enumerate(g.iterrows()):
                lib = f"**{r['library']}**" if i == 0 and r["wins"] > 0 else r["library"]
                L.append(f"| {wid} {r['workload']} | {prec} | {lib} | {r['wins']}/{r['problems']} "
                         f"({r['clear_wins']}, {r['tied_wins']}) | {r['median_speed_vs_winner']:.2f} | {r['worst_speed_vs_winner']:.2f} |")
        dn = [(wid, lib, n) for (wid, lib) in sorted(set(zip(gd["workload_id"], gd["library"])))
              for n in notes_for(dev, wid, lib, notes)]
        if dn:
            L += ["", "Notes:", ""]
            for wid, lib, n in dn:
                L.append(f"- **{wid} / {lib}** ({n.get('status', 'hypothesis')}): {n['note']}"
                         + (f" Evidence: {n['evidence']}" if n.get("evidence") else ""))
        L += ["", "<details><summary>Per-problem detail</summary>", "",
              "| Workload | Problem | Winner | Margin | Tie | Winner % of measured peak | Runner-up |", "|---|---|---|---|---|---|---|"]
        for _, r in report[report["device"] == dev].sort_values(["workload_id", "problem"]).iterrows():
            pk = "" if r["winner_pct_measured_peak"] != r["winner_pct_measured_peak"] else f"{r['winner_pct_measured_peak']:.0f}%"
            L.append(f"| {r['workload_id']} | `{r['problem']}` | **{r['winner']}** ({r['winner_placement']}) | "
                     f"{r['margin_pct']:+.1f}% | {'tie' if r['tie'] else ''} | {pk} | {r['runner_up']} |")
        L += ["", "</details>", ""]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", type=Path, help="also rewrite the library-table block of this README")
    a = ap.parse_args()
    report, summ = build()
    if report.empty:
        print("no multi-library problems found under", OUT)
        return
    notes = _notes()
    var = variant_effects()
    report.to_csv(OUT / "library_report.csv", index=False)
    summ.to_csv(OUT / "library_summary.csv", index=False)
    var.to_csv(OUT / "variant_effects.csv", index=False)
    if a.readme:
        write_readme(a.readme, readme_block(summ, var, notes))
    payload = {"summary": summ.to_dict(orient="records"),
               "problems": json.loads(report.to_json(orient="records")),
               "variants": json.loads(var.to_json(orient="records")) if len(var) else [],
               "notes": notes, "order": _order(), "lib_names": LIB_NAMES, "variant_names": VARIANT_NAMES}
    (OUT / "library_report.json").write_text(json.dumps(payload, indent=1) + "\n")
    (OUT / "library_report.md").write_text(_md(report, summ, notes))
    print(f"library report: {len(report)} multi-library problems on {report['device'].nunique()} device(s), "
          f"{int(report['tie'].sum())} ties -> {OUT}/library_report.*")


if __name__ == "__main__":
    main()
