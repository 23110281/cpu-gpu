#!/usr/bin/env python3
"""Library comparison report: which library wins each problem on each device, by how much,
and whether the win is statistically real -- built from analyze.py's per-library output.

Inputs : results/analysis/<DEVICE>_libraries.csv   (written by analyze.py; every library x placement)
         results/calibration/<DEVICE>.json          (measured ceilings, for % of peak)
         configs/library_notes.json                 (curated explanations; optional)
Outputs: results/analysis/library_report.csv        one row per (device, problem) with >= 2 libraries
         results/analysis/library_summary.csv       per (device, workload, precision, library): wins, ties, speed vs winner
         results/analysis/library_report.json       the same, structured (dashboard)
         results/analysis/library_report.md         readable tables, winners in bold, notes

Rules:
  - A library's entry for a problem is its BEST placement (placement is a separate choice).
  - Winner = highest metric (all metrics are higher-is-better rates).
  - tie = the winner's 95% CI overlaps the runner-up's: not a statistically real win.
  - Notes are only shown as written in library_notes.json; nothing is inferred automatically.
"""
from __future__ import annotations

import fnmatch
import glob
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("ANALYSIS_OUT", ROOT / "results" / "analysis"))
CAL = Path(os.environ.get("ANALYSIS_CAL", ROOT / "results" / "calibration"))
NOTES = Path(os.environ.get("LIBRARY_NOTES", ROOT / "configs" / "library_notes.json"))


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
    rows = []
    for (dev, wid, prob), g in df.groupby(["device", "workload_id", "problem"]):
        best = g.sort_values("metric", ascending=False).groupby("lib", as_index=False).first()
        if len(best) < 2:
            continue                                   # single-library problem: nothing to compare
        cal = _calibration(dev)
        best = best.sort_values("metric", ascending=False).reset_index(drop=True)
        w, ru = best.iloc[0], best.iloc[1]
        rec = {"device": dev, "workload_id": wid, "workload": w["workload"], "bound": w["bound"],
               "precision": w["precision"], "variant": w["variant"], "problem": prob,
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
    report, summ = build()
    if report.empty:
        print("no multi-library problems found under", OUT)
        return
    notes = _notes()
    report.to_csv(OUT / "library_report.csv", index=False)
    summ.to_csv(OUT / "library_summary.csv", index=False)
    payload = {"summary": summ.to_dict(orient="records"),
               "problems": json.loads(report.to_json(orient="records")),
               "notes": notes}
    (OUT / "library_report.json").write_text(json.dumps(payload, indent=1) + "\n")
    (OUT / "library_report.md").write_text(_md(report, summ, notes))
    print(f"library report: {len(report)} multi-library problems on {report['device'].nunique()} device(s), "
          f"{int(report['tie'].sum())} ties -> {OUT}/library_report.*")


if __name__ == "__main__":
    main()
