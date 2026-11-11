#!/usr/bin/env python3
"""Build results/site/index.html from analysis/dashboard.html (a single static page).

Injected into the template:
  pages     results/site/devices/<DEV>/summary.json (dashboard_summary.py): machine and workload pages
  progress  the Project progress matrix: 16 workloads x core machines, each cell one of
              final   -- our final scripts collected validated rows (results/data/<cpu|gpu>/<DEV>/<ID>_*.csv)
              earlier -- no final data, but a correct earlier-script result (configs/legacy.json)
              bad     -- ran, but the earlier script or collection method was wrong (red TODO)
              todo    -- never run (white TODO)
  compare   results/analysis/compare.json (compare.py): data for the CPU vs GPU page
  library   results/analysis/library_report.json (library_report.py)
  todo      configs/workloads.json "todo"
"""
from __future__ import annotations

import csv
import glob
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import legacy  # noqa: E402

SITE = ROOT / "results" / "site"
TEMPLATE = HERE / "dashboard.html"
BOUND_TITLE = {"B": "Limited by memory speed", "C": "Limited by compute", "S": "Limited by synchronization",
               "A": "Full application"}


def _json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def final_rows(cls: str, dev: str, wid: str) -> int:
    n = 0
    for f in glob.glob(str(ROOT / "results" / "data" / cls / dev / f"{wid}_*.csv")):
        with open(f, newline="") as fh:
            n += sum(1 for r in csv.DictReader(fh) if r.get("status") == "ok" and "smoke" not in (r.get("flag") or ""))
    return n


def progress(manifest: dict, pages: dict) -> dict:
    states = legacy.cell_states()
    cols = {side: [d for d in manifest["devices"][side] if d.get("core")] for side in ("cpu", "gpu")}
    groups = []
    for b in ("B", "C", "S", "A"):
        rows = []
        for w in [w for w in manifest["workloads"] if w["bound"] == b]:
            cells = {}
            for side in ("cpu", "gpu"):
                for d in cols[side]:
                    if final_rows(side, d["id"], w["id"]):
                        cells[d["id"]] = {"s": "final", "why": "Collected with the final scripts."}
                    elif w["id"] in states.get(d["id"], {}):
                        st, why = states[d["id"]][w["id"]]
                        cells[d["id"]] = {"s": st, "why": why}
                    else:
                        cells[d["id"]] = {"s": "todo", "why": "Not run on this machine yet."}
            rows.append({"id": w["id"], "name": w.get("plain", w["name"]), "cells": cells})
        groups.append({"bound": b, "title": BOUND_TITLE[b], "rows": rows})
    ws = manifest["workloads"]
    gpu_done = sum(w["gpu"]["impl"] == "done" for w in ws)
    core_ids = {d["id"] for side in cols.values() for d in side}
    core_final = [i for i in core_ids if pages.get(i, {}).get("category") == "final"]
    core_earlier = [i for i in core_ids if pages.get(i, {}).get("category") == "earlier"]
    others = [i for i, p in pages.items() if i not in core_ids]
    counts = {b: sum(w["bound"] == b for w in ws) for b in "BCSA"}
    cards = [
        {"label": "Final workloads", "value": str(len(ws)),
         "sub": f"{counts['B']} memory · {counts['C']} compute · {counts['S']} synchronization · {counts['A']} application"},
        {"label": "CPU implementations", "value": f"{sum(w['cpu']['impl'] == 'done' for w in ws)}/{len(ws)}", "sub": "done"},
        {"label": "GPU implementations", "value": f"{gpu_done}/{len(ws)}",
         "sub": (f"{len(ws) - gpu_done} awaiting bf16 checks on newer GPUs" if gpu_done < len(ws) else "done")},
        {"label": "Machines benchmarked", "value": str(len(core_final) + len(core_earlier)),
         "sub": f"{len(core_final)} with final scripts · {len(core_earlier)} with earlier scripts"
                + (f" · +{len(others)} more machines with earlier-script data" if others else "")},
    ]
    def node(dev):
        spec = _json(ROOT / "configs" / "devices" / f"{dev}.json").get("display", {}).get("spec", "")
        hit = [x for x in spec.split(" · ") if "CloudLab" in x or "superchip" in x]
        return hit[0].replace("GH200 Grace Hopper superchip", "GH200 superchip") if hit else ""
    colinfo = {side: [{"id": d["id"], "name": d["short"], "node": node(d["id"])} for d in cols[side]] for side in cols}
    return {"cards": cards, "groups": groups, "columns": colinfo}


def nav(manifest: dict, pages: dict) -> dict:
    out = {}
    for side in ("cpu", "gpu"):
        ids = [d["id"] for d in manifest["devices"][side] if d["id"] in pages]
        ids += [i for i, p in pages.items() if p["cls"] == side and i not in ids]
        base = list(ids)
        ids.sort(key=lambda i: (pages[i]["category"] != "final", not pages[i].get("core"), base.index(i)))
        out[side] = [{"id": i, "name": pages[i]["name"], "category": pages[i]["category"]} for i in ids]
    return out


def inject(src: str, name: str, value) -> str:
    pattern = re.compile(r"const %s = \{.*?\};" % re.escape(name))
    if not pattern.search(src):
        raise SystemExit(f"template placeholder 'const {name} = {{}};' not found in {TEMPLATE}")
    payload = json.dumps(value, separators=(",", ":")).replace("</", "<\\/")
    return pattern.sub(lambda _m: f"const {name} = {payload};", src, count=1)


def main():
    pages = {}
    for f in sorted((SITE / "devices").glob("*/summary.json")):
        s = json.loads(f.read_text())
        pages[s["device"]] = s
    manifest = _json(ROOT / "configs" / "workloads.json")
    html = TEMPLATE.read_text()
    html = inject(html, "pages", pages)
    html = inject(html, "navData", nav(manifest, pages))
    html = inject(html, "progressData", progress(manifest, pages))
    html = inject(html, "compareData", _json(ROOT / "results" / "analysis" / "compare.json"))
    html = inject(html, "libraryReport", _json(ROOT / "results" / "analysis" / "library_report.json"))
    html = inject(html, "todoData", {"groups": manifest.get("todo", [])})
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "index.html").write_text(html)
    print(f"Generated results/site/index.html: {len(pages)} machine pages "
          f"({sum(p['category'] == 'final' for p in pages.values())} final, "
          f"{sum(p['category'] == 'earlier' for p in pages.values())} earlier scripts).")


if __name__ == "__main__":
    main()
