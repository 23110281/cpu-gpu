"""Build results/plots/index.html (the published dashboard).

Inputs
  results/index.html            template (JS placeholders replaced below)
  results/plots/v2/<DEVICE>/summary.json + *.png   v2 device pages (results/plotting/v2/
                                analyze.py + dashboard_summary.py, run by `make plots`). Legacy
                                plot folders under results/plots/ are NOT shown: legacy data is a
                                placeholder until a device has v2 data.
  results/data/{v2,legacy}/{cpu,gpu}/<DEVICE>/...  raw CSVs (classify devices + data coverage)
  results/workloads.json        final workload suite + implementation status / TODOs

Data coverage is computed, not hand-edited. Per (workload, device) cell:
  'collected'   - v2 harness rows (results/data/v2/..., status ok, not smoke)
  'preliminary' - only frozen legacy data (results/data/legacy/...), valid
  'invalid'     - only legacy data flagged valid=false in workloads.json
  missing       - nothing
"""
import csv
import glob
import json
import os
import re
from pathlib import Path

current_dir = Path(__file__).parent
plots_dir = current_dir / "plots"
data_dir = current_dir / "data"
template_path = current_dir / "index.html"
manifest_path = current_dir / "workloads.json"

csv.field_size_limit(1 << 24)

# ── v2 device pages ──────────────────────────────────────────────────────────
pages = {}
v2_plots = plots_dir / "v2"
if v2_plots.exists():
    for dev_dir in sorted(v2_plots.iterdir()):
        f = dev_dir / "summary.json"
        if dev_dir.is_dir() and f.exists():
            try:
                page = json.loads(f.read_text())
            except ValueError as e:
                print(f"Warning: cannot read {f}: {e}")
                continue
            page["figures"] = [fig for fig in page.get("figures", []) if (dev_dir / fig["file"]).exists()]
            pages[dev_dir.name] = page

# ── Device classification (results/data/{legacy,v2}/cpu|gpu/<DEVICE>) ──────────
DATA_TREES = ("v2", "legacy")          # v2 = publication harness; legacy = frozen, preliminary
device_class = {}
for tree_name in DATA_TREES:
    for side in ("cpu", "gpu"):
        side_dir = data_dir / tree_name / side
        if side_dir.exists():
            for d in sorted(side_dir.iterdir()):
                if d.is_dir() and not d.name.startswith("smoke_"):
                    device_class.setdefault(d.name, side)

# ── Workload status matrix ───────────────────────────────────────────────────
_csv_cache = {}


def _read_rows(path):
    if path not in _csv_cache:
        try:
            with open(path, newline="") as f:
                _csv_cache[path] = list(csv.DictReader(f))
        except Exception as e:  # unreadable CSV counts as no data
            print(f"Warning: cannot read {path}: {e}")
            _csv_cache[path] = []
    return _csv_cache[path]


def _usable(row, v2=False):
    status = row.get("status")
    if v2:
        return status == "ok" and "smoke" not in (row.get("flag") or "")
    return status is None or "ok" in status or "l2-bound" in status


def _device_of(path, side):
    rel = Path(path).relative_to(data_dir).parts  # (tree, 'cpu'|'gpu', DEVICE, ...)
    if rel and rel[0] in DATA_TREES:
        rel = rel[1:]
    return rel[1] if len(rel) > 1 and rel[0] == side else None


def _source_hits(src, side):
    """Return {device: row_count} for one data source."""
    hits = {}
    v2 = src["glob"].startswith("v2/")
    for path in sorted(glob.glob(str(data_dir / src["glob"]))):
        rows = _read_rows(path)
        match = src.get("match", {})
        n = sum(1 for r in rows if _usable(r, v2) and all(r.get(k) in v for k, v in match.items()))
        if n:
            dev = src.get("device") or _device_of(path, side)
            if dev and dev not in src.get("exclude_devices", ()):
                hits[dev] = hits.get(dev, 0) + n
    return hits


def build_status(manifest):
    out = {k: manifest[k] for k in ("plan", "devices", "bounds", "infra_todo", "dropped", "system_axis")}
    # a device's older directory names (e.g. legacy "Tesla_V100-SXM2-16GB" for v2 "NVIDIA_V100")
    alias = {a: d["id"] for side in ("cpu", "gpu") for d in manifest["devices"][side] for a in d.get("aliases", [])}
    known = {s: {d["id"] for d in manifest["devices"][s]} | {a for d in manifest["devices"][s]
             for a in d.get("aliases", [])} for s in ("cpu", "gpu")}
    # Device dirs with data that the manifest does not list (e.g. a new node whose
    # dir is still named after its raw lscpu model) are surfaced, not dropped.
    out["other_devices"] = {
        s: sorted(d for d, c in device_class.items() if c == s and d not in known[s])
        for s in ("cpu", "gpu")
    }
    workloads = []
    for w in manifest["workloads"]:
        cells = {}
        for side in ("cpu", "gpu"):
            per_dev = {}
            sources = [{"glob": f"v2/{side}/*/{w['id']}_*.csv", "v2": True}] + w["data"].get(side, [])
            for src in sources:
                for dev, n in _source_hits(src, side).items():
                    dev = alias.get(dev, dev)
                    cell = per_dev.setdefault(dev, {"rows": 0, "valid_rows": 0, "v2_rows": 0, "notes": []})
                    cell["rows"] += n
                    if src.get("v2"):
                        cell["v2_rows"] += n
                    elif src.get("valid", True):
                        cell["valid_rows"] += n
                    elif src.get("note"):
                        cell["notes"].append(src["note"])
            for dev, c in per_dev.items():
                c["state"] = ("collected" if c["v2_rows"] else
                              "preliminary" if c["valid_rows"] else "invalid")
            cells[side] = per_dev
        entry = {k: v for k, v in w.items() if k != "data"}
        entry["cells"] = cells
        workloads.append(entry)
    out["workloads"] = workloads
    return out


status = {}
if manifest_path.exists():
    with open(manifest_path) as f:
        status = build_status(json.load(f))

# ── Render ───────────────────────────────────────────────────────────────────
with open(template_path, "r") as f:
    html = f.read()


def inject(src, name, value):
    pattern = re.compile(r"const %s = \{.*?\};" % re.escape(name))
    if not pattern.search(src):
        raise SystemExit(f"template placeholder 'const {name} = {{}};' not found in {template_path}")
    payload = json.dumps(value, separators=(",", ":")).replace("</", "<\\/")
    return pattern.sub(lambda _m: f"const {name} = {payload};", src, count=1)


html = inject(html, "pages", pages)
html = inject(html, "deviceClass", device_class)
html = inject(html, "workloadStatus", status)

plots_dir.mkdir(parents=True, exist_ok=True)
with open(plots_dir / "index.html", "w") as f:
    f.write(html)

if status:
    tot = sum(1 for _ in status["workloads"])
    print(f"Workload status: {tot} workloads tracked from {manifest_path.name}")
print(f"Generated results/plots/index.html: {len(pages)} v2 device page(s) ({', '.join(pages) or 'none'}).")
