"""Build results/site/index.html (the published dashboard).

Inputs
  analysis/dashboard.html                 template (JS placeholders replaced below)
  results/site/devices/<DEVICE>/summary.json + *.png   one page per device with our data
                                          (analysis/analyze.py + dashboard_summary.py, `make site`)
  results/data/<cpu|gpu>/<DEVICE>/        our CSVs (data coverage)
  legacy/data/<cpu|gpu>/<DEVICE>/         frozen legacy CSVs (placeholder 'prelim' cells only)
  studies/gh200_c2c/data/                 GH200 study files referenced by configs/workloads.json
  configs/workloads.json                  workload suite, implementation status, TODOs, legacy sources

Data coverage is computed, not hand-edited. Per (workload, device) cell:
  'collected'   - our rows (results/data/..., status ok, not smoke)
  'preliminary' - only frozen legacy data, valid
  'invalid'     - only legacy data flagged valid=false in workloads.json
  missing       - nothing
Legacy plots are never shown: legacy data is a placeholder until a device has our data.
"""
import csv
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
site_dir = ROOT / "results" / "site"
data_dir = ROOT / "results" / "data"
legacy_dir = ROOT / "legacy" / "data"
template_path = ROOT / "analysis" / "dashboard.html"
manifest_path = ROOT / "configs" / "workloads.json"

csv.field_size_limit(1 << 24)

# ── device pages ─────────────────────────────────────────────────────────────
pages = {}
dev_root = site_dir / "devices"
if dev_root.exists():
    for dev_dir in sorted(dev_root.iterdir()):
        f = dev_dir / "summary.json"
        if dev_dir.is_dir() and f.exists():
            try:
                page = json.loads(f.read_text())
            except ValueError as e:
                print(f"Warning: cannot read {f}: {e}")
                continue
            page["figures"] = [fig for fig in page.get("figures", []) if (dev_dir / fig["file"]).exists()]
            pages[dev_dir.name] = page

library = {}
lib_json = ROOT / "results" / "analysis" / "library_report.json"
if lib_json.exists():
    try:
        library = json.loads(lib_json.read_text())
    except ValueError as e:
        print(f"Warning: cannot read {lib_json}: {e}")

# ── device classification (our data first, then legacy) ─────────────────────
device_class = {}
for root in (data_dir, legacy_dir):
    for side in ("cpu", "gpu"):
        side_dir = root / side
        if side_dir.exists():
            for d in sorted(side_dir.iterdir()):
                if d.is_dir() and not d.name.startswith("smoke_"):
                    device_class.setdefault(d.name, side)

# ── workload status matrix ───────────────────────────────────────────────────
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


def _usable(row, ours=False):
    status = row.get("status")
    if ours:
        return status == "ok" and "smoke" not in (row.get("flag") or "")
    return status is None or "ok" in status or "l2-bound" in status


def _device_of(path, side):
    """.../<cpu|gpu>/<DEVICE>/... -> DEVICE (path relative to the repo root)."""
    parts = Path(path).relative_to(ROOT).parts
    for i, p in enumerate(parts[:-2]):
        if p == side:
            return parts[i + 1]
    return None


def _source_hits(src, side):
    """Return {device: row_count} for one data source (glob relative to the repo root)."""
    hits = {}
    for path in sorted(glob.glob(str(ROOT / src["glob"]))):
        rows = _read_rows(path)
        match = src.get("match", {})
        n = sum(1 for r in rows if _usable(r, src.get("ours", False)) and all(r.get(k) in v for k, v in match.items()))
        if n:
            dev = src.get("device") or _device_of(path, side)
            if dev and dev not in src.get("exclude_devices", ()):
                hits[dev] = hits.get(dev, 0) + n
    return hits


def build_status(manifest):
    out = {k: manifest[k] for k in ("plan", "devices", "bounds", "infra_todo", "dropped", "system_axis")}
    # a device's older directory names (e.g. legacy "Tesla_V100-SXM2-16GB" for "NVIDIA_V100")
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
            sources = [{"glob": f"results/data/{side}/*/{w['id']}_*.csv", "ours": True}] + w["data"].get(side, [])
            for src in sources:
                for dev, n in _source_hits(src, side).items():
                    dev = alias.get(dev, dev)
                    cell = per_dev.setdefault(dev, {"rows": 0, "valid_rows": 0, "our_rows": 0, "notes": []})
                    cell["rows"] += n
                    if src.get("ours"):
                        cell["our_rows"] += n
                    elif src.get("valid", True):
                        cell["valid_rows"] += n
                    elif src.get("note"):
                        cell["notes"].append(src["note"])
            for dev, c in per_dev.items():
                c["state"] = ("collected" if c["our_rows"] else
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

# ── render ───────────────────────────────────────────────────────────────────
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
html = inject(html, "libraryReport", library)

site_dir.mkdir(parents=True, exist_ok=True)
with open(site_dir / "index.html", "w") as f:
    f.write(html)

if status:
    print(f"Workload status: {len(status['workloads'])} workloads tracked from {manifest_path.name}")
print(f"Generated results/site/index.html: {len(pages)} device page(s) ({', '.join(pages) or 'none'}), "
      f"library report: {len(library.get('problems', []))} multi-library problems.")
