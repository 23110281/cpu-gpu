"""Build results/plots/index.html (the published dashboard).

Inputs
  results/index.html            template (JS placeholders replaced below)
  results/plots/<DEVICE>/<bench>/*.png   plot tree (built by `make plots`)
  results/data/{cpu,gpu}/<DEVICE>/...    raw CSVs (classify devices + data coverage)
  results/workloads.json        final workload suite + implementation status / TODOs
  results/raw/calibration/peaks.json     optional GH200 calibration

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
peaks_path = current_dir / "raw" / "calibration" / "peaks.json"

csv.field_size_limit(1 << 24)

# ── Plot tree ────────────────────────────────────────────────────────────────
tree = {}
if plots_dir.exists():
    for dev_dir in sorted(plots_dir.iterdir()):
        if dev_dir.is_dir():
            dev = dev_dir.name
            tree[dev] = {}
            for bench_dir in sorted(dev_dir.iterdir()):
                if bench_dir.is_dir():
                    plots = [p.name for p in sorted(bench_dir.glob("*.png"))]
                    if plots:
                        tree[dev][bench_dir.name] = plots

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

# Base limits for standard GPUs
gpu_limits = {
    "NVIDIA_GH200_480GB": {
        "compute": {"sms": 132, "fp32_cores_per_sm": 128, "fp64_cores_per_sm": 64, "tc_per_sm": 4, "tc_flops_per_clock": 948, "boost_mhz": 1980},
        "memory": {"bus_width_bits": 6144, "speed_gbps": 5.238}
    },
    "NVIDIA_A100": {
        "compute": {"sms": 108, "fp32_cores_per_sm": 64, "fp64_cores_per_sm": 32, "tc_per_sm": 4, "tc_flops_per_clock": 512, "boost_mhz": 1410},
        "memory": {"bus_width_bits": 5120, "speed_gbps": 2.43}
    },
    "NVIDIA_L40S": {
        "compute": {"sms": 142, "fp32_cores_per_sm": 128, "fp64_cores_per_sm": 2, "tc_per_sm": 4, "tc_flops_per_clock": 512, "boost_mhz": 2520},
        "memory": {"bus_width_bits": 384, "speed_gbps": 18.0}
    },
    "NVIDIA_A30": {
        "compute": {"sms": 56, "fp32_cores_per_sm": 64, "fp64_cores_per_sm": 32, "tc_per_sm": 4, "tc_flops_per_clock": 512, "boost_mhz": 1440},
        "memory": {"bus_width_bits": 3072, "speed_gbps": 2.43}
    },
    "Tesla_V100-SXM2-16GB": {
        "compute": {"sms": 80, "fp32_cores_per_sm": 64, "fp64_cores_per_sm": 32, "tc_per_sm": 8, "tc_flops_per_clock": 128, "boost_mhz": 1530},
        "memory": {"bus_width_bits": 4096, "speed_gbps": 1.75}
    },
    "Tesla_P100-PCIE-12GB": {
        "compute": {"sms": 56, "fp32_cores_per_sm": 64, "fp64_cores_per_sm": 32, "tc_per_sm": 0, "tc_flops_per_clock": 0, "boost_mhz": 1300},
        "memory": {"bus_width_bits": 3072, "speed_gbps": 1.43}
    }
}

# Calibrate GH200 specs from peaks.json if available
if peaks_path.exists():
    try:
        with open(peaks_path, "r") as f:
            peaks = json.load(f)
            hg = peaks.get("hopper_gpu", {})
            if hg:
                gpu_limits["NVIDIA_GH200_480GB"]["compute"]["sms"] = hg.get("sm_count", 132)
                gpu_limits["NVIDIA_GH200_480GB"]["compute"]["boost_mhz"] = int(hg.get("sm_clock_boost_ghz", 1.980) * 1000)
                gpu_limits["NVIDIA_GH200_480GB"]["memory"]["bus_width_bits"] = hg.get("hbm3_bus_width_bits", 6144)
    except Exception as e:
        print(f"Warning: Failed to parse peaks.json: {e}")


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
    known = {s: {d["id"] for d in manifest["devices"][s]} for s in ("cpu", "gpu")}
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


html = inject(html, "tree", tree)
html = inject(html, "gpuLimits", gpu_limits)
html = inject(html, "deviceClass", device_class)
html = inject(html, "workloadStatus", status)

plots_dir.mkdir(parents=True, exist_ok=True)
with open(plots_dir / "index.html", "w") as f:
    f.write(html)

if status:
    tot = sum(1 for _ in status["workloads"])
    print(f"Workload status: {tot} workloads tracked from {manifest_path.name}")
print("Generated results/plots/index.html successfully with updated taxonomy and plots.")
