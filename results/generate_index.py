import os
import json
from pathlib import Path
import re

current_dir = Path(__file__).parent
plots_dir = current_dir / "plots"
template_path = current_dir / "index.html"
peaks_path = current_dir / "raw" / "calibration" / "peaks.json"

tree = {}
if plots_dir.exists():
    for gpu_dir in sorted(plots_dir.iterdir()):
        if gpu_dir.is_dir():
            gpu = gpu_dir.name
            tree[gpu] = {}
            for bench_dir in sorted(gpu_dir.iterdir()):
                if bench_dir.is_dir():
                    bench = bench_dir.name
                    plots = []
                    for plot_file in sorted(bench_dir.glob("*.png")):
                        plots.append(plot_file.name)
                    if plots:
                        tree[gpu][bench] = plots

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

with open(template_path, "r") as f:
    original_html = f.read()

new_html = re.sub(r'const tree = \{.*?\};', f'const tree = {json.dumps(tree)};', original_html)
new_html = re.sub(r'const gpuLimits = \{.*?\};', f'const gpuLimits = {json.dumps(gpu_limits)};', new_html)

plots_dir.mkdir(parents=True, exist_ok=True)
with open(plots_dir / "index.html", "w") as f:
    f.write(new_html)

print("Generated results/plots/index.html successfully with updated taxonomy and plots.")
