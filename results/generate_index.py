import os
import json
from pathlib import Path
import re

current_dir = Path(__file__).parent
plots_dir = current_dir / "plots"
template_path = current_dir / "index.html"

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

with open(template_path, "r") as f:
    original_html = f.read()

new_html = re.sub(r'const tree = \{.*?\};', f'const tree = {json.dumps(tree)};', original_html)

plots_dir.mkdir(parents=True, exist_ok=True)
with open(plots_dir / "index.html", "w") as f:
    f.write(new_html)

print("Generated results/plots/index.html successfully.")
