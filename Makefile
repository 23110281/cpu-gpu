# CPU vs GPU benchmarks -- primary targets (see docs/RUNNING.md for the full node workflow).
#
#   make setup                     install everything this node needs (CPU or GPU, auto-detected)
#   make test                      harness unit tests
#   make smoke                     every workload, smallest sizes, short timing (CPU or GPU auto)
#   make calibrate DEVICE=<id>     measure the device's ceilings (bandwidth, peak GEMM)
#   make sweep DEVICE=<id>         full collection, 3 trials (foreground)
#   make sweep_dry DEVICE=<id>     show the job list without running anything
#   make campaign DEVICE=<id>      test -> smoke -> calibrate -> sweep, detached (scripts/run_campaign.sh)
#   make analysis                  tables + figures from results/data (results/analysis, results/site)
#   make site                      analysis + the dashboard (results/site/index.html)
#   make readme                    refresh the auto-generated library table in README.md
#   make list                      registered workloads
#   make clean                     remove generated output
#
# <id> is a file name in configs/devices/ (without .json). Native cross-check: make -C crosscheck;
# frozen legacy benches: make -C legacy; GH200 study: make -C studies/gh200_c2c.

PYTHON ?= python3
BENCH  := PYTHONPATH=. $(PYTHON) -m bench
DEVICE ?=
TRIALS ?= 3
need_device = @test -n "$(DEVICE)" || { echo "usage: make $@ DEVICE=<id in configs/devices/>"; exit 2; }
# GPU if a working NVIDIA driver is present, else CPU (override: make smoke SMOKE_DEVICE=cpu)
SMOKE_DEVICE ?= $(shell nvidia-smi -L >/dev/null 2>&1 && echo cuda || echo cpu)

.PHONY: setup test smoke calibrate sweep sweep_dry campaign analysis site readme list venv clean

setup:
	bash scripts/setup_node.sh

test:
	$(PYTHON) -m pytest -q tests

smoke:
	$(BENCH) smoke --device $(SMOKE_DEVICE)

calibrate:
	$(need_device)
	$(PYTHON) scripts/calibrate.py --device-config configs/devices/$(DEVICE).json

sweep:
	$(need_device)
	$(BENCH) sweep --device-config configs/devices/$(DEVICE).json --trials $(TRIALS)

sweep_dry:
	$(need_device)
	$(BENCH) sweep --device-config configs/devices/$(DEVICE).json --trials $(TRIALS) --dry-run

campaign:
	$(need_device)
	bash scripts/run_campaign.sh start $(DEVICE)

analysis:
	SOURCE_DATE_EPOCH=0 $(PYTHON) analysis/analyze.py
	$(PYTHON) analysis/library_report.py
	SOURCE_DATE_EPOCH=0 $(PYTHON) analysis/dashboard_summary.py

site: analysis
	$(PYTHON) analysis/build_dashboard.py

readme: analysis
	$(PYTHON) analysis/library_report.py --readme README.md

list:
	$(BENCH) list

# Analysis/dashboard Python environment (pinned; used by CI): make venv && make site PYTHON=.venv/bin/python
venv: requirements.txt
	@if [ ! -d ".venv" ]; then uv venv .venv; fi
	uv pip install --python .venv/bin/python -r requirements.txt

clean:
	rm -rf results/site results/data/smoke
	find results/analysis -maxdepth 1 -type f \( -name '*.csv' -o -name '*.json' -o -name '*.md' \) -delete
