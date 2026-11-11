# Running the benchmarks on a new machine

This guide takes you from a freshly booked CloudLab machine to a finished, validated full sweep, and then to results on the dashboard. It is the same for CPU and GPU machines; GPU machines have one extra step (4). Every command is meant to be copied as is.

The flow in one line: **connect → get the code → (GPU: install the driver) → set up → start the campaign → check on it → copy the results back → analyse.**

## 1. Before you start

- Book a CloudLab node with the default **Ubuntu 22.04** image.
- Have your CloudLab ssh key, here `~/.ssh/cloudlab`.
- Find the machine's **device id**: a file name in [`configs/devices/`](../configs/devices/) (without `.json`), for example `INTEL_C6620`, `AMD_R6615` or `NVIDIA_V100`. If your machine type has no file yet, step 5 shows how to create one.

## 2. Connect

```bash
ssh -i ~/.ssh/cloudlab <user>@<node-hostname>
```

## 3. Get the code

```bash
git clone https://github.com/23110281/cpu-gpu ~/cpu-gpu
cd ~/cpu-gpu
```

> **Testing commits that are not pushed yet?** On your laptop, `git bundle create /tmp/cpu-gpu.bundle --all`, then copy the bundle over (`scp -i ~/.ssh/cloudlab /tmp/cpu-gpu.bundle <user>@<node>:/tmp/`). On the node, run `git clone /tmp/cpu-gpu.bundle ~/cpu-gpu` instead of the clone above.

## 4. GPU machines only: install the NVIDIA driver and CUDA

```bash
sudo bash scripts/install_cuda.sh
sudo reboot
```

Wait a minute, connect again (step 2), and `cd ~/cpu-gpu`. Check that `nvidia-smi` lists the GPU.

Older GPUs (V100, P100 and earlier) automatically get driver 580, the last driver that supports them.

## 5. Set up the machine

```bash
bash scripts/setup_node.sh
```

This takes about 5–10 minutes. It detects CPU or GPU by itself and installs:
- the compilers and libraries (OpenBLAS, BLIS, oneTBB, OpenSSL, OpenCV);
- the right PyTorch build (on GPUs: the CUDA build that matches the toolkit *and* has kernels for this GPU);
- read access to the power and frequency counters;
- the "performance" CPU frequency mode.

It ends with a short check (PyTorch version, GPU visible, power counters readable).

**New machine type, with no file in `configs/devices/`?**

```bash
python3 -m bench describe --id <NEW_ID> --write     # e.g. --id INTEL_C6620
```

This prints what the harness sees (sockets, cores, NUMA nodes, cache; or the GPU) and saves a suggested `configs/devices/<NEW_ID>.json`. Then add the vendor's datasheet numbers to it (peak bandwidth, peak FP32/FP64), following the format of `configs/devices/AMD_R6615.json` (CPU) or `configs/devices/NVIDIA_V100.json` (GPU). Commit the file.

## 6. Start the campaign

```bash
bash scripts/run_campaign.sh start <DEVICE>
```

That is all. The campaign runs **in the background**: you can close the ssh session and let your laptop sleep. It does these steps in order, and stops at the first one that fails:

| Step | What happens | Typical time |
|---|---|---|
| 1 | Records the machine and code version | seconds |
| 2 | Unit tests | ~1 min |
| 3 | Smoke test: every workload at its smallest size. Any error or wrong answer stops the campaign | 5–10 min |
| 4 | Calibration: measures this machine's real memory bandwidth and peak matrix-multiply speed (skipped if already done; `--recalibrate` redoes it) | 5–15 min |
| 5 | Full sweep: every workload, size and library, 3 independent trials, each result validated before it is timed | V100: ~1 h. CPU sockets: ~6–10 h (EPYC 7402P took ~8 h) |
| 6 | Summary: how many results are ok / skipped / failed | seconds |

Options:
- `--trials N`: number of trials (default 3).
- `--workloads B1,C3`: only some workloads.
- `--dry-run`: print the job list and exit.
- `--skip-tests`: skip the unit tests.

## 7. Check on it

```bash
bash scripts/run_campaign.sh status <DEVICE>
```

This shows:
- whether it is running, finished or stopped;
- the current step;
- jobs done out of the total, with a rough time left;
- how many results so far are `ok`, `oom` (too big for memory, skipped before running), `unsupported` (e.g. bf16 on hardware without bf16) or failed.

To watch it live: `tail -f results/logs/<DEVICE>.log`.

## 8. If it was interrupted

```bash
bash scripts/run_campaign.sh resume <DEVICE>
```

Resume continues the same campaign: finished jobs are skipped, and jobs that crashed are run again. To stop a running campaign on purpose: `bash scripts/run_campaign.sh stop <DEVICE>`.

## 9. Copy the results to your laptop

> **Do this before the reservation ends.** When a CloudLab experiment ends, the machine and everything on it are deleted. One full campaign was lost this way.

Run on your laptop, from the repo folder. Use `gpu` instead of `cpu` for GPU machines:

```bash
NODE=<user>@<node-hostname>; DEV=<DEVICE>; SIDE=cpu
SSH="ssh -i $HOME/.ssh/cloudlab"
mkdir -p results/data/$SIDE/$DEV results/data/sessions/${DEV}_$(date +%Y%m%d)/logs   # rsync does not create parent folders
rsync -av -e "$SSH" $NODE:cpu-gpu/results/data/$SIDE/$DEV/ results/data/$SIDE/$DEV/ </dev/null
rsync -av -e "$SSH" $NODE:cpu-gpu/results/calibration/ results/calibration/ </dev/null
rsync -av -e "$SSH" $NODE:cpu-gpu/results/logs/ results/data/sessions/${DEV}_$(date +%Y%m%d)/logs/ </dev/null
```

Then pack the per-job files (thousands of tiny `samples/` and `env/` files) into two files per machine: `samples.tar` (every raw timing, unchanged; the analysis reads it) and `machine.json` (software versions, machine state and per-placement thread settings: one record per machine, since they are identical across jobs; per-job ids and git commits are already in the CSV rows):

```bash
make pack
```

You can copy while the campaign is still going, as a backup. Copy, then `make pack` again; new files are merged into the existing packs.

## 10. Analyse and publish (on your laptop)

```bash
make site        # analysis tables + figures + dashboard -> open results/site/index.html
make readme      # refresh the "fastest library per machine" table in README.md
git add results/data results/calibration README.md && git commit -m "data: <DEVICE> campaign"
```

`make site` needs Python ≥ 3.11 with `requirements.txt` installed (`make venv` creates one; then use `make site PYTHON=.venv/bin/python`). After a push, GitHub rebuilds the public dashboard automatically.

## Troubleshooting

| What you see | Why | Fix |
|---|---|---|
| An ssh command hangs and never returns | A background job is holding the ssh channel | Add `< /dev/null` to remote commands; `run_campaign.sh` already detaches properly |
| `setup_node.sh` says "GPU present but its driver is not working" | Driver not installed yet, or no reboot after installing | Step 4, then reboot |
| Setup warns "torch wheel ... has no kernels for this GPU" or smoke shows "no kernel image is available" | Newest PyTorch builds dropped old GPUs (V100 needs the CUDA 12.6 build) | `setup_node.sh` tries older builds automatically; re-run it and check its log line |
| Smoke stops with `unsupported: ... extension build failed` | A native kernel did not compile | Read the compiler error in the log (it is printed); usually a missing package, so re-run `setup_node.sh` |
| `status` shows many `oom` rows | Those problem sizes do not fit in this machine's memory | Expected on small-memory GPUs; they are skipped before running, not errors |
| `bf16 ... unsupported` rows | The hardware has no native bf16 (e.g. V100, AMD Zen 2) | Expected: we never time emulated bf16 |
| A CPU library looks absurdly slow (single-core speed) | Its threads were pinned to one core | Already handled in the code (OpenMP OpenBLAS, widened affinity for OpenCV/TBB); report it if it reappears |
| The reservation is about to end | | Step 9 now, then extend the reservation or resume later on a new node |

More background: [`docs/methodology.md`](methodology.md) explains how every number is measured and validated.
