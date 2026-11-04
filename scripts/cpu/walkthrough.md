# CloudLab CPU Benchmarking Walkthrough

This guide details the end-to-end execution workflow for running CPU benchmarks (Dense GEMM, Dense GEMV & Sparse SpMV, Vector SAXPY, Sparse SpGEMM, Monte Carlo Option Pricing, Cryptography SHA-256/SHA-512, PCA, and K-Means Clustering) on CloudLab bare-metal nodes, covering both **Intel Xeon** and **AMD EPYC** architectures.


---

## 1. Initial Access & File Synchronization

### Step 1.1: Connect to CloudLab Node
Establish an SSH connection from your local terminal to your allocated CloudLab node:
```bash
ssh Kushal17@<cloudlab-node-ip>
```

### Step 1.2: Prepare Target Directory on CloudLab
Ensure the base working directory exists on the CloudLab node:
```bash
mkdir -p ~/cpu-gpu
```

### Step 1.3: Copy Code and Build Scripts (From Local Machine)
Run this command from your local project root (`cpu-gpu/`) to transfer the source and build files:
```bash
scp -r src scripts tests Makefile Kushal17@<cloudlab-node-ip>:~/cpu-gpu/
```

---

## 2. Configure Autonomous GitHub Backup (One-Time Setup)

### Step 2.1: Store Backup Personal Access Token (PAT)
Save your GitHub PAT to an unversioned file in your CloudLab home directory to allow the background runners to automatically push completed CSVs to your personal backup repository (`cpu-gpu-proj-course-backup`):
```bash
echo "PAT" > ~/.git_backup_token
chmod 600 ~/.git_backup_token
```

---

## 3. Node Provisioning & Benchmark Execution

---

### Path A: AMD EPYC Nodes

#### Step A.1: Run AMD Provisioning Script
Install OpenBLAS, OpenMP dependencies, and configure hardware monitoring modules:
```bash
cd ~/cpu-gpu
bash scripts/cpu/setup_cloudlab_epyc.sh
```

#### Step A.2: Verify Telemetry & Active Socket Power
Run the telemetry test using `sudo` to ensure kernel MSR access is active and it can read the power consumption data correctly:
```bash
sudo make test_telemetry
```


#### Step A.3: Launch Complete Benchmark Suite in Background
Start all 8 benchmark sweeps detached via `sudo` (logs to `results/logs/suite_epyc_latest.log`):
```bash
sudo make amd_suite_bg
```


#### Step A.4: Monitor Real-Time Progress
Stream the live benchmark throughput, frequencies, and power measurements:
```bash
tail -f results/logs/suite_epyc_latest.log
```

---

### Path B: Intel Xeon Nodes

#### Step B.1: Run Intel Xeon Provisioning Script
Install Intel oneAPI Base Toolkit (oneMKL) and configure Linux powercap drivers:
```bash
cd ~/cpu-gpu
bash scripts/cpu/setup_cloudlab_xeon.sh
```

#### Step B.2: Initialize Intel oneAPI Environment
Source the oneAPI environment variables to put MKL compilers and libraries in your path:
```bash
source /opt/intel/oneapi/setvars.sh
```

#### Step B.3: Verify Telemetry & Active Socket Power
Test Intel RAPL powercap access:
```bash
make test_telemetry
```


#### Step B.4: Launch Complete Benchmark Suite in Background
Start all 8 benchmark sweeps detached in the background (logs to `results/logs/suite_latest.log`):
```bash
make intel_suite_bg
```


#### Step B.5: Monitor Real-Time Progress
Stream the live benchmark throughput, frequencies, and power measurements:
```bash
tail -f results/logs/suite_latest.log
```

---

### Path C: PyTorch Benchmark Suite (AMD & Intel Nodes)

The PyTorch benchmarks port all 4 algebraic workloads (Dense GEMM, Dense GEMV & Sparse SpMV, Vector SAXPY, Sparse SpGEMM) into PyTorch with full CPU thread scaling, hardware telemetry (RAPL/MSR), and identical CSV schemas.

#### Step C.1: Verify PyTorch Runtime
Confirm that PyTorch CPU runtime is installed and operational:
```bash
python3 -c "import torch; print('PyTorch Version:', torch.__version__)"
```

#### Step C.2: Quick Smoke Test (Optional)
Run a quick sanity test on small matrix/vector sizes:
```bash
make run_torch_gemm
```

#### Step C.3: Launch Complete PyTorch Suite in Background
Start all 4 PyTorch benchmark sweeps detached in the background:

- **On AMD EPYC Nodes** (requires sudo for AMD MSR socket energy reads):
```bash
sudo make torch_amd_suite_bg
```

- **On Intel Xeon Nodes**:
```bash
make torch_intel_suite_bg
```

> **Safe Disconnection**: You can safely close your laptop or disconnect your SSH session immediately after launching. The benchmarks run completely detached via `nohup` with `SIGHUP` immunity.

#### Step C.4: Monitor Real-Time Progress
Stream the live PyTorch benchmark throughput, frequencies, and power measurements:
```bash
tail -f results/logs/suite_torch_latest.log
```

Upon completion, all 4 PyTorch sweeps are timestamped into `results/data/cpu/<CPU_MODEL>/` and automatically committed to your backup GitHub repository.


---

## 4. Verification & Results Retrieval

### Step 4.1: Verify Offsite GitHub Backup
Upon benchmark completion, the runner automatically commits and pushes this machine's CSV data. Verify your files online at:
```text
https://github.com/KushalMehta17/cpu-gpu-proj-course-backup/tree/main/data/
```

### Step 4.2: Direct Local Retrieval (Alternative via SCP)
If copying data directly to your laptop via SCP, ensure read permissions on CloudLab:
```bash
sudo chmod -R a+r results/data
```
Then run this from your local machine to copy directly into your local folder:
```bash
scp -r Kushal17@<cloudlab-node-ip>:~/cpu-gpu/results/data/cpu/* "/home/kushal/cpu gpu project/"
```

### Step 4.3: Generate Local Visualization Plots
Once CSVs are on your laptop, run the visualization suite to generate all figures:
```bash
python results/plotting/cpu/generate_cpu_gemm_plots.py && \
python results/plotting/cpu/generate_cpu_gemv_plots.py && \
python results/plotting/cpu/generate_cpu_saxpy_plots.py && \
python results/plotting/cpu/generate_cpu_spgemm_plots.py
```
Generated plots are saved into `results/plots/<CPU_MODEL>/`.
