# Unified CPU + GPU benchmark Makefile
#   GPU (CUDA) harnesses : src/gpu/, GH200-only C2C benches: src/gpu/gh200/
#   CPU (C/OpenMP/BLAS)  : src/cpu/   (see '# ── CPU Benchmarks' section below)
#   PyTorch (cpu|cuda)   : src/torch/
#   Shared headers       : src/common/
#
# GPU section — cuBLAS GEMM benchmark, two build flavours
#
# Self-contained: NOT wired into the top-level dgemm Makefile (that build is
# CPU-only, clang, -Werror -Wpedantic C11 and must stay GPU-free).
#
# Targets (nvcc / CUDA C++ build, default):
#   make            -> bin/gpu_gemm_bench_cu   (nvcc, CUDA C++17, sm_80)
#   make run        quick smoke: nvcc build, 1 GPU, size=256..4096
#   make sweep      full sweep via scripts/gpu/run_sweep.sh (which now calls the cu binary)
#
# Targets (gcc / C11 build, explicit fallback):
#   make bench_gcc  -> bin/gpu_gemm_bench      (gcc, C11)
#   make run_gcc    quick smoke: gcc build, 1 GPU, small sizes
#   make sweep_gcc  full sweep via scripts/gpu/run_sweep_gcc.sh
#
# Combined:
#   make both       builds both binaries
#   make clean      removes bin/

CUDA_HOME ?= /usr/local/cuda
CC        ?= gcc
NVCC      ?= $(CUDA_HOME)/bin/nvcc

CFLAGS  := -std=c11 -O3 -march=native -Wall -Wextra -fopenmp \
           -Wno-deprecated-declarations \
           -I src/common -I src/gpu -I $(CUDA_HOME)/include
# stubs/ supplies link-time libnvidia-ml.so; the real lib loads at runtime.
LDFLAGS := -L $(CUDA_HOME)/lib64 -L $(CUDA_HOME)/lib64/stubs \
           -Wl,-rpath,$(CUDA_HOME)/lib64
LDLIBS  := -lcublas -lcublasLt -lcusparse -lcudart -lnvidia-ml -lpthread -lm -fopenmp

# ── nvcc / CUDA C++17 (default) ──────────────────────────────────────────────
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),aarch64)
ARCH           ?= sm_90
CUSPARSELT_LIB ?= /usr/lib/aarch64-linux-gnu/libcusparseLt/12
NVCC_ARCH_LDFLAGS := -L /usr/lib/aarch64-linux-gnu
NVJITLINK_LIB  := -lnvJitLink
else
ARCH           ?= sm_80
CUSPARSELT_LIB ?= /usr/lib/x86_64-linux-gnu/libcusparseLt/12
NVCC_ARCH_LDFLAGS :=
NVJITLINK_LIB  :=
endif

CUSPARSELT_INC ?= /usr/include/libcusparseLt/12

# Singularity override: download and use local cuSPARSELt
ifeq ($(HOST),SINGULARITY)
CUSPARSELT_DIR  := $(CURDIR)/third_party/cusparselt/libcusparse_lt-linux-x86_64-0.8.1.1_cuda12-archive
CUSPARSELT_INC  := $(CUSPARSELT_DIR)/include
CUSPARSELT_LIB  := $(CUSPARSELT_DIR)/lib
CUSPARSELT_DEPS := $(CUSPARSELT_INC)/cusparseLt.h
else
CUSPARSELT_DEPS :=
endif
NVCC_FLAGS   := -std=c++17 -O3 -arch=$(ARCH) \
                -Xcompiler "-Wall -Wextra -fopenmp -Wno-deprecated-declarations -mcpu=native" \
                -I src/common -I src/gpu -I src/gpu/gh200 -I $(CUDA_HOME)/include -I $(CUSPARSELT_INC)
NVCC_LDFLAGS := -L $(CUDA_HOME)/lib64 -L $(CUDA_HOME)/lib64/stubs -L $(CUSPARSELT_LIB) $(NVCC_ARCH_LDFLAGS) \
                -Xlinker -rpath,$(CUDA_HOME)/lib64 -Xlinker -rpath,$(CUSPARSELT_LIB)
NVCC_LDLIBS  := -lcublas -lcublasLt -lcusparse -lcusparseLt $(NVJITLINK_LIB) -lnvidia-ml -lpthread -lm \
                -Xcompiler -fopenmp

BIN := bin/gpu_gemm_bench
SRC := src/gpu/gpu_gemm_bench.cu

# ── gcc / C11 (legacy fallback) ──────────────────────────────────────────────
CFLAGS  := -std=c11 -O3 -march=native -Wall -Wextra -fopenmp \
           -Wno-deprecated-declarations \
           -I src/common -I src/gpu -I $(CUDA_HOME)/include
LDFLAGS := -L $(CUDA_HOME)/lib64 -L $(CUDA_HOME)/lib64/stubs \
           -Wl,-rpath,$(CUDA_HOME)/lib64
LDLIBS  := -lcublas -lcublasLt -lcusparse -lcudart -lnvidia-ml -lpthread -lm -fopenmp

BIN_GCC := bin/gpu_gemm_bench_gcc
SRC_GCC := src/gpu/gpu_gemm_bench.c
HDR_GCC := src/common/bf16_cvt.h

.PHONY: all bench_gcc both run run_gcc sweep sweep_gcc gemv run_gemv sweep_gemv saxpy run_saxpy sweep_saxpy spgemm run_spgemm sweep_spgemm venv plots clean
all: $(BIN) gemv saxpy spgemm

both: $(BIN) $(BIN_GCC)

$(BIN): $(SRC) $(CUSPARSELT_DEPS)
	@mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LDFLAGS) -o $@ $(SRC) $(NVCC_LDLIBS)
	@echo "built $@"

# Auto-download cuSPARSELt when on Singularity
$(CUSPARSELT_DIR)/include/cusparseLt.h:
	@mkdir -p third_party/cusparselt
	@echo "Downloading cuSPARSELt to third_party/cusparselt..."
	curl -sSL -o third_party/cusparselt/cusparselt.tar.xz https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-x86_64/libcusparse_lt-linux-x86_64-0.8.1.1_cuda12-archive.tar.xz
	tar -xf third_party/cusparselt/cusparselt.tar.xz -C third_party/cusparselt/

bench_gcc: $(BIN_GCC)
$(BIN_GCC): $(SRC_GCC) $(HDR_GCC)
	@mkdir -p bin
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(SRC_GCC) $(LDLIBS)
	@echo "built $@"

run: $(BIN)
	./$(BIN) --sizes 256,512,1024,2048,4096 --gpus 1 --validate --iters 20

run_gcc: $(BIN_GCC)
	./$(BIN_GCC) --sizes 512,1024,2048,4096 --gpus 1 --validate --iters 20

sweep: $(BIN)
	./scripts/gpu/run_sweep.sh

sweep_gcc: $(BIN_GCC)
	./scripts/gpu/run_sweep_gcc.sh

# ── GEMV/SpMV benchmark (dense cuBLAS gemv + sparse cuSPARSE SpMV) ─────────
# No cuSPARSELt dependency -- regular cuSPARSE only, see src/gpu/gpu_gemv_bench.cu
# header comment for why.
GEMV_BIN := bin/gpu_gemv_bench
GEMV_SRC := src/gpu/gpu_gemv_bench.cu
GEMV_LDLIBS := -lcublas -lcublasLt -lcusparse $(NVJITLINK_LIB) -lnvidia-ml -lpthread -lm

gemv: $(GEMV_BIN)
$(GEMV_BIN): $(GEMV_SRC) src/common/gemv_host.h src/common/bf16_cvt.h
	@mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LDFLAGS) -o $@ $(GEMV_SRC) $(GEMV_LDLIBS)
	@echo "built $@"

run_gemv: $(GEMV_BIN)
	./$(GEMV_BIN) --sizes 256,512,1024,2048,4096 --gpus 1 --validate --iters 20

sweep_gemv: $(GEMV_BIN)
	./scripts/gpu/run_sweep_gemv.sh

# ── SAXPY benchmark (cuBLAS Level-1 AXPY) ──────────────────────────────────
SAXPY_BIN := bin/gpu_saxpy_bench
SAXPY_SRC := src/gpu/gpu_saxpy_bench.cu
SAXPY_LDLIBS := -lcublas -lnvidia-ml -lpthread -lm

saxpy: $(SAXPY_BIN)
$(SAXPY_BIN): $(SAXPY_SRC) src/common/bf16_cvt.h
	@mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LDFLAGS) -o $@ $(SAXPY_SRC) $(SAXPY_LDLIBS)
	@echo "built $@"

run_saxpy: $(SAXPY_BIN)
	./$(SAXPY_BIN) --sizes 33554432,67108864,134217728 --gpus 1 --validate --iters 20

sweep_saxpy: $(SAXPY_BIN)
	./scripts/gpu/run_sweep_saxpy.sh

# ── SpGEMM benchmark (cuSPARSE Generic API) ────────────────────────────────
SPGEMM_BIN := bin/gpu_spgemm_bench
SPGEMM_SRC := src/gpu/gpu_spgemm_bench.cu
SPGEMM_LDLIBS := -lcusparse $(NVJITLINK_LIB) -lnvidia-ml -lpthread -lm

spgemm: $(SPGEMM_BIN)
$(SPGEMM_BIN): $(SPGEMM_SRC) src/common/gemv_host.h src/common/bf16_cvt.h
	@mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LDFLAGS) -o $@ $(SPGEMM_SRC) $(SPGEMM_LDLIBS)
	@echo "built $@"

run_spgemm: $(SPGEMM_BIN)
	./$(SPGEMM_BIN) --sizes 1024,2048,4096 --gpus 1 --validate --iters 10

sweep_spgemm: $(SPGEMM_BIN)
	./scripts/gpu/run_sweep_spgemm.sh

# ── Cycle 1: NVLink-C2C Streaming Benchmark (SAXPY + GEMV across 4 NUMA Modes)
C2C_STREAM_BIN := bin/gpu_c2c_streaming_bench
C2C_STREAM_SRC := src/gpu/gh200/gpu_c2c_streaming_bench.cu
C2C_STREAM_LDLIBS := -lcublas -lopenblas -lnvidia-ml -lpthread -lm -Xcompiler -fopenmp

c2c_streaming: $(C2C_STREAM_BIN)
$(C2C_STREAM_BIN): $(C2C_STREAM_SRC) src/gpu/gh200/bench_harness.h src/gpu/gpu_device_specs.h
	@mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LDFLAGS) -o $@ $(C2C_STREAM_SRC) $(C2C_STREAM_LDLIBS)
	@echo "built $@"

# ── Cycle 2: Arithmetic Compute Benchmark (GEMM + Monte Carlo across 4 NUMA Modes)
C2C_COMPUTE_BIN := bin/gpu_c2c_compute_bench
C2C_COMPUTE_SRC := src/gpu/gh200/gpu_c2c_compute_bench.cu
C2C_COMPUTE_LDLIBS := -lcublas -lopenblas -lnvidia-ml -lpthread -lm -Xcompiler -fopenmp

c2c_compute: $(C2C_COMPUTE_BIN)
$(C2C_COMPUTE_BIN): $(C2C_COMPUTE_SRC) src/gpu/gh200/bench_harness.h src/gpu/gpu_device_specs.h
	@mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LDFLAGS) -o $@ $(C2C_COMPUTE_SRC) $(C2C_COMPUTE_LDLIBS)
	@echo "built $@"

# ── Cycle 2B: cuFFT vs FFTW3 Benchmark (1D & 2D FFT)
C2C_FFT_BIN := bin/gpu_c2c_fft_bench
C2C_FFT_SRC := src/gpu/gh200/gpu_c2c_fft_bench.cu
C2C_FFT_LDLIBS := -lcufft -lfftw3f -lfftw3f_threads -lnvidia-ml -lpthread -lm -Xcompiler -fopenmp

c2c_fft: $(C2C_FFT_BIN)
$(C2C_FFT_BIN): $(C2C_FFT_SRC) src/gpu/gh200/bench_harness.h src/gpu/gpu_device_specs.h
	@mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LDFLAGS) -o $@ $(C2C_FFT_SRC) $(C2C_FFT_LDLIBS)
	@echo "built $@"

# ── Cycle 3: PyTorch CUDAPluggableAllocator (libtorch_c2c_allocator.so) ───────
C2C_ALLOC_LIB := bin/libtorch_c2c_allocator.so
C2C_ALLOC_SRC := src/gpu/gh200/torch_c2c_allocator.cu

c2c_allocator: $(C2C_ALLOC_LIB)
$(C2C_ALLOC_LIB): $(C2C_ALLOC_SRC)
	@mkdir -p bin
	$(NVCC) -std=c++17 -O3 -arch=$(ARCH) --compiler-options '-fPIC' -shared -o $@ $(C2C_ALLOC_SRC)
	@echo "built $@"

# ── Cycle 4: Regime 3 Irregular Benchmark (SpMV + SpGEMM + Thrust Search) ────
C2C_IRREG_BIN := bin/gpu_c2c_irregular_bench
C2C_IRREG_SRC := src/gpu/gh200/gpu_c2c_irregular_bench.cu
C2C_IRREG_LDLIBS := -lcusparse $(NVJITLINK_LIB) -lnvidia-ml -lpthread -lm -Xcompiler -fopenmp

c2c_irregular: $(C2C_IRREG_BIN)
$(C2C_IRREG_BIN): $(C2C_IRREG_SRC) src/gpu/gh200/bench_harness.h src/gpu/gpu_device_specs.h src/common/gemv_host.h
	@mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LDFLAGS) -o $@ $(C2C_IRREG_SRC) $(C2C_IRREG_LDLIBS)
	@echo "built $@"

# ── Cycle 5: Regime 4 Divergent & Crypto Benchmark (Sort + Hist + AES) ───────
C2C_DIV_BIN := bin/gpu_c2c_divergence_bench
C2C_DIV_SRC := src/gpu/gh200/gpu_c2c_divergence_bench.cu
C2C_DIV_LDLIBS := -lnvidia-ml -lpthread -lm -Xcompiler -fopenmp

c2c_divergence: $(C2C_DIV_BIN)
$(C2C_DIV_BIN): $(C2C_DIV_SRC) src/gpu/gh200/bench_harness.h src/gpu/gpu_device_specs.h src/common/gemv_host.h
	@mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LDFLAGS) -o $@ $(C2C_DIV_SRC) $(C2C_DIV_LDLIBS)
	@echo "built $@"

# ── Cycle 6: Cooperative CPU + GPU Concurrent & Contention Benchmark ──────────
C2C_COOP_BIN := bin/gpu_c2c_cooperative_bench
C2C_COOP_SRC := src/gpu/gh200/gpu_c2c_cooperative_bench.cu
C2C_COOP_LDLIBS := -lcublas -lopenblas -lnvidia-ml -lpthread -lm -Xcompiler -fopenmp

c2c_cooperative: $(C2C_COOP_BIN)
$(C2C_COOP_BIN): $(C2C_COOP_SRC) src/gpu/gh200/bench_harness.h
	@mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LDFLAGS) -o $@ $(C2C_COOP_SRC) $(C2C_COOP_LDLIBS)
	@echo "built $@"

# ── Sweep 7: Fine-Grained UVM Advise & 96GB Cliff Benchmark ───────────────────
C2C_CLIFF_BIN := bin/gpu_c2c_uvm_advise_bench
C2C_CLIFF_SRC := src/gpu/gh200/gpu_c2c_uvm_advise_bench.cu
C2C_CLIFF_LDLIBS := -lcublas -lnvidia-ml -lpthread -lm

c2c_uvm_advise: $(C2C_CLIFF_BIN)
$(C2C_CLIFF_BIN): $(C2C_CLIFF_SRC) src/gpu/gh200/bench_harness.h
	@mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LDFLAGS) -o $@ $(C2C_CLIFF_SRC) $(C2C_CLIFF_LDLIBS)
	@echo "built $@"

# ── Python Environment and Plotting ──────────────────────────────────────────
venv: requirements.txt
	@if [ ! -d ".venv" ]; then uv venv .venv; fi
	uv pip install --python .venv/bin/python -r requirements.txt

plots: venv
	SOURCE_DATE_EPOCH=0 .venv/bin/python results/plotting/gpu/generate_plots.py
	SOURCE_DATE_EPOCH=0 .venv/bin/python results/plotting/gpu/generate_gemv_plots.py
	SOURCE_DATE_EPOCH=0 .venv/bin/python results/plotting/gpu/generate_saxpy_plots.py
	SOURCE_DATE_EPOCH=0 .venv/bin/python results/plotting/gpu/generate_spgemm_plots.py
	SOURCE_DATE_EPOCH=0 .venv/bin/python results/plotting/gpu/gh200/generate_all_workload_and_coherence_plots.py
	SOURCE_DATE_EPOCH=0 .venv/bin/python results/plotting/cpu/generate_cpu_gemm_plots.py
	SOURCE_DATE_EPOCH=0 .venv/bin/python results/plotting/cpu/generate_cpu_gemv_plots.py
	SOURCE_DATE_EPOCH=0 .venv/bin/python results/plotting/cpu/generate_cpu_saxpy_plots.py
	SOURCE_DATE_EPOCH=0 .venv/bin/python results/plotting/cpu/generate_cpu_spgemm_plots.py
	SOURCE_DATE_EPOCH=0 .venv/bin/python results/plotting/v2/analyze.py
	.venv/bin/python results/generate_index.py

# v2 analysis only: headline tables, library comparison, bound checks, ratios, plots
analysis:
	$(PYTHON) results/plotting/v2/analyze.py
	$(PYTHON) results/plotting/v2/library_report.py

# Regenerate only the dashboard (index.html + workload status/TODO matrix)
index: venv
	.venv/bin/python results/generate_index.py



# ── CPU Benchmarks (Intel oneMKL or OpenBLAS) ──────────────────────────────
# BLAS can be explicitly set: make BLAS=openblas or make BLAS=mkl
# Default is 'auto': uses MKL if /opt/intel/oneapi/mkl/latest exists, else OpenBLAS.
BLAS ?= auto

ifeq ($(BLAS),openblas)
    CPU_CFLAGS  := -std=c11 -O3 -march=native -Wall -Wextra -fopenmp -DUSE_OPENBLAS=1 -I src/common -I src/cpu -I /usr/include -I /usr/include/x86_64-linux-gnu
    CPU_LDFLAGS :=
    CPU_LDLIBS  := -lopenblas -lpthread -lm -ldl
else ifeq ($(BLAS),mkl)
    MKLROOT ?= /opt/intel/oneapi/mkl/latest
    MKL_INC ?= $(MKLROOT)/include
    MKL_LIB ?= $(MKLROOT)/lib/intel64
    CPU_CFLAGS  := -std=c11 -O3 -march=native -Wall -Wextra -fopenmp -DUSE_MKL=1 \
                   -Wno-deprecated-declarations -I src/common -I src/cpu -I $(MKL_INC)
    CPU_LDFLAGS := -L $(MKL_LIB) -Wl,-rpath,$(MKL_LIB)
    CPU_LDLIBS  := -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
else ifeq ($(shell test -d /opt/intel/oneapi/mkl/latest && echo yes),yes)
    MKLROOT ?= /opt/intel/oneapi/mkl/latest
    MKL_INC ?= $(MKLROOT)/include
    MKL_LIB ?= $(MKLROOT)/lib/intel64
    CPU_CFLAGS  := -std=c11 -O3 -march=native -Wall -Wextra -fopenmp -DUSE_MKL=1 \
                   -Wno-deprecated-declarations -I src/common -I src/cpu -I $(MKL_INC)
    CPU_LDFLAGS := -L $(MKL_LIB) -Wl,-rpath,$(MKL_LIB)
    CPU_LDLIBS  := -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
else
    # Fallback to OpenBLAS
    CPU_CFLAGS  := -std=c11 -O3 -march=native -Wall -Wextra -fopenmp -DUSE_OPENBLAS=1 -I src/common -I src/cpu -I /usr/include -I /usr/include/x86_64-linux-gnu
    CPU_LDFLAGS :=
    CPU_LDLIBS  := -lopenblas -lpthread -lm -ldl
endif

CPU_GEMM_BIN   := bin/cpu_gemm_bench
CPU_TELEM_TEST := bin/test_cpu_telemetry

cpu_gemm: $(CPU_GEMM_BIN)
$(CPU_GEMM_BIN): src/cpu/cpu_gemm_bench.c src/common/gemv_host.h src/common/bf16_cvt.h src/cpu/cpu_telemetry.h
	@mkdir -p bin
	$(CC) $(CPU_CFLAGS) $(CPU_LDFLAGS) -o $@ $< $(CPU_LDLIBS)
	@echo "built $@"

run_cpu_gemm: $(CPU_GEMM_BIN)
	./$(CPU_GEMM_BIN) --sizes 512,1024,2048 --threads 1,2,4,8 --validate --iters 10

sweep_cpu_gemm: $(CPU_GEMM_BIN)
	bash ./scripts/cpu/run_gemm_sweep.sh

plots_cpu_gemm:
	python3 results/plotting/cpu/generate_cpu_gemm_plots.py

# ── CPU GEMV / SpMV ──────────────────────────────────────────────────────────
CPU_GEMV_BIN := bin/cpu_gemv_bench

cpu_gemv: $(CPU_GEMV_BIN)
$(CPU_GEMV_BIN): src/cpu/cpu_gemv_bench.c src/common/gemv_host.h src/common/bf16_cvt.h src/cpu/cpu_telemetry.h
	@mkdir -p bin
	$(CC) $(CPU_CFLAGS) $(CPU_LDFLAGS) -o $@ $< $(CPU_LDLIBS)
	@echo "built $@"

run_cpu_gemv: $(CPU_GEMV_BIN)
	./$(CPU_GEMV_BIN) --sizes 1024,2048,4096 --threads 1,2,4,8 --validate --iters 10

sweep_cpu_gemv: $(CPU_GEMV_BIN)
	bash ./scripts/cpu/run_gemv_sweep.sh

plots_cpu_gemv:
	python3 results/plotting/cpu/generate_cpu_gemv_plots.py

# ── CPU SAXPY ────────────────────────────────────────────────────────────────
CPU_SAXPY_BIN := bin/cpu_saxpy_bench

cpu_saxpy: $(CPU_SAXPY_BIN)
$(CPU_SAXPY_BIN): src/cpu/cpu_saxpy_bench.c src/common/gemv_host.h src/cpu/cpu_telemetry.h
	@mkdir -p bin
	$(CC) $(CPU_CFLAGS) $(CPU_LDFLAGS) -o $@ $< $(CPU_LDLIBS)
	@echo "built $@"

run_cpu_saxpy: $(CPU_SAXPY_BIN)
	./$(CPU_SAXPY_BIN) --sizes 1048576,4194304,16777216 --threads 1,2,4,8 --validate --iters 10

sweep_cpu_saxpy: $(CPU_SAXPY_BIN)
	bash ./scripts/cpu/run_saxpy_sweep.sh

plots_cpu_saxpy:
	python3 results/plotting/cpu/generate_cpu_saxpy_plots.py

# ── CPU SpGEMM (Sparse x Sparse) ─────────────────────────────────────────────
CPU_SPGEMM_BIN := bin/cpu_spgemm_bench

cpu_spgemm: $(CPU_SPGEMM_BIN)
$(CPU_SPGEMM_BIN): src/cpu/cpu_spgemm_bench.c src/common/gemv_host.h src/cpu/cpu_telemetry.h
	@mkdir -p bin
	$(CC) $(CPU_CFLAGS) $(CPU_LDFLAGS) -o $@ $< $(CPU_LDLIBS)
	@echo "built $@"

run_cpu_spgemm: $(CPU_SPGEMM_BIN)
	./$(CPU_SPGEMM_BIN) --sizes 1024,2048 --threads 1,2,4,8 --validate --iters 5

sweep_cpu_spgemm: $(CPU_SPGEMM_BIN)
	bash ./scripts/cpu/run_spgemm_sweep.sh

plots_cpu_spgemm:
	python3 results/plotting/cpu/generate_cpu_spgemm_plots.py

# ── CPU Monte Carlo Option Pricing ──────────────────────────────────────────
CPU_MONTECARLO_BIN := bin/cpu_montecarlo_bench

cpu_montecarlo: $(CPU_MONTECARLO_BIN)
$(CPU_MONTECARLO_BIN): src/cpu/cpu_montecarlo_bench.c src/cpu/cpu_telemetry.h
	@mkdir -p bin
	$(CC) $(CPU_CFLAGS) $(CPU_LDFLAGS) -o $@ $< $(CPU_LDLIBS) -lm
	@echo "built $@"

run_cpu_montecarlo: $(CPU_MONTECARLO_BIN)
	./$(CPU_MONTECARLO_BIN) --sizes 1000000,5000000 --threads 1,2,4,8 --validate --iters 5

sweep_cpu_montecarlo: $(CPU_MONTECARLO_BIN)
	bash ./scripts/cpu/run_montecarlo_sweep.sh

cpu_all: cpu_gemm cpu_gemv cpu_saxpy cpu_spgemm cpu_montecarlo   # native cross-check benches


# ── Intel Xeon Benchmark Suite ──────────────────────────────────────────────
intel_suite:
	bash ./scripts/cpu/run_all_intel.sh

intel_suite_bg:
	@mkdir -p results/logs
	@nohup bash ./scripts/cpu/run_all_intel.sh > /dev/null 2>&1 & \
	 echo "==> Intel Suite launched in background! You can safely close your terminal/laptop." && \
	 echo "==> Live monitor with: tail -f results/logs/suite_latest.log"

cpu_suite: intel_suite
cpu_suite_bg: intel_suite_bg

# ── AMD EPYC Benchmark Suite ────────────────────────────────────────────────
amd_suite:
	bash ./scripts/cpu/run_all_amd.sh

amd_suite_bg:
	@mkdir -p results/logs
	@nohup bash ./scripts/cpu/run_all_amd.sh > /dev/null 2>&1 & \
	 echo "==> AMD Suite launched in background! You can safely close your terminal/laptop." && \
	 echo "==> Live monitor with: tail -f results/logs/suite_epyc_latest.log"

epyc_suite: amd_suite
epyc_suite_bg: amd_suite_bg

# ── PyTorch suite = v2 publication harness (see "v2 publication harness" below) ──
# The pre-v2 torch_*_bench.py sweeps were retired: they crashed on the 2nd config and
# did not meet docs/methodology.md. torch_gemv/torch_spgemm remain only as porting refs.
PYTHON ?= python3

torch_suite:
	@test -n "$(DEVICE)" || { echo "usage: make torch_suite DEVICE=<id in configs/devices/>"; exit 2; }
	PYTHONPATH=src/torch $(PYTHON) -m bench sweep --device-config configs/devices/$(DEVICE).json --trials 3

torch_suite_bg:
	@test -n "$(DEVICE)" || { echo "usage: make torch_suite_bg DEVICE=<id in configs/devices/>"; exit 2; }
	@mkdir -p results/logs
	@nohup env PYTHONPATH=src/torch $(PYTHON) -m bench sweep --device-config configs/devices/$(DEVICE).json \
	   --trials 3 > results/logs/bench_$(DEVICE).log 2>&1 & \
	 echo "==> v2 sweep for $(DEVICE) launched in background." && \
	 echo "==> Live monitor with: tail -f results/logs/bench_$(DEVICE).log"

test_telemetry: $(CPU_TELEM_TEST)
	./$(CPU_TELEM_TEST)


$(CPU_TELEM_TEST): tests/cpu/test_cpu_telemetry.c src/cpu/cpu_telemetry.h
	@mkdir -p bin
	$(CC) -std=c11 -O2 -Wall -I src/cpu -o $@ tests/cpu/test_cpu_telemetry.c -lm
	@echo "built $@"

# ── v2 publication harness (src/torch/bench, protocol: docs/methodology.md) ──
BENCH := PYTHONPATH=src/torch $(PYTHON) -m bench
DEVICE ?=

bench_list:
	$(BENCH) list

# Every registered workload, smallest configs, short timing -> results/data/smoke (git-ignored)
bench_smoke:
	$(BENCH) smoke --device cpu

bench_smoke_cuda:
	$(BENCH) smoke --device cuda

# Full collection on this node:  make bench_suite DEVICE=INTEL_C6620   (configs/devices/<DEVICE>.json)
bench_suite:
	@test -n "$(DEVICE)" || { echo "usage: make bench_suite DEVICE=<id in configs/devices/>"; exit 2; }
	$(BENCH) sweep --device-config configs/devices/$(DEVICE).json --trials 3

bench_suite_dry:
	@test -n "$(DEVICE)" || { echo "usage: make bench_suite_dry DEVICE=<id>"; exit 2; }
	$(BENCH) sweep --device-config configs/devices/$(DEVICE).json --trials 3 --dry-run

test_bench:
	$(PYTHON) -m pytest -q tests/torch

# ── Common header unit test ─────────────────────────────────────────────────
test_gemv_host: tests/common/test_gemv_host.c src/common/gemv_host.h
	@mkdir -p bin
	$(CC) -std=c11 -O2 -Wall -I src/common -o bin/test_gemv_host $< -lm
	./bin/test_gemv_host

clean:
	rm -rf bin
