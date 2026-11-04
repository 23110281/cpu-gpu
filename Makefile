# third_party/gpu_bench/Makefile — cuBLAS GEMM benchmark, two build flavours
#
# Self-contained: NOT wired into the top-level dgemm Makefile (that build is
# CPU-only, clang, -Werror -Wpedantic C11 and must stay GPU-free).
#
# Targets (nvcc / CUDA C++ build, default):
#   make            -> bin/gpu_gemm_bench_cu   (nvcc, CUDA C++17, sm_80)
#   make run        quick smoke: nvcc build, 1 GPU, size=256..4096
#   make sweep      full sweep via scripts/run_sweep.sh (which now calls the cu binary)
#
# Targets (gcc / C11 build, explicit fallback):
#   make bench_gcc  -> bin/gpu_gemm_bench      (gcc, C11)
#   make run_gcc    quick smoke: gcc build, 1 GPU, small sizes
#   make sweep_gcc  full sweep via scripts/run_sweep_gcc.sh
#
# Combined:
#   make both       builds both binaries
#   make clean      removes bin/

CUDA_HOME ?= /usr/local/cuda
CC        ?= gcc
NVCC      ?= $(CUDA_HOME)/bin/nvcc

CFLAGS  := -std=c11 -O3 -march=native -Wall -Wextra -fopenmp \
           -Wno-deprecated-declarations \
           -I src -I $(CUDA_HOME)/include
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
                -I src -I $(CUDA_HOME)/include -I $(CUSPARSELT_INC)
NVCC_LDFLAGS := -L $(CUDA_HOME)/lib64 -L $(CUDA_HOME)/lib64/stubs -L $(CUSPARSELT_LIB) $(NVCC_ARCH_LDFLAGS) \
                -Xlinker -rpath,$(CUDA_HOME)/lib64 -Xlinker -rpath,$(CUSPARSELT_LIB)
NVCC_LDLIBS  := -lcublas -lcublasLt -lcusparse -lcusparseLt $(NVJITLINK_LIB) -lnvidia-ml -lpthread -lm \
                -Xcompiler -fopenmp

BIN := bin/gpu_gemm_bench
SRC := src/gpu_gemm_bench.cu

# ── gcc / C11 (legacy fallback) ──────────────────────────────────────────────
CFLAGS  := -std=c11 -O3 -march=native -Wall -Wextra -fopenmp \
           -Wno-deprecated-declarations \
           -I src -I $(CUDA_HOME)/include
LDFLAGS := -L $(CUDA_HOME)/lib64 -L $(CUDA_HOME)/lib64/stubs \
           -Wl,-rpath,$(CUDA_HOME)/lib64
LDLIBS  := -lcublas -lcublasLt -lcusparse -lcudart -lnvidia-ml -lpthread -lm -fopenmp

BIN_GCC := bin/gpu_gemm_bench_gcc
SRC_GCC := src/gpu_gemm_bench.c
HDR_GCC := src/bf16_cvt.h

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
	./scripts/run_sweep.sh

sweep_gcc: $(BIN_GCC)
	./scripts/run_sweep_gcc.sh

# ── GEMV/SpMV benchmark (dense cuBLAS gemv + sparse cuSPARSE SpMV) ─────────
# No cuSPARSELt dependency -- regular cuSPARSE only, see src/gpu_gemv_bench.cu
# header comment for why.
GEMV_BIN := bin/gpu_gemv_bench
GEMV_SRC := src/gpu_gemv_bench.cu
GEMV_LDLIBS := -lcublas -lcublasLt -lcusparse $(NVJITLINK_LIB) -lnvidia-ml -lpthread -lm

gemv: $(GEMV_BIN)
$(GEMV_BIN): $(GEMV_SRC) src/gemv_host.h src/bf16_cvt.h
	@mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LDFLAGS) -o $@ $(GEMV_SRC) $(GEMV_LDLIBS)
	@echo "built $@"

run_gemv: $(GEMV_BIN)
	./$(GEMV_BIN) --sizes 256,512,1024,2048,4096 --gpus 1 --validate --iters 20

sweep_gemv: $(GEMV_BIN)
	./scripts/run_sweep_gemv.sh

# ── SAXPY benchmark (cuBLAS Level-1 AXPY) ──────────────────────────────────
SAXPY_BIN := bin/gpu_saxpy_bench
SAXPY_SRC := src/gpu_saxpy_bench.cu
SAXPY_LDLIBS := -lcublas -lnvidia-ml -lpthread -lm

saxpy: $(SAXPY_BIN)
$(SAXPY_BIN): $(SAXPY_SRC) src/bf16_cvt.h
	@mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LDFLAGS) -o $@ $(SAXPY_SRC) $(SAXPY_LDLIBS)
	@echo "built $@"

run_saxpy: $(SAXPY_BIN)
	./$(SAXPY_BIN) --sizes 33554432,67108864,134217728 --gpus 1 --validate --iters 20

sweep_saxpy: $(SAXPY_BIN)
	./scripts/run_sweep_saxpy.sh

# ── SpGEMM benchmark (cuSPARSE Generic API) ────────────────────────────────
SPGEMM_BIN := bin/gpu_spgemm_bench
SPGEMM_SRC := src/gpu_spgemm_bench.cu
SPGEMM_LDLIBS := -lcusparse $(NVJITLINK_LIB) -lnvidia-ml -lpthread -lm

spgemm: $(SPGEMM_BIN)
$(SPGEMM_BIN): $(SPGEMM_SRC) src/gemv_host.h src/bf16_cvt.h
	@mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LDFLAGS) -o $@ $(SPGEMM_SRC) $(SPGEMM_LDLIBS)
	@echo "built $@"

run_spgemm: $(SPGEMM_BIN)
	./$(SPGEMM_BIN) --sizes 1024,2048,4096 --gpus 1 --validate --iters 10

sweep_spgemm: $(SPGEMM_BIN)
	./scripts/run_sweep_spgemm.sh

# ── Cycle 1: NVLink-C2C Streaming Benchmark (SAXPY + GEMV across 4 NUMA Modes)
C2C_STREAM_BIN := bin/gpu_c2c_streaming_bench
C2C_STREAM_SRC := src/gpu_c2c_streaming_bench.cu
C2C_STREAM_LDLIBS := -lcublas -lopenblas -lnvidia-ml -lpthread -lm -Xcompiler -fopenmp

c2c_streaming: $(C2C_STREAM_BIN)
$(C2C_STREAM_BIN): $(C2C_STREAM_SRC) src/bench_harness.h src/gpu_device_specs.h
	@mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LDFLAGS) -o $@ $(C2C_STREAM_SRC) $(C2C_STREAM_LDLIBS)
	@echo "built $@"

# ── Cycle 2: Arithmetic Compute Benchmark (GEMM + Monte Carlo across 4 NUMA Modes)
C2C_COMPUTE_BIN := bin/gpu_c2c_compute_bench
C2C_COMPUTE_SRC := src/gpu_c2c_compute_bench.cu
C2C_COMPUTE_LDLIBS := -lcublas -lopenblas -lnvidia-ml -lpthread -lm -Xcompiler -fopenmp

c2c_compute: $(C2C_COMPUTE_BIN)
$(C2C_COMPUTE_BIN): $(C2C_COMPUTE_SRC) src/bench_harness.h src/gpu_device_specs.h
	@mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LDFLAGS) -o $@ $(C2C_COMPUTE_SRC) $(C2C_COMPUTE_LDLIBS)
	@echo "built $@"

# ── Cycle 2B: cuFFT vs FFTW3 Benchmark (1D & 2D FFT)
C2C_FFT_BIN := bin/gpu_c2c_fft_bench
C2C_FFT_SRC := src/gpu_c2c_fft_bench.cu
C2C_FFT_LDLIBS := -lcufft -lfftw3f -lfftw3f_threads -lnvidia-ml -lpthread -lm -Xcompiler -fopenmp

c2c_fft: $(C2C_FFT_BIN)
$(C2C_FFT_BIN): $(C2C_FFT_SRC) src/bench_harness.h src/gpu_device_specs.h
	@mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LDFLAGS) -o $@ $(C2C_FFT_SRC) $(C2C_FFT_LDLIBS)
	@echo "built $@"

# ── Cycle 3: PyTorch CUDAPluggableAllocator (libtorch_c2c_allocator.so) ───────
C2C_ALLOC_LIB := bin/libtorch_c2c_allocator.so
C2C_ALLOC_SRC := src/torch_c2c_allocator.cu

c2c_allocator: $(C2C_ALLOC_LIB)
$(C2C_ALLOC_LIB): $(C2C_ALLOC_SRC)
	@mkdir -p bin
	$(NVCC) -std=c++17 -O3 -arch=$(ARCH) --compiler-options '-fPIC' -shared -o $@ $(C2C_ALLOC_SRC)
	@echo "built $@"

# ── Cycle 4: Regime 3 Irregular Benchmark (SpMV + SpGEMM + Thrust Search) ────
C2C_IRREG_BIN := bin/gpu_c2c_irregular_bench
C2C_IRREG_SRC := src/gpu_c2c_irregular_bench.cu
C2C_IRREG_LDLIBS := -lcusparse $(NVJITLINK_LIB) -lnvidia-ml -lpthread -lm -Xcompiler -fopenmp

c2c_irregular: $(C2C_IRREG_BIN)
$(C2C_IRREG_BIN): $(C2C_IRREG_SRC) src/bench_harness.h src/gpu_device_specs.h src/gemv_host.h
	@mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LDFLAGS) -o $@ $(C2C_IRREG_SRC) $(C2C_IRREG_LDLIBS)
	@echo "built $@"

# ── Cycle 5: Regime 4 Divergent & Crypto Benchmark (Sort + Hist + AES) ───────
C2C_DIV_BIN := bin/gpu_c2c_divergence_bench
C2C_DIV_SRC := src/gpu_c2c_divergence_bench.cu
C2C_DIV_LDLIBS := -lnvidia-ml -lpthread -lm -Xcompiler -fopenmp

c2c_divergence: $(C2C_DIV_BIN)
$(C2C_DIV_BIN): $(C2C_DIV_SRC) src/bench_harness.h src/gpu_device_specs.h src/gemv_host.h
	@mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LDFLAGS) -o $@ $(C2C_DIV_SRC) $(C2C_DIV_LDLIBS)
	@echo "built $@"

# ── Cycle 6: Cooperative CPU + GPU Concurrent & Contention Benchmark ──────────
C2C_COOP_BIN := bin/gpu_c2c_cooperative_bench
C2C_COOP_SRC := src/gpu_c2c_cooperative_bench.cu
C2C_COOP_LDLIBS := -lcublas -lopenblas -lnvidia-ml -lpthread -lm -Xcompiler -fopenmp

c2c_cooperative: $(C2C_COOP_BIN)
$(C2C_COOP_BIN): $(C2C_COOP_SRC) src/bench_harness.h
	@mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LDFLAGS) -o $@ $(C2C_COOP_SRC) $(C2C_COOP_LDLIBS)
	@echo "built $@"

# ── Sweep 7: Fine-Grained UVM Advise & 96GB Cliff Benchmark ───────────────────
C2C_CLIFF_BIN := bin/gpu_c2c_uvm_advise_bench
C2C_CLIFF_SRC := src/gpu_c2c_uvm_advise_bench.cu
C2C_CLIFF_LDLIBS := -lcublas -lnvidia-ml -lpthread -lm

c2c_uvm_advise: $(C2C_CLIFF_BIN)
$(C2C_CLIFF_BIN): $(C2C_CLIFF_SRC) src/bench_harness.h
	@mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LDFLAGS) -o $@ $(C2C_CLIFF_SRC) $(C2C_CLIFF_LDLIBS)
	@echo "built $@"

# ── Python Environment and Plotting ──────────────────────────────────────────
venv: requirements.txt
	@if [ ! -d ".venv" ]; then uv venv .venv; fi
	uv pip install --python .venv/bin/python -r requirements.txt

plots: venv
	SOURCE_DATE_EPOCH=0 .venv/bin/python results/generate_plots.py
	SOURCE_DATE_EPOCH=0 .venv/bin/python results/generate_gemv_plots.py
	SOURCE_DATE_EPOCH=0 .venv/bin/python results/generate_saxpy_plots.py
	SOURCE_DATE_EPOCH=0 .venv/bin/python results/generate_spgemm_plots.py
	SOURCE_DATE_EPOCH=0 .venv/bin/python scripts/generate_all_workload_and_coherence_plots.py
	.venv/bin/python results/generate_index.py


clean:
	rm -rf bin
# Trigger CI
