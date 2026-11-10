# studies/gh200_c2c/Makefile -- GH200 NVLink-C2C residency study (NOT fixed; see README.md).
# Run on a GH200 as `make -C studies/gh200_c2c <target>`; binaries go to studies/gh200_c2c/bin/.
#   c2c_streaming c2c_compute c2c_fft c2c_irregular c2c_divergence c2c_cooperative c2c_uvm_advise
#   c2c_allocator (PyTorch pluggable allocator .so)
# protocol_benchmark_suite.cu has no target yet (known defect, README.md).

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

.PHONY: all clean c2c_streaming c2c_compute c2c_fft c2c_irregular c2c_divergence c2c_cooperative c2c_uvm_advise c2c_allocator
all: c2c_streaming c2c_compute c2c_fft c2c_irregular c2c_divergence c2c_cooperative c2c_uvm_advise c2c_allocator

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

clean:
	rm -rf bin
