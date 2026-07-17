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
# sm_80 = A100 SXM4; extend with -gencode arch=compute_90,code=sm_90 for H100.
# -Xcompiler passes flags to the host compiler (gcc) through nvcc.
#
# CUSPARSELT_DIR: the cuda-repo apt packages (libcusparselt0-dev-cuda-12) do
# NOT install into $(CUDA_HOME)/include|lib64 like the rest of the toolkit --
# they land under a versioned /usr/include|lib path. Override if yours differs.
CUSPARSELT_INC ?= /usr/include/libcusparseLt/12
CUSPARSELT_LIB ?= /usr/lib/x86_64-linux-gnu/libcusparseLt/12
NVCC_FLAGS   := -std=c++17 -O3 -arch=sm_80 \
                -Xcompiler "-Wall -Wextra -fopenmp -Wno-deprecated-declarations" \
                -I src -I $(CUDA_HOME)/include -I $(CUSPARSELT_INC)
NVCC_LDFLAGS := -L $(CUDA_HOME)/lib64 -L $(CUDA_HOME)/lib64/stubs -L $(CUSPARSELT_LIB) \
                -Xlinker -rpath,$(CUDA_HOME)/lib64 -Xlinker -rpath,$(CUSPARSELT_LIB)
NVCC_LDLIBS  := -lcublas -lcublasLt -lcusparse -lcusparseLt -lnvidia-ml -lpthread -lm \
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

.PHONY: all bench_gcc both run run_gcc sweep sweep_gcc gemv run_gemv sweep_gemv clean
all: $(BIN)

both: $(BIN) $(BIN_GCC)

$(BIN): $(SRC)
	@mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LDFLAGS) -o $@ $(SRC) $(NVCC_LDLIBS)
	@echo "built $@"

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
GEMV_LDLIBS := -lcublas -lcublasLt -lcusparse -lnvidia-ml -lpthread -lm

gemv: $(GEMV_BIN)
$(GEMV_BIN): $(GEMV_SRC) src/gemv_host.h src/bf16_cvt.h
	@mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LDFLAGS) -o $@ $(GEMV_SRC) $(GEMV_LDLIBS)
	@echo "built $@"

run_gemv: $(GEMV_BIN)
	./$(GEMV_BIN) --sizes 256,512,1024,2048,4096 --gpus 1 --validate --iters 20

sweep_gemv: $(GEMV_BIN)
	./scripts/run_sweep_gemv.sh

clean:
	rm -rf bin
