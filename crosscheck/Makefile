# Native cross-check benches (used by crosscheck/crosscheck.py):
#   make -C crosscheck              -> bin/cpu_gemm_bench, bin/cpu_saxpy_bench
#   make -C crosscheck BLAS=mkl     -> link Intel oneMKL instead of OpenBLAS (needs /opt/intel/oneapi)
# OpenBLAS is the default because the harness's `lib=openblas` rows are the like-for-like
# comparison (crosscheck.py CASES).
CC   ?= gcc
BLAS ?= openblas

ifeq ($(BLAS),mkl)
    MKLROOT ?= /opt/intel/oneapi/mkl/latest
    CFLAGS  := -std=c11 -O3 -march=native -Wall -Wextra -fopenmp -DUSE_MKL=1 -Wno-deprecated-declarations \
               -I src -I $(MKLROOT)/include
    LDFLAGS := -L $(MKLROOT)/lib/intel64 -Wl,-rpath,$(MKLROOT)/lib/intel64
    LDLIBS  := -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
else
    CFLAGS  := -std=c11 -O3 -march=native -Wall -Wextra -fopenmp -DUSE_OPENBLAS=1 \
               -I src -I /usr/include -I /usr/include/$(shell uname -m)-linux-gnu
    LDFLAGS :=
    LDLIBS  := -lopenblas -lpthread -lm -ldl
endif

.PHONY: all cpu_gemm cpu_saxpy clean
all: cpu_gemm cpu_saxpy

cpu_gemm: bin/cpu_gemm_bench
bin/cpu_gemm_bench: src/cpu_gemm_bench.c src/gemv_host.h src/bf16_cvt.h src/cpu_telemetry.h
	@mkdir -p bin
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

cpu_saxpy: bin/cpu_saxpy_bench
bin/cpu_saxpy_bench: src/cpu_saxpy_bench.c src/gemv_host.h src/cpu_telemetry.h
	@mkdir -p bin
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -rf bin
