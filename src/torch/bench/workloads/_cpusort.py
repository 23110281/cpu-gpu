"""Multithreaded CPU sort libraries for C6 (fastest-library rule, methodology §7).

ATen's CPU sort scales poorly (measured EPYC 7402P, 2^27 pairs: 19 -> 100 Mpairs/s from 1 to
24 cores), so the well-known parallel CPU sorts are timed on the same data:
  gnu_parallel : libstdc++ parallel mode __gnu_parallel::sort (OpenMP multiway mergesort, GCC)
  tbb          : oneTBB tbb::parallel_sort
Contract identical to torch.sort(stable=True): sorted int32 keys + int64 source index.
Each (key, idx) is packed into one uint64 = (key ^ 0x80000000) << 32 | idx; sorting these
words orders by key then by index -- exactly the stable-sort output. Packing/unpacking are
parallel loops inside the call (they are part of producing the required output).
Built once per node with torch.utils.cpp_extension.load_inline (cached in
~/.cache/torch_extensions); thread count = OMP_NUM_THREADS (gnu) / tbb::global_control.
"""
from __future__ import annotations

from functools import lru_cache

from ..core import Unsupported

_SRC = r"""
#include <torch/extension.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <cstdint>
#include <parallel/algorithm>
#ifdef HAVE_TBB
#include <oneapi/tbb/parallel_sort.h>
#include <oneapi/tbb/global_control.h>
#endif

static void pack(const int32_t* k, uint64_t* w, int64_t n) {
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; ++i)
    w[i] = (uint64_t(uint32_t(k[i]) ^ 0x80000000u) << 32) | uint64_t(uint32_t(i));
}
static void unpack(const uint64_t* w, int32_t* k, int64_t* idx, int64_t n) {
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; ++i) {
    k[i] = int32_t(uint32_t(w[i] >> 32) ^ 0x80000000u);
    idx[i] = int64_t(w[i] & 0xFFFFFFFFull);
  }
}
void sort_gnu(torch::Tensor keys, torch::Tensor out_k, torch::Tensor out_i, torch::Tensor scratch) {
  const int64_t n = keys.numel();
  auto* w = reinterpret_cast<uint64_t*>(scratch.data_ptr<int64_t>());
  pack(keys.data_ptr<int32_t>(), w, n);
  __gnu_parallel::sort(w, w + n, __gnu_parallel::multiway_mergesort_tag(omp_get_max_threads()));
  unpack(w, out_k.data_ptr<int32_t>(), out_i.data_ptr<int64_t>(), n);
}
bool has_tbb() {
#ifdef HAVE_TBB
  return true;
#else
  return false;
#endif
}
// OpenMP binding (OMP_PROC_BIND) pins the calling thread to ONE core; pthread-based
// libraries (TBB) create workers that inherit that mask -> all workers on one core.
// Widen the caller to the run's full placement for the duration of the call.
struct WidenAffinity {
  cpu_set_t saved;
  explicit WidenAffinity(torch::Tensor cpus) {
    pthread_getaffinity_np(pthread_self(), sizeof(saved), &saved);
    cpu_set_t s; CPU_ZERO(&s);
    auto* c = cpus.data_ptr<int32_t>();
    for (int64_t i = 0; i < cpus.numel(); ++i) CPU_SET(c[i], &s);
    pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
  }
  ~WidenAffinity() { pthread_setaffinity_np(pthread_self(), sizeof(saved), &saved); }
};

void sort_tbb(torch::Tensor keys, torch::Tensor out_k, torch::Tensor out_i, torch::Tensor scratch,
              int64_t threads, torch::Tensor cpus) {
#ifdef HAVE_TBB
  const int64_t n = keys.numel();
  auto* w = reinterpret_cast<uint64_t*>(scratch.data_ptr<int64_t>());
  pack(keys.data_ptr<int32_t>(), w, n);
  {
    WidenAffinity wa(cpus);
    oneapi::tbb::global_control gc(oneapi::tbb::global_control::max_allowed_parallelism, (size_t)threads);
    oneapi::tbb::parallel_sort(w, w + n);
  }
  unpack(w, out_k.data_ptr<int32_t>(), out_i.data_ptr<int64_t>(), n);
#else
  TORCH_CHECK(false, "built without oneTBB");
#endif
}
"""

_BINDINGS = ["sort_gnu", "sort_tbb", "has_tbb"]


def _ensure_ninja_on_path():
    """pip installs ninja into ~/.local/bin, often absent from non-interactive PATHs."""
    import os
    import shutil
    if shutil.which("ninja"):
        return
    try:
        import ninja
        os.environ["PATH"] = ninja.BIN_DIR + os.pathsep + os.environ.get("PATH", "")
    except ImportError:
        pass


@lru_cache(maxsize=1)
def ext():
    import os
    from torch.utils.cpp_extension import load_inline
    _ensure_ninja_on_path()
    tbb = any(os.path.exists(p) for p in ("/usr/include/oneapi/tbb/parallel_sort.h",
                                          "/usr/include/tbb/parallel_sort.h"))
    try:
        return load_inline(
            name="bench_cpusort_v2" + ("_tbb" if tbb else ""), cpp_sources=_SRC, functions=_BINDINGS,
            extra_cflags=["-O3", "-march=native", "-fopenmp"] + (["-DHAVE_TBB"] if tbb else []),
            extra_ldflags=["-fopenmp"] + (["-ltbb"] if tbb else []), verbose=False)
    except Exception as e:  # compiler / ninja missing -> the variant is unavailable on this node
        raise Unsupported(f"cpu sort extension build failed: {type(e).__name__}: {str(e)[:120]}") from e


def version(lib: str) -> str:
    import subprocess
    if lib == "gnu_parallel":
        try:
            v = subprocess.run(["g++", "-dumpfullversion"], capture_output=True, text=True).stdout.strip()
        except OSError:
            v = "?"
        return f"libstdc++-parallel(g++ {v})/multiway_mergesort"
    try:
        out = subprocess.run(["sh", "-c", "dpkg-query -W -f='${Version}' libtbb-dev 2>/dev/null || true"],
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        out = ""
    return f"oneTBB {out or '?'}/parallel_sort"
