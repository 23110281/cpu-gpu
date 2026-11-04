#!/usr/bin/env python3
"""
bench_llama32_layers.py

Benchmark suite for all distinct layer types of Llama-3.2 (1B & 3B shapes)
using official PyTorch canonical operations:
  - Token Embedding (Regime 3 - Irregular Gather)
  - RMSNorm vs. LayerNorm (Regime 2 - Streaming BW)
  - RoPE (Rotary Position Embeddings, Regime 2)
  - QKV & Output Linear Projections (Regime 1 Compute / Regime 2 Decode)
  - Grouped-Query Attention SDPA (Regime 1 Prefill / Regime 2-3 Decode)
  - SwiGLU FFN (Regime 1 Compute)
  - LM Head + Top-K / Softmax (Regime 4 - Selection & Sort)

Tested across:
  - Modes: Prefill (S in [512, 2048, 8192], B=1) and Decode (S=1, B in [1, 8, 32])
  - Residency: CPU_LOCAL (72 threads), GPU_EXPLICIT_COPY, GPU_ZERO_COPY, GPU_UM_MIGRATE
    (GPU_ZERO_COPY and GPU_UM_MIGRATE use PyTorch CUDAPluggableAllocator with libtorch_c2c_allocator.so)
Outputs unified CSV schema.
"""

import sys
import os
import time
import argparse
import subprocess
import torch
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.recompile_limit = 128
except Exception:
    pass

# Model Metadata definitions from uncommitted/llm_matrix_workload/model_metadata_raw.txt
LLAMA_CONFIGS = {
    "llama3.2_1b": {
        "vocab_size": 128256,
        "hidden_dim": 2048,
        "intermediate_dim": 8192,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 64,
        "num_layers": 16,
    },
    "llama3.2_3b": {
        "vocab_size": 128256,
        "hidden_dim": 3072,
        "intermediate_dim": 8192,
        "num_heads": 24,
        "num_kv_heads": 8,
        "head_dim": 128,
        "num_layers": 28,
    },
    "llama3_8b": {
        "vocab_size": 128256,
        "hidden_dim": 4096,
        "intermediate_dim": 14336,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "num_layers": 32,
    }
}

def setup_allocator(residency_mode, so_path):
    import torch
    if residency_mode == "gpu_zero_copy":
        if os.path.exists(so_path):
            alloc = torch.cuda.CUDAPluggableAllocator(so_path, "c2c_zero_copy_malloc", "c2c_zero_copy_free")
            torch.cuda.change_current_allocator(alloc)
            print(f"[PluggableAllocator] Initialized GPU_ZERO_COPY (LPDDR5X via C2C) using {so_path}")
        else:
            print(f"[WARNING] {so_path} not found! Falling back to standard CUDA allocator.")
    elif residency_mode == "gpu_um_migrate":
        if os.path.exists(so_path):
            alloc = torch.cuda.CUDAPluggableAllocator(so_path, "c2c_um_malloc", "c2c_um_free")
            torch.cuda.change_current_allocator(alloc)
            print(f"[PluggableAllocator] Initialized GPU_UM_MIGRATE (UVM via C2C) using {so_path}")
        else:
            print(f"[WARNING] {so_path} not found! Falling back to standard CUDA allocator.")

def log_record(csv_file, workload, regime, precision, residency_mode, working_set_bytes, problem_size,
               threads_or_sms, t_alloc_ms, t_h2d_ms, t_kernel_cold_ms, t_kernel_warm_ms, t_d2h_ms,
               t_e2e_ms, eff_bw_gbps, c2c_bw_gbps, gflops, pct_peak, status="ok"):
    line = f"{workload},{regime},{precision},{residency_mode},{working_set_bytes},\"{problem_size}\",{threads_or_sms}," \
           f"{t_alloc_ms:.3f},{t_h2d_ms:.3f},{t_kernel_cold_ms:.3f},{t_kernel_warm_ms:.3f},{t_d2h_ms:.3f}," \
           f"{t_e2e_ms:.3f},{eff_bw_gbps:.2f},{c2c_bw_gbps:.2f},{gflops:.2f},{pct_peak:.1f},{status}\n"
    if csv_file:
        with open(csv_file, "a") as f:
            f.write(line)
    print(f"[{workload:<20}] {residency_mode:<14} {problem_size:<20} | warm={t_kernel_warm_ms:7.3f}ms e2e={t_e2e_ms:7.3f}ms | {eff_bw_gbps:7.1f} GB/s ({gflops:7.1f} GFLOPS)")

def measure_op(fn, in_tensor, residency_mode, iters=10, warmup=3):
    import torch
    device = "cuda" if "gpu" in residency_mode else "cpu"
    
    t_h2d = 0.0
    t_d2h = 0.0
    
    if residency_mode == "gpu_explicit":
        # Timed H2D transfer
        torch.cuda.synchronize()
        e_h2d0 = torch.cuda.Event(enable_timing=True)
        e_h2d1 = torch.cuda.Event(enable_timing=True)
        e_h2d0.record()
        gpu_in = in_tensor.to("cuda", non_blocking=True)
        e_h2d1.record()
        torch.cuda.synchronize()
        t_h2d = e_h2d0.elapsed_time(e_h2d1)
        
        # Cold kernel
        e_k0 = torch.cuda.Event(enable_timing=True)
        e_k1 = torch.cuda.Event(enable_timing=True)
        e_k0.record()
        gpu_out = fn(gpu_in)
        e_k1.record()
        torch.cuda.synchronize()
        t_kernel_cold = e_k0.elapsed_time(e_k1)
        
        # D2H transfer
        e_d2h0 = torch.cuda.Event(enable_timing=True)
        e_d2h1 = torch.cuda.Event(enable_timing=True)
        e_d2h0.record()
        cpu_out = gpu_out.to("cpu", non_blocking=True) if isinstance(gpu_out, torch.Tensor) else None
        e_d2h1.record()
        torch.cuda.synchronize()
        t_d2h = e_d2h0.elapsed_time(e_d2h1)
        
        # Warmup
        for _ in range(warmup):
            fn(gpu_in)
        torch.cuda.synchronize()
        
        # Timed warm loop
        e_k0.record()
        for _ in range(iters):
            fn(gpu_in)
        e_k1.record()
        torch.cuda.synchronize()
        t_kernel_warm = e_k0.elapsed_time(e_k1) / iters
        
    elif residency_mode in ("gpu_zero_copy", "gpu_um_migrate"):
        # Tensors were allocated via CUDAPluggableAllocator
        torch.cuda.synchronize()
        e_k0 = torch.cuda.Event(enable_timing=True)
        e_k1 = torch.cuda.Event(enable_timing=True)
        
        e_k0.record()
        out = fn(in_tensor)
        e_k1.record()
        torch.cuda.synchronize()
        t_kernel_cold = e_k0.elapsed_time(e_k1)
        
        for _ in range(warmup):
            fn(in_tensor)
        torch.cuda.synchronize()
        
        e_k0.record()
        for _ in range(iters):
            fn(in_tensor)
        e_k1.record()
        torch.cuda.synchronize()
        t_kernel_warm = e_k0.elapsed_time(e_k1) / iters
        
    else: # cpu_local
        t0 = time.perf_counter()
        out = fn(in_tensor)
        t_kernel_cold = (time.perf_counter() - t0) * 1000.0
        
        for _ in range(warmup):
            fn(in_tensor)
            
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(in_tensor)
        t_kernel_warm = ((time.perf_counter() - t0) * 1000.0) / iters

    if residency_mode == "gpu_um_migrate":
        t_e2e = t_kernel_cold + t_kernel_warm
    elif residency_mode == "gpu_zero_copy":
        t_e2e = t_kernel_warm
    else:
        t_e2e = t_h2d + t_kernel_warm + t_d2h
    return t_h2d, t_kernel_cold, t_kernel_warm, t_d2h, t_e2e

def run_llama_layer_benchmarks(res_mode, model_name="llama3.2_1b", csv_file=None, iters=10, warmup=3, so_path=None, compile_mode=False):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    cfg = LLAMA_CONFIGS[model_name]
    H = cfg["hidden_dim"]
    FFN = cfg["intermediate_dim"]
    V = cfg["vocab_size"]
    N_heads = cfg["num_heads"]
    N_kv = cfg["num_kv_heads"]
    D_head = cfg["head_dim"]
    
    print(f"\n==========================================================================================")
    print(f" Llama-3.2 Layer Benchmark: {model_name} | Mode={res_mode}")
    print(f" Hidden={H}, FFN={FFN}, Heads={N_heads}/{N_kv}, D={D_head} | Target CSV: {csv_file}")
    print(f"==========================================================================================")
    
    dev = "cuda" if "gpu" in res_mode else "cpu"
    if dev == "cpu":
        torch.set_num_threads(72)
    threads_or_sms = 132 if dev == "cuda" else 72
    
    dtype = torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    prec_str = "bf16" if dtype == torch.bfloat16 else "fp32"
    
    # Sweep Prefill and Decode across fine-grained batch sizes and sequence lengths
    test_cases = [
        ("prefill", 1, 128),
        ("prefill", 1, 512),
        ("prefill", 1, 2048),
        ("prefill", 1, 8192),
        ("prefill", 1, 16384),
        ("prefill", 4, 2048),
        ("prefill", 16, 512),
        ("decode", 1, 1),
        ("decode", 4, 1),
        ("decode", 16, 1),
        ("decode", 32, 1),
        ("decode", 64, 1)
    ]
    
    for phase, B, S in test_cases:
        tag = "_compiled" if compile_mode else ""
        p_size_str = f"{phase}_B{B}_S{S}{tag}"
        print(f"\n--- Phase: {phase.upper()} (B={B}, S={S}{', COMPILED' if compile_mode else ''}) ---")
        
        # 1. Token Embedding (Regime 3 - Gather)
        in_device = "cpu" if res_mode == "gpu_explicit" else dev
        input_ids = torch.randint(0, V, (B, S), dtype=torch.long, device=in_device)
        emb = nn.Embedding(V, H).to(dtype=dtype, device=dev)
        if compile_mode and dev == "cuda":
            try:
                emb = torch.compile(emb)
            except Exception as e:
                print(f"Warning: torch.compile(emb) failed: {e}")
        t_h2d, t_cold, t_warm, t_d2h, t_e2e = measure_op(emb, input_ids, res_mode, iters, warmup)
        bytes_emb = B * S * H * (2 if dtype == torch.bfloat16 else 4)
        bw = (bytes_emb / (t_e2e * 1e6)) if t_e2e > 0 else 0
        log_record(csv_file, "llama32_embedding", 3, prec_str, res_mode, bytes_emb, p_size_str, threads_or_sms,
                   0, t_h2d, t_cold, t_warm, t_d2h, t_e2e, bw, 0, 0, 0)
        
        # 2. RMSNorm (Regime 2 - Streaming BW)
        x = torch.randn((B, S, H), dtype=dtype, device=in_device)
        rmsnorm = nn.RMSNorm(H).to(dtype=dtype, device=dev)
        if compile_mode and dev == "cuda":
            try:
                rmsnorm = torch.compile(rmsnorm)
            except Exception as e:
                print(f"Warning: torch.compile(rmsnorm) failed: {e}")
        t_h2d, t_cold, t_warm, t_d2h, t_e2e = measure_op(rmsnorm, x, res_mode, iters, warmup)
        bytes_norm = 2 * B * S * H * (2 if dtype == torch.bfloat16 else 4)
        bw = (bytes_norm / (t_e2e * 1e6)) if t_e2e > 0 else 0
        log_record(csv_file, "llama32_rmsnorm", 2, prec_str, res_mode, bytes_norm, p_size_str, threads_or_sms,
                   0, t_h2d, t_cold, t_warm, t_d2h, t_e2e, bw, 0, 0, 0)
                   
        # 3. RoPE (Regime 2 - Complex Rotation)
        q = torch.randn((B, N_heads, S, D_head), dtype=dtype, device=in_device)
        freqs_cos = torch.cos(torch.arange(0, S, dtype=dtype, device=dev)[:, None]).repeat(1, D_head)
        def rope_fn(act):
            return act * freqs_cos
        if compile_mode and dev == "cuda":
            try:
                rope_fn = torch.compile(rope_fn)
            except Exception as e:
                print(f"Warning: torch.compile(rope_fn) failed: {e}")
        t_h2d, t_cold, t_warm, t_d2h, t_e2e = measure_op(rope_fn, q, res_mode, iters, warmup)
        bytes_rope = 2 * B * N_heads * S * D_head * (2 if dtype == torch.bfloat16 else 4)
        bw = bytes_rope / (t_e2e * 1e6) if t_e2e > 0 else 0
        log_record(csv_file, "llama32_rope", 2, prec_str, res_mode, bytes_rope, p_size_str, threads_or_sms,
                   0, t_h2d, t_cold, t_warm, t_d2h, t_e2e, bw, 0, 0, 0)

        # 4. QKV Projections (Regime 1 Compute in Prefill / Regime 2 in Decode)
        qkv_dim = (N_heads + 2 * N_kv) * D_head
        qkv_proj = nn.Linear(H, qkv_dim, bias=False).to(dtype=dtype, device=dev)
        if compile_mode and dev == "cuda":
            try:
                qkv_proj = torch.compile(qkv_proj)
            except Exception as e:
                print(f"Warning: torch.compile(qkv_proj) failed: {e}")
        t_h2d, t_cold, t_warm, t_d2h, t_e2e = measure_op(qkv_proj, x, res_mode, iters, warmup)
        flops = 2.0 * B * S * H * qkv_dim
        gflops = flops / (t_warm * 1e6) if t_warm > 0 else 0
        bytes_qkv = (B * S * H + H * qkv_dim + B * S * qkv_dim) * (2 if dtype == torch.bfloat16 else 4)
        bw = bytes_qkv / (t_e2e * 1e6) if t_e2e > 0 else 0
        regime = 1 if S > 1 else 2
        log_record(csv_file, "llama32_qkv_proj", regime, prec_str, res_mode, bytes_qkv, p_size_str, threads_or_sms,
                   0, t_h2d, t_cold, t_warm, t_d2h, t_e2e, bw, 0, gflops, 0)
                   
        # 5. GQA Scaled Dot-Product Attention (SDPA, FlashAttention-2 / Memory-Efficient)
        k_in = torch.randn((B, N_kv, S, D_head), dtype=dtype, device=dev)
        v_in = torch.randn((B, N_kv, S, D_head), dtype=dtype, device=dev)
        if N_heads != N_kv:
            k_in = k_in.repeat_interleave(N_heads // N_kv, dim=1)
            v_in = v_in.repeat_interleave(N_heads // N_kv, dim=1)
        
        q_attn = torch.randn((B, N_heads, S, D_head), dtype=dtype, device=in_device)
        def sdpa_fn(act):
            return F.scaled_dot_product_attention(act if dev == "cpu" else act.cuda(), k_in, v_in)
        if compile_mode and dev == "cuda":
            try:
                sdpa_fn = torch.compile(sdpa_fn)
            except Exception as e:
                print(f"Warning: torch.compile(sdpa_fn) failed: {e}")
            
        t_h2d, t_cold, t_warm, t_d2h, t_e2e = measure_op(sdpa_fn, q_attn, res_mode, iters, warmup)
        flops_attn = 4.0 * B * N_heads * S * S * D_head
        gflops_attn = flops_attn / (t_warm * 1e6) if t_warm > 0 else 0
        bytes_attn = B * N_heads * S * D_head * 4 * (2 if dtype == torch.bfloat16 else 4)
        bw = bytes_attn / (t_e2e * 1e6) if t_e2e > 0 else 0
        log_record(csv_file, "llama32_sdpa_attention", 1, prec_str, res_mode, bytes_attn, p_size_str, threads_or_sms,
                   0, t_h2d, t_cold, t_warm, t_d2h, t_e2e, bw, 0, gflops_attn, 0)
                   
        # 6. SwiGLU FFN (Regime 1 Compute)
        gate = nn.Linear(H, FFN, bias=False).to(dtype=dtype, device=dev)
        up = nn.Linear(H, FFN, bias=False).to(dtype=dtype, device=dev)
        down = nn.Linear(FFN, H, bias=False).to(dtype=dtype, device=dev)
        
        def swiglu_fn(act):
            return down(F.silu(gate(act)) * up(act))
        if compile_mode and dev == "cuda":
            try:
                swiglu_fn = torch.compile(swiglu_fn)
            except Exception as e:
                print(f"Warning: torch.compile(swiglu_fn) failed: {e}")
            
        t_h2d, t_cold, t_warm, t_d2h, t_e2e = measure_op(swiglu_fn, x, res_mode, iters, warmup)
        flops_ffn = 3.0 * (2.0 * B * S * H * FFN)
        gflops_ffn = flops_ffn / (t_warm * 1e6) if t_warm > 0 else 0
        bytes_ffn = (B * S * H + 3 * H * FFN + B * S * H) * (2 if dtype == torch.bfloat16 else 4)
        bw = bytes_ffn / (t_e2e * 1e6) if t_e2e > 0 else 0
        log_record(csv_file, "llama32_swiglu_ffn", 1, prec_str, res_mode, bytes_ffn, p_size_str, threads_or_sms,
                   0, t_h2d, t_cold, t_warm, t_d2h, t_e2e, bw, 0, gflops_ffn, 0)
                   
        # 7. LM Head + Top-K (Regime 4 Selection & Sort)
        lm_head = nn.Linear(H, V, bias=False).to(dtype=dtype, device=dev)
        def lm_head_topk(act):
            logits = lm_head(act)
            probs = F.softmax(logits, dim=-1)
            return torch.topk(probs, k=50, dim=-1)
        if compile_mode and dev == "cuda":
            try:
                lm_head_topk = torch.compile(lm_head_topk)
            except Exception as e:
                print(f"Warning: torch.compile(lm_head_topk) failed: {e}")
            
        t_h2d, t_cold, t_warm, t_d2h, t_e2e = measure_op(lm_head_topk, x, res_mode, iters, warmup)
        bytes_lm = (B * S * H + H * V + B * S * 50 * 2) * (2 if dtype == torch.bfloat16 else 4)
        bw = bytes_lm / (t_e2e * 1e6) if t_e2e > 0 else 0
        log_record(csv_file, "llama32_lmhead_topk", 4, prec_str, res_mode, bytes_lm, p_size_str, threads_or_sms,
                   0, t_h2d, t_cold, t_warm, t_d2h, t_e2e, bw, 0, 0, 0)

def run_kv_cache_offload_bench(res_mode, csv_file=None, iters=10, warmup=3, so_path=None):
    import torch
    import torch.nn.functional as F

    dev = "cuda" if "gpu" in res_mode else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    prec_str = "bf16" if dtype == torch.bfloat16 else "fp32"
    elem_bytes = 2 if dtype == torch.bfloat16 else 4

    print(f"\n==========================================================================================")
    print(f" Dedicated Zero-Copy KV-Cache Offload Benchmark (1GB to 128GB in Grace LPDDR5X) | Mode={res_mode}")
    print(f"==========================================================================================")

    B = 1
    N_heads = 32
    N_kv = 8
    D_head = 128

    kv_targets = [
        ("1GB", 262144),
        ("4GB", 1048576),
        ("16GB", 4194304),
        ("32GB", 8388608),
        ("64GB", 16777216),
        ("128GB", 33554432)
    ]

    for label, S_kv in kv_targets:
        bytes_kv = 2 * B * N_kv * S_kv * D_head * elem_bytes
        p_size_str = f"kv_offload_{label}_S{S_kv}"

        try:
            q = torch.randn((B, N_heads, 1, D_head), dtype=dtype, device=dev)
            k = torch.randn((B, N_kv, S_kv, D_head), dtype=dtype, device=dev)
            v = torch.randn((B, N_kv, S_kv, D_head), dtype=dtype, device=dev)

            def attn_step(q_in):
                return F.scaled_dot_product_attention(q_in, k, v, enable_gqa=True)

            t_h2d, t_cold, t_warm, t_d2h, t_e2e = measure_op(attn_step, q, res_mode, iters, warmup)
            bw = (bytes_kv / (t_e2e * 1e6)) if t_e2e > 0 else 0
            flops = 4.0 * B * N_heads * 1 * S_kv * D_head
            gflops = (flops / (t_warm * 1e6)) if t_warm > 0 else 0

            log_record(csv_file, "llama3_kv_cache_offload", 2, prec_str, res_mode, bytes_kv, p_size_str,
                       132 if dev == "cuda" else 72, 0, t_h2d, t_cold, t_warm, t_d2h, t_e2e, bw, bw, gflops, 0, "ok")
            del q, k, v
            if dev == "cuda":
                torch.cuda.empty_cache()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(f"OOM/Error at KV cache size {label}: {e}")
            log_record(csv_file, "llama3_kv_cache_offload", 2, prec_str, res_mode, bytes_kv, p_size_str,
                       132 if dev == "cuda" else 72, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "oom-skip")

def run_coop_kv_partition_bench(csv_file=None, iters=5, warmup=2, so_path=None):
    import torch
    import torch.nn.functional as F
    import concurrent.futures

    print(f"\n==========================================================================================")
    print(f" GH200 Cooperative KV-Cache Attention Partitioning (128GB working set) | CPU + GPU Split")
    print(f"==========================================================================================")

    if so_path and os.path.exists(so_path):
        setup_allocator("gpu_zero_copy", os.path.abspath(so_path))

    B = 1
    N_heads = 32
    N_kv = 8
    D_head = 128
    S_kv = 33554432
    elem_bytes = 2
    bytes_kv = 2 * B * N_kv * S_kv * D_head * elem_bytes

    alphas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]

    # Check already logged alphas in CSV
    done_alphas = set()
    if csv_file and os.path.exists(csv_file):
        with open(csv_file, "r") as f:
            for line in f:
                if "coop_partition_kv_offload" in line:
                    for part in line.split(","):
                        if "alpha=" in part:
                            try:
                                a_str = part.strip('"').split("_")[0].split("=")[1]
                                done_alphas.add(round(float(a_str), 2))
                            except Exception:
                                pass

    for alpha in alphas:
        if round(alpha, 2) in done_alphas:
            print(f"[Partition KV 128GB] alpha={alpha:.2f} already completed in CSV, skipping.")
            continue

        S_cpu = int(alpha * S_kv)
        S_gpu = S_kv - S_cpu
        p_size_str = f"alpha={alpha:.2f}_Scpu={S_cpu}_Sgpu={S_gpu}"

        try:
            q_cpu = k_cpu = v_cpu = None
            q_gpu = k_gpu = v_gpu = None

            if S_cpu > 0:
                torch.set_num_threads(72)
                # Use float32 on CPU for native ARM NEON SIMD vectorization (matches Sweep 5 cpu_local: ~140 GB/s)
                q_cpu = torch.zeros((B, N_heads, 1, D_head), dtype=torch.float32, device="cpu")
                k_cpu = torch.zeros((B, N_kv, S_cpu, D_head), dtype=torch.float32, device="cpu")
                v_cpu = torch.zeros((B, N_kv, S_cpu, D_head), dtype=torch.float32, device="cpu")

            if S_gpu > 0:
                # GPU executes bfloat16 over NVLink-C2C via pluggable allocator
                q_gpu = torch.zeros((B, N_heads, 1, D_head), dtype=torch.bfloat16, device="cuda")
                k_gpu = torch.zeros((B, N_kv, S_gpu, D_head), dtype=torch.bfloat16, device="cuda")
                v_gpu = torch.zeros((B, N_kv, S_gpu, D_head), dtype=torch.bfloat16, device="cuda")

            def step():
                if S_gpu > 0:
                    # Asynchronous non-blocking CUDA launch on current stream
                    _ = F.scaled_dot_product_attention(q_gpu, k_gpu, v_gpu, enable_gqa=True)
                if S_cpu > 0:
                    # Run CPU attention directly across 72 Grace cores on main thread
                    torch.set_num_threads(72)
                    _ = F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu, enable_gqa=True)
                if S_gpu > 0:
                    torch.cuda.synchronize()

            for _ in range(warmup):
                step()

            t0 = time.perf_counter()
            for _ in range(iters):
                step()
            wall_ms = ((time.perf_counter() - t0) * 1000.0) / iters

            res_mode = "cpu_local" if alpha >= 0.999 else "gpu_zero_copy"
            threads_or_sms = 72 if alpha >= 0.999 else (132 if alpha <= 0.001 else int(alpha * 72 + (1.0 - alpha) * 132))
            prec_str = "fp32" if alpha >= 0.999 else "bf16"
            eff_bytes_kv = (2 * B * N_kv * S_cpu * D_head * 4) + (2 * B * N_kv * S_gpu * D_head * 2)

            bw = (eff_bytes_kv / (wall_ms * 1e6)) if wall_ms > 0 else 0
            flops = 4.0 * B * N_heads * 1 * S_kv * D_head
            gflops = (flops / (wall_ms * 1e6)) if wall_ms > 0 else 0

            log_record(csv_file, "coop_partition_kv_offload", 2, prec_str, res_mode, eff_bytes_kv, p_size_str,
                       threads_or_sms, 0.0, 0.0, 0.0, wall_ms, 0.0, wall_ms, bw, bw, gflops, 0.0, "ok")
            print(f"[Partition KV 128GB] alpha={alpha:.2f} ({res_mode}) | Wall={wall_ms:7.2f} ms | Eff BW={bw:6.2f} GB/s")

            del q_cpu, k_cpu, v_cpu, q_gpu, k_gpu, v_gpu
            torch.cuda.empty_cache()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(f"OOM/Error at alpha {alpha:.2f}: {e}")
            res_mode = "cpu_local" if alpha >= 0.999 else "gpu_zero_copy"
            threads_or_sms = 72 if alpha >= 0.999 else (132 if alpha <= 0.001 else int(alpha * 72 + (1.0 - alpha) * 132))
            prec_str = "fp32" if alpha >= 0.999 else "bf16"
            eff_bytes_kv = (2 * B * N_kv * S_cpu * D_head * 4) + (2 * B * N_kv * S_gpu * D_head * 2)
            log_record(csv_file, "coop_partition_kv_offload", 2, prec_str, res_mode, eff_bytes_kv, p_size_str,
                       threads_or_sms, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "oom-skip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama-3 Layer Benchmark")
    parser.add_argument("--model", type=str, default="llama3.2_1b", choices=["llama3.2_1b", "llama3.2_3b", "llama3_8b"])
    parser.add_argument("--residency", type=str, default="all", choices=["all", "cpu_local", "gpu_explicit", "gpu_zero_copy", "gpu_um_migrate"])
    parser.add_argument("--csv", type=str, default="llama32_layers_sweep.csv")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--so-path", type=str, default="bin/libtorch_c2c_allocator.so")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--kv-offload", action="store_true", help="Run dedicated KV cache offload sweep")
    parser.add_argument("--coop-kv-partition", action="store_true", help="Run cooperative CPU+GPU KV cache partition sweep")
    args = parser.parse_args()
    
    if args.csv and not os.path.exists(args.csv):
        with open(args.csv, "w") as f:
            f.write("workload,regime,precision,residency_mode,working_set_bytes,problem_size,threads_or_sms,"
                    "t_alloc_ms,t_h2d_ms,t_kernel_cold_ms,t_kernel_warm_ms,t_d2h_ms,t_e2e_ms,"
                    "eff_bw_gbps,c2c_bw_gbps,gflops,pct_peak,status\n")

    if args.coop_kv_partition:
        run_coop_kv_partition_bench(args.csv, args.iters, args.warmup, args.so_path)
    elif args.residency == "all":
        modes = ["cpu_local", "gpu_explicit", "gpu_zero_copy", "gpu_um_migrate"]
        for m in modes:
            cmd = [
                sys.executable, sys.argv[0],
                "--model", args.model,
                "--residency", m,
                "--csv", args.csv,
                "--iters", str(args.iters),
                "--warmup", str(args.warmup),
                "--so-path", args.so_path
            ]
            if args.compile:
                cmd.append("--compile")
            if args.kv_offload:
                cmd.append("--kv-offload")
            print(f"\n>>> Running subprocess for residency mode: {m}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Subprocess for {m} exited with code {e.returncode}")
    else:
        so_full = os.path.abspath(args.so_path)
        setup_allocator(args.residency, so_full)
        if args.kv_offload:
            run_kv_cache_offload_bench(args.residency, args.csv, args.iters, args.warmup, so_full)
        else:
            run_llama_layer_benchmarks(args.residency, args.model, args.csv, args.iters, args.warmup, so_full, args.compile)
