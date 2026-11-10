"""Native crypto kernels for C8 (AES-128-CTR) and C9 (SHA-256 multi-buffer).

CPU: OpenSSL libcrypto called from C++ with OpenMP (AES: each thread encrypts a contiguous
16-byte-aligned span with its own EVP context and the CTR counter advanced to that span;
SHA-256: messages distributed over threads, low-level SHA256_Init/Update/Final per message). OpenSSL selects
AES-NI/VAES, SHA-NI, ARMv8 crypto extensions at runtime; OPENSSL_ia32cap / OPENSSL_armcap
(set before the library initialises, via the config key "_env") disable them for the
"no hardware crypto" variant.
GPU: custom kernels (no well-known GPU crypto library exists):
  AES-128-CTR: T-table implementation (FIPS-197), Te0..Te3 + S-box in shared memory, one
               16-byte counter block per thread; round keys expanded on the host.
  SHA-256:     one message per thread (FIPS 180-4), full padding.
Python references (aes_ref / sha256 via hashlib) define the expected outputs; the GPU
kernels mirror aes_ref exactly and are validated against OpenSSL output byte for byte.
"""
from __future__ import annotations

from functools import lru_cache

from ._common import build_error_summary
from ..core import Unsupported

# ── reference AES-128 (T-table form, mirrors the CUDA kernel) ───────────────
SBOX = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
]
RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]


def _xt(b):
    b <<= 1
    return (b ^ 0x11B) if b & 0x100 else b


def te_tables():
    te0 = [((_xt(s) << 24) | (s << 16) | (s << 8) | (_xt(s) ^ s)) for s in SBOX]
    ror = lambda w, r: ((w >> r) | (w << (32 - r))) & 0xFFFFFFFF  # noqa: E731
    return te0, [ror(w, 8) for w in te0], [ror(w, 16) for w in te0], [ror(w, 24) for w in te0]


def expand_key(key: bytes) -> list[int]:
    w = [int.from_bytes(key[4 * i:4 * i + 4], "big") for i in range(4)]
    for i in range(4, 44):
        t = w[i - 1]
        if i % 4 == 0:
            t = ((t << 8) | (t >> 24)) & 0xFFFFFFFF
            t = (SBOX[t >> 24] << 24) | (SBOX[(t >> 16) & 255] << 16) | (SBOX[(t >> 8) & 255] << 8) | SBOX[t & 255]
            t ^= RCON[i // 4 - 1] << 24
        w.append(w[i - 4] ^ t)
    return w


def aes_encrypt_block_ref(block: bytes, rk: list[int]) -> bytes:
    """T-table AES-128 encryption of one block -- the exact algorithm of the CUDA kernel."""
    te0, te1, te2, te3 = te_tables()
    s = [int.from_bytes(block[4 * i:4 * i + 4], "big") ^ rk[i] for i in range(4)]
    for r in range(1, 10):
        s = [te0[s[i] >> 24] ^ te1[(s[(i + 1) % 4] >> 16) & 255] ^ te2[(s[(i + 2) % 4] >> 8) & 255]
             ^ te3[s[(i + 3) % 4] & 255] ^ rk[4 * r + i] for i in range(4)]
    out = [((SBOX[s[i] >> 24] << 24) | (SBOX[(s[(i + 1) % 4] >> 16) & 255] << 16)
            | (SBOX[(s[(i + 2) % 4] >> 8) & 255] << 8) | SBOX[s[(i + 3) % 4] & 255]) ^ rk[40 + i]
           for i in range(4)]
    return b"".join(x.to_bytes(4, "big") for x in out)


def aes_ctr_ref(key: bytes, iv: bytes, data: bytes) -> bytes:
    rk, ctr, out = expand_key(key), int.from_bytes(iv, "big"), bytearray()
    for i in range(0, len(data), 16):
        ks = aes_encrypt_block_ref(((ctr + i // 16) % (1 << 128)).to_bytes(16, "big"), rk)
        out += bytes(a ^ b for a, b in zip(data[i:i + 16], ks))
    return bytes(out)


# ── native sources ──────────────────────────────────────────────────────────
_CPU_SRC = r"""
#include <torch/extension.h>
#include <omp.h>
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <openssl/crypto.h>
#include <cstring>

static void ctr_add(const uint8_t iv[16], uint64_t blocks, uint8_t out[16]) {
  // 128-bit big-endian iv + blocks
  std::memcpy(out, iv, 16);
  uint64_t carry = blocks;
  for (int i = 15; i >= 0 && carry; --i) {
    uint64_t v = (uint64_t)out[i] + (carry & 0xFF);
    out[i] = (uint8_t)v;
    carry = (carry >> 8) + (v >> 8);
  }
}

void aes128_ctr_cpu(torch::Tensor key, torch::Tensor iv, torch::Tensor in, torch::Tensor out) {
  const uint8_t* k = key.data_ptr<uint8_t>();
  const uint8_t* v = iv.data_ptr<uint8_t>();
  const uint8_t* src = in.data_ptr<uint8_t>();
  uint8_t* dst = out.data_ptr<uint8_t>();
  const int64_t n = in.numel(), nblk = (n + 15) / 16;
  #pragma omp parallel
  {
    const int t = omp_get_thread_num(), T = omp_get_num_threads();
    const int64_t b0 = nblk * t / T, b1 = nblk * (t + 1) / T;
    if (b1 > b0) {
      uint8_t ivt[16];
      ctr_add(v, (uint64_t)b0, ivt);
      EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
      EVP_EncryptInit_ex(ctx, EVP_aes_128_ctr(), nullptr, k, ivt);
      int outl = 0;
      const int64_t off = b0 * 16, len = std::min<int64_t>(n, b1 * 16) - off;
      EVP_EncryptUpdate(ctx, dst + off, &outl, src + off, (int)len);
      EVP_CIPHER_CTX_free(ctx);
    }
  }
}

// Low-level SHA256_Init/Update/Final (deprecated in OpenSSL 3 but supported): the one-shot
// SHA256() goes through the provider layer (algorithm fetch + global locks per call), which
// measured 82 MB/s/core on EPYC 7402P with 24 threads -- lock contention, not hashing.
// The low-level API dispatches to the same sha256_block_data_order (SHA-NI / ARMv8 SHA2 by
// OPENSSL_ia32cap/armcap) without per-call provider overhead.
void sha256_multi_cpu(torch::Tensor msgs, int64_t msg_len, torch::Tensor digests) {
  const uint8_t* m = msgs.data_ptr<uint8_t>();
  uint8_t* d = digests.data_ptr<uint8_t>();
  const int64_t count = msgs.numel() / msg_len;
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < count; ++i) {
    SHA256_CTX c;
    SHA256_Init(&c);
    SHA256_Update(&c, m + i * msg_len, (size_t)msg_len);
    SHA256_Final(d + i * 32, &c);
  }
}

std::string openssl_version() { return OpenSSL_version(OPENSSL_VERSION); }
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void aes128_ctr_kernel(const uint32_t* __restrict__ tables, const uint32_t* __restrict__ rk_g,
                                  const uint8_t* __restrict__ iv, const uint4* __restrict__ in,
                                  uint4* __restrict__ out, int64_t nblk) {
  __shared__ uint32_t Te[4][256];
  __shared__ uint8_t S[256];
  __shared__ uint32_t rk[44];
  for (int i = threadIdx.x; i < 1024; i += blockDim.x) Te[i >> 8][i & 255] = tables[i];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) S[i] = (uint8_t)tables[1024 + i];
  for (int i = threadIdx.x; i < 44; i += blockDim.x) rk[i] = rk_g[i];
  __syncthreads();
  uint64_t ivh = 0, ivl = 0;
  for (int i = 0; i < 8; ++i) { ivh = (ivh << 8) | iv[i]; ivl = (ivl << 8) | iv[8 + i]; }
  for (int64_t b = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; b < nblk; b += (int64_t)gridDim.x * blockDim.x) {
    uint64_t lo = ivl + (uint64_t)b, hi = ivh + (lo < ivl ? 1 : 0);
    uint32_t s0 = (uint32_t)(hi >> 32) ^ rk[0], s1 = (uint32_t)hi ^ rk[1];
    uint32_t s2 = (uint32_t)(lo >> 32) ^ rk[2], s3 = (uint32_t)lo ^ rk[3];
    #pragma unroll
    for (int r = 1; r < 10; ++r) {
      uint32_t t0 = Te[0][s0 >> 24] ^ Te[1][(s1 >> 16) & 255] ^ Te[2][(s2 >> 8) & 255] ^ Te[3][s3 & 255] ^ rk[4 * r];
      uint32_t t1 = Te[0][s1 >> 24] ^ Te[1][(s2 >> 16) & 255] ^ Te[2][(s3 >> 8) & 255] ^ Te[3][s0 & 255] ^ rk[4 * r + 1];
      uint32_t t2 = Te[0][s2 >> 24] ^ Te[1][(s3 >> 16) & 255] ^ Te[2][(s0 >> 8) & 255] ^ Te[3][s1 & 255] ^ rk[4 * r + 2];
      uint32_t t3 = Te[0][s3 >> 24] ^ Te[1][(s0 >> 16) & 255] ^ Te[2][(s1 >> 8) & 255] ^ Te[3][s2 & 255] ^ rk[4 * r + 3];
      s0 = t0; s1 = t1; s2 = t2; s3 = t3;
    }
    uint32_t c[4];
    c[0] = ((uint32_t)S[s0 >> 24] << 24 | (uint32_t)S[(s1 >> 16) & 255] << 16 | (uint32_t)S[(s2 >> 8) & 255] << 8 | S[s3 & 255]) ^ rk[40];
    c[1] = ((uint32_t)S[s1 >> 24] << 24 | (uint32_t)S[(s2 >> 16) & 255] << 16 | (uint32_t)S[(s3 >> 8) & 255] << 8 | S[s0 & 255]) ^ rk[41];
    c[2] = ((uint32_t)S[s2 >> 24] << 24 | (uint32_t)S[(s3 >> 16) & 255] << 16 | (uint32_t)S[(s0 >> 8) & 255] << 8 | S[s1 & 255]) ^ rk[42];
    c[3] = ((uint32_t)S[s3 >> 24] << 24 | (uint32_t)S[(s0 >> 16) & 255] << 16 | (uint32_t)S[(s1 >> 8) & 255] << 8 | S[s2 & 255]) ^ rk[43];
    uint4 p = in[b];   // keystream (big-endian words) XOR plaintext (little-endian loads)
    p.x ^= __byte_perm(c[0], 0, 0x0123); p.y ^= __byte_perm(c[1], 0, 0x0123);
    p.z ^= __byte_perm(c[2], 0, 0x0123); p.w ^= __byte_perm(c[3], 0, 0x0123);
    out[b] = p;
  }
}

__constant__ uint32_t K256[64] = {
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
__device__ void sha256_block(uint32_t h[8], const uint32_t w_in[16]) {
  uint32_t w[64];
  #pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = w_in[i];
  #pragma unroll
  for (int i = 16; i < 64; ++i) {
    uint32_t s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3);
    uint32_t s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10);
    w[i] = w[i-16] + s0 + w[i-7] + s1;
  }
  uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
  #pragma unroll
  for (int i = 0; i < 64; ++i) {
    uint32_t t1 = hh + (rotr(e,6)^rotr(e,11)^rotr(e,25)) + ((e&f)^(~e&g)) + K256[i] + w[i];
    uint32_t t2 = (rotr(a,2)^rotr(a,13)^rotr(a,22)) + ((a&b)^(a&c)^(b&c));
    hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
  }
  h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
}
__global__ void sha256_multi_kernel(const uint8_t* __restrict__ msgs, int64_t msg_len, int64_t count,
                                    uint8_t* __restrict__ digests) {
  for (int64_t m = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; m < count; m += (int64_t)gridDim.x * blockDim.x) {
    const uint8_t* p = msgs + m * msg_len;
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint32_t w[16];
    int64_t full = msg_len / 64;
    for (int64_t blk = 0; blk < full; ++blk) {
      for (int i = 0; i < 16; ++i) { const uint8_t* q = p + blk*64 + 4*i; w[i] = (uint32_t)q[0]<<24|(uint32_t)q[1]<<16|(uint32_t)q[2]<<8|q[3]; }
      sha256_block(h, w);
    }
    // padding: remaining bytes + 0x80 + zeros + 64-bit big-endian bit length (1 or 2 blocks)
    uint8_t tail[128];
    int rem = (int)(msg_len - full * 64);
    for (int i = 0; i < rem; ++i) tail[i] = p[full * 64 + i];
    tail[rem] = 0x80;
    int tl = (rem + 1 + 8 <= 64) ? 64 : 128;
    for (int i = rem + 1; i < tl - 8; ++i) tail[i] = 0;
    uint64_t bits = (uint64_t)msg_len * 8;
    for (int i = 0; i < 8; ++i) tail[tl - 1 - i] = (uint8_t)(bits >> (8 * i));
    for (int blk = 0; blk < tl / 64; ++blk) {
      for (int i = 0; i < 16; ++i) { const uint8_t* q = tail + blk*64 + 4*i; w[i] = (uint32_t)q[0]<<24|(uint32_t)q[1]<<16|(uint32_t)q[2]<<8|q[3]; }
      sha256_block(h, w);
    }
    uint8_t* d = digests + m * 32;
    for (int i = 0; i < 8; ++i) { d[4*i]=h[i]>>24; d[4*i+1]=h[i]>>16; d[4*i+2]=h[i]>>8; d[4*i+3]=h[i]; }
  }
}

void aes128_ctr_cuda(torch::Tensor tables, torch::Tensor rk, torch::Tensor iv, torch::Tensor in, torch::Tensor out, int64_t grid) {
  const int64_t nblk = in.numel() / 16;
  aes128_ctr_kernel<<<(int)grid, 256>>>((const uint32_t*)tables.data_ptr<int32_t>(), (const uint32_t*)rk.data_ptr<int32_t>(),
      iv.data_ptr<uint8_t>(), (const uint4*)in.data_ptr<uint8_t>(), (uint4*)out.data_ptr<uint8_t>(), nblk);
}
void sha256_multi_cuda(torch::Tensor msgs, int64_t msg_len, torch::Tensor digests, int64_t grid) {
  sha256_multi_kernel<<<(int)grid, 128>>>(msgs.data_ptr<uint8_t>(), msg_len, msgs.numel() / msg_len, digests.data_ptr<uint8_t>());
}
"""


@lru_cache(maxsize=1)
def cpu():
    from torch.utils.cpp_extension import load_inline
    from ._cpusort import _ensure_ninja_on_path
    _ensure_ninja_on_path()
    try:
        return load_inline(name="bench_crypto_cpu_v2", cpp_sources=_CPU_SRC,
                           functions=["aes128_ctr_cpu", "sha256_multi_cpu", "openssl_version"],
                           extra_cflags=["-O3", "-march=native", "-fopenmp", "-Wno-deprecated-declarations"],
                           extra_ldflags=["-fopenmp", "-lcrypto"], verbose=False)
    except Exception as e:
        raise Unsupported(f"crypto CPU extension build failed (needs libssl-dev): {build_error_summary(e)}") from e


@lru_cache(maxsize=1)
def cuda():
    from torch.utils.cpp_extension import load_inline
    from ._cpusort import _ensure_ninja_on_path
    _ensure_ninja_on_path()
    try:
        from ._common import cuda_declarations
        fns = ["aes128_ctr_cuda", "sha256_multi_cuda"]
        return load_inline(name="bench_crypto_cuda_v2", cpp_sources=cuda_declarations(_CUDA_SRC, fns),
                           cuda_sources=_CUDA_SRC, functions=fns, extra_cuda_cflags=["-O3"], verbose=False)
    except Exception as e:
        raise Unsupported(f"crypto CUDA extension build failed: {build_error_summary(e)}") from e


def tables_tensor():
    """Te0..Te3 (1024 words) followed by the S-box (256 words), int32 for torch."""
    import torch
    te = te_tables()
    words = [w for t in te for w in t] + SBOX
    return torch.tensor([w - (1 << 32) if w >= 1 << 31 else w for w in words], dtype=torch.int32)


def round_keys_tensor(key: bytes):
    import torch
    return torch.tensor([w - (1 << 32) if w >= 1 << 31 else w for w in expand_key(key)], dtype=torch.int32)
