"""C8 — AES-128-CTR encryption of a buffer   (compute-bound, integer; fixed-function on CPU)

Equal work: CTR-mode encryption (FIPS-197 AES-128, NIST SP 800-38A CTR, 128-bit big-endian
counter) of the same N-byte plaintext with the same key/IV; full ciphertext materialised.
CPU: OpenSSL libcrypto EVP_aes_128_ctr from C++/OpenMP (AES-NI/VAES or ARMv8 crypto chosen
     by OpenSSL). Variant `sw`: hardware AES disabled via OPENSSL_ia32cap / OPENSSL_armcap
     (applied to the process environment before libcrypto initialises) -> table-based AES.
GPU: custom T-table kernel (_crypto_ext.py; no well-known GPU library).
Validation (exact): NIST SP 800-38A F.5.1 vector, then the FULL ciphertext compared byte for
byte with python `cryptography` (OpenSSL) on the CPU -- so CPU and GPU outputs are identical.
Metric: GB/s of plaintext.
"""
from __future__ import annotations

import torch

from ..core import Ctx, Unsupported, Workload
from ..validate import exact
from . import _crypto_ext as X

SIZES = [64 << 20, 1 << 30, 4 << 30]
KEY = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
IV = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff")
NIST_PT = bytes.fromhex("6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e51"
                        "30c81c46a35ce411e5fbc1191a0a52eff69f2445df4f9b17ad2b417be66c3710")
NIST_CT = bytes.fromhex("874d6191b620e3261bef6864990db6ce9806f66b7970fdff8617187bb9fffdff"
                        "5ae4df3edbd5d35e5b4f09020db03eab1e031dda2fbe03d1792170a0f3009cee")
# OpenSSL capability masks: x86 clear AES-NI+PCLMULQDQ (word 1) and SHA-NI+VAES (word 2); ARM: all off
NO_HW_CRYPTO_ENV = {"OPENSSL_ia32cap": "~0x200000200000000:~0x20020000000", "OPENSSL_armcap": "0"}


def _u8(b: bytes, device):
    return torch.frombuffer(bytearray(b), dtype=torch.uint8).to(device)


class AES(Workload):
    id, name, bound, field_ = "C8", "aes128ctr", "C", "Security"

    def configs(self, device_class=None):
        out = [{"precision": "uint8", "variant": "base", "bytes": n} for n in SIZES]
        if device_class in (None, "cpu"):
            out += [{"precision": "uint8", "variant": "sw", "bytes": n, "_env": NO_HW_CRYPTO_ENV} for n in SIZES]
        return out

    def working_set_bytes(self, p):
        return 2 * p["bytes"]

    def work(self, p):
        return {"flops": 0.0, "bytes": 2.0 * p["bytes"], "metric_name": "GB/s",
                "metric_units_per_call": p["bytes"] / 1e9}

    def library(self, ctx):
        if ctx.device_class == "cpu":
            return f"{X.cpu().openssl_version()}/EVP_aes_128_ctr+OpenMP"
        return "custom CUDA T-table AES-128-CTR"

    def setup(self, p, ctx: Ctx):
        n = p["bytes"]
        if p["variant"] == "sw" and ctx.device_class != "cpu":
            raise Unsupported("software-AES variant is CPU-only")
        pt = torch.empty(n, dtype=torch.uint8, device=ctx.device)
        pt.fill_(0)
        pt.copy_(torch.randint(0, 256, (n,), generator=ctx.generator(1), dtype=torch.uint8, device=ctx.device))
        out = torch.empty_like(pt)
        key, iv = _u8(KEY, ctx.device), _u8(IV, ctx.device)
        if ctx.device_class == "cpu":
            ext = X.cpu()
            fn = lambda a, b: ext.aes128_ctr_cpu(key, iv, a, b)  # noqa: E731
        else:
            ext = X.cuda()
            tables, rk = X.tables_tensor().to(ctx.device), X.round_keys_tensor(KEY).to(ctx.device)
            grid = torch.cuda.get_device_properties(ctx.gpu_index).multi_processor_count * 8
            fn = lambda a, b: ext.aes128_ctr_cuda(tables, rk, iv, a, b, grid)  # noqa: E731
        return {"pt": pt, "out": out, "fn": fn, "device": ctx.device}

    def call(self, s):
        s["fn"](s["pt"], s["out"])

    def validate(self, s, ctx):
        dev = s["device"]
        nist_in, nist_out = _u8(NIST_PT, dev), torch.empty(len(NIST_PT), dtype=torch.uint8, device=dev)
        s["fn"](nist_in, nist_out)
        nist_ok = bytes(nist_out.cpu().numpy()) == NIST_CT
        self.call(s)
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        ref = Cipher(algorithms.AES(KEY), modes.CTR(IV)).encryptor().update(s["pt"].cpu().numpy().tobytes())
        got = s["out"].cpu().numpy().tobytes()
        mism = 0 if got == ref else sum(1 for i in range(0, len(ref), 1 << 20) if got[i:i + (1 << 20)] != ref[i:i + (1 << 20)])
        return exact(nist_ok and mism == 0, mism + (0 if nist_ok else 1))

    def transfers(self, s):
        return [s["pt"]], [s["out"]]


WORKLOAD = AES()
