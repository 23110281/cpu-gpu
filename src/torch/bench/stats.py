"""Summary statistics for timing samples (docs/methodology.md §5)."""
from __future__ import annotations

import numpy as np

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20101  # fixed so CIs are reproducible
UNSTABLE_CV = 0.05


def summarize(samples_ms) -> dict:
    x = np.asarray(samples_ms, dtype=np.float64)
    if x.size == 0:
        raise ValueError("no samples")
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    mean = x.mean()
    cv = float(x.std(ddof=1) / mean) if x.size > 1 and mean > 0 else 0.0
    lo, hi = bootstrap_ci_median(x)
    return {
        "n_samples": int(x.size),
        "median_ms": float(med),
        "iqr_ms": float(q3 - q1),
        "min_ms": float(x.min()),
        "cv": cv,
        "ci95_lo_ms": lo,
        "ci95_hi_ms": hi,
    }


def bootstrap_ci_median(x, alpha: float = 0.05, resamples: int = BOOTSTRAP_RESAMPLES,
                        seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """Percentile bootstrap CI of the median."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        v = float(x[0])
        return v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(resamples, x.size))
    meds = np.median(x[idx], axis=1)
    lo, hi = np.percentile(meds, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def ratio_ci(a_ms, b_ms, alpha: float = 0.05, resamples: int = BOOTSTRAP_RESAMPLES,
             seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    """Speed-up of B over A = median(a)/median(b), with an independent-bootstrap CI."""
    a = np.asarray(a_ms, dtype=np.float64)
    b = np.asarray(b_ms, dtype=np.float64)
    rng = np.random.default_rng(seed)
    ma = np.median(a[rng.integers(0, a.size, size=(resamples, a.size))], axis=1)
    mb = np.median(b[rng.integers(0, b.size, size=(resamples, b.size))], axis=1)
    r = ma / mb
    lo, hi = np.percentile(r, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.median(a) / np.median(b)), float(lo), float(hi)
