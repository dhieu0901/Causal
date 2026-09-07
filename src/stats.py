"""Statistics for the break-even study: McNemar power, bootstrap CI for k*."""
from __future__ import annotations
import numpy as np
from scipy import stats


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar on the discordant cells."""
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5)))


def mcnemar_power(n, p_a, p_b, rho=0.6, alpha=0.05, sims=4000, seed=0):
    """Power of the paired test at sample size n.

    rho is the correlation between the two conditions' per-item correctness;
    paired designs on the same items are strongly correlated, which is exactly
    why the paired test buys power over two independent samples.
    """
    rng = np.random.default_rng(seed)
    # Gaussian copula -> correlated Bernoulli pair with the requested marginals
    cov = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(cov)
    za, zb = stats.norm.ppf(p_a), stats.norm.ppf(p_b)
    hits = 0
    for _ in range(sims):
        z = rng.standard_normal((n, 2)) @ L.T
        a, b = z[:, 0] < za, z[:, 1] < zb
        n01 = int(np.sum(a & ~b))
        n10 = int(np.sum(~a & b))
        if mcnemar_exact_p(n01, n10) < alpha:
            hits += 1
    return hits / sims


def break_even_k(ks, accs, floor):
    """Linear interpolation of where the accuracy curve crosses the floor."""
    ks, accs = np.asarray(ks, float), np.asarray(accs, float)
    for i in range(len(ks) - 1):
        a0, a1 = accs[i], accs[i + 1]
        if (a0 - floor) * (a1 - floor) <= 0 and a0 != a1:
            return float(ks[i] + (a0 - floor) / (a0 - a1) * (ks[i + 1] - ks[i]))
    if accs[-1] > floor:                      # never crosses inside the measured range
        slope = (accs[0] - accs[-1]) / (ks[-1] - ks[0])
        return float(ks[-1] + (accs[-1] - floor) / slope) if slope > 0 else np.nan
    return float(ks[0])


def bootstrap_break_even(correct_by_k, floor_correct, reps=10000, seed=0):
    """Percentile CI for k*, resampling items (not observations) to keep pairing.

    correct_by_k: dict k -> 0/1 array over the SAME items, in the same order.
    floor_correct: 0/1 array over those items for the no-graph condition.
    """
    rng = np.random.default_rng(seed)
    ks = sorted(correct_by_k)
    mats = np.stack([np.asarray(correct_by_k[k], float) for k in ks])
    floor = np.asarray(floor_correct, float)
    n = mats.shape[1]
    out = np.empty(reps)
    for r in range(reps):
        idx = rng.integers(0, n, n)
        out[r] = break_even_k(ks, mats[:, idx].mean(axis=1) * 100, floor[idx].mean() * 100)
    out = out[np.isfinite(out)]
    return {
        "k_star": float(break_even_k(ks, mats.mean(axis=1) * 100, floor.mean() * 100)),
        "lo": float(np.percentile(out, 2.5)),
        "hi": float(np.percentile(out, 97.5)),
        "n_valid": int(out.size),
    }


def resolution(n, p=0.70, slope_pp_per_edge=11.8, alpha=0.05):
    """How tightly a sample of n localises k*, given the curve's slope."""
    se = np.sqrt(p * (1 - p) / n) * 100
    half = stats.norm.ppf(1 - alpha / 2) * se
    return {"se_pp": se, "ci_half_pp": half, "k_resolution": half / slope_pp_per_edge}
